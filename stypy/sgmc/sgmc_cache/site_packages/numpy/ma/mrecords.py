
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: ''':mod:`numpy.ma..mrecords`
2: 
3: Defines the equivalent of :class:`numpy.recarrays` for masked arrays,
4: where fields can be accessed as attributes.
5: Note that :class:`numpy.ma.MaskedArray` already supports structured datatypes
6: and the masking of individual fields.
7: 
8: .. moduleauthor:: Pierre Gerard-Marchant
9: 
10: '''
11: from __future__ import division, absolute_import, print_function
12: 
13: #  We should make sure that no field is called '_mask','mask','_fieldmask',
14: #  or whatever restricted keywords.  An idea would be to no bother in the
15: #  first place, and then rename the invalid fields with a trailing
16: #  underscore. Maybe we could just overload the parser function ?
17: 
18: import sys
19: import warnings
20: 
21: import numpy as np
22: import numpy.core.numerictypes as ntypes
23: from numpy.compat import basestring
24: from numpy import (
25:         bool_, dtype, ndarray, recarray, array as narray
26:         )
27: from numpy.core.records import (
28:         fromarrays as recfromarrays, fromrecords as recfromrecords
29:         )
30: 
31: _byteorderconv = np.core.records._byteorderconv
32: _typestr = ntypes._typestr
33: 
34: import numpy.ma as ma
35: from numpy.ma import (
36:         MAError, MaskedArray, masked, nomask, masked_array, getdata,
37:         getmaskarray, filled
38:         )
39: 
40: _check_fill_value = ma.core._check_fill_value
41: 
42: 
43: __all__ = [
44:     'MaskedRecords', 'mrecarray', 'fromarrays', 'fromrecords',
45:     'fromtextfile', 'addfield',
46:     ]
47: 
48: reserved_fields = ['_data', '_mask', '_fieldmask', 'dtype']
49: 
50: 
51: def _getformats(data):
52:     '''
53:     Returns the formats of arrays in arraylist as a comma-separated string.
54: 
55:     '''
56:     if hasattr(data, 'dtype'):
57:         return ",".join([desc[1] for desc in data.dtype.descr])
58: 
59:     formats = ''
60:     for obj in data:
61:         obj = np.asarray(obj)
62:         formats += _typestr[obj.dtype.type]
63:         if issubclass(obj.dtype.type, ntypes.flexible):
64:             formats += repr(obj.itemsize)
65:         formats += ','
66:     return formats[:-1]
67: 
68: 
69: def _checknames(descr, names=None):
70:     '''
71:     Checks that field names ``descr`` are not reserved keywords.
72: 
73:     If this is the case, a default 'f%i' is substituted.  If the argument
74:     `names` is not None, updates the field names to valid names.
75: 
76:     '''
77:     ndescr = len(descr)
78:     default_names = ['f%i' % i for i in range(ndescr)]
79:     if names is None:
80:         new_names = default_names
81:     else:
82:         if isinstance(names, (tuple, list)):
83:             new_names = names
84:         elif isinstance(names, str):
85:             new_names = names.split(',')
86:         else:
87:             raise NameError("illegal input names %s" % repr(names))
88:         nnames = len(new_names)
89:         if nnames < ndescr:
90:             new_names += default_names[nnames:]
91:     ndescr = []
92:     for (n, d, t) in zip(new_names, default_names, descr.descr):
93:         if n in reserved_fields:
94:             if t[0] in reserved_fields:
95:                 ndescr.append((d, t[1]))
96:             else:
97:                 ndescr.append(t)
98:         else:
99:             ndescr.append((n, t[1]))
100:     return np.dtype(ndescr)
101: 
102: 
103: def _get_fieldmask(self):
104:     mdescr = [(n, '|b1') for n in self.dtype.names]
105:     fdmask = np.empty(self.shape, dtype=mdescr)
106:     fdmask.flat = tuple([False] * len(mdescr))
107:     return fdmask
108: 
109: 
110: class MaskedRecords(MaskedArray, object):
111:     '''
112: 
113:     Attributes
114:     ----------
115:     _data : recarray
116:         Underlying data, as a record array.
117:     _mask : boolean array
118:         Mask of the records. A record is masked when all its fields are
119:         masked.
120:     _fieldmask : boolean recarray
121:         Record array of booleans, setting the mask of each individual field
122:         of each record.
123:     _fill_value : record
124:         Filling values for each field.
125: 
126:     '''
127: 
128:     def __new__(cls, shape, dtype=None, buf=None, offset=0, strides=None,
129:                 formats=None, names=None, titles=None,
130:                 byteorder=None, aligned=False,
131:                 mask=nomask, hard_mask=False, fill_value=None, keep_mask=True,
132:                 copy=False,
133:                 **options):
134: 
135:         self = recarray.__new__(cls, shape, dtype=dtype, buf=buf, offset=offset,
136:                                 strides=strides, formats=formats, names=names,
137:                                 titles=titles, byteorder=byteorder,
138:                                 aligned=aligned,)
139: 
140:         mdtype = ma.make_mask_descr(self.dtype)
141:         if mask is nomask or not np.size(mask):
142:             if not keep_mask:
143:                 self._mask = tuple([False] * len(mdtype))
144:         else:
145:             mask = np.array(mask, copy=copy)
146:             if mask.shape != self.shape:
147:                 (nd, nm) = (self.size, mask.size)
148:                 if nm == 1:
149:                     mask = np.resize(mask, self.shape)
150:                 elif nm == nd:
151:                     mask = np.reshape(mask, self.shape)
152:                 else:
153:                     msg = "Mask and data not compatible: data size is %i, " + \
154:                           "mask size is %i."
155:                     raise MAError(msg % (nd, nm))
156:                 copy = True
157:             if not keep_mask:
158:                 self.__setmask__(mask)
159:                 self._sharedmask = True
160:             else:
161:                 if mask.dtype == mdtype:
162:                     _mask = mask
163:                 else:
164:                     _mask = np.array([tuple([m] * len(mdtype)) for m in mask],
165:                                      dtype=mdtype)
166:                 self._mask = _mask
167:         return self
168: 
169:     def __array_finalize__(self, obj):
170:         # Make sure we have a _fieldmask by default
171:         _mask = getattr(obj, '_mask', None)
172:         if _mask is None:
173:             objmask = getattr(obj, '_mask', nomask)
174:             _dtype = ndarray.__getattribute__(self, 'dtype')
175:             if objmask is nomask:
176:                 _mask = ma.make_mask_none(self.shape, dtype=_dtype)
177:             else:
178:                 mdescr = ma.make_mask_descr(_dtype)
179:                 _mask = narray([tuple([m] * len(mdescr)) for m in objmask],
180:                                dtype=mdescr).view(recarray)
181:         # Update some of the attributes
182:         _dict = self.__dict__
183:         _dict.update(_mask=_mask)
184:         self._update_from(obj)
185:         if _dict['_baseclass'] == ndarray:
186:             _dict['_baseclass'] = recarray
187:         return
188: 
189:     def _getdata(self):
190:         '''
191:         Returns the data as a recarray.
192: 
193:         '''
194:         return ndarray.view(self, recarray)
195: 
196:     _data = property(fget=_getdata)
197: 
198:     def _getfieldmask(self):
199:         '''
200:         Alias to mask.
201: 
202:         '''
203:         return self._mask
204: 
205:     _fieldmask = property(fget=_getfieldmask)
206: 
207:     def __len__(self):
208:         '''
209:         Returns the length
210: 
211:         '''
212:         # We have more than one record
213:         if self.ndim:
214:             return len(self._data)
215:         # We have only one record: return the nb of fields
216:         return len(self.dtype)
217: 
218:     def __getattribute__(self, attr):
219:         try:
220:             return object.__getattribute__(self, attr)
221:         except AttributeError:
222:             # attr must be a fieldname
223:             pass
224:         fielddict = ndarray.__getattribute__(self, 'dtype').fields
225:         try:
226:             res = fielddict[attr][:2]
227:         except (TypeError, KeyError):
228:             raise AttributeError("record array has no attribute %s" % attr)
229:         # So far, so good
230:         _localdict = ndarray.__getattribute__(self, '__dict__')
231:         _data = ndarray.view(self, _localdict['_baseclass'])
232:         obj = _data.getfield(*res)
233:         if obj.dtype.fields:
234:             raise NotImplementedError("MaskedRecords is currently limited to"
235:                                       "simple records.")
236:         # Get some special attributes
237:         # Reset the object's mask
238:         hasmasked = False
239:         _mask = _localdict.get('_mask', None)
240:         if _mask is not None:
241:             try:
242:                 _mask = _mask[attr]
243:             except IndexError:
244:                 # Couldn't find a mask: use the default (nomask)
245:                 pass
246:             hasmasked = _mask.view((np.bool, (len(_mask.dtype) or 1))).any()
247:         if (obj.shape or hasmasked):
248:             obj = obj.view(MaskedArray)
249:             obj._baseclass = ndarray
250:             obj._isfield = True
251:             obj._mask = _mask
252:             # Reset the field values
253:             _fill_value = _localdict.get('_fill_value', None)
254:             if _fill_value is not None:
255:                 try:
256:                     obj._fill_value = _fill_value[attr]
257:                 except ValueError:
258:                     obj._fill_value = None
259:         else:
260:             obj = obj.item()
261:         return obj
262: 
263:     def __setattr__(self, attr, val):
264:         '''
265:         Sets the attribute attr to the value val.
266: 
267:         '''
268:         # Should we call __setmask__ first ?
269:         if attr in ['mask', 'fieldmask']:
270:             self.__setmask__(val)
271:             return
272:         # Create a shortcut (so that we don't have to call getattr all the time)
273:         _localdict = object.__getattribute__(self, '__dict__')
274:         # Check whether we're creating a new field
275:         newattr = attr not in _localdict
276:         try:
277:             # Is attr a generic attribute ?
278:             ret = object.__setattr__(self, attr, val)
279:         except:
280:             # Not a generic attribute: exit if it's not a valid field
281:             fielddict = ndarray.__getattribute__(self, 'dtype').fields or {}
282:             optinfo = ndarray.__getattribute__(self, '_optinfo') or {}
283:             if not (attr in fielddict or attr in optinfo):
284:                 exctype, value = sys.exc_info()[:2]
285:                 raise exctype(value)
286:         else:
287:             # Get the list of names
288:             fielddict = ndarray.__getattribute__(self, 'dtype').fields or {}
289:             # Check the attribute
290:             if attr not in fielddict:
291:                 return ret
292:             if newattr:
293:                 # We just added this one or this setattr worked on an
294:                 # internal attribute.
295:                 try:
296:                     object.__delattr__(self, attr)
297:                 except:
298:                     return ret
299:         # Let's try to set the field
300:         try:
301:             res = fielddict[attr][:2]
302:         except (TypeError, KeyError):
303:             raise AttributeError("record array has no attribute %s" % attr)
304: 
305:         if val is masked:
306:             _fill_value = _localdict['_fill_value']
307:             if _fill_value is not None:
308:                 dval = _localdict['_fill_value'][attr]
309:             else:
310:                 dval = val
311:             mval = True
312:         else:
313:             dval = filled(val)
314:             mval = getmaskarray(val)
315:         obj = ndarray.__getattribute__(self, '_data').setfield(dval, *res)
316:         _localdict['_mask'].__setitem__(attr, mval)
317:         return obj
318: 
319:     def __getitem__(self, indx):
320:         '''
321:         Returns all the fields sharing the same fieldname base.
322: 
323:         The fieldname base is either `_data` or `_mask`.
324: 
325:         '''
326:         _localdict = self.__dict__
327:         _mask = ndarray.__getattribute__(self, '_mask')
328:         _data = ndarray.view(self, _localdict['_baseclass'])
329:         # We want a field
330:         if isinstance(indx, basestring):
331:             # Make sure _sharedmask is True to propagate back to _fieldmask
332:             # Don't use _set_mask, there are some copies being made that
333:             # break propagation Don't force the mask to nomask, that wreaks
334:             # easy masking
335:             obj = _data[indx].view(MaskedArray)
336:             obj._mask = _mask[indx]
337:             obj._sharedmask = True
338:             fval = _localdict['_fill_value']
339:             if fval is not None:
340:                 obj._fill_value = fval[indx]
341:             # Force to masked if the mask is True
342:             if not obj.ndim and obj._mask:
343:                 return masked
344:             return obj
345:         # We want some elements.
346:         # First, the data.
347:         obj = np.array(_data[indx], copy=False).view(mrecarray)
348:         obj._mask = np.array(_mask[indx], copy=False).view(recarray)
349:         return obj
350: 
351:     def __setitem__(self, indx, value):
352:         '''
353:         Sets the given record to value.
354: 
355:         '''
356:         MaskedArray.__setitem__(self, indx, value)
357:         if isinstance(indx, basestring):
358:             self._mask[indx] = ma.getmaskarray(value)
359: 
360:     def __str__(self):
361:         '''
362:         Calculates the string representation.
363: 
364:         '''
365:         if self.size > 1:
366:             mstr = ["(%s)" % ",".join([str(i) for i in s])
367:                     for s in zip(*[getattr(self, f) for f in self.dtype.names])]
368:             return "[%s]" % ", ".join(mstr)
369:         else:
370:             mstr = ["%s" % ",".join([str(i) for i in s])
371:                     for s in zip([getattr(self, f) for f in self.dtype.names])]
372:             return "(%s)" % ", ".join(mstr)
373: 
374:     def __repr__(self):
375:         '''
376:         Calculates the repr representation.
377: 
378:         '''
379:         _names = self.dtype.names
380:         fmt = "%%%is : %%s" % (max([len(n) for n in _names]) + 4,)
381:         reprstr = [fmt % (f, getattr(self, f)) for f in self.dtype.names]
382:         reprstr.insert(0, 'masked_records(')
383:         reprstr.extend([fmt % ('    fill_value', self.fill_value),
384:                          '              )'])
385:         return str("\n".join(reprstr))
386: 
387:     def view(self, dtype=None, type=None):
388:         '''
389:         Returns a view of the mrecarray.
390: 
391:         '''
392:         # OK, basic copy-paste from MaskedArray.view.
393:         if dtype is None:
394:             if type is None:
395:                 output = ndarray.view(self)
396:             else:
397:                 output = ndarray.view(self, type)
398:         # Here again.
399:         elif type is None:
400:             try:
401:                 if issubclass(dtype, ndarray):
402:                     output = ndarray.view(self, dtype)
403:                     dtype = None
404:                 else:
405:                     output = ndarray.view(self, dtype)
406:             # OK, there's the change
407:             except TypeError:
408:                 dtype = np.dtype(dtype)
409:                 # we need to revert to MaskedArray, but keeping the possibility
410:                 # of subclasses (eg, TimeSeriesRecords), so we'll force a type
411:                 # set to the first parent
412:                 if dtype.fields is None:
413:                     basetype = self.__class__.__bases__[0]
414:                     output = self.__array__().view(dtype, basetype)
415:                     output._update_from(self)
416:                 else:
417:                     output = ndarray.view(self, dtype)
418:                 output._fill_value = None
419:         else:
420:             output = ndarray.view(self, dtype, type)
421:         # Update the mask, just like in MaskedArray.view
422:         if (getattr(output, '_mask', nomask) is not nomask):
423:             mdtype = ma.make_mask_descr(output.dtype)
424:             output._mask = self._mask.view(mdtype, ndarray)
425:             output._mask.shape = output.shape
426:         return output
427: 
428:     def harden_mask(self):
429:         '''
430:         Forces the mask to hard.
431: 
432:         '''
433:         self._hardmask = True
434: 
435:     def soften_mask(self):
436:         '''
437:         Forces the mask to soft
438: 
439:         '''
440:         self._hardmask = False
441: 
442:     def copy(self):
443:         '''
444:         Returns a copy of the masked record.
445: 
446:         '''
447:         copied = self._data.copy().view(type(self))
448:         copied._mask = self._mask.copy()
449:         return copied
450: 
451:     def tolist(self, fill_value=None):
452:         '''
453:         Return the data portion of the array as a list.
454: 
455:         Data items are converted to the nearest compatible Python type.
456:         Masked values are converted to fill_value. If fill_value is None,
457:         the corresponding entries in the output list will be ``None``.
458: 
459:         '''
460:         if fill_value is not None:
461:             return self.filled(fill_value).tolist()
462:         result = narray(self.filled().tolist(), dtype=object)
463:         mask = narray(self._mask.tolist())
464:         result[mask] = None
465:         return result.tolist()
466: 
467:     def __getstate__(self):
468:         '''Return the internal state of the masked array.
469: 
470:         This is for pickling.
471: 
472:         '''
473:         state = (1,
474:                  self.shape,
475:                  self.dtype,
476:                  self.flags.fnc,
477:                  self._data.tobytes(),
478:                  self._mask.tobytes(),
479:                  self._fill_value,
480:                  )
481:         return state
482: 
483:     def __setstate__(self, state):
484:         '''
485:         Restore the internal state of the masked array.
486: 
487:         This is for pickling.  ``state`` is typically the output of the
488:         ``__getstate__`` output, and is a 5-tuple:
489: 
490:         - class name
491:         - a tuple giving the shape of the data
492:         - a typecode for the data
493:         - a binary string for the data
494:         - a binary string for the mask.
495: 
496:         '''
497:         (ver, shp, typ, isf, raw, msk, flv) = state
498:         ndarray.__setstate__(self, (shp, typ, isf, raw))
499:         mdtype = dtype([(k, bool_) for (k, _) in self.dtype.descr])
500:         self.__dict__['_mask'].__setstate__((shp, mdtype, isf, msk))
501:         self.fill_value = flv
502: 
503:     def __reduce__(self):
504:         '''
505:         Return a 3-tuple for pickling a MaskedArray.
506: 
507:         '''
508:         return (_mrreconstruct,
509:                 (self.__class__, self._baseclass, (0,), 'b',),
510:                 self.__getstate__())
511: 
512: def _mrreconstruct(subtype, baseclass, baseshape, basetype,):
513:     '''
514:     Build a new MaskedArray from the information stored in a pickle.
515: 
516:     '''
517:     _data = ndarray.__new__(baseclass, baseshape, basetype).view(subtype)
518:     _mask = ndarray.__new__(ndarray, baseshape, 'b1')
519:     return subtype.__new__(subtype, _data, mask=_mask, dtype=basetype,)
520: 
521: mrecarray = MaskedRecords
522: 
523: 
524: ###############################################################################
525: #                             Constructors                                    #
526: ###############################################################################
527: 
528: 
529: def fromarrays(arraylist, dtype=None, shape=None, formats=None,
530:                names=None, titles=None, aligned=False, byteorder=None,
531:                fill_value=None):
532:     '''
533:     Creates a mrecarray from a (flat) list of masked arrays.
534: 
535:     Parameters
536:     ----------
537:     arraylist : sequence
538:         A list of (masked) arrays. Each element of the sequence is first converted
539:         to a masked array if needed. If a 2D array is passed as argument, it is
540:         processed line by line
541:     dtype : {None, dtype}, optional
542:         Data type descriptor.
543:     shape : {None, integer}, optional
544:         Number of records. If None, shape is defined from the shape of the
545:         first array in the list.
546:     formats : {None, sequence}, optional
547:         Sequence of formats for each individual field. If None, the formats will
548:         be autodetected by inspecting the fields and selecting the highest dtype
549:         possible.
550:     names : {None, sequence}, optional
551:         Sequence of the names of each field.
552:     fill_value : {None, sequence}, optional
553:         Sequence of data to be used as filling values.
554: 
555:     Notes
556:     -----
557:     Lists of tuples should be preferred over lists of lists for faster processing.
558: 
559:     '''
560:     datalist = [getdata(x) for x in arraylist]
561:     masklist = [np.atleast_1d(getmaskarray(x)) for x in arraylist]
562:     _array = recfromarrays(datalist,
563:                            dtype=dtype, shape=shape, formats=formats,
564:                            names=names, titles=titles, aligned=aligned,
565:                            byteorder=byteorder).view(mrecarray)
566:     _array._mask.flat = list(zip(*masklist))
567:     if fill_value is not None:
568:         _array.fill_value = fill_value
569:     return _array
570: 
571: 
572: def fromrecords(reclist, dtype=None, shape=None, formats=None, names=None,
573:                 titles=None, aligned=False, byteorder=None,
574:                 fill_value=None, mask=nomask):
575:     '''
576:     Creates a MaskedRecords from a list of records.
577: 
578:     Parameters
579:     ----------
580:     reclist : sequence
581:         A list of records. Each element of the sequence is first converted
582:         to a masked array if needed. If a 2D array is passed as argument, it is
583:         processed line by line
584:     dtype : {None, dtype}, optional
585:         Data type descriptor.
586:     shape : {None,int}, optional
587:         Number of records. If None, ``shape`` is defined from the shape of the
588:         first array in the list.
589:     formats : {None, sequence}, optional
590:         Sequence of formats for each individual field. If None, the formats will
591:         be autodetected by inspecting the fields and selecting the highest dtype
592:         possible.
593:     names : {None, sequence}, optional
594:         Sequence of the names of each field.
595:     fill_value : {None, sequence}, optional
596:         Sequence of data to be used as filling values.
597:     mask : {nomask, sequence}, optional.
598:         External mask to apply on the data.
599: 
600:     Notes
601:     -----
602:     Lists of tuples should be preferred over lists of lists for faster processing.
603: 
604:     '''
605:     # Grab the initial _fieldmask, if needed:
606:     _mask = getattr(reclist, '_mask', None)
607:     # Get the list of records.
608:     if isinstance(reclist, ndarray):
609:         # Make sure we don't have some hidden mask
610:         if isinstance(reclist, MaskedArray):
611:             reclist = reclist.filled().view(ndarray)
612:         # Grab the initial dtype, just in case
613:         if dtype is None:
614:             dtype = reclist.dtype
615:         reclist = reclist.tolist()
616:     mrec = recfromrecords(reclist, dtype=dtype, shape=shape, formats=formats,
617:                           names=names, titles=titles,
618:                           aligned=aligned, byteorder=byteorder).view(mrecarray)
619:     # Set the fill_value if needed
620:     if fill_value is not None:
621:         mrec.fill_value = fill_value
622:     # Now, let's deal w/ the mask
623:     if mask is not nomask:
624:         mask = np.array(mask, copy=False)
625:         maskrecordlength = len(mask.dtype)
626:         if maskrecordlength:
627:             mrec._mask.flat = mask
628:         elif len(mask.shape) == 2:
629:             mrec._mask.flat = [tuple(m) for m in mask]
630:         else:
631:             mrec.__setmask__(mask)
632:     if _mask is not None:
633:         mrec._mask[:] = _mask
634:     return mrec
635: 
636: 
637: def _guessvartypes(arr):
638:     '''
639:     Tries to guess the dtypes of the str_ ndarray `arr`.
640: 
641:     Guesses by testing element-wise conversion. Returns a list of dtypes.
642:     The array is first converted to ndarray. If the array is 2D, the test
643:     is performed on the first line. An exception is raised if the file is
644:     3D or more.
645: 
646:     '''
647:     vartypes = []
648:     arr = np.asarray(arr)
649:     if len(arr.shape) == 2:
650:         arr = arr[0]
651:     elif len(arr.shape) > 2:
652:         raise ValueError("The array should be 2D at most!")
653:     # Start the conversion loop.
654:     for f in arr:
655:         try:
656:             int(f)
657:         except ValueError:
658:             try:
659:                 float(f)
660:             except ValueError:
661:                 try:
662:                     complex(f)
663:                 except ValueError:
664:                     vartypes.append(arr.dtype)
665:                 else:
666:                     vartypes.append(np.dtype(complex))
667:             else:
668:                 vartypes.append(np.dtype(float))
669:         else:
670:             vartypes.append(np.dtype(int))
671:     return vartypes
672: 
673: 
674: def openfile(fname):
675:     '''
676:     Opens the file handle of file `fname`.
677: 
678:     '''
679:     # A file handle
680:     if hasattr(fname, 'readline'):
681:         return fname
682:     # Try to open the file and guess its type
683:     try:
684:         f = open(fname)
685:     except IOError:
686:         raise IOError("No such file: '%s'" % fname)
687:     if f.readline()[:2] != "\\x":
688:         f.seek(0, 0)
689:         return f
690:     f.close()
691:     raise NotImplementedError("Wow, binary file")
692: 
693: 
694: def fromtextfile(fname, delimitor=None, commentchar='#', missingchar='',
695:                  varnames=None, vartypes=None):
696:     '''
697:     Creates a mrecarray from data stored in the file `filename`.
698: 
699:     Parameters
700:     ----------
701:     fname : {file name/handle}
702:         Handle of an opened file.
703:     delimitor : {None, string}, optional
704:         Alphanumeric character used to separate columns in the file.
705:         If None, any (group of) white spacestring(s) will be used.
706:     commentchar : {'#', string}, optional
707:         Alphanumeric character used to mark the start of a comment.
708:     missingchar : {'', string}, optional
709:         String indicating missing data, and used to create the masks.
710:     varnames : {None, sequence}, optional
711:         Sequence of the variable names. If None, a list will be created from
712:         the first non empty line of the file.
713:     vartypes : {None, sequence}, optional
714:         Sequence of the variables dtypes. If None, it will be estimated from
715:         the first non-commented line.
716: 
717: 
718:     Ultra simple: the varnames are in the header, one line'''
719:     # Try to open the file.
720:     ftext = openfile(fname)
721: 
722:     # Get the first non-empty line as the varnames
723:     while True:
724:         line = ftext.readline()
725:         firstline = line[:line.find(commentchar)].strip()
726:         _varnames = firstline.split(delimitor)
727:         if len(_varnames) > 1:
728:             break
729:     if varnames is None:
730:         varnames = _varnames
731: 
732:     # Get the data.
733:     _variables = masked_array([line.strip().split(delimitor) for line in ftext
734:                                if line[0] != commentchar and len(line) > 1])
735:     (_, nfields) = _variables.shape
736:     ftext.close()
737: 
738:     # Try to guess the dtype.
739:     if vartypes is None:
740:         vartypes = _guessvartypes(_variables[0])
741:     else:
742:         vartypes = [np.dtype(v) for v in vartypes]
743:         if len(vartypes) != nfields:
744:             msg = "Attempting to %i dtypes for %i fields!"
745:             msg += " Reverting to default."
746:             warnings.warn(msg % (len(vartypes), nfields))
747:             vartypes = _guessvartypes(_variables[0])
748: 
749:     # Construct the descriptor.
750:     mdescr = [(n, f) for (n, f) in zip(varnames, vartypes)]
751:     mfillv = [ma.default_fill_value(f) for f in vartypes]
752: 
753:     # Get the data and the mask.
754:     # We just need a list of masked_arrays. It's easier to create it like that:
755:     _mask = (_variables.T == missingchar)
756:     _datalist = [masked_array(a, mask=m, dtype=t, fill_value=f)
757:                  for (a, m, t, f) in zip(_variables.T, _mask, vartypes, mfillv)]
758: 
759:     return fromarrays(_datalist, dtype=mdescr)
760: 
761: 
762: def addfield(mrecord, newfield, newfieldname=None):
763:     '''Adds a new field to the masked record array
764: 
765:     Uses `newfield` as data and `newfieldname` as name. If `newfieldname`
766:     is None, the new field name is set to 'fi', where `i` is the number of
767:     existing fields.
768: 
769:     '''
770:     _data = mrecord._data
771:     _mask = mrecord._mask
772:     if newfieldname is None or newfieldname in reserved_fields:
773:         newfieldname = 'f%i' % len(_data.dtype)
774:     newfield = ma.array(newfield)
775:     # Get the new data.
776:     # Create a new empty recarray
777:     newdtype = np.dtype(_data.dtype.descr + [(newfieldname, newfield.dtype)])
778:     newdata = recarray(_data.shape, newdtype)
779:     # Add the exisintg field
780:     [newdata.setfield(_data.getfield(*f), *f)
781:          for f in _data.dtype.fields.values()]
782:     # Add the new field
783:     newdata.setfield(newfield._data, *newdata.dtype.fields[newfieldname])
784:     newdata = newdata.view(MaskedRecords)
785:     # Get the new mask
786:     # Create a new empty recarray
787:     newmdtype = np.dtype([(n, bool_) for n in newdtype.names])
788:     newmask = recarray(_data.shape, newmdtype)
789:     # Add the old masks
790:     [newmask.setfield(_mask.getfield(*f), *f)
791:          for f in _mask.dtype.fields.values()]
792:     # Add the mask of the new field
793:     newmask.setfield(getmaskarray(newfield),
794:                      *newmask.dtype.fields[newfieldname])
795:     newdata._mask = newmask
796:     return newdata
797: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_154986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, (-1)), 'str', ':mod:`numpy.ma..mrecords`\n\nDefines the equivalent of :class:`numpy.recarrays` for masked arrays,\nwhere fields can be accessed as attributes.\nNote that :class:`numpy.ma.MaskedArray` already supports structured datatypes\nand the masking of individual fields.\n\n.. moduleauthor:: Pierre Gerard-Marchant\n\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 18, 0))

# 'import sys' statement (line 18)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 19, 0))

# 'import warnings' statement (line 19)
import warnings

import_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'warnings', warnings, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 21, 0))

# 'import numpy' statement (line 21)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/ma/')
import_154987 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'numpy')

if (type(import_154987) is not StypyTypeError):

    if (import_154987 != 'pyd_module'):
        __import__(import_154987)
        sys_modules_154988 = sys.modules[import_154987]
        import_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'np', sys_modules_154988.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'numpy', import_154987)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/ma/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 22, 0))

# 'import numpy.core.numerictypes' statement (line 22)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/ma/')
import_154989 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'numpy.core.numerictypes')

if (type(import_154989) is not StypyTypeError):

    if (import_154989 != 'pyd_module'):
        __import__(import_154989)
        sys_modules_154990 = sys.modules[import_154989]
        import_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'ntypes', sys_modules_154990.module_type_store, module_type_store)
    else:
        import numpy.core.numerictypes as ntypes

        import_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'ntypes', numpy.core.numerictypes, module_type_store)

else:
    # Assigning a type to the variable 'numpy.core.numerictypes' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'numpy.core.numerictypes', import_154989)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/ma/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 23, 0))

# 'from numpy.compat import basestring' statement (line 23)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/ma/')
import_154991 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 23, 0), 'numpy.compat')

if (type(import_154991) is not StypyTypeError):

    if (import_154991 != 'pyd_module'):
        __import__(import_154991)
        sys_modules_154992 = sys.modules[import_154991]
        import_from_module(stypy.reporting.localization.Localization(__file__, 23, 0), 'numpy.compat', sys_modules_154992.module_type_store, module_type_store, ['basestring'])
        nest_module(stypy.reporting.localization.Localization(__file__, 23, 0), __file__, sys_modules_154992, sys_modules_154992.module_type_store, module_type_store)
    else:
        from numpy.compat import basestring

        import_from_module(stypy.reporting.localization.Localization(__file__, 23, 0), 'numpy.compat', None, module_type_store, ['basestring'], [basestring])

else:
    # Assigning a type to the variable 'numpy.compat' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'numpy.compat', import_154991)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/ma/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 24, 0))

# 'from numpy import bool_, dtype, ndarray, recarray, narray' statement (line 24)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/ma/')
import_154993 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'numpy')

if (type(import_154993) is not StypyTypeError):

    if (import_154993 != 'pyd_module'):
        __import__(import_154993)
        sys_modules_154994 = sys.modules[import_154993]
        import_from_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'numpy', sys_modules_154994.module_type_store, module_type_store, ['bool_', 'dtype', 'ndarray', 'recarray', 'array'])
        nest_module(stypy.reporting.localization.Localization(__file__, 24, 0), __file__, sys_modules_154994, sys_modules_154994.module_type_store, module_type_store)
    else:
        from numpy import bool_, dtype, ndarray, recarray, array as narray

        import_from_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'numpy', None, module_type_store, ['bool_', 'dtype', 'ndarray', 'recarray', 'array'], [bool_, dtype, ndarray, recarray, narray])

else:
    # Assigning a type to the variable 'numpy' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'numpy', import_154993)

# Adding an alias
module_type_store.add_alias('narray', 'array')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/ma/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 27, 0))

# 'from numpy.core.records import recfromarrays, recfromrecords' statement (line 27)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/ma/')
import_154995 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 27, 0), 'numpy.core.records')

if (type(import_154995) is not StypyTypeError):

    if (import_154995 != 'pyd_module'):
        __import__(import_154995)
        sys_modules_154996 = sys.modules[import_154995]
        import_from_module(stypy.reporting.localization.Localization(__file__, 27, 0), 'numpy.core.records', sys_modules_154996.module_type_store, module_type_store, ['fromarrays', 'fromrecords'])
        nest_module(stypy.reporting.localization.Localization(__file__, 27, 0), __file__, sys_modules_154996, sys_modules_154996.module_type_store, module_type_store)
    else:
        from numpy.core.records import fromarrays as recfromarrays, fromrecords as recfromrecords

        import_from_module(stypy.reporting.localization.Localization(__file__, 27, 0), 'numpy.core.records', None, module_type_store, ['fromarrays', 'fromrecords'], [recfromarrays, recfromrecords])

else:
    # Assigning a type to the variable 'numpy.core.records' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 0), 'numpy.core.records', import_154995)

# Adding an alias
module_type_store.add_alias('recfromrecords', 'fromrecords')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/ma/')


# Assigning a Attribute to a Name (line 31):

# Assigning a Attribute to a Name (line 31):
# Getting the type of 'np' (line 31)
np_154997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 17), 'np')
# Obtaining the member 'core' of a type (line 31)
core_154998 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 17), np_154997, 'core')
# Obtaining the member 'records' of a type (line 31)
records_154999 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 17), core_154998, 'records')
# Obtaining the member '_byteorderconv' of a type (line 31)
_byteorderconv_155000 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 17), records_154999, '_byteorderconv')
# Assigning a type to the variable '_byteorderconv' (line 31)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 0), '_byteorderconv', _byteorderconv_155000)

# Assigning a Attribute to a Name (line 32):

# Assigning a Attribute to a Name (line 32):
# Getting the type of 'ntypes' (line 32)
ntypes_155001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 11), 'ntypes')
# Obtaining the member '_typestr' of a type (line 32)
_typestr_155002 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 11), ntypes_155001, '_typestr')
# Assigning a type to the variable '_typestr' (line 32)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 0), '_typestr', _typestr_155002)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 34, 0))

# 'import numpy.ma' statement (line 34)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/ma/')
import_155003 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 34, 0), 'numpy.ma')

if (type(import_155003) is not StypyTypeError):

    if (import_155003 != 'pyd_module'):
        __import__(import_155003)
        sys_modules_155004 = sys.modules[import_155003]
        import_module(stypy.reporting.localization.Localization(__file__, 34, 0), 'ma', sys_modules_155004.module_type_store, module_type_store)
    else:
        import numpy.ma as ma

        import_module(stypy.reporting.localization.Localization(__file__, 34, 0), 'ma', numpy.ma, module_type_store)

else:
    # Assigning a type to the variable 'numpy.ma' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 0), 'numpy.ma', import_155003)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/ma/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 35, 0))

# 'from numpy.ma import MAError, MaskedArray, masked, nomask, masked_array, getdata, getmaskarray, filled' statement (line 35)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/ma/')
import_155005 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 35, 0), 'numpy.ma')

if (type(import_155005) is not StypyTypeError):

    if (import_155005 != 'pyd_module'):
        __import__(import_155005)
        sys_modules_155006 = sys.modules[import_155005]
        import_from_module(stypy.reporting.localization.Localization(__file__, 35, 0), 'numpy.ma', sys_modules_155006.module_type_store, module_type_store, ['MAError', 'MaskedArray', 'masked', 'nomask', 'masked_array', 'getdata', 'getmaskarray', 'filled'])
        nest_module(stypy.reporting.localization.Localization(__file__, 35, 0), __file__, sys_modules_155006, sys_modules_155006.module_type_store, module_type_store)
    else:
        from numpy.ma import MAError, MaskedArray, masked, nomask, masked_array, getdata, getmaskarray, filled

        import_from_module(stypy.reporting.localization.Localization(__file__, 35, 0), 'numpy.ma', None, module_type_store, ['MAError', 'MaskedArray', 'masked', 'nomask', 'masked_array', 'getdata', 'getmaskarray', 'filled'], [MAError, MaskedArray, masked, nomask, masked_array, getdata, getmaskarray, filled])

else:
    # Assigning a type to the variable 'numpy.ma' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 0), 'numpy.ma', import_155005)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/ma/')


# Assigning a Attribute to a Name (line 40):

# Assigning a Attribute to a Name (line 40):
# Getting the type of 'ma' (line 40)
ma_155007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 20), 'ma')
# Obtaining the member 'core' of a type (line 40)
core_155008 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 20), ma_155007, 'core')
# Obtaining the member '_check_fill_value' of a type (line 40)
_check_fill_value_155009 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 20), core_155008, '_check_fill_value')
# Assigning a type to the variable '_check_fill_value' (line 40)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 0), '_check_fill_value', _check_fill_value_155009)

# Assigning a List to a Name (line 43):

# Assigning a List to a Name (line 43):
__all__ = ['MaskedRecords', 'mrecarray', 'fromarrays', 'fromrecords', 'fromtextfile', 'addfield']
module_type_store.set_exportable_members(['MaskedRecords', 'mrecarray', 'fromarrays', 'fromrecords', 'fromtextfile', 'addfield'])

# Obtaining an instance of the builtin type 'list' (line 43)
list_155010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 43)
# Adding element type (line 43)
str_155011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 4), 'str', 'MaskedRecords')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 10), list_155010, str_155011)
# Adding element type (line 43)
str_155012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 21), 'str', 'mrecarray')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 10), list_155010, str_155012)
# Adding element type (line 43)
str_155013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 34), 'str', 'fromarrays')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 10), list_155010, str_155013)
# Adding element type (line 43)
str_155014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 48), 'str', 'fromrecords')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 10), list_155010, str_155014)
# Adding element type (line 43)
str_155015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 4), 'str', 'fromtextfile')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 10), list_155010, str_155015)
# Adding element type (line 43)
str_155016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 20), 'str', 'addfield')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 10), list_155010, str_155016)

# Assigning a type to the variable '__all__' (line 43)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 0), '__all__', list_155010)

# Assigning a List to a Name (line 48):

# Assigning a List to a Name (line 48):

# Obtaining an instance of the builtin type 'list' (line 48)
list_155017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 18), 'list')
# Adding type elements to the builtin type 'list' instance (line 48)
# Adding element type (line 48)
str_155018 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 19), 'str', '_data')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 18), list_155017, str_155018)
# Adding element type (line 48)
str_155019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 28), 'str', '_mask')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 18), list_155017, str_155019)
# Adding element type (line 48)
str_155020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 37), 'str', '_fieldmask')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 18), list_155017, str_155020)
# Adding element type (line 48)
str_155021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 51), 'str', 'dtype')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 18), list_155017, str_155021)

# Assigning a type to the variable 'reserved_fields' (line 48)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 0), 'reserved_fields', list_155017)

@norecursion
def _getformats(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_getformats'
    module_type_store = module_type_store.open_function_context('_getformats', 51, 0, False)
    
    # Passed parameters checking function
    _getformats.stypy_localization = localization
    _getformats.stypy_type_of_self = None
    _getformats.stypy_type_store = module_type_store
    _getformats.stypy_function_name = '_getformats'
    _getformats.stypy_param_names_list = ['data']
    _getformats.stypy_varargs_param_name = None
    _getformats.stypy_kwargs_param_name = None
    _getformats.stypy_call_defaults = defaults
    _getformats.stypy_call_varargs = varargs
    _getformats.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_getformats', ['data'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_getformats', localization, ['data'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_getformats(...)' code ##################

    str_155022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, (-1)), 'str', '\n    Returns the formats of arrays in arraylist as a comma-separated string.\n\n    ')
    
    # Type idiom detected: calculating its left and rigth part (line 56)
    str_155023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 21), 'str', 'dtype')
    # Getting the type of 'data' (line 56)
    data_155024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 15), 'data')
    
    (may_be_155025, more_types_in_union_155026) = may_provide_member(str_155023, data_155024)

    if may_be_155025:

        if more_types_in_union_155026:
            # Runtime conditional SSA (line 56)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'data' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'data', remove_not_member_provider_from_union(data_155024, 'dtype'))
        
        # Call to join(...): (line 57)
        # Processing the call arguments (line 57)
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'data' (line 57)
        data_155033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 45), 'data', False)
        # Obtaining the member 'dtype' of a type (line 57)
        dtype_155034 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 45), data_155033, 'dtype')
        # Obtaining the member 'descr' of a type (line 57)
        descr_155035 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 45), dtype_155034, 'descr')
        comprehension_155036 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 25), descr_155035)
        # Assigning a type to the variable 'desc' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 25), 'desc', comprehension_155036)
        
        # Obtaining the type of the subscript
        int_155029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 30), 'int')
        # Getting the type of 'desc' (line 57)
        desc_155030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 25), 'desc', False)
        # Obtaining the member '__getitem__' of a type (line 57)
        getitem___155031 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 25), desc_155030, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 57)
        subscript_call_result_155032 = invoke(stypy.reporting.localization.Localization(__file__, 57, 25), getitem___155031, int_155029)
        
        list_155037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 25), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 25), list_155037, subscript_call_result_155032)
        # Processing the call keyword arguments (line 57)
        kwargs_155038 = {}
        str_155027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 15), 'str', ',')
        # Obtaining the member 'join' of a type (line 57)
        join_155028 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 15), str_155027, 'join')
        # Calling join(args, kwargs) (line 57)
        join_call_result_155039 = invoke(stypy.reporting.localization.Localization(__file__, 57, 15), join_155028, *[list_155037], **kwargs_155038)
        
        # Assigning a type to the variable 'stypy_return_type' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'stypy_return_type', join_call_result_155039)

        if more_types_in_union_155026:
            # SSA join for if statement (line 56)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Str to a Name (line 59):
    
    # Assigning a Str to a Name (line 59):
    str_155040 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 14), 'str', '')
    # Assigning a type to the variable 'formats' (line 59)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 'formats', str_155040)
    
    # Getting the type of 'data' (line 60)
    data_155041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 15), 'data')
    # Testing the type of a for loop iterable (line 60)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 60, 4), data_155041)
    # Getting the type of the for loop variable (line 60)
    for_loop_var_155042 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 60, 4), data_155041)
    # Assigning a type to the variable 'obj' (line 60)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'obj', for_loop_var_155042)
    # SSA begins for a for statement (line 60)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 61):
    
    # Assigning a Call to a Name (line 61):
    
    # Call to asarray(...): (line 61)
    # Processing the call arguments (line 61)
    # Getting the type of 'obj' (line 61)
    obj_155045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 25), 'obj', False)
    # Processing the call keyword arguments (line 61)
    kwargs_155046 = {}
    # Getting the type of 'np' (line 61)
    np_155043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 14), 'np', False)
    # Obtaining the member 'asarray' of a type (line 61)
    asarray_155044 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 14), np_155043, 'asarray')
    # Calling asarray(args, kwargs) (line 61)
    asarray_call_result_155047 = invoke(stypy.reporting.localization.Localization(__file__, 61, 14), asarray_155044, *[obj_155045], **kwargs_155046)
    
    # Assigning a type to the variable 'obj' (line 61)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'obj', asarray_call_result_155047)
    
    # Getting the type of 'formats' (line 62)
    formats_155048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'formats')
    
    # Obtaining the type of the subscript
    # Getting the type of 'obj' (line 62)
    obj_155049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 28), 'obj')
    # Obtaining the member 'dtype' of a type (line 62)
    dtype_155050 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 28), obj_155049, 'dtype')
    # Obtaining the member 'type' of a type (line 62)
    type_155051 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 28), dtype_155050, 'type')
    # Getting the type of '_typestr' (line 62)
    _typestr_155052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 19), '_typestr')
    # Obtaining the member '__getitem__' of a type (line 62)
    getitem___155053 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 19), _typestr_155052, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 62)
    subscript_call_result_155054 = invoke(stypy.reporting.localization.Localization(__file__, 62, 19), getitem___155053, type_155051)
    
    # Applying the binary operator '+=' (line 62)
    result_iadd_155055 = python_operator(stypy.reporting.localization.Localization(__file__, 62, 8), '+=', formats_155048, subscript_call_result_155054)
    # Assigning a type to the variable 'formats' (line 62)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'formats', result_iadd_155055)
    
    
    
    # Call to issubclass(...): (line 63)
    # Processing the call arguments (line 63)
    # Getting the type of 'obj' (line 63)
    obj_155057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 22), 'obj', False)
    # Obtaining the member 'dtype' of a type (line 63)
    dtype_155058 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 22), obj_155057, 'dtype')
    # Obtaining the member 'type' of a type (line 63)
    type_155059 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 22), dtype_155058, 'type')
    # Getting the type of 'ntypes' (line 63)
    ntypes_155060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 38), 'ntypes', False)
    # Obtaining the member 'flexible' of a type (line 63)
    flexible_155061 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 38), ntypes_155060, 'flexible')
    # Processing the call keyword arguments (line 63)
    kwargs_155062 = {}
    # Getting the type of 'issubclass' (line 63)
    issubclass_155056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 11), 'issubclass', False)
    # Calling issubclass(args, kwargs) (line 63)
    issubclass_call_result_155063 = invoke(stypy.reporting.localization.Localization(__file__, 63, 11), issubclass_155056, *[type_155059, flexible_155061], **kwargs_155062)
    
    # Testing the type of an if condition (line 63)
    if_condition_155064 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 63, 8), issubclass_call_result_155063)
    # Assigning a type to the variable 'if_condition_155064' (line 63)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'if_condition_155064', if_condition_155064)
    # SSA begins for if statement (line 63)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'formats' (line 64)
    formats_155065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 12), 'formats')
    
    # Call to repr(...): (line 64)
    # Processing the call arguments (line 64)
    # Getting the type of 'obj' (line 64)
    obj_155067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 28), 'obj', False)
    # Obtaining the member 'itemsize' of a type (line 64)
    itemsize_155068 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 28), obj_155067, 'itemsize')
    # Processing the call keyword arguments (line 64)
    kwargs_155069 = {}
    # Getting the type of 'repr' (line 64)
    repr_155066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 23), 'repr', False)
    # Calling repr(args, kwargs) (line 64)
    repr_call_result_155070 = invoke(stypy.reporting.localization.Localization(__file__, 64, 23), repr_155066, *[itemsize_155068], **kwargs_155069)
    
    # Applying the binary operator '+=' (line 64)
    result_iadd_155071 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 12), '+=', formats_155065, repr_call_result_155070)
    # Assigning a type to the variable 'formats' (line 64)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 12), 'formats', result_iadd_155071)
    
    # SSA join for if statement (line 63)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'formats' (line 65)
    formats_155072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'formats')
    str_155073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 19), 'str', ',')
    # Applying the binary operator '+=' (line 65)
    result_iadd_155074 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 8), '+=', formats_155072, str_155073)
    # Assigning a type to the variable 'formats' (line 65)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'formats', result_iadd_155074)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining the type of the subscript
    int_155075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 20), 'int')
    slice_155076 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 66, 11), None, int_155075, None)
    # Getting the type of 'formats' (line 66)
    formats_155077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 11), 'formats')
    # Obtaining the member '__getitem__' of a type (line 66)
    getitem___155078 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 11), formats_155077, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 66)
    subscript_call_result_155079 = invoke(stypy.reporting.localization.Localization(__file__, 66, 11), getitem___155078, slice_155076)
    
    # Assigning a type to the variable 'stypy_return_type' (line 66)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'stypy_return_type', subscript_call_result_155079)
    
    # ################# End of '_getformats(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_getformats' in the type store
    # Getting the type of 'stypy_return_type' (line 51)
    stypy_return_type_155080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_155080)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_getformats'
    return stypy_return_type_155080

# Assigning a type to the variable '_getformats' (line 51)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 0), '_getformats', _getformats)

@norecursion
def _checknames(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 69)
    None_155081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 29), 'None')
    defaults = [None_155081]
    # Create a new context for function '_checknames'
    module_type_store = module_type_store.open_function_context('_checknames', 69, 0, False)
    
    # Passed parameters checking function
    _checknames.stypy_localization = localization
    _checknames.stypy_type_of_self = None
    _checknames.stypy_type_store = module_type_store
    _checknames.stypy_function_name = '_checknames'
    _checknames.stypy_param_names_list = ['descr', 'names']
    _checknames.stypy_varargs_param_name = None
    _checknames.stypy_kwargs_param_name = None
    _checknames.stypy_call_defaults = defaults
    _checknames.stypy_call_varargs = varargs
    _checknames.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_checknames', ['descr', 'names'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_checknames', localization, ['descr', 'names'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_checknames(...)' code ##################

    str_155082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, (-1)), 'str', "\n    Checks that field names ``descr`` are not reserved keywords.\n\n    If this is the case, a default 'f%i' is substituted.  If the argument\n    `names` is not None, updates the field names to valid names.\n\n    ")
    
    # Assigning a Call to a Name (line 77):
    
    # Assigning a Call to a Name (line 77):
    
    # Call to len(...): (line 77)
    # Processing the call arguments (line 77)
    # Getting the type of 'descr' (line 77)
    descr_155084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 17), 'descr', False)
    # Processing the call keyword arguments (line 77)
    kwargs_155085 = {}
    # Getting the type of 'len' (line 77)
    len_155083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 13), 'len', False)
    # Calling len(args, kwargs) (line 77)
    len_call_result_155086 = invoke(stypy.reporting.localization.Localization(__file__, 77, 13), len_155083, *[descr_155084], **kwargs_155085)
    
    # Assigning a type to the variable 'ndescr' (line 77)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 4), 'ndescr', len_call_result_155086)
    
    # Assigning a ListComp to a Name (line 78):
    
    # Assigning a ListComp to a Name (line 78):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to range(...): (line 78)
    # Processing the call arguments (line 78)
    # Getting the type of 'ndescr' (line 78)
    ndescr_155091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 46), 'ndescr', False)
    # Processing the call keyword arguments (line 78)
    kwargs_155092 = {}
    # Getting the type of 'range' (line 78)
    range_155090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 40), 'range', False)
    # Calling range(args, kwargs) (line 78)
    range_call_result_155093 = invoke(stypy.reporting.localization.Localization(__file__, 78, 40), range_155090, *[ndescr_155091], **kwargs_155092)
    
    comprehension_155094 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 21), range_call_result_155093)
    # Assigning a type to the variable 'i' (line 78)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 21), 'i', comprehension_155094)
    str_155087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 21), 'str', 'f%i')
    # Getting the type of 'i' (line 78)
    i_155088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 29), 'i')
    # Applying the binary operator '%' (line 78)
    result_mod_155089 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 21), '%', str_155087, i_155088)
    
    list_155095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 21), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 21), list_155095, result_mod_155089)
    # Assigning a type to the variable 'default_names' (line 78)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'default_names', list_155095)
    
    # Type idiom detected: calculating its left and rigth part (line 79)
    # Getting the type of 'names' (line 79)
    names_155096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 7), 'names')
    # Getting the type of 'None' (line 79)
    None_155097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 16), 'None')
    
    (may_be_155098, more_types_in_union_155099) = may_be_none(names_155096, None_155097)

    if may_be_155098:

        if more_types_in_union_155099:
            # Runtime conditional SSA (line 79)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Name to a Name (line 80):
        
        # Assigning a Name to a Name (line 80):
        # Getting the type of 'default_names' (line 80)
        default_names_155100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 20), 'default_names')
        # Assigning a type to the variable 'new_names' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'new_names', default_names_155100)

        if more_types_in_union_155099:
            # Runtime conditional SSA for else branch (line 79)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_155098) or more_types_in_union_155099):
        
        
        # Call to isinstance(...): (line 82)
        # Processing the call arguments (line 82)
        # Getting the type of 'names' (line 82)
        names_155102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 22), 'names', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 82)
        tuple_155103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 30), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 82)
        # Adding element type (line 82)
        # Getting the type of 'tuple' (line 82)
        tuple_155104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 30), 'tuple', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 30), tuple_155103, tuple_155104)
        # Adding element type (line 82)
        # Getting the type of 'list' (line 82)
        list_155105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 37), 'list', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 30), tuple_155103, list_155105)
        
        # Processing the call keyword arguments (line 82)
        kwargs_155106 = {}
        # Getting the type of 'isinstance' (line 82)
        isinstance_155101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 11), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 82)
        isinstance_call_result_155107 = invoke(stypy.reporting.localization.Localization(__file__, 82, 11), isinstance_155101, *[names_155102, tuple_155103], **kwargs_155106)
        
        # Testing the type of an if condition (line 82)
        if_condition_155108 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 82, 8), isinstance_call_result_155107)
        # Assigning a type to the variable 'if_condition_155108' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'if_condition_155108', if_condition_155108)
        # SSA begins for if statement (line 82)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 83):
        
        # Assigning a Name to a Name (line 83):
        # Getting the type of 'names' (line 83)
        names_155109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 24), 'names')
        # Assigning a type to the variable 'new_names' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 12), 'new_names', names_155109)
        # SSA branch for the else part of an if statement (line 82)
        module_type_store.open_ssa_branch('else')
        
        # Type idiom detected: calculating its left and rigth part (line 84)
        # Getting the type of 'str' (line 84)
        str_155110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 31), 'str')
        # Getting the type of 'names' (line 84)
        names_155111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 24), 'names')
        
        (may_be_155112, more_types_in_union_155113) = may_be_subtype(str_155110, names_155111)

        if may_be_155112:

            if more_types_in_union_155113:
                # Runtime conditional SSA (line 84)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'names' (line 84)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 13), 'names', remove_not_subtype_from_union(names_155111, str))
            
            # Assigning a Call to a Name (line 85):
            
            # Assigning a Call to a Name (line 85):
            
            # Call to split(...): (line 85)
            # Processing the call arguments (line 85)
            str_155116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 36), 'str', ',')
            # Processing the call keyword arguments (line 85)
            kwargs_155117 = {}
            # Getting the type of 'names' (line 85)
            names_155114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 24), 'names', False)
            # Obtaining the member 'split' of a type (line 85)
            split_155115 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 24), names_155114, 'split')
            # Calling split(args, kwargs) (line 85)
            split_call_result_155118 = invoke(stypy.reporting.localization.Localization(__file__, 85, 24), split_155115, *[str_155116], **kwargs_155117)
            
            # Assigning a type to the variable 'new_names' (line 85)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 12), 'new_names', split_call_result_155118)

            if more_types_in_union_155113:
                # Runtime conditional SSA for else branch (line 84)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_155112) or more_types_in_union_155113):
            # Assigning a type to the variable 'names' (line 84)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 13), 'names', remove_subtype_from_union(names_155111, str))
            
            # Call to NameError(...): (line 87)
            # Processing the call arguments (line 87)
            str_155120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 28), 'str', 'illegal input names %s')
            
            # Call to repr(...): (line 87)
            # Processing the call arguments (line 87)
            # Getting the type of 'names' (line 87)
            names_155122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 60), 'names', False)
            # Processing the call keyword arguments (line 87)
            kwargs_155123 = {}
            # Getting the type of 'repr' (line 87)
            repr_155121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 55), 'repr', False)
            # Calling repr(args, kwargs) (line 87)
            repr_call_result_155124 = invoke(stypy.reporting.localization.Localization(__file__, 87, 55), repr_155121, *[names_155122], **kwargs_155123)
            
            # Applying the binary operator '%' (line 87)
            result_mod_155125 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 28), '%', str_155120, repr_call_result_155124)
            
            # Processing the call keyword arguments (line 87)
            kwargs_155126 = {}
            # Getting the type of 'NameError' (line 87)
            NameError_155119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 18), 'NameError', False)
            # Calling NameError(args, kwargs) (line 87)
            NameError_call_result_155127 = invoke(stypy.reporting.localization.Localization(__file__, 87, 18), NameError_155119, *[result_mod_155125], **kwargs_155126)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 87, 12), NameError_call_result_155127, 'raise parameter', BaseException)

            if (may_be_155112 and more_types_in_union_155113):
                # SSA join for if statement (line 84)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for if statement (line 82)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 88):
        
        # Assigning a Call to a Name (line 88):
        
        # Call to len(...): (line 88)
        # Processing the call arguments (line 88)
        # Getting the type of 'new_names' (line 88)
        new_names_155129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 21), 'new_names', False)
        # Processing the call keyword arguments (line 88)
        kwargs_155130 = {}
        # Getting the type of 'len' (line 88)
        len_155128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 17), 'len', False)
        # Calling len(args, kwargs) (line 88)
        len_call_result_155131 = invoke(stypy.reporting.localization.Localization(__file__, 88, 17), len_155128, *[new_names_155129], **kwargs_155130)
        
        # Assigning a type to the variable 'nnames' (line 88)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'nnames', len_call_result_155131)
        
        
        # Getting the type of 'nnames' (line 89)
        nnames_155132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 11), 'nnames')
        # Getting the type of 'ndescr' (line 89)
        ndescr_155133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 20), 'ndescr')
        # Applying the binary operator '<' (line 89)
        result_lt_155134 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 11), '<', nnames_155132, ndescr_155133)
        
        # Testing the type of an if condition (line 89)
        if_condition_155135 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 89, 8), result_lt_155134)
        # Assigning a type to the variable 'if_condition_155135' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'if_condition_155135', if_condition_155135)
        # SSA begins for if statement (line 89)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'new_names' (line 90)
        new_names_155136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 12), 'new_names')
        
        # Obtaining the type of the subscript
        # Getting the type of 'nnames' (line 90)
        nnames_155137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 39), 'nnames')
        slice_155138 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 90, 25), nnames_155137, None, None)
        # Getting the type of 'default_names' (line 90)
        default_names_155139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 25), 'default_names')
        # Obtaining the member '__getitem__' of a type (line 90)
        getitem___155140 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 25), default_names_155139, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 90)
        subscript_call_result_155141 = invoke(stypy.reporting.localization.Localization(__file__, 90, 25), getitem___155140, slice_155138)
        
        # Applying the binary operator '+=' (line 90)
        result_iadd_155142 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 12), '+=', new_names_155136, subscript_call_result_155141)
        # Assigning a type to the variable 'new_names' (line 90)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 12), 'new_names', result_iadd_155142)
        
        # SSA join for if statement (line 89)
        module_type_store = module_type_store.join_ssa_context()
        

        if (may_be_155098 and more_types_in_union_155099):
            # SSA join for if statement (line 79)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a List to a Name (line 91):
    
    # Assigning a List to a Name (line 91):
    
    # Obtaining an instance of the builtin type 'list' (line 91)
    list_155143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 13), 'list')
    # Adding type elements to the builtin type 'list' instance (line 91)
    
    # Assigning a type to the variable 'ndescr' (line 91)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 4), 'ndescr', list_155143)
    
    
    # Call to zip(...): (line 92)
    # Processing the call arguments (line 92)
    # Getting the type of 'new_names' (line 92)
    new_names_155145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 25), 'new_names', False)
    # Getting the type of 'default_names' (line 92)
    default_names_155146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 36), 'default_names', False)
    # Getting the type of 'descr' (line 92)
    descr_155147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 51), 'descr', False)
    # Obtaining the member 'descr' of a type (line 92)
    descr_155148 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 51), descr_155147, 'descr')
    # Processing the call keyword arguments (line 92)
    kwargs_155149 = {}
    # Getting the type of 'zip' (line 92)
    zip_155144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 21), 'zip', False)
    # Calling zip(args, kwargs) (line 92)
    zip_call_result_155150 = invoke(stypy.reporting.localization.Localization(__file__, 92, 21), zip_155144, *[new_names_155145, default_names_155146, descr_155148], **kwargs_155149)
    
    # Testing the type of a for loop iterable (line 92)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 92, 4), zip_call_result_155150)
    # Getting the type of the for loop variable (line 92)
    for_loop_var_155151 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 92, 4), zip_call_result_155150)
    # Assigning a type to the variable 'n' (line 92)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 4), 'n', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 4), for_loop_var_155151))
    # Assigning a type to the variable 'd' (line 92)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 4), 'd', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 4), for_loop_var_155151))
    # Assigning a type to the variable 't' (line 92)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 4), 't', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 4), for_loop_var_155151))
    # SSA begins for a for statement (line 92)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Getting the type of 'n' (line 93)
    n_155152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 11), 'n')
    # Getting the type of 'reserved_fields' (line 93)
    reserved_fields_155153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 16), 'reserved_fields')
    # Applying the binary operator 'in' (line 93)
    result_contains_155154 = python_operator(stypy.reporting.localization.Localization(__file__, 93, 11), 'in', n_155152, reserved_fields_155153)
    
    # Testing the type of an if condition (line 93)
    if_condition_155155 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 93, 8), result_contains_155154)
    # Assigning a type to the variable 'if_condition_155155' (line 93)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'if_condition_155155', if_condition_155155)
    # SSA begins for if statement (line 93)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    
    # Obtaining the type of the subscript
    int_155156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 17), 'int')
    # Getting the type of 't' (line 94)
    t_155157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 15), 't')
    # Obtaining the member '__getitem__' of a type (line 94)
    getitem___155158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 15), t_155157, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 94)
    subscript_call_result_155159 = invoke(stypy.reporting.localization.Localization(__file__, 94, 15), getitem___155158, int_155156)
    
    # Getting the type of 'reserved_fields' (line 94)
    reserved_fields_155160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 23), 'reserved_fields')
    # Applying the binary operator 'in' (line 94)
    result_contains_155161 = python_operator(stypy.reporting.localization.Localization(__file__, 94, 15), 'in', subscript_call_result_155159, reserved_fields_155160)
    
    # Testing the type of an if condition (line 94)
    if_condition_155162 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 94, 12), result_contains_155161)
    # Assigning a type to the variable 'if_condition_155162' (line 94)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 12), 'if_condition_155162', if_condition_155162)
    # SSA begins for if statement (line 94)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 95)
    # Processing the call arguments (line 95)
    
    # Obtaining an instance of the builtin type 'tuple' (line 95)
    tuple_155165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 31), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 95)
    # Adding element type (line 95)
    # Getting the type of 'd' (line 95)
    d_155166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 31), 'd', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 31), tuple_155165, d_155166)
    # Adding element type (line 95)
    
    # Obtaining the type of the subscript
    int_155167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 36), 'int')
    # Getting the type of 't' (line 95)
    t_155168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 34), 't', False)
    # Obtaining the member '__getitem__' of a type (line 95)
    getitem___155169 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 34), t_155168, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 95)
    subscript_call_result_155170 = invoke(stypy.reporting.localization.Localization(__file__, 95, 34), getitem___155169, int_155167)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 31), tuple_155165, subscript_call_result_155170)
    
    # Processing the call keyword arguments (line 95)
    kwargs_155171 = {}
    # Getting the type of 'ndescr' (line 95)
    ndescr_155163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 16), 'ndescr', False)
    # Obtaining the member 'append' of a type (line 95)
    append_155164 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 16), ndescr_155163, 'append')
    # Calling append(args, kwargs) (line 95)
    append_call_result_155172 = invoke(stypy.reporting.localization.Localization(__file__, 95, 16), append_155164, *[tuple_155165], **kwargs_155171)
    
    # SSA branch for the else part of an if statement (line 94)
    module_type_store.open_ssa_branch('else')
    
    # Call to append(...): (line 97)
    # Processing the call arguments (line 97)
    # Getting the type of 't' (line 97)
    t_155175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 30), 't', False)
    # Processing the call keyword arguments (line 97)
    kwargs_155176 = {}
    # Getting the type of 'ndescr' (line 97)
    ndescr_155173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 16), 'ndescr', False)
    # Obtaining the member 'append' of a type (line 97)
    append_155174 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 16), ndescr_155173, 'append')
    # Calling append(args, kwargs) (line 97)
    append_call_result_155177 = invoke(stypy.reporting.localization.Localization(__file__, 97, 16), append_155174, *[t_155175], **kwargs_155176)
    
    # SSA join for if statement (line 94)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 93)
    module_type_store.open_ssa_branch('else')
    
    # Call to append(...): (line 99)
    # Processing the call arguments (line 99)
    
    # Obtaining an instance of the builtin type 'tuple' (line 99)
    tuple_155180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 27), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 99)
    # Adding element type (line 99)
    # Getting the type of 'n' (line 99)
    n_155181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 27), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 27), tuple_155180, n_155181)
    # Adding element type (line 99)
    
    # Obtaining the type of the subscript
    int_155182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 32), 'int')
    # Getting the type of 't' (line 99)
    t_155183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 30), 't', False)
    # Obtaining the member '__getitem__' of a type (line 99)
    getitem___155184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 30), t_155183, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 99)
    subscript_call_result_155185 = invoke(stypy.reporting.localization.Localization(__file__, 99, 30), getitem___155184, int_155182)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 27), tuple_155180, subscript_call_result_155185)
    
    # Processing the call keyword arguments (line 99)
    kwargs_155186 = {}
    # Getting the type of 'ndescr' (line 99)
    ndescr_155178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 12), 'ndescr', False)
    # Obtaining the member 'append' of a type (line 99)
    append_155179 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 12), ndescr_155178, 'append')
    # Calling append(args, kwargs) (line 99)
    append_call_result_155187 = invoke(stypy.reporting.localization.Localization(__file__, 99, 12), append_155179, *[tuple_155180], **kwargs_155186)
    
    # SSA join for if statement (line 93)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to dtype(...): (line 100)
    # Processing the call arguments (line 100)
    # Getting the type of 'ndescr' (line 100)
    ndescr_155190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 20), 'ndescr', False)
    # Processing the call keyword arguments (line 100)
    kwargs_155191 = {}
    # Getting the type of 'np' (line 100)
    np_155188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 11), 'np', False)
    # Obtaining the member 'dtype' of a type (line 100)
    dtype_155189 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 11), np_155188, 'dtype')
    # Calling dtype(args, kwargs) (line 100)
    dtype_call_result_155192 = invoke(stypy.reporting.localization.Localization(__file__, 100, 11), dtype_155189, *[ndescr_155190], **kwargs_155191)
    
    # Assigning a type to the variable 'stypy_return_type' (line 100)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'stypy_return_type', dtype_call_result_155192)
    
    # ################# End of '_checknames(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_checknames' in the type store
    # Getting the type of 'stypy_return_type' (line 69)
    stypy_return_type_155193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_155193)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_checknames'
    return stypy_return_type_155193

# Assigning a type to the variable '_checknames' (line 69)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 0), '_checknames', _checknames)

@norecursion
def _get_fieldmask(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_get_fieldmask'
    module_type_store = module_type_store.open_function_context('_get_fieldmask', 103, 0, False)
    
    # Passed parameters checking function
    _get_fieldmask.stypy_localization = localization
    _get_fieldmask.stypy_type_of_self = None
    _get_fieldmask.stypy_type_store = module_type_store
    _get_fieldmask.stypy_function_name = '_get_fieldmask'
    _get_fieldmask.stypy_param_names_list = ['self']
    _get_fieldmask.stypy_varargs_param_name = None
    _get_fieldmask.stypy_kwargs_param_name = None
    _get_fieldmask.stypy_call_defaults = defaults
    _get_fieldmask.stypy_call_varargs = varargs
    _get_fieldmask.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_get_fieldmask', ['self'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_get_fieldmask', localization, ['self'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_get_fieldmask(...)' code ##################

    
    # Assigning a ListComp to a Name (line 104):
    
    # Assigning a ListComp to a Name (line 104):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'self' (line 104)
    self_155197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 34), 'self')
    # Obtaining the member 'dtype' of a type (line 104)
    dtype_155198 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 34), self_155197, 'dtype')
    # Obtaining the member 'names' of a type (line 104)
    names_155199 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 34), dtype_155198, 'names')
    comprehension_155200 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 14), names_155199)
    # Assigning a type to the variable 'n' (line 104)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 14), 'n', comprehension_155200)
    
    # Obtaining an instance of the builtin type 'tuple' (line 104)
    tuple_155194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 104)
    # Adding element type (line 104)
    # Getting the type of 'n' (line 104)
    n_155195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 15), 'n')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 15), tuple_155194, n_155195)
    # Adding element type (line 104)
    str_155196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 18), 'str', '|b1')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 15), tuple_155194, str_155196)
    
    list_155201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 14), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 14), list_155201, tuple_155194)
    # Assigning a type to the variable 'mdescr' (line 104)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 4), 'mdescr', list_155201)
    
    # Assigning a Call to a Name (line 105):
    
    # Assigning a Call to a Name (line 105):
    
    # Call to empty(...): (line 105)
    # Processing the call arguments (line 105)
    # Getting the type of 'self' (line 105)
    self_155204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 22), 'self', False)
    # Obtaining the member 'shape' of a type (line 105)
    shape_155205 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 22), self_155204, 'shape')
    # Processing the call keyword arguments (line 105)
    # Getting the type of 'mdescr' (line 105)
    mdescr_155206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 40), 'mdescr', False)
    keyword_155207 = mdescr_155206
    kwargs_155208 = {'dtype': keyword_155207}
    # Getting the type of 'np' (line 105)
    np_155202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 13), 'np', False)
    # Obtaining the member 'empty' of a type (line 105)
    empty_155203 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 13), np_155202, 'empty')
    # Calling empty(args, kwargs) (line 105)
    empty_call_result_155209 = invoke(stypy.reporting.localization.Localization(__file__, 105, 13), empty_155203, *[shape_155205], **kwargs_155208)
    
    # Assigning a type to the variable 'fdmask' (line 105)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 4), 'fdmask', empty_call_result_155209)
    
    # Assigning a Call to a Attribute (line 106):
    
    # Assigning a Call to a Attribute (line 106):
    
    # Call to tuple(...): (line 106)
    # Processing the call arguments (line 106)
    
    # Obtaining an instance of the builtin type 'list' (line 106)
    list_155211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 106)
    # Adding element type (line 106)
    # Getting the type of 'False' (line 106)
    False_155212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 25), 'False', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 24), list_155211, False_155212)
    
    
    # Call to len(...): (line 106)
    # Processing the call arguments (line 106)
    # Getting the type of 'mdescr' (line 106)
    mdescr_155214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 38), 'mdescr', False)
    # Processing the call keyword arguments (line 106)
    kwargs_155215 = {}
    # Getting the type of 'len' (line 106)
    len_155213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 34), 'len', False)
    # Calling len(args, kwargs) (line 106)
    len_call_result_155216 = invoke(stypy.reporting.localization.Localization(__file__, 106, 34), len_155213, *[mdescr_155214], **kwargs_155215)
    
    # Applying the binary operator '*' (line 106)
    result_mul_155217 = python_operator(stypy.reporting.localization.Localization(__file__, 106, 24), '*', list_155211, len_call_result_155216)
    
    # Processing the call keyword arguments (line 106)
    kwargs_155218 = {}
    # Getting the type of 'tuple' (line 106)
    tuple_155210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 18), 'tuple', False)
    # Calling tuple(args, kwargs) (line 106)
    tuple_call_result_155219 = invoke(stypy.reporting.localization.Localization(__file__, 106, 18), tuple_155210, *[result_mul_155217], **kwargs_155218)
    
    # Getting the type of 'fdmask' (line 106)
    fdmask_155220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 4), 'fdmask')
    # Setting the type of the member 'flat' of a type (line 106)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 4), fdmask_155220, 'flat', tuple_call_result_155219)
    # Getting the type of 'fdmask' (line 107)
    fdmask_155221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 11), 'fdmask')
    # Assigning a type to the variable 'stypy_return_type' (line 107)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 4), 'stypy_return_type', fdmask_155221)
    
    # ################# End of '_get_fieldmask(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_get_fieldmask' in the type store
    # Getting the type of 'stypy_return_type' (line 103)
    stypy_return_type_155222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_155222)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_get_fieldmask'
    return stypy_return_type_155222

# Assigning a type to the variable '_get_fieldmask' (line 103)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 0), '_get_fieldmask', _get_fieldmask)
# Declaration of the 'MaskedRecords' class
# Getting the type of 'MaskedArray' (line 110)
MaskedArray_155223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 20), 'MaskedArray')
# Getting the type of 'object' (line 110)
object_155224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 33), 'object')

class MaskedRecords(MaskedArray_155223, object_155224, ):
    str_155225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, (-1)), 'str', '\n\n    Attributes\n    ----------\n    _data : recarray\n        Underlying data, as a record array.\n    _mask : boolean array\n        Mask of the records. A record is masked when all its fields are\n        masked.\n    _fieldmask : boolean recarray\n        Record array of booleans, setting the mask of each individual field\n        of each record.\n    _fill_value : record\n        Filling values for each field.\n\n    ')

    @norecursion
    def __new__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 128)
        None_155226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 34), 'None')
        # Getting the type of 'None' (line 128)
        None_155227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 44), 'None')
        int_155228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 57), 'int')
        # Getting the type of 'None' (line 128)
        None_155229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 68), 'None')
        # Getting the type of 'None' (line 129)
        None_155230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 24), 'None')
        # Getting the type of 'None' (line 129)
        None_155231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 36), 'None')
        # Getting the type of 'None' (line 129)
        None_155232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 49), 'None')
        # Getting the type of 'None' (line 130)
        None_155233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 26), 'None')
        # Getting the type of 'False' (line 130)
        False_155234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 40), 'False')
        # Getting the type of 'nomask' (line 131)
        nomask_155235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 21), 'nomask')
        # Getting the type of 'False' (line 131)
        False_155236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 39), 'False')
        # Getting the type of 'None' (line 131)
        None_155237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 57), 'None')
        # Getting the type of 'True' (line 131)
        True_155238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 73), 'True')
        # Getting the type of 'False' (line 132)
        False_155239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 21), 'False')
        defaults = [None_155226, None_155227, int_155228, None_155229, None_155230, None_155231, None_155232, None_155233, False_155234, nomask_155235, False_155236, None_155237, True_155238, False_155239]
        # Create a new context for function '__new__'
        module_type_store = module_type_store.open_function_context('__new__', 128, 4, False)
        # Assigning a type to the variable 'self' (line 129)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MaskedRecords.__new__.__dict__.__setitem__('stypy_localization', localization)
        MaskedRecords.__new__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MaskedRecords.__new__.__dict__.__setitem__('stypy_type_store', module_type_store)
        MaskedRecords.__new__.__dict__.__setitem__('stypy_function_name', 'MaskedRecords.__new__')
        MaskedRecords.__new__.__dict__.__setitem__('stypy_param_names_list', ['shape', 'dtype', 'buf', 'offset', 'strides', 'formats', 'names', 'titles', 'byteorder', 'aligned', 'mask', 'hard_mask', 'fill_value', 'keep_mask', 'copy'])
        MaskedRecords.__new__.__dict__.__setitem__('stypy_varargs_param_name', None)
        MaskedRecords.__new__.__dict__.__setitem__('stypy_kwargs_param_name', 'options')
        MaskedRecords.__new__.__dict__.__setitem__('stypy_call_defaults', defaults)
        MaskedRecords.__new__.__dict__.__setitem__('stypy_call_varargs', varargs)
        MaskedRecords.__new__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MaskedRecords.__new__.__dict__.__setitem__('stypy_declared_arg_number', 16)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MaskedRecords.__new__', ['shape', 'dtype', 'buf', 'offset', 'strides', 'formats', 'names', 'titles', 'byteorder', 'aligned', 'mask', 'hard_mask', 'fill_value', 'keep_mask', 'copy'], None, 'options', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__new__', localization, ['shape', 'dtype', 'buf', 'offset', 'strides', 'formats', 'names', 'titles', 'byteorder', 'aligned', 'mask', 'hard_mask', 'fill_value', 'keep_mask', 'copy'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__new__(...)' code ##################

        
        # Assigning a Call to a Name (line 135):
        
        # Assigning a Call to a Name (line 135):
        
        # Call to __new__(...): (line 135)
        # Processing the call arguments (line 135)
        # Getting the type of 'cls' (line 135)
        cls_155242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 32), 'cls', False)
        # Getting the type of 'shape' (line 135)
        shape_155243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 37), 'shape', False)
        # Processing the call keyword arguments (line 135)
        # Getting the type of 'dtype' (line 135)
        dtype_155244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 50), 'dtype', False)
        keyword_155245 = dtype_155244
        # Getting the type of 'buf' (line 135)
        buf_155246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 61), 'buf', False)
        keyword_155247 = buf_155246
        # Getting the type of 'offset' (line 135)
        offset_155248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 73), 'offset', False)
        keyword_155249 = offset_155248
        # Getting the type of 'strides' (line 136)
        strides_155250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 40), 'strides', False)
        keyword_155251 = strides_155250
        # Getting the type of 'formats' (line 136)
        formats_155252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 57), 'formats', False)
        keyword_155253 = formats_155252
        # Getting the type of 'names' (line 136)
        names_155254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 72), 'names', False)
        keyword_155255 = names_155254
        # Getting the type of 'titles' (line 137)
        titles_155256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 39), 'titles', False)
        keyword_155257 = titles_155256
        # Getting the type of 'byteorder' (line 137)
        byteorder_155258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 57), 'byteorder', False)
        keyword_155259 = byteorder_155258
        # Getting the type of 'aligned' (line 138)
        aligned_155260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 40), 'aligned', False)
        keyword_155261 = aligned_155260
        kwargs_155262 = {'dtype': keyword_155245, 'offset': keyword_155249, 'strides': keyword_155251, 'titles': keyword_155257, 'names': keyword_155255, 'formats': keyword_155253, 'aligned': keyword_155261, 'buf': keyword_155247, 'byteorder': keyword_155259}
        # Getting the type of 'recarray' (line 135)
        recarray_155240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 15), 'recarray', False)
        # Obtaining the member '__new__' of a type (line 135)
        new___155241 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 15), recarray_155240, '__new__')
        # Calling __new__(args, kwargs) (line 135)
        new___call_result_155263 = invoke(stypy.reporting.localization.Localization(__file__, 135, 15), new___155241, *[cls_155242, shape_155243], **kwargs_155262)
        
        # Assigning a type to the variable 'self' (line 135)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'self', new___call_result_155263)
        
        # Assigning a Call to a Name (line 140):
        
        # Assigning a Call to a Name (line 140):
        
        # Call to make_mask_descr(...): (line 140)
        # Processing the call arguments (line 140)
        # Getting the type of 'self' (line 140)
        self_155266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 36), 'self', False)
        # Obtaining the member 'dtype' of a type (line 140)
        dtype_155267 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 36), self_155266, 'dtype')
        # Processing the call keyword arguments (line 140)
        kwargs_155268 = {}
        # Getting the type of 'ma' (line 140)
        ma_155264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 17), 'ma', False)
        # Obtaining the member 'make_mask_descr' of a type (line 140)
        make_mask_descr_155265 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 17), ma_155264, 'make_mask_descr')
        # Calling make_mask_descr(args, kwargs) (line 140)
        make_mask_descr_call_result_155269 = invoke(stypy.reporting.localization.Localization(__file__, 140, 17), make_mask_descr_155265, *[dtype_155267], **kwargs_155268)
        
        # Assigning a type to the variable 'mdtype' (line 140)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 8), 'mdtype', make_mask_descr_call_result_155269)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'mask' (line 141)
        mask_155270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 11), 'mask')
        # Getting the type of 'nomask' (line 141)
        nomask_155271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 19), 'nomask')
        # Applying the binary operator 'is' (line 141)
        result_is__155272 = python_operator(stypy.reporting.localization.Localization(__file__, 141, 11), 'is', mask_155270, nomask_155271)
        
        
        
        # Call to size(...): (line 141)
        # Processing the call arguments (line 141)
        # Getting the type of 'mask' (line 141)
        mask_155275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 41), 'mask', False)
        # Processing the call keyword arguments (line 141)
        kwargs_155276 = {}
        # Getting the type of 'np' (line 141)
        np_155273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 33), 'np', False)
        # Obtaining the member 'size' of a type (line 141)
        size_155274 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 33), np_155273, 'size')
        # Calling size(args, kwargs) (line 141)
        size_call_result_155277 = invoke(stypy.reporting.localization.Localization(__file__, 141, 33), size_155274, *[mask_155275], **kwargs_155276)
        
        # Applying the 'not' unary operator (line 141)
        result_not__155278 = python_operator(stypy.reporting.localization.Localization(__file__, 141, 29), 'not', size_call_result_155277)
        
        # Applying the binary operator 'or' (line 141)
        result_or_keyword_155279 = python_operator(stypy.reporting.localization.Localization(__file__, 141, 11), 'or', result_is__155272, result_not__155278)
        
        # Testing the type of an if condition (line 141)
        if_condition_155280 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 141, 8), result_or_keyword_155279)
        # Assigning a type to the variable 'if_condition_155280' (line 141)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 8), 'if_condition_155280', if_condition_155280)
        # SSA begins for if statement (line 141)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Getting the type of 'keep_mask' (line 142)
        keep_mask_155281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 19), 'keep_mask')
        # Applying the 'not' unary operator (line 142)
        result_not__155282 = python_operator(stypy.reporting.localization.Localization(__file__, 142, 15), 'not', keep_mask_155281)
        
        # Testing the type of an if condition (line 142)
        if_condition_155283 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 142, 12), result_not__155282)
        # Assigning a type to the variable 'if_condition_155283' (line 142)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 12), 'if_condition_155283', if_condition_155283)
        # SSA begins for if statement (line 142)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Attribute (line 143):
        
        # Assigning a Call to a Attribute (line 143):
        
        # Call to tuple(...): (line 143)
        # Processing the call arguments (line 143)
        
        # Obtaining an instance of the builtin type 'list' (line 143)
        list_155285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 35), 'list')
        # Adding type elements to the builtin type 'list' instance (line 143)
        # Adding element type (line 143)
        # Getting the type of 'False' (line 143)
        False_155286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 36), 'False', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 143, 35), list_155285, False_155286)
        
        
        # Call to len(...): (line 143)
        # Processing the call arguments (line 143)
        # Getting the type of 'mdtype' (line 143)
        mdtype_155288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 49), 'mdtype', False)
        # Processing the call keyword arguments (line 143)
        kwargs_155289 = {}
        # Getting the type of 'len' (line 143)
        len_155287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 45), 'len', False)
        # Calling len(args, kwargs) (line 143)
        len_call_result_155290 = invoke(stypy.reporting.localization.Localization(__file__, 143, 45), len_155287, *[mdtype_155288], **kwargs_155289)
        
        # Applying the binary operator '*' (line 143)
        result_mul_155291 = python_operator(stypy.reporting.localization.Localization(__file__, 143, 35), '*', list_155285, len_call_result_155290)
        
        # Processing the call keyword arguments (line 143)
        kwargs_155292 = {}
        # Getting the type of 'tuple' (line 143)
        tuple_155284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 29), 'tuple', False)
        # Calling tuple(args, kwargs) (line 143)
        tuple_call_result_155293 = invoke(stypy.reporting.localization.Localization(__file__, 143, 29), tuple_155284, *[result_mul_155291], **kwargs_155292)
        
        # Getting the type of 'self' (line 143)
        self_155294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 16), 'self')
        # Setting the type of the member '_mask' of a type (line 143)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 16), self_155294, '_mask', tuple_call_result_155293)
        # SSA join for if statement (line 142)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 141)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 145):
        
        # Assigning a Call to a Name (line 145):
        
        # Call to array(...): (line 145)
        # Processing the call arguments (line 145)
        # Getting the type of 'mask' (line 145)
        mask_155297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 28), 'mask', False)
        # Processing the call keyword arguments (line 145)
        # Getting the type of 'copy' (line 145)
        copy_155298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 39), 'copy', False)
        keyword_155299 = copy_155298
        kwargs_155300 = {'copy': keyword_155299}
        # Getting the type of 'np' (line 145)
        np_155295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 19), 'np', False)
        # Obtaining the member 'array' of a type (line 145)
        array_155296 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 19), np_155295, 'array')
        # Calling array(args, kwargs) (line 145)
        array_call_result_155301 = invoke(stypy.reporting.localization.Localization(__file__, 145, 19), array_155296, *[mask_155297], **kwargs_155300)
        
        # Assigning a type to the variable 'mask' (line 145)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 12), 'mask', array_call_result_155301)
        
        
        # Getting the type of 'mask' (line 146)
        mask_155302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 15), 'mask')
        # Obtaining the member 'shape' of a type (line 146)
        shape_155303 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 15), mask_155302, 'shape')
        # Getting the type of 'self' (line 146)
        self_155304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 29), 'self')
        # Obtaining the member 'shape' of a type (line 146)
        shape_155305 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 29), self_155304, 'shape')
        # Applying the binary operator '!=' (line 146)
        result_ne_155306 = python_operator(stypy.reporting.localization.Localization(__file__, 146, 15), '!=', shape_155303, shape_155305)
        
        # Testing the type of an if condition (line 146)
        if_condition_155307 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 146, 12), result_ne_155306)
        # Assigning a type to the variable 'if_condition_155307' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 12), 'if_condition_155307', if_condition_155307)
        # SSA begins for if statement (line 146)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Tuple to a Tuple (line 147):
        
        # Assigning a Attribute to a Name (line 147):
        # Getting the type of 'self' (line 147)
        self_155308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 28), 'self')
        # Obtaining the member 'size' of a type (line 147)
        size_155309 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 28), self_155308, 'size')
        # Assigning a type to the variable 'tuple_assignment_154973' (line 147)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 16), 'tuple_assignment_154973', size_155309)
        
        # Assigning a Attribute to a Name (line 147):
        # Getting the type of 'mask' (line 147)
        mask_155310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 39), 'mask')
        # Obtaining the member 'size' of a type (line 147)
        size_155311 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 39), mask_155310, 'size')
        # Assigning a type to the variable 'tuple_assignment_154974' (line 147)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 16), 'tuple_assignment_154974', size_155311)
        
        # Assigning a Name to a Name (line 147):
        # Getting the type of 'tuple_assignment_154973' (line 147)
        tuple_assignment_154973_155312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 16), 'tuple_assignment_154973')
        # Assigning a type to the variable 'nd' (line 147)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 17), 'nd', tuple_assignment_154973_155312)
        
        # Assigning a Name to a Name (line 147):
        # Getting the type of 'tuple_assignment_154974' (line 147)
        tuple_assignment_154974_155313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 16), 'tuple_assignment_154974')
        # Assigning a type to the variable 'nm' (line 147)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 21), 'nm', tuple_assignment_154974_155313)
        
        
        # Getting the type of 'nm' (line 148)
        nm_155314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 19), 'nm')
        int_155315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 25), 'int')
        # Applying the binary operator '==' (line 148)
        result_eq_155316 = python_operator(stypy.reporting.localization.Localization(__file__, 148, 19), '==', nm_155314, int_155315)
        
        # Testing the type of an if condition (line 148)
        if_condition_155317 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 148, 16), result_eq_155316)
        # Assigning a type to the variable 'if_condition_155317' (line 148)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 16), 'if_condition_155317', if_condition_155317)
        # SSA begins for if statement (line 148)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 149):
        
        # Assigning a Call to a Name (line 149):
        
        # Call to resize(...): (line 149)
        # Processing the call arguments (line 149)
        # Getting the type of 'mask' (line 149)
        mask_155320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 37), 'mask', False)
        # Getting the type of 'self' (line 149)
        self_155321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 43), 'self', False)
        # Obtaining the member 'shape' of a type (line 149)
        shape_155322 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 43), self_155321, 'shape')
        # Processing the call keyword arguments (line 149)
        kwargs_155323 = {}
        # Getting the type of 'np' (line 149)
        np_155318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 27), 'np', False)
        # Obtaining the member 'resize' of a type (line 149)
        resize_155319 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 27), np_155318, 'resize')
        # Calling resize(args, kwargs) (line 149)
        resize_call_result_155324 = invoke(stypy.reporting.localization.Localization(__file__, 149, 27), resize_155319, *[mask_155320, shape_155322], **kwargs_155323)
        
        # Assigning a type to the variable 'mask' (line 149)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 20), 'mask', resize_call_result_155324)
        # SSA branch for the else part of an if statement (line 148)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'nm' (line 150)
        nm_155325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 21), 'nm')
        # Getting the type of 'nd' (line 150)
        nd_155326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 27), 'nd')
        # Applying the binary operator '==' (line 150)
        result_eq_155327 = python_operator(stypy.reporting.localization.Localization(__file__, 150, 21), '==', nm_155325, nd_155326)
        
        # Testing the type of an if condition (line 150)
        if_condition_155328 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 150, 21), result_eq_155327)
        # Assigning a type to the variable 'if_condition_155328' (line 150)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 21), 'if_condition_155328', if_condition_155328)
        # SSA begins for if statement (line 150)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 151):
        
        # Assigning a Call to a Name (line 151):
        
        # Call to reshape(...): (line 151)
        # Processing the call arguments (line 151)
        # Getting the type of 'mask' (line 151)
        mask_155331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 38), 'mask', False)
        # Getting the type of 'self' (line 151)
        self_155332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 44), 'self', False)
        # Obtaining the member 'shape' of a type (line 151)
        shape_155333 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 44), self_155332, 'shape')
        # Processing the call keyword arguments (line 151)
        kwargs_155334 = {}
        # Getting the type of 'np' (line 151)
        np_155329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 27), 'np', False)
        # Obtaining the member 'reshape' of a type (line 151)
        reshape_155330 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 27), np_155329, 'reshape')
        # Calling reshape(args, kwargs) (line 151)
        reshape_call_result_155335 = invoke(stypy.reporting.localization.Localization(__file__, 151, 27), reshape_155330, *[mask_155331, shape_155333], **kwargs_155334)
        
        # Assigning a type to the variable 'mask' (line 151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 20), 'mask', reshape_call_result_155335)
        # SSA branch for the else part of an if statement (line 150)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a BinOp to a Name (line 153):
        
        # Assigning a BinOp to a Name (line 153):
        str_155336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 26), 'str', 'Mask and data not compatible: data size is %i, ')
        str_155337 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 26), 'str', 'mask size is %i.')
        # Applying the binary operator '+' (line 153)
        result_add_155338 = python_operator(stypy.reporting.localization.Localization(__file__, 153, 26), '+', str_155336, str_155337)
        
        # Assigning a type to the variable 'msg' (line 153)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 20), 'msg', result_add_155338)
        
        # Call to MAError(...): (line 155)
        # Processing the call arguments (line 155)
        # Getting the type of 'msg' (line 155)
        msg_155340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 34), 'msg', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 155)
        tuple_155341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 41), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 155)
        # Adding element type (line 155)
        # Getting the type of 'nd' (line 155)
        nd_155342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 41), 'nd', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 155, 41), tuple_155341, nd_155342)
        # Adding element type (line 155)
        # Getting the type of 'nm' (line 155)
        nm_155343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 45), 'nm', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 155, 41), tuple_155341, nm_155343)
        
        # Applying the binary operator '%' (line 155)
        result_mod_155344 = python_operator(stypy.reporting.localization.Localization(__file__, 155, 34), '%', msg_155340, tuple_155341)
        
        # Processing the call keyword arguments (line 155)
        kwargs_155345 = {}
        # Getting the type of 'MAError' (line 155)
        MAError_155339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 26), 'MAError', False)
        # Calling MAError(args, kwargs) (line 155)
        MAError_call_result_155346 = invoke(stypy.reporting.localization.Localization(__file__, 155, 26), MAError_155339, *[result_mod_155344], **kwargs_155345)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 155, 20), MAError_call_result_155346, 'raise parameter', BaseException)
        # SSA join for if statement (line 150)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 148)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Name (line 156):
        
        # Assigning a Name to a Name (line 156):
        # Getting the type of 'True' (line 156)
        True_155347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 23), 'True')
        # Assigning a type to the variable 'copy' (line 156)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 16), 'copy', True_155347)
        # SSA join for if statement (line 146)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'keep_mask' (line 157)
        keep_mask_155348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 19), 'keep_mask')
        # Applying the 'not' unary operator (line 157)
        result_not__155349 = python_operator(stypy.reporting.localization.Localization(__file__, 157, 15), 'not', keep_mask_155348)
        
        # Testing the type of an if condition (line 157)
        if_condition_155350 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 157, 12), result_not__155349)
        # Assigning a type to the variable 'if_condition_155350' (line 157)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 12), 'if_condition_155350', if_condition_155350)
        # SSA begins for if statement (line 157)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to __setmask__(...): (line 158)
        # Processing the call arguments (line 158)
        # Getting the type of 'mask' (line 158)
        mask_155353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 33), 'mask', False)
        # Processing the call keyword arguments (line 158)
        kwargs_155354 = {}
        # Getting the type of 'self' (line 158)
        self_155351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 16), 'self', False)
        # Obtaining the member '__setmask__' of a type (line 158)
        setmask___155352 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 16), self_155351, '__setmask__')
        # Calling __setmask__(args, kwargs) (line 158)
        setmask___call_result_155355 = invoke(stypy.reporting.localization.Localization(__file__, 158, 16), setmask___155352, *[mask_155353], **kwargs_155354)
        
        
        # Assigning a Name to a Attribute (line 159):
        
        # Assigning a Name to a Attribute (line 159):
        # Getting the type of 'True' (line 159)
        True_155356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 35), 'True')
        # Getting the type of 'self' (line 159)
        self_155357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 16), 'self')
        # Setting the type of the member '_sharedmask' of a type (line 159)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 16), self_155357, '_sharedmask', True_155356)
        # SSA branch for the else part of an if statement (line 157)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'mask' (line 161)
        mask_155358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 19), 'mask')
        # Obtaining the member 'dtype' of a type (line 161)
        dtype_155359 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 19), mask_155358, 'dtype')
        # Getting the type of 'mdtype' (line 161)
        mdtype_155360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 33), 'mdtype')
        # Applying the binary operator '==' (line 161)
        result_eq_155361 = python_operator(stypy.reporting.localization.Localization(__file__, 161, 19), '==', dtype_155359, mdtype_155360)
        
        # Testing the type of an if condition (line 161)
        if_condition_155362 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 161, 16), result_eq_155361)
        # Assigning a type to the variable 'if_condition_155362' (line 161)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 16), 'if_condition_155362', if_condition_155362)
        # SSA begins for if statement (line 161)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 162):
        
        # Assigning a Name to a Name (line 162):
        # Getting the type of 'mask' (line 162)
        mask_155363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 28), 'mask')
        # Assigning a type to the variable '_mask' (line 162)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 20), '_mask', mask_155363)
        # SSA branch for the else part of an if statement (line 161)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 164):
        
        # Assigning a Call to a Name (line 164):
        
        # Call to array(...): (line 164)
        # Processing the call arguments (line 164)
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'mask' (line 164)
        mask_155376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 72), 'mask', False)
        comprehension_155377 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 38), mask_155376)
        # Assigning a type to the variable 'm' (line 164)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 38), 'm', comprehension_155377)
        
        # Call to tuple(...): (line 164)
        # Processing the call arguments (line 164)
        
        # Obtaining an instance of the builtin type 'list' (line 164)
        list_155367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 44), 'list')
        # Adding type elements to the builtin type 'list' instance (line 164)
        # Adding element type (line 164)
        # Getting the type of 'm' (line 164)
        m_155368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 45), 'm', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 44), list_155367, m_155368)
        
        
        # Call to len(...): (line 164)
        # Processing the call arguments (line 164)
        # Getting the type of 'mdtype' (line 164)
        mdtype_155370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 54), 'mdtype', False)
        # Processing the call keyword arguments (line 164)
        kwargs_155371 = {}
        # Getting the type of 'len' (line 164)
        len_155369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 50), 'len', False)
        # Calling len(args, kwargs) (line 164)
        len_call_result_155372 = invoke(stypy.reporting.localization.Localization(__file__, 164, 50), len_155369, *[mdtype_155370], **kwargs_155371)
        
        # Applying the binary operator '*' (line 164)
        result_mul_155373 = python_operator(stypy.reporting.localization.Localization(__file__, 164, 44), '*', list_155367, len_call_result_155372)
        
        # Processing the call keyword arguments (line 164)
        kwargs_155374 = {}
        # Getting the type of 'tuple' (line 164)
        tuple_155366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 38), 'tuple', False)
        # Calling tuple(args, kwargs) (line 164)
        tuple_call_result_155375 = invoke(stypy.reporting.localization.Localization(__file__, 164, 38), tuple_155366, *[result_mul_155373], **kwargs_155374)
        
        list_155378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 38), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 38), list_155378, tuple_call_result_155375)
        # Processing the call keyword arguments (line 164)
        # Getting the type of 'mdtype' (line 165)
        mdtype_155379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 43), 'mdtype', False)
        keyword_155380 = mdtype_155379
        kwargs_155381 = {'dtype': keyword_155380}
        # Getting the type of 'np' (line 164)
        np_155364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 28), 'np', False)
        # Obtaining the member 'array' of a type (line 164)
        array_155365 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 28), np_155364, 'array')
        # Calling array(args, kwargs) (line 164)
        array_call_result_155382 = invoke(stypy.reporting.localization.Localization(__file__, 164, 28), array_155365, *[list_155378], **kwargs_155381)
        
        # Assigning a type to the variable '_mask' (line 164)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 20), '_mask', array_call_result_155382)
        # SSA join for if statement (line 161)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 166):
        
        # Assigning a Name to a Attribute (line 166):
        # Getting the type of '_mask' (line 166)
        _mask_155383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 29), '_mask')
        # Getting the type of 'self' (line 166)
        self_155384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 16), 'self')
        # Setting the type of the member '_mask' of a type (line 166)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 16), self_155384, '_mask', _mask_155383)
        # SSA join for if statement (line 157)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 141)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'self' (line 167)
        self_155385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 15), 'self')
        # Assigning a type to the variable 'stypy_return_type' (line 167)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 'stypy_return_type', self_155385)
        
        # ################# End of '__new__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__new__' in the type store
        # Getting the type of 'stypy_return_type' (line 128)
        stypy_return_type_155386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_155386)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__new__'
        return stypy_return_type_155386


    @norecursion
    def __array_finalize__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__array_finalize__'
        module_type_store = module_type_store.open_function_context('__array_finalize__', 169, 4, False)
        # Assigning a type to the variable 'self' (line 170)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MaskedRecords.__array_finalize__.__dict__.__setitem__('stypy_localization', localization)
        MaskedRecords.__array_finalize__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MaskedRecords.__array_finalize__.__dict__.__setitem__('stypy_type_store', module_type_store)
        MaskedRecords.__array_finalize__.__dict__.__setitem__('stypy_function_name', 'MaskedRecords.__array_finalize__')
        MaskedRecords.__array_finalize__.__dict__.__setitem__('stypy_param_names_list', ['obj'])
        MaskedRecords.__array_finalize__.__dict__.__setitem__('stypy_varargs_param_name', None)
        MaskedRecords.__array_finalize__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MaskedRecords.__array_finalize__.__dict__.__setitem__('stypy_call_defaults', defaults)
        MaskedRecords.__array_finalize__.__dict__.__setitem__('stypy_call_varargs', varargs)
        MaskedRecords.__array_finalize__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MaskedRecords.__array_finalize__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MaskedRecords.__array_finalize__', ['obj'], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Call to a Name (line 171):
        
        # Assigning a Call to a Name (line 171):
        
        # Call to getattr(...): (line 171)
        # Processing the call arguments (line 171)
        # Getting the type of 'obj' (line 171)
        obj_155388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 24), 'obj', False)
        str_155389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 29), 'str', '_mask')
        # Getting the type of 'None' (line 171)
        None_155390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 38), 'None', False)
        # Processing the call keyword arguments (line 171)
        kwargs_155391 = {}
        # Getting the type of 'getattr' (line 171)
        getattr_155387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 16), 'getattr', False)
        # Calling getattr(args, kwargs) (line 171)
        getattr_call_result_155392 = invoke(stypy.reporting.localization.Localization(__file__, 171, 16), getattr_155387, *[obj_155388, str_155389, None_155390], **kwargs_155391)
        
        # Assigning a type to the variable '_mask' (line 171)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 8), '_mask', getattr_call_result_155392)
        
        # Type idiom detected: calculating its left and rigth part (line 172)
        # Getting the type of '_mask' (line 172)
        _mask_155393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 11), '_mask')
        # Getting the type of 'None' (line 172)
        None_155394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 20), 'None')
        
        (may_be_155395, more_types_in_union_155396) = may_be_none(_mask_155393, None_155394)

        if may_be_155395:

            if more_types_in_union_155396:
                # Runtime conditional SSA (line 172)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 173):
            
            # Assigning a Call to a Name (line 173):
            
            # Call to getattr(...): (line 173)
            # Processing the call arguments (line 173)
            # Getting the type of 'obj' (line 173)
            obj_155398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 30), 'obj', False)
            str_155399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 35), 'str', '_mask')
            # Getting the type of 'nomask' (line 173)
            nomask_155400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 44), 'nomask', False)
            # Processing the call keyword arguments (line 173)
            kwargs_155401 = {}
            # Getting the type of 'getattr' (line 173)
            getattr_155397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 22), 'getattr', False)
            # Calling getattr(args, kwargs) (line 173)
            getattr_call_result_155402 = invoke(stypy.reporting.localization.Localization(__file__, 173, 22), getattr_155397, *[obj_155398, str_155399, nomask_155400], **kwargs_155401)
            
            # Assigning a type to the variable 'objmask' (line 173)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 12), 'objmask', getattr_call_result_155402)
            
            # Assigning a Call to a Name (line 174):
            
            # Assigning a Call to a Name (line 174):
            
            # Call to __getattribute__(...): (line 174)
            # Processing the call arguments (line 174)
            # Getting the type of 'self' (line 174)
            self_155405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 46), 'self', False)
            str_155406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 52), 'str', 'dtype')
            # Processing the call keyword arguments (line 174)
            kwargs_155407 = {}
            # Getting the type of 'ndarray' (line 174)
            ndarray_155403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 21), 'ndarray', False)
            # Obtaining the member '__getattribute__' of a type (line 174)
            getattribute___155404 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 21), ndarray_155403, '__getattribute__')
            # Calling __getattribute__(args, kwargs) (line 174)
            getattribute___call_result_155408 = invoke(stypy.reporting.localization.Localization(__file__, 174, 21), getattribute___155404, *[self_155405, str_155406], **kwargs_155407)
            
            # Assigning a type to the variable '_dtype' (line 174)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 12), '_dtype', getattribute___call_result_155408)
            
            
            # Getting the type of 'objmask' (line 175)
            objmask_155409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 15), 'objmask')
            # Getting the type of 'nomask' (line 175)
            nomask_155410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 26), 'nomask')
            # Applying the binary operator 'is' (line 175)
            result_is__155411 = python_operator(stypy.reporting.localization.Localization(__file__, 175, 15), 'is', objmask_155409, nomask_155410)
            
            # Testing the type of an if condition (line 175)
            if_condition_155412 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 175, 12), result_is__155411)
            # Assigning a type to the variable 'if_condition_155412' (line 175)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 12), 'if_condition_155412', if_condition_155412)
            # SSA begins for if statement (line 175)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 176):
            
            # Assigning a Call to a Name (line 176):
            
            # Call to make_mask_none(...): (line 176)
            # Processing the call arguments (line 176)
            # Getting the type of 'self' (line 176)
            self_155415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 42), 'self', False)
            # Obtaining the member 'shape' of a type (line 176)
            shape_155416 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 42), self_155415, 'shape')
            # Processing the call keyword arguments (line 176)
            # Getting the type of '_dtype' (line 176)
            _dtype_155417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 60), '_dtype', False)
            keyword_155418 = _dtype_155417
            kwargs_155419 = {'dtype': keyword_155418}
            # Getting the type of 'ma' (line 176)
            ma_155413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 24), 'ma', False)
            # Obtaining the member 'make_mask_none' of a type (line 176)
            make_mask_none_155414 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 24), ma_155413, 'make_mask_none')
            # Calling make_mask_none(args, kwargs) (line 176)
            make_mask_none_call_result_155420 = invoke(stypy.reporting.localization.Localization(__file__, 176, 24), make_mask_none_155414, *[shape_155416], **kwargs_155419)
            
            # Assigning a type to the variable '_mask' (line 176)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 16), '_mask', make_mask_none_call_result_155420)
            # SSA branch for the else part of an if statement (line 175)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Call to a Name (line 178):
            
            # Assigning a Call to a Name (line 178):
            
            # Call to make_mask_descr(...): (line 178)
            # Processing the call arguments (line 178)
            # Getting the type of '_dtype' (line 178)
            _dtype_155423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 44), '_dtype', False)
            # Processing the call keyword arguments (line 178)
            kwargs_155424 = {}
            # Getting the type of 'ma' (line 178)
            ma_155421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 25), 'ma', False)
            # Obtaining the member 'make_mask_descr' of a type (line 178)
            make_mask_descr_155422 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 25), ma_155421, 'make_mask_descr')
            # Calling make_mask_descr(args, kwargs) (line 178)
            make_mask_descr_call_result_155425 = invoke(stypy.reporting.localization.Localization(__file__, 178, 25), make_mask_descr_155422, *[_dtype_155423], **kwargs_155424)
            
            # Assigning a type to the variable 'mdescr' (line 178)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 16), 'mdescr', make_mask_descr_call_result_155425)
            
            # Assigning a Call to a Name (line 179):
            
            # Assigning a Call to a Name (line 179):
            
            # Call to view(...): (line 179)
            # Processing the call arguments (line 179)
            # Getting the type of 'recarray' (line 180)
            recarray_155445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 50), 'recarray', False)
            # Processing the call keyword arguments (line 179)
            kwargs_155446 = {}
            
            # Call to narray(...): (line 179)
            # Processing the call arguments (line 179)
            # Calculating list comprehension
            # Calculating comprehension expression
            # Getting the type of 'objmask' (line 179)
            objmask_155437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 66), 'objmask', False)
            comprehension_155438 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 32), objmask_155437)
            # Assigning a type to the variable 'm' (line 179)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 32), 'm', comprehension_155438)
            
            # Call to tuple(...): (line 179)
            # Processing the call arguments (line 179)
            
            # Obtaining an instance of the builtin type 'list' (line 179)
            list_155428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 38), 'list')
            # Adding type elements to the builtin type 'list' instance (line 179)
            # Adding element type (line 179)
            # Getting the type of 'm' (line 179)
            m_155429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 39), 'm', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 38), list_155428, m_155429)
            
            
            # Call to len(...): (line 179)
            # Processing the call arguments (line 179)
            # Getting the type of 'mdescr' (line 179)
            mdescr_155431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 48), 'mdescr', False)
            # Processing the call keyword arguments (line 179)
            kwargs_155432 = {}
            # Getting the type of 'len' (line 179)
            len_155430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 44), 'len', False)
            # Calling len(args, kwargs) (line 179)
            len_call_result_155433 = invoke(stypy.reporting.localization.Localization(__file__, 179, 44), len_155430, *[mdescr_155431], **kwargs_155432)
            
            # Applying the binary operator '*' (line 179)
            result_mul_155434 = python_operator(stypy.reporting.localization.Localization(__file__, 179, 38), '*', list_155428, len_call_result_155433)
            
            # Processing the call keyword arguments (line 179)
            kwargs_155435 = {}
            # Getting the type of 'tuple' (line 179)
            tuple_155427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 32), 'tuple', False)
            # Calling tuple(args, kwargs) (line 179)
            tuple_call_result_155436 = invoke(stypy.reporting.localization.Localization(__file__, 179, 32), tuple_155427, *[result_mul_155434], **kwargs_155435)
            
            list_155439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 32), 'list')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 32), list_155439, tuple_call_result_155436)
            # Processing the call keyword arguments (line 179)
            # Getting the type of 'mdescr' (line 180)
            mdescr_155440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 37), 'mdescr', False)
            keyword_155441 = mdescr_155440
            kwargs_155442 = {'dtype': keyword_155441}
            # Getting the type of 'narray' (line 179)
            narray_155426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 24), 'narray', False)
            # Calling narray(args, kwargs) (line 179)
            narray_call_result_155443 = invoke(stypy.reporting.localization.Localization(__file__, 179, 24), narray_155426, *[list_155439], **kwargs_155442)
            
            # Obtaining the member 'view' of a type (line 179)
            view_155444 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 24), narray_call_result_155443, 'view')
            # Calling view(args, kwargs) (line 179)
            view_call_result_155447 = invoke(stypy.reporting.localization.Localization(__file__, 179, 24), view_155444, *[recarray_155445], **kwargs_155446)
            
            # Assigning a type to the variable '_mask' (line 179)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 16), '_mask', view_call_result_155447)
            # SSA join for if statement (line 175)
            module_type_store = module_type_store.join_ssa_context()
            

            if more_types_in_union_155396:
                # SSA join for if statement (line 172)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Attribute to a Name (line 182):
        
        # Assigning a Attribute to a Name (line 182):
        # Getting the type of 'self' (line 182)
        self_155448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 16), 'self')
        # Obtaining the member '__dict__' of a type (line 182)
        dict___155449 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 16), self_155448, '__dict__')
        # Assigning a type to the variable '_dict' (line 182)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), '_dict', dict___155449)
        
        # Call to update(...): (line 183)
        # Processing the call keyword arguments (line 183)
        # Getting the type of '_mask' (line 183)
        _mask_155452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 27), '_mask', False)
        keyword_155453 = _mask_155452
        kwargs_155454 = {'_mask': keyword_155453}
        # Getting the type of '_dict' (line 183)
        _dict_155450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 8), '_dict', False)
        # Obtaining the member 'update' of a type (line 183)
        update_155451 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 8), _dict_155450, 'update')
        # Calling update(args, kwargs) (line 183)
        update_call_result_155455 = invoke(stypy.reporting.localization.Localization(__file__, 183, 8), update_155451, *[], **kwargs_155454)
        
        
        # Call to _update_from(...): (line 184)
        # Processing the call arguments (line 184)
        # Getting the type of 'obj' (line 184)
        obj_155458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 26), 'obj', False)
        # Processing the call keyword arguments (line 184)
        kwargs_155459 = {}
        # Getting the type of 'self' (line 184)
        self_155456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 'self', False)
        # Obtaining the member '_update_from' of a type (line 184)
        _update_from_155457 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 8), self_155456, '_update_from')
        # Calling _update_from(args, kwargs) (line 184)
        _update_from_call_result_155460 = invoke(stypy.reporting.localization.Localization(__file__, 184, 8), _update_from_155457, *[obj_155458], **kwargs_155459)
        
        
        
        
        # Obtaining the type of the subscript
        str_155461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 17), 'str', '_baseclass')
        # Getting the type of '_dict' (line 185)
        _dict_155462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 11), '_dict')
        # Obtaining the member '__getitem__' of a type (line 185)
        getitem___155463 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 11), _dict_155462, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 185)
        subscript_call_result_155464 = invoke(stypy.reporting.localization.Localization(__file__, 185, 11), getitem___155463, str_155461)
        
        # Getting the type of 'ndarray' (line 185)
        ndarray_155465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 34), 'ndarray')
        # Applying the binary operator '==' (line 185)
        result_eq_155466 = python_operator(stypy.reporting.localization.Localization(__file__, 185, 11), '==', subscript_call_result_155464, ndarray_155465)
        
        # Testing the type of an if condition (line 185)
        if_condition_155467 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 185, 8), result_eq_155466)
        # Assigning a type to the variable 'if_condition_155467' (line 185)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 8), 'if_condition_155467', if_condition_155467)
        # SSA begins for if statement (line 185)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Subscript (line 186):
        
        # Assigning a Name to a Subscript (line 186):
        # Getting the type of 'recarray' (line 186)
        recarray_155468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 34), 'recarray')
        # Getting the type of '_dict' (line 186)
        _dict_155469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 12), '_dict')
        str_155470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 18), 'str', '_baseclass')
        # Storing an element on a container (line 186)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 186, 12), _dict_155469, (str_155470, recarray_155468))
        # SSA join for if statement (line 185)
        module_type_store = module_type_store.join_ssa_context()
        
        # Assigning a type to the variable 'stypy_return_type' (line 187)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 8), 'stypy_return_type', types.NoneType)
        
        # ################# End of '__array_finalize__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__array_finalize__' in the type store
        # Getting the type of 'stypy_return_type' (line 169)
        stypy_return_type_155471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_155471)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__array_finalize__'
        return stypy_return_type_155471


    @norecursion
    def _getdata(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_getdata'
        module_type_store = module_type_store.open_function_context('_getdata', 189, 4, False)
        # Assigning a type to the variable 'self' (line 190)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MaskedRecords._getdata.__dict__.__setitem__('stypy_localization', localization)
        MaskedRecords._getdata.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MaskedRecords._getdata.__dict__.__setitem__('stypy_type_store', module_type_store)
        MaskedRecords._getdata.__dict__.__setitem__('stypy_function_name', 'MaskedRecords._getdata')
        MaskedRecords._getdata.__dict__.__setitem__('stypy_param_names_list', [])
        MaskedRecords._getdata.__dict__.__setitem__('stypy_varargs_param_name', None)
        MaskedRecords._getdata.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MaskedRecords._getdata.__dict__.__setitem__('stypy_call_defaults', defaults)
        MaskedRecords._getdata.__dict__.__setitem__('stypy_call_varargs', varargs)
        MaskedRecords._getdata.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MaskedRecords._getdata.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MaskedRecords._getdata', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_getdata', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_getdata(...)' code ##################

        str_155472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, (-1)), 'str', '\n        Returns the data as a recarray.\n\n        ')
        
        # Call to view(...): (line 194)
        # Processing the call arguments (line 194)
        # Getting the type of 'self' (line 194)
        self_155475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 28), 'self', False)
        # Getting the type of 'recarray' (line 194)
        recarray_155476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 34), 'recarray', False)
        # Processing the call keyword arguments (line 194)
        kwargs_155477 = {}
        # Getting the type of 'ndarray' (line 194)
        ndarray_155473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 15), 'ndarray', False)
        # Obtaining the member 'view' of a type (line 194)
        view_155474 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 15), ndarray_155473, 'view')
        # Calling view(args, kwargs) (line 194)
        view_call_result_155478 = invoke(stypy.reporting.localization.Localization(__file__, 194, 15), view_155474, *[self_155475, recarray_155476], **kwargs_155477)
        
        # Assigning a type to the variable 'stypy_return_type' (line 194)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 8), 'stypy_return_type', view_call_result_155478)
        
        # ################# End of '_getdata(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_getdata' in the type store
        # Getting the type of 'stypy_return_type' (line 189)
        stypy_return_type_155479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_155479)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_getdata'
        return stypy_return_type_155479

    
    # Assigning a Call to a Name (line 196):

    @norecursion
    def _getfieldmask(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_getfieldmask'
        module_type_store = module_type_store.open_function_context('_getfieldmask', 198, 4, False)
        # Assigning a type to the variable 'self' (line 199)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MaskedRecords._getfieldmask.__dict__.__setitem__('stypy_localization', localization)
        MaskedRecords._getfieldmask.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MaskedRecords._getfieldmask.__dict__.__setitem__('stypy_type_store', module_type_store)
        MaskedRecords._getfieldmask.__dict__.__setitem__('stypy_function_name', 'MaskedRecords._getfieldmask')
        MaskedRecords._getfieldmask.__dict__.__setitem__('stypy_param_names_list', [])
        MaskedRecords._getfieldmask.__dict__.__setitem__('stypy_varargs_param_name', None)
        MaskedRecords._getfieldmask.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MaskedRecords._getfieldmask.__dict__.__setitem__('stypy_call_defaults', defaults)
        MaskedRecords._getfieldmask.__dict__.__setitem__('stypy_call_varargs', varargs)
        MaskedRecords._getfieldmask.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MaskedRecords._getfieldmask.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MaskedRecords._getfieldmask', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_getfieldmask', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_getfieldmask(...)' code ##################

        str_155480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, (-1)), 'str', '\n        Alias to mask.\n\n        ')
        # Getting the type of 'self' (line 203)
        self_155481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 15), 'self')
        # Obtaining the member '_mask' of a type (line 203)
        _mask_155482 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 15), self_155481, '_mask')
        # Assigning a type to the variable 'stypy_return_type' (line 203)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 8), 'stypy_return_type', _mask_155482)
        
        # ################# End of '_getfieldmask(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_getfieldmask' in the type store
        # Getting the type of 'stypy_return_type' (line 198)
        stypy_return_type_155483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_155483)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_getfieldmask'
        return stypy_return_type_155483

    
    # Assigning a Call to a Name (line 205):

    @norecursion
    def __len__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__len__'
        module_type_store = module_type_store.open_function_context('__len__', 207, 4, False)
        # Assigning a type to the variable 'self' (line 208)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MaskedRecords.__len__.__dict__.__setitem__('stypy_localization', localization)
        MaskedRecords.__len__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MaskedRecords.__len__.__dict__.__setitem__('stypy_type_store', module_type_store)
        MaskedRecords.__len__.__dict__.__setitem__('stypy_function_name', 'MaskedRecords.__len__')
        MaskedRecords.__len__.__dict__.__setitem__('stypy_param_names_list', [])
        MaskedRecords.__len__.__dict__.__setitem__('stypy_varargs_param_name', None)
        MaskedRecords.__len__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MaskedRecords.__len__.__dict__.__setitem__('stypy_call_defaults', defaults)
        MaskedRecords.__len__.__dict__.__setitem__('stypy_call_varargs', varargs)
        MaskedRecords.__len__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MaskedRecords.__len__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MaskedRecords.__len__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__len__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__len__(...)' code ##################

        str_155484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, (-1)), 'str', '\n        Returns the length\n\n        ')
        
        # Getting the type of 'self' (line 213)
        self_155485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 11), 'self')
        # Obtaining the member 'ndim' of a type (line 213)
        ndim_155486 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 11), self_155485, 'ndim')
        # Testing the type of an if condition (line 213)
        if_condition_155487 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 213, 8), ndim_155486)
        # Assigning a type to the variable 'if_condition_155487' (line 213)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 8), 'if_condition_155487', if_condition_155487)
        # SSA begins for if statement (line 213)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to len(...): (line 214)
        # Processing the call arguments (line 214)
        # Getting the type of 'self' (line 214)
        self_155489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 23), 'self', False)
        # Obtaining the member '_data' of a type (line 214)
        _data_155490 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 23), self_155489, '_data')
        # Processing the call keyword arguments (line 214)
        kwargs_155491 = {}
        # Getting the type of 'len' (line 214)
        len_155488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 19), 'len', False)
        # Calling len(args, kwargs) (line 214)
        len_call_result_155492 = invoke(stypy.reporting.localization.Localization(__file__, 214, 19), len_155488, *[_data_155490], **kwargs_155491)
        
        # Assigning a type to the variable 'stypy_return_type' (line 214)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 12), 'stypy_return_type', len_call_result_155492)
        # SSA join for if statement (line 213)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to len(...): (line 216)
        # Processing the call arguments (line 216)
        # Getting the type of 'self' (line 216)
        self_155494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 19), 'self', False)
        # Obtaining the member 'dtype' of a type (line 216)
        dtype_155495 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 19), self_155494, 'dtype')
        # Processing the call keyword arguments (line 216)
        kwargs_155496 = {}
        # Getting the type of 'len' (line 216)
        len_155493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 15), 'len', False)
        # Calling len(args, kwargs) (line 216)
        len_call_result_155497 = invoke(stypy.reporting.localization.Localization(__file__, 216, 15), len_155493, *[dtype_155495], **kwargs_155496)
        
        # Assigning a type to the variable 'stypy_return_type' (line 216)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 8), 'stypy_return_type', len_call_result_155497)
        
        # ################# End of '__len__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__len__' in the type store
        # Getting the type of 'stypy_return_type' (line 207)
        stypy_return_type_155498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_155498)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__len__'
        return stypy_return_type_155498


    @norecursion
    def __getattribute__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__getattribute__'
        module_type_store = module_type_store.open_function_context('__getattribute__', 218, 4, False)
        # Assigning a type to the variable 'self' (line 219)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MaskedRecords.__getattribute__.__dict__.__setitem__('stypy_localization', localization)
        MaskedRecords.__getattribute__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MaskedRecords.__getattribute__.__dict__.__setitem__('stypy_type_store', module_type_store)
        MaskedRecords.__getattribute__.__dict__.__setitem__('stypy_function_name', 'MaskedRecords.__getattribute__')
        MaskedRecords.__getattribute__.__dict__.__setitem__('stypy_param_names_list', ['attr'])
        MaskedRecords.__getattribute__.__dict__.__setitem__('stypy_varargs_param_name', None)
        MaskedRecords.__getattribute__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MaskedRecords.__getattribute__.__dict__.__setitem__('stypy_call_defaults', defaults)
        MaskedRecords.__getattribute__.__dict__.__setitem__('stypy_call_varargs', varargs)
        MaskedRecords.__getattribute__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MaskedRecords.__getattribute__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MaskedRecords.__getattribute__', ['attr'], None, None, defaults, varargs, kwargs)

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

        
        
        # SSA begins for try-except statement (line 219)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to __getattribute__(...): (line 220)
        # Processing the call arguments (line 220)
        # Getting the type of 'self' (line 220)
        self_155501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 43), 'self', False)
        # Getting the type of 'attr' (line 220)
        attr_155502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 49), 'attr', False)
        # Processing the call keyword arguments (line 220)
        kwargs_155503 = {}
        # Getting the type of 'object' (line 220)
        object_155499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 19), 'object', False)
        # Obtaining the member '__getattribute__' of a type (line 220)
        getattribute___155500 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 19), object_155499, '__getattribute__')
        # Calling __getattribute__(args, kwargs) (line 220)
        getattribute___call_result_155504 = invoke(stypy.reporting.localization.Localization(__file__, 220, 19), getattribute___155500, *[self_155501, attr_155502], **kwargs_155503)
        
        # Assigning a type to the variable 'stypy_return_type' (line 220)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 12), 'stypy_return_type', getattribute___call_result_155504)
        # SSA branch for the except part of a try statement (line 219)
        # SSA branch for the except 'AttributeError' branch of a try statement (line 219)
        module_type_store.open_ssa_branch('except')
        pass
        # SSA join for try-except statement (line 219)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Attribute to a Name (line 224):
        
        # Assigning a Attribute to a Name (line 224):
        
        # Call to __getattribute__(...): (line 224)
        # Processing the call arguments (line 224)
        # Getting the type of 'self' (line 224)
        self_155507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 45), 'self', False)
        str_155508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 51), 'str', 'dtype')
        # Processing the call keyword arguments (line 224)
        kwargs_155509 = {}
        # Getting the type of 'ndarray' (line 224)
        ndarray_155505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 20), 'ndarray', False)
        # Obtaining the member '__getattribute__' of a type (line 224)
        getattribute___155506 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 20), ndarray_155505, '__getattribute__')
        # Calling __getattribute__(args, kwargs) (line 224)
        getattribute___call_result_155510 = invoke(stypy.reporting.localization.Localization(__file__, 224, 20), getattribute___155506, *[self_155507, str_155508], **kwargs_155509)
        
        # Obtaining the member 'fields' of a type (line 224)
        fields_155511 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 20), getattribute___call_result_155510, 'fields')
        # Assigning a type to the variable 'fielddict' (line 224)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 8), 'fielddict', fields_155511)
        
        
        # SSA begins for try-except statement (line 225)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Subscript to a Name (line 226):
        
        # Assigning a Subscript to a Name (line 226):
        
        # Obtaining the type of the subscript
        int_155512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 35), 'int')
        slice_155513 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 226, 18), None, int_155512, None)
        
        # Obtaining the type of the subscript
        # Getting the type of 'attr' (line 226)
        attr_155514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 28), 'attr')
        # Getting the type of 'fielddict' (line 226)
        fielddict_155515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 18), 'fielddict')
        # Obtaining the member '__getitem__' of a type (line 226)
        getitem___155516 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 18), fielddict_155515, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 226)
        subscript_call_result_155517 = invoke(stypy.reporting.localization.Localization(__file__, 226, 18), getitem___155516, attr_155514)
        
        # Obtaining the member '__getitem__' of a type (line 226)
        getitem___155518 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 18), subscript_call_result_155517, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 226)
        subscript_call_result_155519 = invoke(stypy.reporting.localization.Localization(__file__, 226, 18), getitem___155518, slice_155513)
        
        # Assigning a type to the variable 'res' (line 226)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 12), 'res', subscript_call_result_155519)
        # SSA branch for the except part of a try statement (line 225)
        # SSA branch for the except 'Tuple' branch of a try statement (line 225)
        module_type_store.open_ssa_branch('except')
        
        # Call to AttributeError(...): (line 228)
        # Processing the call arguments (line 228)
        str_155521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 33), 'str', 'record array has no attribute %s')
        # Getting the type of 'attr' (line 228)
        attr_155522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 70), 'attr', False)
        # Applying the binary operator '%' (line 228)
        result_mod_155523 = python_operator(stypy.reporting.localization.Localization(__file__, 228, 33), '%', str_155521, attr_155522)
        
        # Processing the call keyword arguments (line 228)
        kwargs_155524 = {}
        # Getting the type of 'AttributeError' (line 228)
        AttributeError_155520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 18), 'AttributeError', False)
        # Calling AttributeError(args, kwargs) (line 228)
        AttributeError_call_result_155525 = invoke(stypy.reporting.localization.Localization(__file__, 228, 18), AttributeError_155520, *[result_mod_155523], **kwargs_155524)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 228, 12), AttributeError_call_result_155525, 'raise parameter', BaseException)
        # SSA join for try-except statement (line 225)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 230):
        
        # Assigning a Call to a Name (line 230):
        
        # Call to __getattribute__(...): (line 230)
        # Processing the call arguments (line 230)
        # Getting the type of 'self' (line 230)
        self_155528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 46), 'self', False)
        str_155529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 52), 'str', '__dict__')
        # Processing the call keyword arguments (line 230)
        kwargs_155530 = {}
        # Getting the type of 'ndarray' (line 230)
        ndarray_155526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 21), 'ndarray', False)
        # Obtaining the member '__getattribute__' of a type (line 230)
        getattribute___155527 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 21), ndarray_155526, '__getattribute__')
        # Calling __getattribute__(args, kwargs) (line 230)
        getattribute___call_result_155531 = invoke(stypy.reporting.localization.Localization(__file__, 230, 21), getattribute___155527, *[self_155528, str_155529], **kwargs_155530)
        
        # Assigning a type to the variable '_localdict' (line 230)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 8), '_localdict', getattribute___call_result_155531)
        
        # Assigning a Call to a Name (line 231):
        
        # Assigning a Call to a Name (line 231):
        
        # Call to view(...): (line 231)
        # Processing the call arguments (line 231)
        # Getting the type of 'self' (line 231)
        self_155534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 29), 'self', False)
        
        # Obtaining the type of the subscript
        str_155535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 46), 'str', '_baseclass')
        # Getting the type of '_localdict' (line 231)
        _localdict_155536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 35), '_localdict', False)
        # Obtaining the member '__getitem__' of a type (line 231)
        getitem___155537 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 35), _localdict_155536, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 231)
        subscript_call_result_155538 = invoke(stypy.reporting.localization.Localization(__file__, 231, 35), getitem___155537, str_155535)
        
        # Processing the call keyword arguments (line 231)
        kwargs_155539 = {}
        # Getting the type of 'ndarray' (line 231)
        ndarray_155532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 16), 'ndarray', False)
        # Obtaining the member 'view' of a type (line 231)
        view_155533 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 16), ndarray_155532, 'view')
        # Calling view(args, kwargs) (line 231)
        view_call_result_155540 = invoke(stypy.reporting.localization.Localization(__file__, 231, 16), view_155533, *[self_155534, subscript_call_result_155538], **kwargs_155539)
        
        # Assigning a type to the variable '_data' (line 231)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 8), '_data', view_call_result_155540)
        
        # Assigning a Call to a Name (line 232):
        
        # Assigning a Call to a Name (line 232):
        
        # Call to getfield(...): (line 232)
        # Getting the type of 'res' (line 232)
        res_155543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 30), 'res', False)
        # Processing the call keyword arguments (line 232)
        kwargs_155544 = {}
        # Getting the type of '_data' (line 232)
        _data_155541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 14), '_data', False)
        # Obtaining the member 'getfield' of a type (line 232)
        getfield_155542 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 14), _data_155541, 'getfield')
        # Calling getfield(args, kwargs) (line 232)
        getfield_call_result_155545 = invoke(stypy.reporting.localization.Localization(__file__, 232, 14), getfield_155542, *[res_155543], **kwargs_155544)
        
        # Assigning a type to the variable 'obj' (line 232)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 8), 'obj', getfield_call_result_155545)
        
        # Getting the type of 'obj' (line 233)
        obj_155546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 11), 'obj')
        # Obtaining the member 'dtype' of a type (line 233)
        dtype_155547 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 11), obj_155546, 'dtype')
        # Obtaining the member 'fields' of a type (line 233)
        fields_155548 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 11), dtype_155547, 'fields')
        # Testing the type of an if condition (line 233)
        if_condition_155549 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 233, 8), fields_155548)
        # Assigning a type to the variable 'if_condition_155549' (line 233)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 8), 'if_condition_155549', if_condition_155549)
        # SSA begins for if statement (line 233)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to NotImplementedError(...): (line 234)
        # Processing the call arguments (line 234)
        str_155551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 38), 'str', 'MaskedRecords is currently limited tosimple records.')
        # Processing the call keyword arguments (line 234)
        kwargs_155552 = {}
        # Getting the type of 'NotImplementedError' (line 234)
        NotImplementedError_155550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 18), 'NotImplementedError', False)
        # Calling NotImplementedError(args, kwargs) (line 234)
        NotImplementedError_call_result_155553 = invoke(stypy.reporting.localization.Localization(__file__, 234, 18), NotImplementedError_155550, *[str_155551], **kwargs_155552)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 234, 12), NotImplementedError_call_result_155553, 'raise parameter', BaseException)
        # SSA join for if statement (line 233)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Name (line 238):
        
        # Assigning a Name to a Name (line 238):
        # Getting the type of 'False' (line 238)
        False_155554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 20), 'False')
        # Assigning a type to the variable 'hasmasked' (line 238)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 8), 'hasmasked', False_155554)
        
        # Assigning a Call to a Name (line 239):
        
        # Assigning a Call to a Name (line 239):
        
        # Call to get(...): (line 239)
        # Processing the call arguments (line 239)
        str_155557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 31), 'str', '_mask')
        # Getting the type of 'None' (line 239)
        None_155558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 40), 'None', False)
        # Processing the call keyword arguments (line 239)
        kwargs_155559 = {}
        # Getting the type of '_localdict' (line 239)
        _localdict_155555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 16), '_localdict', False)
        # Obtaining the member 'get' of a type (line 239)
        get_155556 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 16), _localdict_155555, 'get')
        # Calling get(args, kwargs) (line 239)
        get_call_result_155560 = invoke(stypy.reporting.localization.Localization(__file__, 239, 16), get_155556, *[str_155557, None_155558], **kwargs_155559)
        
        # Assigning a type to the variable '_mask' (line 239)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 8), '_mask', get_call_result_155560)
        
        # Type idiom detected: calculating its left and rigth part (line 240)
        # Getting the type of '_mask' (line 240)
        _mask_155561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), '_mask')
        # Getting the type of 'None' (line 240)
        None_155562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 24), 'None')
        
        (may_be_155563, more_types_in_union_155564) = may_not_be_none(_mask_155561, None_155562)

        if may_be_155563:

            if more_types_in_union_155564:
                # Runtime conditional SSA (line 240)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            
            # SSA begins for try-except statement (line 241)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
            
            # Assigning a Subscript to a Name (line 242):
            
            # Assigning a Subscript to a Name (line 242):
            
            # Obtaining the type of the subscript
            # Getting the type of 'attr' (line 242)
            attr_155565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 30), 'attr')
            # Getting the type of '_mask' (line 242)
            _mask_155566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 24), '_mask')
            # Obtaining the member '__getitem__' of a type (line 242)
            getitem___155567 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 24), _mask_155566, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 242)
            subscript_call_result_155568 = invoke(stypy.reporting.localization.Localization(__file__, 242, 24), getitem___155567, attr_155565)
            
            # Assigning a type to the variable '_mask' (line 242)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 16), '_mask', subscript_call_result_155568)
            # SSA branch for the except part of a try statement (line 241)
            # SSA branch for the except 'IndexError' branch of a try statement (line 241)
            module_type_store.open_ssa_branch('except')
            pass
            # SSA join for try-except statement (line 241)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Assigning a Call to a Name (line 246):
            
            # Assigning a Call to a Name (line 246):
            
            # Call to any(...): (line 246)
            # Processing the call keyword arguments (line 246)
            kwargs_155584 = {}
            
            # Call to view(...): (line 246)
            # Processing the call arguments (line 246)
            
            # Obtaining an instance of the builtin type 'tuple' (line 246)
            tuple_155571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 36), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 246)
            # Adding element type (line 246)
            # Getting the type of 'np' (line 246)
            np_155572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 36), 'np', False)
            # Obtaining the member 'bool' of a type (line 246)
            bool_155573 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 36), np_155572, 'bool')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 246, 36), tuple_155571, bool_155573)
            # Adding element type (line 246)
            
            # Evaluating a boolean operation
            
            # Call to len(...): (line 246)
            # Processing the call arguments (line 246)
            # Getting the type of '_mask' (line 246)
            _mask_155575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 50), '_mask', False)
            # Obtaining the member 'dtype' of a type (line 246)
            dtype_155576 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 50), _mask_155575, 'dtype')
            # Processing the call keyword arguments (line 246)
            kwargs_155577 = {}
            # Getting the type of 'len' (line 246)
            len_155574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 46), 'len', False)
            # Calling len(args, kwargs) (line 246)
            len_call_result_155578 = invoke(stypy.reporting.localization.Localization(__file__, 246, 46), len_155574, *[dtype_155576], **kwargs_155577)
            
            int_155579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 66), 'int')
            # Applying the binary operator 'or' (line 246)
            result_or_keyword_155580 = python_operator(stypy.reporting.localization.Localization(__file__, 246, 46), 'or', len_call_result_155578, int_155579)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 246, 36), tuple_155571, result_or_keyword_155580)
            
            # Processing the call keyword arguments (line 246)
            kwargs_155581 = {}
            # Getting the type of '_mask' (line 246)
            _mask_155569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 24), '_mask', False)
            # Obtaining the member 'view' of a type (line 246)
            view_155570 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 24), _mask_155569, 'view')
            # Calling view(args, kwargs) (line 246)
            view_call_result_155582 = invoke(stypy.reporting.localization.Localization(__file__, 246, 24), view_155570, *[tuple_155571], **kwargs_155581)
            
            # Obtaining the member 'any' of a type (line 246)
            any_155583 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 24), view_call_result_155582, 'any')
            # Calling any(args, kwargs) (line 246)
            any_call_result_155585 = invoke(stypy.reporting.localization.Localization(__file__, 246, 24), any_155583, *[], **kwargs_155584)
            
            # Assigning a type to the variable 'hasmasked' (line 246)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 12), 'hasmasked', any_call_result_155585)

            if more_types_in_union_155564:
                # SSA join for if statement (line 240)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        # Evaluating a boolean operation
        # Getting the type of 'obj' (line 247)
        obj_155586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 12), 'obj')
        # Obtaining the member 'shape' of a type (line 247)
        shape_155587 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 12), obj_155586, 'shape')
        # Getting the type of 'hasmasked' (line 247)
        hasmasked_155588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 25), 'hasmasked')
        # Applying the binary operator 'or' (line 247)
        result_or_keyword_155589 = python_operator(stypy.reporting.localization.Localization(__file__, 247, 12), 'or', shape_155587, hasmasked_155588)
        
        # Testing the type of an if condition (line 247)
        if_condition_155590 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 247, 8), result_or_keyword_155589)
        # Assigning a type to the variable 'if_condition_155590' (line 247)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 8), 'if_condition_155590', if_condition_155590)
        # SSA begins for if statement (line 247)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 248):
        
        # Assigning a Call to a Name (line 248):
        
        # Call to view(...): (line 248)
        # Processing the call arguments (line 248)
        # Getting the type of 'MaskedArray' (line 248)
        MaskedArray_155593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 27), 'MaskedArray', False)
        # Processing the call keyword arguments (line 248)
        kwargs_155594 = {}
        # Getting the type of 'obj' (line 248)
        obj_155591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 18), 'obj', False)
        # Obtaining the member 'view' of a type (line 248)
        view_155592 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 18), obj_155591, 'view')
        # Calling view(args, kwargs) (line 248)
        view_call_result_155595 = invoke(stypy.reporting.localization.Localization(__file__, 248, 18), view_155592, *[MaskedArray_155593], **kwargs_155594)
        
        # Assigning a type to the variable 'obj' (line 248)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 12), 'obj', view_call_result_155595)
        
        # Assigning a Name to a Attribute (line 249):
        
        # Assigning a Name to a Attribute (line 249):
        # Getting the type of 'ndarray' (line 249)
        ndarray_155596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 29), 'ndarray')
        # Getting the type of 'obj' (line 249)
        obj_155597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 12), 'obj')
        # Setting the type of the member '_baseclass' of a type (line 249)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 12), obj_155597, '_baseclass', ndarray_155596)
        
        # Assigning a Name to a Attribute (line 250):
        
        # Assigning a Name to a Attribute (line 250):
        # Getting the type of 'True' (line 250)
        True_155598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 27), 'True')
        # Getting the type of 'obj' (line 250)
        obj_155599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 12), 'obj')
        # Setting the type of the member '_isfield' of a type (line 250)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 12), obj_155599, '_isfield', True_155598)
        
        # Assigning a Name to a Attribute (line 251):
        
        # Assigning a Name to a Attribute (line 251):
        # Getting the type of '_mask' (line 251)
        _mask_155600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 24), '_mask')
        # Getting the type of 'obj' (line 251)
        obj_155601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 12), 'obj')
        # Setting the type of the member '_mask' of a type (line 251)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 12), obj_155601, '_mask', _mask_155600)
        
        # Assigning a Call to a Name (line 253):
        
        # Assigning a Call to a Name (line 253):
        
        # Call to get(...): (line 253)
        # Processing the call arguments (line 253)
        str_155604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 41), 'str', '_fill_value')
        # Getting the type of 'None' (line 253)
        None_155605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 56), 'None', False)
        # Processing the call keyword arguments (line 253)
        kwargs_155606 = {}
        # Getting the type of '_localdict' (line 253)
        _localdict_155602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 26), '_localdict', False)
        # Obtaining the member 'get' of a type (line 253)
        get_155603 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 26), _localdict_155602, 'get')
        # Calling get(args, kwargs) (line 253)
        get_call_result_155607 = invoke(stypy.reporting.localization.Localization(__file__, 253, 26), get_155603, *[str_155604, None_155605], **kwargs_155606)
        
        # Assigning a type to the variable '_fill_value' (line 253)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 12), '_fill_value', get_call_result_155607)
        
        # Type idiom detected: calculating its left and rigth part (line 254)
        # Getting the type of '_fill_value' (line 254)
        _fill_value_155608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 12), '_fill_value')
        # Getting the type of 'None' (line 254)
        None_155609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 34), 'None')
        
        (may_be_155610, more_types_in_union_155611) = may_not_be_none(_fill_value_155608, None_155609)

        if may_be_155610:

            if more_types_in_union_155611:
                # Runtime conditional SSA (line 254)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            
            # SSA begins for try-except statement (line 255)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
            
            # Assigning a Subscript to a Attribute (line 256):
            
            # Assigning a Subscript to a Attribute (line 256):
            
            # Obtaining the type of the subscript
            # Getting the type of 'attr' (line 256)
            attr_155612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 50), 'attr')
            # Getting the type of '_fill_value' (line 256)
            _fill_value_155613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 38), '_fill_value')
            # Obtaining the member '__getitem__' of a type (line 256)
            getitem___155614 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 38), _fill_value_155613, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 256)
            subscript_call_result_155615 = invoke(stypy.reporting.localization.Localization(__file__, 256, 38), getitem___155614, attr_155612)
            
            # Getting the type of 'obj' (line 256)
            obj_155616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 20), 'obj')
            # Setting the type of the member '_fill_value' of a type (line 256)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 20), obj_155616, '_fill_value', subscript_call_result_155615)
            # SSA branch for the except part of a try statement (line 255)
            # SSA branch for the except 'ValueError' branch of a try statement (line 255)
            module_type_store.open_ssa_branch('except')
            
            # Assigning a Name to a Attribute (line 258):
            
            # Assigning a Name to a Attribute (line 258):
            # Getting the type of 'None' (line 258)
            None_155617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 38), 'None')
            # Getting the type of 'obj' (line 258)
            obj_155618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 20), 'obj')
            # Setting the type of the member '_fill_value' of a type (line 258)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 20), obj_155618, '_fill_value', None_155617)
            # SSA join for try-except statement (line 255)
            module_type_store = module_type_store.join_ssa_context()
            

            if more_types_in_union_155611:
                # SSA join for if statement (line 254)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA branch for the else part of an if statement (line 247)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 260):
        
        # Assigning a Call to a Name (line 260):
        
        # Call to item(...): (line 260)
        # Processing the call keyword arguments (line 260)
        kwargs_155621 = {}
        # Getting the type of 'obj' (line 260)
        obj_155619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 18), 'obj', False)
        # Obtaining the member 'item' of a type (line 260)
        item_155620 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 18), obj_155619, 'item')
        # Calling item(args, kwargs) (line 260)
        item_call_result_155622 = invoke(stypy.reporting.localization.Localization(__file__, 260, 18), item_155620, *[], **kwargs_155621)
        
        # Assigning a type to the variable 'obj' (line 260)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 12), 'obj', item_call_result_155622)
        # SSA join for if statement (line 247)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'obj' (line 261)
        obj_155623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 15), 'obj')
        # Assigning a type to the variable 'stypy_return_type' (line 261)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 8), 'stypy_return_type', obj_155623)
        
        # ################# End of '__getattribute__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__getattribute__' in the type store
        # Getting the type of 'stypy_return_type' (line 218)
        stypy_return_type_155624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_155624)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__getattribute__'
        return stypy_return_type_155624


    @norecursion
    def __setattr__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__setattr__'
        module_type_store = module_type_store.open_function_context('__setattr__', 263, 4, False)
        # Assigning a type to the variable 'self' (line 264)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MaskedRecords.__setattr__.__dict__.__setitem__('stypy_localization', localization)
        MaskedRecords.__setattr__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MaskedRecords.__setattr__.__dict__.__setitem__('stypy_type_store', module_type_store)
        MaskedRecords.__setattr__.__dict__.__setitem__('stypy_function_name', 'MaskedRecords.__setattr__')
        MaskedRecords.__setattr__.__dict__.__setitem__('stypy_param_names_list', ['attr', 'val'])
        MaskedRecords.__setattr__.__dict__.__setitem__('stypy_varargs_param_name', None)
        MaskedRecords.__setattr__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MaskedRecords.__setattr__.__dict__.__setitem__('stypy_call_defaults', defaults)
        MaskedRecords.__setattr__.__dict__.__setitem__('stypy_call_varargs', varargs)
        MaskedRecords.__setattr__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MaskedRecords.__setattr__.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MaskedRecords.__setattr__', ['attr', 'val'], None, None, defaults, varargs, kwargs)

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

        str_155625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, (-1)), 'str', '\n        Sets the attribute attr to the value val.\n\n        ')
        
        
        # Getting the type of 'attr' (line 269)
        attr_155626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 11), 'attr')
        
        # Obtaining an instance of the builtin type 'list' (line 269)
        list_155627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 269)
        # Adding element type (line 269)
        str_155628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 20), 'str', 'mask')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 269, 19), list_155627, str_155628)
        # Adding element type (line 269)
        str_155629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 28), 'str', 'fieldmask')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 269, 19), list_155627, str_155629)
        
        # Applying the binary operator 'in' (line 269)
        result_contains_155630 = python_operator(stypy.reporting.localization.Localization(__file__, 269, 11), 'in', attr_155626, list_155627)
        
        # Testing the type of an if condition (line 269)
        if_condition_155631 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 269, 8), result_contains_155630)
        # Assigning a type to the variable 'if_condition_155631' (line 269)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 8), 'if_condition_155631', if_condition_155631)
        # SSA begins for if statement (line 269)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to __setmask__(...): (line 270)
        # Processing the call arguments (line 270)
        # Getting the type of 'val' (line 270)
        val_155634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 29), 'val', False)
        # Processing the call keyword arguments (line 270)
        kwargs_155635 = {}
        # Getting the type of 'self' (line 270)
        self_155632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 12), 'self', False)
        # Obtaining the member '__setmask__' of a type (line 270)
        setmask___155633 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 12), self_155632, '__setmask__')
        # Calling __setmask__(args, kwargs) (line 270)
        setmask___call_result_155636 = invoke(stypy.reporting.localization.Localization(__file__, 270, 12), setmask___155633, *[val_155634], **kwargs_155635)
        
        # Assigning a type to the variable 'stypy_return_type' (line 271)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 12), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 269)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 273):
        
        # Assigning a Call to a Name (line 273):
        
        # Call to __getattribute__(...): (line 273)
        # Processing the call arguments (line 273)
        # Getting the type of 'self' (line 273)
        self_155639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 45), 'self', False)
        str_155640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 51), 'str', '__dict__')
        # Processing the call keyword arguments (line 273)
        kwargs_155641 = {}
        # Getting the type of 'object' (line 273)
        object_155637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 21), 'object', False)
        # Obtaining the member '__getattribute__' of a type (line 273)
        getattribute___155638 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 21), object_155637, '__getattribute__')
        # Calling __getattribute__(args, kwargs) (line 273)
        getattribute___call_result_155642 = invoke(stypy.reporting.localization.Localization(__file__, 273, 21), getattribute___155638, *[self_155639, str_155640], **kwargs_155641)
        
        # Assigning a type to the variable '_localdict' (line 273)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 8), '_localdict', getattribute___call_result_155642)
        
        # Assigning a Compare to a Name (line 275):
        
        # Assigning a Compare to a Name (line 275):
        
        # Getting the type of 'attr' (line 275)
        attr_155643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 18), 'attr')
        # Getting the type of '_localdict' (line 275)
        _localdict_155644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 30), '_localdict')
        # Applying the binary operator 'notin' (line 275)
        result_contains_155645 = python_operator(stypy.reporting.localization.Localization(__file__, 275, 18), 'notin', attr_155643, _localdict_155644)
        
        # Assigning a type to the variable 'newattr' (line 275)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 8), 'newattr', result_contains_155645)
        
        
        # SSA begins for try-except statement (line 276)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Name (line 278):
        
        # Assigning a Call to a Name (line 278):
        
        # Call to __setattr__(...): (line 278)
        # Processing the call arguments (line 278)
        # Getting the type of 'self' (line 278)
        self_155648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 37), 'self', False)
        # Getting the type of 'attr' (line 278)
        attr_155649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 43), 'attr', False)
        # Getting the type of 'val' (line 278)
        val_155650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 49), 'val', False)
        # Processing the call keyword arguments (line 278)
        kwargs_155651 = {}
        # Getting the type of 'object' (line 278)
        object_155646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 18), 'object', False)
        # Obtaining the member '__setattr__' of a type (line 278)
        setattr___155647 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 18), object_155646, '__setattr__')
        # Calling __setattr__(args, kwargs) (line 278)
        setattr___call_result_155652 = invoke(stypy.reporting.localization.Localization(__file__, 278, 18), setattr___155647, *[self_155648, attr_155649, val_155650], **kwargs_155651)
        
        # Assigning a type to the variable 'ret' (line 278)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 12), 'ret', setattr___call_result_155652)
        # SSA branch for the except part of a try statement (line 276)
        # SSA branch for the except '<any exception>' branch of a try statement (line 276)
        module_type_store.open_ssa_branch('except')
        
        # Assigning a BoolOp to a Name (line 281):
        
        # Assigning a BoolOp to a Name (line 281):
        
        # Evaluating a boolean operation
        
        # Call to __getattribute__(...): (line 281)
        # Processing the call arguments (line 281)
        # Getting the type of 'self' (line 281)
        self_155655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 49), 'self', False)
        str_155656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 55), 'str', 'dtype')
        # Processing the call keyword arguments (line 281)
        kwargs_155657 = {}
        # Getting the type of 'ndarray' (line 281)
        ndarray_155653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 24), 'ndarray', False)
        # Obtaining the member '__getattribute__' of a type (line 281)
        getattribute___155654 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 24), ndarray_155653, '__getattribute__')
        # Calling __getattribute__(args, kwargs) (line 281)
        getattribute___call_result_155658 = invoke(stypy.reporting.localization.Localization(__file__, 281, 24), getattribute___155654, *[self_155655, str_155656], **kwargs_155657)
        
        # Obtaining the member 'fields' of a type (line 281)
        fields_155659 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 24), getattribute___call_result_155658, 'fields')
        
        # Obtaining an instance of the builtin type 'dict' (line 281)
        dict_155660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 74), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 281)
        
        # Applying the binary operator 'or' (line 281)
        result_or_keyword_155661 = python_operator(stypy.reporting.localization.Localization(__file__, 281, 24), 'or', fields_155659, dict_155660)
        
        # Assigning a type to the variable 'fielddict' (line 281)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 12), 'fielddict', result_or_keyword_155661)
        
        # Assigning a BoolOp to a Name (line 282):
        
        # Assigning a BoolOp to a Name (line 282):
        
        # Evaluating a boolean operation
        
        # Call to __getattribute__(...): (line 282)
        # Processing the call arguments (line 282)
        # Getting the type of 'self' (line 282)
        self_155664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 47), 'self', False)
        str_155665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 53), 'str', '_optinfo')
        # Processing the call keyword arguments (line 282)
        kwargs_155666 = {}
        # Getting the type of 'ndarray' (line 282)
        ndarray_155662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 22), 'ndarray', False)
        # Obtaining the member '__getattribute__' of a type (line 282)
        getattribute___155663 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 22), ndarray_155662, '__getattribute__')
        # Calling __getattribute__(args, kwargs) (line 282)
        getattribute___call_result_155667 = invoke(stypy.reporting.localization.Localization(__file__, 282, 22), getattribute___155663, *[self_155664, str_155665], **kwargs_155666)
        
        
        # Obtaining an instance of the builtin type 'dict' (line 282)
        dict_155668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 68), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 282)
        
        # Applying the binary operator 'or' (line 282)
        result_or_keyword_155669 = python_operator(stypy.reporting.localization.Localization(__file__, 282, 22), 'or', getattribute___call_result_155667, dict_155668)
        
        # Assigning a type to the variable 'optinfo' (line 282)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 12), 'optinfo', result_or_keyword_155669)
        
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'attr' (line 283)
        attr_155670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 20), 'attr')
        # Getting the type of 'fielddict' (line 283)
        fielddict_155671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 28), 'fielddict')
        # Applying the binary operator 'in' (line 283)
        result_contains_155672 = python_operator(stypy.reporting.localization.Localization(__file__, 283, 20), 'in', attr_155670, fielddict_155671)
        
        
        # Getting the type of 'attr' (line 283)
        attr_155673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 41), 'attr')
        # Getting the type of 'optinfo' (line 283)
        optinfo_155674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 49), 'optinfo')
        # Applying the binary operator 'in' (line 283)
        result_contains_155675 = python_operator(stypy.reporting.localization.Localization(__file__, 283, 41), 'in', attr_155673, optinfo_155674)
        
        # Applying the binary operator 'or' (line 283)
        result_or_keyword_155676 = python_operator(stypy.reporting.localization.Localization(__file__, 283, 20), 'or', result_contains_155672, result_contains_155675)
        
        # Applying the 'not' unary operator (line 283)
        result_not__155677 = python_operator(stypy.reporting.localization.Localization(__file__, 283, 15), 'not', result_or_keyword_155676)
        
        # Testing the type of an if condition (line 283)
        if_condition_155678 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 283, 12), result_not__155677)
        # Assigning a type to the variable 'if_condition_155678' (line 283)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 12), 'if_condition_155678', if_condition_155678)
        # SSA begins for if statement (line 283)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Tuple (line 284):
        
        # Assigning a Subscript to a Name (line 284):
        
        # Obtaining the type of the subscript
        int_155679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 16), 'int')
        
        # Obtaining the type of the subscript
        int_155680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 49), 'int')
        slice_155681 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 284, 33), None, int_155680, None)
        
        # Call to exc_info(...): (line 284)
        # Processing the call keyword arguments (line 284)
        kwargs_155684 = {}
        # Getting the type of 'sys' (line 284)
        sys_155682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 33), 'sys', False)
        # Obtaining the member 'exc_info' of a type (line 284)
        exc_info_155683 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 33), sys_155682, 'exc_info')
        # Calling exc_info(args, kwargs) (line 284)
        exc_info_call_result_155685 = invoke(stypy.reporting.localization.Localization(__file__, 284, 33), exc_info_155683, *[], **kwargs_155684)
        
        # Obtaining the member '__getitem__' of a type (line 284)
        getitem___155686 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 33), exc_info_call_result_155685, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 284)
        subscript_call_result_155687 = invoke(stypy.reporting.localization.Localization(__file__, 284, 33), getitem___155686, slice_155681)
        
        # Obtaining the member '__getitem__' of a type (line 284)
        getitem___155688 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 16), subscript_call_result_155687, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 284)
        subscript_call_result_155689 = invoke(stypy.reporting.localization.Localization(__file__, 284, 16), getitem___155688, int_155679)
        
        # Assigning a type to the variable 'tuple_var_assignment_154975' (line 284)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 16), 'tuple_var_assignment_154975', subscript_call_result_155689)
        
        # Assigning a Subscript to a Name (line 284):
        
        # Obtaining the type of the subscript
        int_155690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 16), 'int')
        
        # Obtaining the type of the subscript
        int_155691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 49), 'int')
        slice_155692 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 284, 33), None, int_155691, None)
        
        # Call to exc_info(...): (line 284)
        # Processing the call keyword arguments (line 284)
        kwargs_155695 = {}
        # Getting the type of 'sys' (line 284)
        sys_155693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 33), 'sys', False)
        # Obtaining the member 'exc_info' of a type (line 284)
        exc_info_155694 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 33), sys_155693, 'exc_info')
        # Calling exc_info(args, kwargs) (line 284)
        exc_info_call_result_155696 = invoke(stypy.reporting.localization.Localization(__file__, 284, 33), exc_info_155694, *[], **kwargs_155695)
        
        # Obtaining the member '__getitem__' of a type (line 284)
        getitem___155697 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 33), exc_info_call_result_155696, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 284)
        subscript_call_result_155698 = invoke(stypy.reporting.localization.Localization(__file__, 284, 33), getitem___155697, slice_155692)
        
        # Obtaining the member '__getitem__' of a type (line 284)
        getitem___155699 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 16), subscript_call_result_155698, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 284)
        subscript_call_result_155700 = invoke(stypy.reporting.localization.Localization(__file__, 284, 16), getitem___155699, int_155690)
        
        # Assigning a type to the variable 'tuple_var_assignment_154976' (line 284)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 16), 'tuple_var_assignment_154976', subscript_call_result_155700)
        
        # Assigning a Name to a Name (line 284):
        # Getting the type of 'tuple_var_assignment_154975' (line 284)
        tuple_var_assignment_154975_155701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 16), 'tuple_var_assignment_154975')
        # Assigning a type to the variable 'exctype' (line 284)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 16), 'exctype', tuple_var_assignment_154975_155701)
        
        # Assigning a Name to a Name (line 284):
        # Getting the type of 'tuple_var_assignment_154976' (line 284)
        tuple_var_assignment_154976_155702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 16), 'tuple_var_assignment_154976')
        # Assigning a type to the variable 'value' (line 284)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 25), 'value', tuple_var_assignment_154976_155702)
        
        # Call to exctype(...): (line 285)
        # Processing the call arguments (line 285)
        # Getting the type of 'value' (line 285)
        value_155704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 30), 'value', False)
        # Processing the call keyword arguments (line 285)
        kwargs_155705 = {}
        # Getting the type of 'exctype' (line 285)
        exctype_155703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 22), 'exctype', False)
        # Calling exctype(args, kwargs) (line 285)
        exctype_call_result_155706 = invoke(stypy.reporting.localization.Localization(__file__, 285, 22), exctype_155703, *[value_155704], **kwargs_155705)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 285, 16), exctype_call_result_155706, 'raise parameter', BaseException)
        # SSA join for if statement (line 283)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else branch of a try statement (line 276)
        module_type_store.open_ssa_branch('except else')
        
        # Assigning a BoolOp to a Name (line 288):
        
        # Assigning a BoolOp to a Name (line 288):
        
        # Evaluating a boolean operation
        
        # Call to __getattribute__(...): (line 288)
        # Processing the call arguments (line 288)
        # Getting the type of 'self' (line 288)
        self_155709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 49), 'self', False)
        str_155710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 55), 'str', 'dtype')
        # Processing the call keyword arguments (line 288)
        kwargs_155711 = {}
        # Getting the type of 'ndarray' (line 288)
        ndarray_155707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 24), 'ndarray', False)
        # Obtaining the member '__getattribute__' of a type (line 288)
        getattribute___155708 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 24), ndarray_155707, '__getattribute__')
        # Calling __getattribute__(args, kwargs) (line 288)
        getattribute___call_result_155712 = invoke(stypy.reporting.localization.Localization(__file__, 288, 24), getattribute___155708, *[self_155709, str_155710], **kwargs_155711)
        
        # Obtaining the member 'fields' of a type (line 288)
        fields_155713 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 24), getattribute___call_result_155712, 'fields')
        
        # Obtaining an instance of the builtin type 'dict' (line 288)
        dict_155714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 74), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 288)
        
        # Applying the binary operator 'or' (line 288)
        result_or_keyword_155715 = python_operator(stypy.reporting.localization.Localization(__file__, 288, 24), 'or', fields_155713, dict_155714)
        
        # Assigning a type to the variable 'fielddict' (line 288)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 12), 'fielddict', result_or_keyword_155715)
        
        
        # Getting the type of 'attr' (line 290)
        attr_155716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 15), 'attr')
        # Getting the type of 'fielddict' (line 290)
        fielddict_155717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 27), 'fielddict')
        # Applying the binary operator 'notin' (line 290)
        result_contains_155718 = python_operator(stypy.reporting.localization.Localization(__file__, 290, 15), 'notin', attr_155716, fielddict_155717)
        
        # Testing the type of an if condition (line 290)
        if_condition_155719 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 290, 12), result_contains_155718)
        # Assigning a type to the variable 'if_condition_155719' (line 290)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 12), 'if_condition_155719', if_condition_155719)
        # SSA begins for if statement (line 290)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'ret' (line 291)
        ret_155720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 23), 'ret')
        # Assigning a type to the variable 'stypy_return_type' (line 291)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 16), 'stypy_return_type', ret_155720)
        # SSA join for if statement (line 290)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'newattr' (line 292)
        newattr_155721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 15), 'newattr')
        # Testing the type of an if condition (line 292)
        if_condition_155722 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 292, 12), newattr_155721)
        # Assigning a type to the variable 'if_condition_155722' (line 292)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 12), 'if_condition_155722', if_condition_155722)
        # SSA begins for if statement (line 292)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # SSA begins for try-except statement (line 295)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to __delattr__(...): (line 296)
        # Processing the call arguments (line 296)
        # Getting the type of 'self' (line 296)
        self_155725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 39), 'self', False)
        # Getting the type of 'attr' (line 296)
        attr_155726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 45), 'attr', False)
        # Processing the call keyword arguments (line 296)
        kwargs_155727 = {}
        # Getting the type of 'object' (line 296)
        object_155723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 20), 'object', False)
        # Obtaining the member '__delattr__' of a type (line 296)
        delattr___155724 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 20), object_155723, '__delattr__')
        # Calling __delattr__(args, kwargs) (line 296)
        delattr___call_result_155728 = invoke(stypy.reporting.localization.Localization(__file__, 296, 20), delattr___155724, *[self_155725, attr_155726], **kwargs_155727)
        
        # SSA branch for the except part of a try statement (line 295)
        # SSA branch for the except '<any exception>' branch of a try statement (line 295)
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'ret' (line 298)
        ret_155729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 27), 'ret')
        # Assigning a type to the variable 'stypy_return_type' (line 298)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 20), 'stypy_return_type', ret_155729)
        # SSA join for try-except statement (line 295)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 292)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for try-except statement (line 276)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # SSA begins for try-except statement (line 300)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Subscript to a Name (line 301):
        
        # Assigning a Subscript to a Name (line 301):
        
        # Obtaining the type of the subscript
        int_155730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 35), 'int')
        slice_155731 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 301, 18), None, int_155730, None)
        
        # Obtaining the type of the subscript
        # Getting the type of 'attr' (line 301)
        attr_155732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 28), 'attr')
        # Getting the type of 'fielddict' (line 301)
        fielddict_155733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 18), 'fielddict')
        # Obtaining the member '__getitem__' of a type (line 301)
        getitem___155734 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 18), fielddict_155733, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 301)
        subscript_call_result_155735 = invoke(stypy.reporting.localization.Localization(__file__, 301, 18), getitem___155734, attr_155732)
        
        # Obtaining the member '__getitem__' of a type (line 301)
        getitem___155736 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 18), subscript_call_result_155735, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 301)
        subscript_call_result_155737 = invoke(stypy.reporting.localization.Localization(__file__, 301, 18), getitem___155736, slice_155731)
        
        # Assigning a type to the variable 'res' (line 301)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 12), 'res', subscript_call_result_155737)
        # SSA branch for the except part of a try statement (line 300)
        # SSA branch for the except 'Tuple' branch of a try statement (line 300)
        module_type_store.open_ssa_branch('except')
        
        # Call to AttributeError(...): (line 303)
        # Processing the call arguments (line 303)
        str_155739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 33), 'str', 'record array has no attribute %s')
        # Getting the type of 'attr' (line 303)
        attr_155740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 70), 'attr', False)
        # Applying the binary operator '%' (line 303)
        result_mod_155741 = python_operator(stypy.reporting.localization.Localization(__file__, 303, 33), '%', str_155739, attr_155740)
        
        # Processing the call keyword arguments (line 303)
        kwargs_155742 = {}
        # Getting the type of 'AttributeError' (line 303)
        AttributeError_155738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 18), 'AttributeError', False)
        # Calling AttributeError(args, kwargs) (line 303)
        AttributeError_call_result_155743 = invoke(stypy.reporting.localization.Localization(__file__, 303, 18), AttributeError_155738, *[result_mod_155741], **kwargs_155742)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 303, 12), AttributeError_call_result_155743, 'raise parameter', BaseException)
        # SSA join for try-except statement (line 300)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'val' (line 305)
        val_155744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 11), 'val')
        # Getting the type of 'masked' (line 305)
        masked_155745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 18), 'masked')
        # Applying the binary operator 'is' (line 305)
        result_is__155746 = python_operator(stypy.reporting.localization.Localization(__file__, 305, 11), 'is', val_155744, masked_155745)
        
        # Testing the type of an if condition (line 305)
        if_condition_155747 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 305, 8), result_is__155746)
        # Assigning a type to the variable 'if_condition_155747' (line 305)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 305, 8), 'if_condition_155747', if_condition_155747)
        # SSA begins for if statement (line 305)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Name (line 306):
        
        # Assigning a Subscript to a Name (line 306):
        
        # Obtaining the type of the subscript
        str_155748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 37), 'str', '_fill_value')
        # Getting the type of '_localdict' (line 306)
        _localdict_155749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 26), '_localdict')
        # Obtaining the member '__getitem__' of a type (line 306)
        getitem___155750 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 26), _localdict_155749, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 306)
        subscript_call_result_155751 = invoke(stypy.reporting.localization.Localization(__file__, 306, 26), getitem___155750, str_155748)
        
        # Assigning a type to the variable '_fill_value' (line 306)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 12), '_fill_value', subscript_call_result_155751)
        
        # Type idiom detected: calculating its left and rigth part (line 307)
        # Getting the type of '_fill_value' (line 307)
        _fill_value_155752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 12), '_fill_value')
        # Getting the type of 'None' (line 307)
        None_155753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 34), 'None')
        
        (may_be_155754, more_types_in_union_155755) = may_not_be_none(_fill_value_155752, None_155753)

        if may_be_155754:

            if more_types_in_union_155755:
                # Runtime conditional SSA (line 307)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Subscript to a Name (line 308):
            
            # Assigning a Subscript to a Name (line 308):
            
            # Obtaining the type of the subscript
            # Getting the type of 'attr' (line 308)
            attr_155756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 49), 'attr')
            
            # Obtaining the type of the subscript
            str_155757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, 34), 'str', '_fill_value')
            # Getting the type of '_localdict' (line 308)
            _localdict_155758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 23), '_localdict')
            # Obtaining the member '__getitem__' of a type (line 308)
            getitem___155759 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 23), _localdict_155758, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 308)
            subscript_call_result_155760 = invoke(stypy.reporting.localization.Localization(__file__, 308, 23), getitem___155759, str_155757)
            
            # Obtaining the member '__getitem__' of a type (line 308)
            getitem___155761 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 23), subscript_call_result_155760, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 308)
            subscript_call_result_155762 = invoke(stypy.reporting.localization.Localization(__file__, 308, 23), getitem___155761, attr_155756)
            
            # Assigning a type to the variable 'dval' (line 308)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 308, 16), 'dval', subscript_call_result_155762)

            if more_types_in_union_155755:
                # Runtime conditional SSA for else branch (line 307)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_155754) or more_types_in_union_155755):
            
            # Assigning a Name to a Name (line 310):
            
            # Assigning a Name to a Name (line 310):
            # Getting the type of 'val' (line 310)
            val_155763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 23), 'val')
            # Assigning a type to the variable 'dval' (line 310)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 16), 'dval', val_155763)

            if (may_be_155754 and more_types_in_union_155755):
                # SSA join for if statement (line 307)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Name to a Name (line 311):
        
        # Assigning a Name to a Name (line 311):
        # Getting the type of 'True' (line 311)
        True_155764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 19), 'True')
        # Assigning a type to the variable 'mval' (line 311)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 12), 'mval', True_155764)
        # SSA branch for the else part of an if statement (line 305)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 313):
        
        # Assigning a Call to a Name (line 313):
        
        # Call to filled(...): (line 313)
        # Processing the call arguments (line 313)
        # Getting the type of 'val' (line 313)
        val_155766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 26), 'val', False)
        # Processing the call keyword arguments (line 313)
        kwargs_155767 = {}
        # Getting the type of 'filled' (line 313)
        filled_155765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 19), 'filled', False)
        # Calling filled(args, kwargs) (line 313)
        filled_call_result_155768 = invoke(stypy.reporting.localization.Localization(__file__, 313, 19), filled_155765, *[val_155766], **kwargs_155767)
        
        # Assigning a type to the variable 'dval' (line 313)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 313, 12), 'dval', filled_call_result_155768)
        
        # Assigning a Call to a Name (line 314):
        
        # Assigning a Call to a Name (line 314):
        
        # Call to getmaskarray(...): (line 314)
        # Processing the call arguments (line 314)
        # Getting the type of 'val' (line 314)
        val_155770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 32), 'val', False)
        # Processing the call keyword arguments (line 314)
        kwargs_155771 = {}
        # Getting the type of 'getmaskarray' (line 314)
        getmaskarray_155769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 19), 'getmaskarray', False)
        # Calling getmaskarray(args, kwargs) (line 314)
        getmaskarray_call_result_155772 = invoke(stypy.reporting.localization.Localization(__file__, 314, 19), getmaskarray_155769, *[val_155770], **kwargs_155771)
        
        # Assigning a type to the variable 'mval' (line 314)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 314, 12), 'mval', getmaskarray_call_result_155772)
        # SSA join for if statement (line 305)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 315):
        
        # Assigning a Call to a Name (line 315):
        
        # Call to setfield(...): (line 315)
        # Processing the call arguments (line 315)
        # Getting the type of 'dval' (line 315)
        dval_155780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 63), 'dval', False)
        # Getting the type of 'res' (line 315)
        res_155781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 70), 'res', False)
        # Processing the call keyword arguments (line 315)
        kwargs_155782 = {}
        
        # Call to __getattribute__(...): (line 315)
        # Processing the call arguments (line 315)
        # Getting the type of 'self' (line 315)
        self_155775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 39), 'self', False)
        str_155776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 45), 'str', '_data')
        # Processing the call keyword arguments (line 315)
        kwargs_155777 = {}
        # Getting the type of 'ndarray' (line 315)
        ndarray_155773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 14), 'ndarray', False)
        # Obtaining the member '__getattribute__' of a type (line 315)
        getattribute___155774 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 315, 14), ndarray_155773, '__getattribute__')
        # Calling __getattribute__(args, kwargs) (line 315)
        getattribute___call_result_155778 = invoke(stypy.reporting.localization.Localization(__file__, 315, 14), getattribute___155774, *[self_155775, str_155776], **kwargs_155777)
        
        # Obtaining the member 'setfield' of a type (line 315)
        setfield_155779 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 315, 14), getattribute___call_result_155778, 'setfield')
        # Calling setfield(args, kwargs) (line 315)
        setfield_call_result_155783 = invoke(stypy.reporting.localization.Localization(__file__, 315, 14), setfield_155779, *[dval_155780, res_155781], **kwargs_155782)
        
        # Assigning a type to the variable 'obj' (line 315)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 8), 'obj', setfield_call_result_155783)
        
        # Call to __setitem__(...): (line 316)
        # Processing the call arguments (line 316)
        # Getting the type of 'attr' (line 316)
        attr_155789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 40), 'attr', False)
        # Getting the type of 'mval' (line 316)
        mval_155790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 46), 'mval', False)
        # Processing the call keyword arguments (line 316)
        kwargs_155791 = {}
        
        # Obtaining the type of the subscript
        str_155784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 316, 19), 'str', '_mask')
        # Getting the type of '_localdict' (line 316)
        _localdict_155785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 8), '_localdict', False)
        # Obtaining the member '__getitem__' of a type (line 316)
        getitem___155786 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 8), _localdict_155785, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 316)
        subscript_call_result_155787 = invoke(stypy.reporting.localization.Localization(__file__, 316, 8), getitem___155786, str_155784)
        
        # Obtaining the member '__setitem__' of a type (line 316)
        setitem___155788 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 8), subscript_call_result_155787, '__setitem__')
        # Calling __setitem__(args, kwargs) (line 316)
        setitem___call_result_155792 = invoke(stypy.reporting.localization.Localization(__file__, 316, 8), setitem___155788, *[attr_155789, mval_155790], **kwargs_155791)
        
        # Getting the type of 'obj' (line 317)
        obj_155793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 15), 'obj')
        # Assigning a type to the variable 'stypy_return_type' (line 317)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 317, 8), 'stypy_return_type', obj_155793)
        
        # ################# End of '__setattr__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__setattr__' in the type store
        # Getting the type of 'stypy_return_type' (line 263)
        stypy_return_type_155794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_155794)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__setattr__'
        return stypy_return_type_155794


    @norecursion
    def __getitem__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__getitem__'
        module_type_store = module_type_store.open_function_context('__getitem__', 319, 4, False)
        # Assigning a type to the variable 'self' (line 320)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 320, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MaskedRecords.__getitem__.__dict__.__setitem__('stypy_localization', localization)
        MaskedRecords.__getitem__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MaskedRecords.__getitem__.__dict__.__setitem__('stypy_type_store', module_type_store)
        MaskedRecords.__getitem__.__dict__.__setitem__('stypy_function_name', 'MaskedRecords.__getitem__')
        MaskedRecords.__getitem__.__dict__.__setitem__('stypy_param_names_list', ['indx'])
        MaskedRecords.__getitem__.__dict__.__setitem__('stypy_varargs_param_name', None)
        MaskedRecords.__getitem__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MaskedRecords.__getitem__.__dict__.__setitem__('stypy_call_defaults', defaults)
        MaskedRecords.__getitem__.__dict__.__setitem__('stypy_call_varargs', varargs)
        MaskedRecords.__getitem__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MaskedRecords.__getitem__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MaskedRecords.__getitem__', ['indx'], None, None, defaults, varargs, kwargs)

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

        str_155795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, (-1)), 'str', '\n        Returns all the fields sharing the same fieldname base.\n\n        The fieldname base is either `_data` or `_mask`.\n\n        ')
        
        # Assigning a Attribute to a Name (line 326):
        
        # Assigning a Attribute to a Name (line 326):
        # Getting the type of 'self' (line 326)
        self_155796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 21), 'self')
        # Obtaining the member '__dict__' of a type (line 326)
        dict___155797 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 21), self_155796, '__dict__')
        # Assigning a type to the variable '_localdict' (line 326)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 8), '_localdict', dict___155797)
        
        # Assigning a Call to a Name (line 327):
        
        # Assigning a Call to a Name (line 327):
        
        # Call to __getattribute__(...): (line 327)
        # Processing the call arguments (line 327)
        # Getting the type of 'self' (line 327)
        self_155800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 41), 'self', False)
        str_155801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 47), 'str', '_mask')
        # Processing the call keyword arguments (line 327)
        kwargs_155802 = {}
        # Getting the type of 'ndarray' (line 327)
        ndarray_155798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 16), 'ndarray', False)
        # Obtaining the member '__getattribute__' of a type (line 327)
        getattribute___155799 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 16), ndarray_155798, '__getattribute__')
        # Calling __getattribute__(args, kwargs) (line 327)
        getattribute___call_result_155803 = invoke(stypy.reporting.localization.Localization(__file__, 327, 16), getattribute___155799, *[self_155800, str_155801], **kwargs_155802)
        
        # Assigning a type to the variable '_mask' (line 327)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 327, 8), '_mask', getattribute___call_result_155803)
        
        # Assigning a Call to a Name (line 328):
        
        # Assigning a Call to a Name (line 328):
        
        # Call to view(...): (line 328)
        # Processing the call arguments (line 328)
        # Getting the type of 'self' (line 328)
        self_155806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 29), 'self', False)
        
        # Obtaining the type of the subscript
        str_155807 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 46), 'str', '_baseclass')
        # Getting the type of '_localdict' (line 328)
        _localdict_155808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 35), '_localdict', False)
        # Obtaining the member '__getitem__' of a type (line 328)
        getitem___155809 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 328, 35), _localdict_155808, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 328)
        subscript_call_result_155810 = invoke(stypy.reporting.localization.Localization(__file__, 328, 35), getitem___155809, str_155807)
        
        # Processing the call keyword arguments (line 328)
        kwargs_155811 = {}
        # Getting the type of 'ndarray' (line 328)
        ndarray_155804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 16), 'ndarray', False)
        # Obtaining the member 'view' of a type (line 328)
        view_155805 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 328, 16), ndarray_155804, 'view')
        # Calling view(args, kwargs) (line 328)
        view_call_result_155812 = invoke(stypy.reporting.localization.Localization(__file__, 328, 16), view_155805, *[self_155806, subscript_call_result_155810], **kwargs_155811)
        
        # Assigning a type to the variable '_data' (line 328)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 328, 8), '_data', view_call_result_155812)
        
        # Type idiom detected: calculating its left and rigth part (line 330)
        # Getting the type of 'basestring' (line 330)
        basestring_155813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 28), 'basestring')
        # Getting the type of 'indx' (line 330)
        indx_155814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 22), 'indx')
        
        (may_be_155815, more_types_in_union_155816) = may_be_subtype(basestring_155813, indx_155814)

        if may_be_155815:

            if more_types_in_union_155816:
                # Runtime conditional SSA (line 330)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'indx' (line 330)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 8), 'indx', remove_not_subtype_from_union(indx_155814, basestring))
            
            # Assigning a Call to a Name (line 335):
            
            # Assigning a Call to a Name (line 335):
            
            # Call to view(...): (line 335)
            # Processing the call arguments (line 335)
            # Getting the type of 'MaskedArray' (line 335)
            MaskedArray_155822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 35), 'MaskedArray', False)
            # Processing the call keyword arguments (line 335)
            kwargs_155823 = {}
            
            # Obtaining the type of the subscript
            # Getting the type of 'indx' (line 335)
            indx_155817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 24), 'indx', False)
            # Getting the type of '_data' (line 335)
            _data_155818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 18), '_data', False)
            # Obtaining the member '__getitem__' of a type (line 335)
            getitem___155819 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 18), _data_155818, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 335)
            subscript_call_result_155820 = invoke(stypy.reporting.localization.Localization(__file__, 335, 18), getitem___155819, indx_155817)
            
            # Obtaining the member 'view' of a type (line 335)
            view_155821 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 18), subscript_call_result_155820, 'view')
            # Calling view(args, kwargs) (line 335)
            view_call_result_155824 = invoke(stypy.reporting.localization.Localization(__file__, 335, 18), view_155821, *[MaskedArray_155822], **kwargs_155823)
            
            # Assigning a type to the variable 'obj' (line 335)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 12), 'obj', view_call_result_155824)
            
            # Assigning a Subscript to a Attribute (line 336):
            
            # Assigning a Subscript to a Attribute (line 336):
            
            # Obtaining the type of the subscript
            # Getting the type of 'indx' (line 336)
            indx_155825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 30), 'indx')
            # Getting the type of '_mask' (line 336)
            _mask_155826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 24), '_mask')
            # Obtaining the member '__getitem__' of a type (line 336)
            getitem___155827 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 24), _mask_155826, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 336)
            subscript_call_result_155828 = invoke(stypy.reporting.localization.Localization(__file__, 336, 24), getitem___155827, indx_155825)
            
            # Getting the type of 'obj' (line 336)
            obj_155829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 12), 'obj')
            # Setting the type of the member '_mask' of a type (line 336)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 12), obj_155829, '_mask', subscript_call_result_155828)
            
            # Assigning a Name to a Attribute (line 337):
            
            # Assigning a Name to a Attribute (line 337):
            # Getting the type of 'True' (line 337)
            True_155830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 30), 'True')
            # Getting the type of 'obj' (line 337)
            obj_155831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 12), 'obj')
            # Setting the type of the member '_sharedmask' of a type (line 337)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 12), obj_155831, '_sharedmask', True_155830)
            
            # Assigning a Subscript to a Name (line 338):
            
            # Assigning a Subscript to a Name (line 338):
            
            # Obtaining the type of the subscript
            str_155832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 30), 'str', '_fill_value')
            # Getting the type of '_localdict' (line 338)
            _localdict_155833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 19), '_localdict')
            # Obtaining the member '__getitem__' of a type (line 338)
            getitem___155834 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 19), _localdict_155833, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 338)
            subscript_call_result_155835 = invoke(stypy.reporting.localization.Localization(__file__, 338, 19), getitem___155834, str_155832)
            
            # Assigning a type to the variable 'fval' (line 338)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 338, 12), 'fval', subscript_call_result_155835)
            
            # Type idiom detected: calculating its left and rigth part (line 339)
            # Getting the type of 'fval' (line 339)
            fval_155836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 12), 'fval')
            # Getting the type of 'None' (line 339)
            None_155837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 27), 'None')
            
            (may_be_155838, more_types_in_union_155839) = may_not_be_none(fval_155836, None_155837)

            if may_be_155838:

                if more_types_in_union_155839:
                    # Runtime conditional SSA (line 339)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                
                # Assigning a Subscript to a Attribute (line 340):
                
                # Assigning a Subscript to a Attribute (line 340):
                
                # Obtaining the type of the subscript
                # Getting the type of 'indx' (line 340)
                indx_155840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 39), 'indx')
                # Getting the type of 'fval' (line 340)
                fval_155841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 34), 'fval')
                # Obtaining the member '__getitem__' of a type (line 340)
                getitem___155842 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 340, 34), fval_155841, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 340)
                subscript_call_result_155843 = invoke(stypy.reporting.localization.Localization(__file__, 340, 34), getitem___155842, indx_155840)
                
                # Getting the type of 'obj' (line 340)
                obj_155844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 16), 'obj')
                # Setting the type of the member '_fill_value' of a type (line 340)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 340, 16), obj_155844, '_fill_value', subscript_call_result_155843)

                if more_types_in_union_155839:
                    # SSA join for if statement (line 339)
                    module_type_store = module_type_store.join_ssa_context()


            
            
            
            # Evaluating a boolean operation
            
            # Getting the type of 'obj' (line 342)
            obj_155845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 19), 'obj')
            # Obtaining the member 'ndim' of a type (line 342)
            ndim_155846 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 19), obj_155845, 'ndim')
            # Applying the 'not' unary operator (line 342)
            result_not__155847 = python_operator(stypy.reporting.localization.Localization(__file__, 342, 15), 'not', ndim_155846)
            
            # Getting the type of 'obj' (line 342)
            obj_155848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 32), 'obj')
            # Obtaining the member '_mask' of a type (line 342)
            _mask_155849 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 32), obj_155848, '_mask')
            # Applying the binary operator 'and' (line 342)
            result_and_keyword_155850 = python_operator(stypy.reporting.localization.Localization(__file__, 342, 15), 'and', result_not__155847, _mask_155849)
            
            # Testing the type of an if condition (line 342)
            if_condition_155851 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 342, 12), result_and_keyword_155850)
            # Assigning a type to the variable 'if_condition_155851' (line 342)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 342, 12), 'if_condition_155851', if_condition_155851)
            # SSA begins for if statement (line 342)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'masked' (line 343)
            masked_155852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 23), 'masked')
            # Assigning a type to the variable 'stypy_return_type' (line 343)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 16), 'stypy_return_type', masked_155852)
            # SSA join for if statement (line 342)
            module_type_store = module_type_store.join_ssa_context()
            
            # Getting the type of 'obj' (line 344)
            obj_155853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 19), 'obj')
            # Assigning a type to the variable 'stypy_return_type' (line 344)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 344, 12), 'stypy_return_type', obj_155853)

            if more_types_in_union_155816:
                # SSA join for if statement (line 330)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Name (line 347):
        
        # Assigning a Call to a Name (line 347):
        
        # Call to view(...): (line 347)
        # Processing the call arguments (line 347)
        # Getting the type of 'mrecarray' (line 347)
        mrecarray_155865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 53), 'mrecarray', False)
        # Processing the call keyword arguments (line 347)
        kwargs_155866 = {}
        
        # Call to array(...): (line 347)
        # Processing the call arguments (line 347)
        
        # Obtaining the type of the subscript
        # Getting the type of 'indx' (line 347)
        indx_155856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 29), 'indx', False)
        # Getting the type of '_data' (line 347)
        _data_155857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 23), '_data', False)
        # Obtaining the member '__getitem__' of a type (line 347)
        getitem___155858 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 347, 23), _data_155857, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 347)
        subscript_call_result_155859 = invoke(stypy.reporting.localization.Localization(__file__, 347, 23), getitem___155858, indx_155856)
        
        # Processing the call keyword arguments (line 347)
        # Getting the type of 'False' (line 347)
        False_155860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 41), 'False', False)
        keyword_155861 = False_155860
        kwargs_155862 = {'copy': keyword_155861}
        # Getting the type of 'np' (line 347)
        np_155854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 14), 'np', False)
        # Obtaining the member 'array' of a type (line 347)
        array_155855 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 347, 14), np_155854, 'array')
        # Calling array(args, kwargs) (line 347)
        array_call_result_155863 = invoke(stypy.reporting.localization.Localization(__file__, 347, 14), array_155855, *[subscript_call_result_155859], **kwargs_155862)
        
        # Obtaining the member 'view' of a type (line 347)
        view_155864 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 347, 14), array_call_result_155863, 'view')
        # Calling view(args, kwargs) (line 347)
        view_call_result_155867 = invoke(stypy.reporting.localization.Localization(__file__, 347, 14), view_155864, *[mrecarray_155865], **kwargs_155866)
        
        # Assigning a type to the variable 'obj' (line 347)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 347, 8), 'obj', view_call_result_155867)
        
        # Assigning a Call to a Attribute (line 348):
        
        # Assigning a Call to a Attribute (line 348):
        
        # Call to view(...): (line 348)
        # Processing the call arguments (line 348)
        # Getting the type of 'recarray' (line 348)
        recarray_155879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 59), 'recarray', False)
        # Processing the call keyword arguments (line 348)
        kwargs_155880 = {}
        
        # Call to array(...): (line 348)
        # Processing the call arguments (line 348)
        
        # Obtaining the type of the subscript
        # Getting the type of 'indx' (line 348)
        indx_155870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 35), 'indx', False)
        # Getting the type of '_mask' (line 348)
        _mask_155871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 29), '_mask', False)
        # Obtaining the member '__getitem__' of a type (line 348)
        getitem___155872 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 29), _mask_155871, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 348)
        subscript_call_result_155873 = invoke(stypy.reporting.localization.Localization(__file__, 348, 29), getitem___155872, indx_155870)
        
        # Processing the call keyword arguments (line 348)
        # Getting the type of 'False' (line 348)
        False_155874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 47), 'False', False)
        keyword_155875 = False_155874
        kwargs_155876 = {'copy': keyword_155875}
        # Getting the type of 'np' (line 348)
        np_155868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 20), 'np', False)
        # Obtaining the member 'array' of a type (line 348)
        array_155869 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 20), np_155868, 'array')
        # Calling array(args, kwargs) (line 348)
        array_call_result_155877 = invoke(stypy.reporting.localization.Localization(__file__, 348, 20), array_155869, *[subscript_call_result_155873], **kwargs_155876)
        
        # Obtaining the member 'view' of a type (line 348)
        view_155878 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 20), array_call_result_155877, 'view')
        # Calling view(args, kwargs) (line 348)
        view_call_result_155881 = invoke(stypy.reporting.localization.Localization(__file__, 348, 20), view_155878, *[recarray_155879], **kwargs_155880)
        
        # Getting the type of 'obj' (line 348)
        obj_155882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 8), 'obj')
        # Setting the type of the member '_mask' of a type (line 348)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 8), obj_155882, '_mask', view_call_result_155881)
        # Getting the type of 'obj' (line 349)
        obj_155883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 15), 'obj')
        # Assigning a type to the variable 'stypy_return_type' (line 349)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 8), 'stypy_return_type', obj_155883)
        
        # ################# End of '__getitem__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__getitem__' in the type store
        # Getting the type of 'stypy_return_type' (line 319)
        stypy_return_type_155884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_155884)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__getitem__'
        return stypy_return_type_155884


    @norecursion
    def __setitem__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__setitem__'
        module_type_store = module_type_store.open_function_context('__setitem__', 351, 4, False)
        # Assigning a type to the variable 'self' (line 352)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 352, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MaskedRecords.__setitem__.__dict__.__setitem__('stypy_localization', localization)
        MaskedRecords.__setitem__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MaskedRecords.__setitem__.__dict__.__setitem__('stypy_type_store', module_type_store)
        MaskedRecords.__setitem__.__dict__.__setitem__('stypy_function_name', 'MaskedRecords.__setitem__')
        MaskedRecords.__setitem__.__dict__.__setitem__('stypy_param_names_list', ['indx', 'value'])
        MaskedRecords.__setitem__.__dict__.__setitem__('stypy_varargs_param_name', None)
        MaskedRecords.__setitem__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MaskedRecords.__setitem__.__dict__.__setitem__('stypy_call_defaults', defaults)
        MaskedRecords.__setitem__.__dict__.__setitem__('stypy_call_varargs', varargs)
        MaskedRecords.__setitem__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MaskedRecords.__setitem__.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MaskedRecords.__setitem__', ['indx', 'value'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__setitem__', localization, ['indx', 'value'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__setitem__(...)' code ##################

        str_155885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, (-1)), 'str', '\n        Sets the given record to value.\n\n        ')
        
        # Call to __setitem__(...): (line 356)
        # Processing the call arguments (line 356)
        # Getting the type of 'self' (line 356)
        self_155888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 32), 'self', False)
        # Getting the type of 'indx' (line 356)
        indx_155889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 38), 'indx', False)
        # Getting the type of 'value' (line 356)
        value_155890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 44), 'value', False)
        # Processing the call keyword arguments (line 356)
        kwargs_155891 = {}
        # Getting the type of 'MaskedArray' (line 356)
        MaskedArray_155886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 8), 'MaskedArray', False)
        # Obtaining the member '__setitem__' of a type (line 356)
        setitem___155887 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 356, 8), MaskedArray_155886, '__setitem__')
        # Calling __setitem__(args, kwargs) (line 356)
        setitem___call_result_155892 = invoke(stypy.reporting.localization.Localization(__file__, 356, 8), setitem___155887, *[self_155888, indx_155889, value_155890], **kwargs_155891)
        
        
        # Type idiom detected: calculating its left and rigth part (line 357)
        # Getting the type of 'basestring' (line 357)
        basestring_155893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 28), 'basestring')
        # Getting the type of 'indx' (line 357)
        indx_155894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 22), 'indx')
        
        (may_be_155895, more_types_in_union_155896) = may_be_subtype(basestring_155893, indx_155894)

        if may_be_155895:

            if more_types_in_union_155896:
                # Runtime conditional SSA (line 357)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'indx' (line 357)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 357, 8), 'indx', remove_not_subtype_from_union(indx_155894, basestring))
            
            # Assigning a Call to a Subscript (line 358):
            
            # Assigning a Call to a Subscript (line 358):
            
            # Call to getmaskarray(...): (line 358)
            # Processing the call arguments (line 358)
            # Getting the type of 'value' (line 358)
            value_155899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 47), 'value', False)
            # Processing the call keyword arguments (line 358)
            kwargs_155900 = {}
            # Getting the type of 'ma' (line 358)
            ma_155897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 31), 'ma', False)
            # Obtaining the member 'getmaskarray' of a type (line 358)
            getmaskarray_155898 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 31), ma_155897, 'getmaskarray')
            # Calling getmaskarray(args, kwargs) (line 358)
            getmaskarray_call_result_155901 = invoke(stypy.reporting.localization.Localization(__file__, 358, 31), getmaskarray_155898, *[value_155899], **kwargs_155900)
            
            # Getting the type of 'self' (line 358)
            self_155902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 12), 'self')
            # Obtaining the member '_mask' of a type (line 358)
            _mask_155903 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 12), self_155902, '_mask')
            # Getting the type of 'indx' (line 358)
            indx_155904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 23), 'indx')
            # Storing an element on a container (line 358)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 358, 12), _mask_155903, (indx_155904, getmaskarray_call_result_155901))

            if more_types_in_union_155896:
                # SSA join for if statement (line 357)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # ################# End of '__setitem__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__setitem__' in the type store
        # Getting the type of 'stypy_return_type' (line 351)
        stypy_return_type_155905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_155905)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__setitem__'
        return stypy_return_type_155905


    @norecursion
    def stypy__str__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__str__'
        module_type_store = module_type_store.open_function_context('__str__', 360, 4, False)
        # Assigning a type to the variable 'self' (line 361)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 361, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MaskedRecords.stypy__str__.__dict__.__setitem__('stypy_localization', localization)
        MaskedRecords.stypy__str__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MaskedRecords.stypy__str__.__dict__.__setitem__('stypy_type_store', module_type_store)
        MaskedRecords.stypy__str__.__dict__.__setitem__('stypy_function_name', 'MaskedRecords.__str__')
        MaskedRecords.stypy__str__.__dict__.__setitem__('stypy_param_names_list', [])
        MaskedRecords.stypy__str__.__dict__.__setitem__('stypy_varargs_param_name', None)
        MaskedRecords.stypy__str__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MaskedRecords.stypy__str__.__dict__.__setitem__('stypy_call_defaults', defaults)
        MaskedRecords.stypy__str__.__dict__.__setitem__('stypy_call_varargs', varargs)
        MaskedRecords.stypy__str__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MaskedRecords.stypy__str__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MaskedRecords.__str__', [], None, None, defaults, varargs, kwargs)

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

        str_155906 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, (-1)), 'str', '\n        Calculates the string representation.\n\n        ')
        
        
        # Getting the type of 'self' (line 365)
        self_155907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 11), 'self')
        # Obtaining the member 'size' of a type (line 365)
        size_155908 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 365, 11), self_155907, 'size')
        int_155909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 365, 23), 'int')
        # Applying the binary operator '>' (line 365)
        result_gt_155910 = python_operator(stypy.reporting.localization.Localization(__file__, 365, 11), '>', size_155908, int_155909)
        
        # Testing the type of an if condition (line 365)
        if_condition_155911 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 365, 8), result_gt_155910)
        # Assigning a type to the variable 'if_condition_155911' (line 365)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 365, 8), 'if_condition_155911', if_condition_155911)
        # SSA begins for if statement (line 365)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a ListComp to a Name (line 366):
        
        # Assigning a ListComp to a Name (line 366):
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to zip(...): (line 367)
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'self' (line 367)
        self_155931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 61), 'self', False)
        # Obtaining the member 'dtype' of a type (line 367)
        dtype_155932 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 367, 61), self_155931, 'dtype')
        # Obtaining the member 'names' of a type (line 367)
        names_155933 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 367, 61), dtype_155932, 'names')
        comprehension_155934 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 367, 35), names_155933)
        # Assigning a type to the variable 'f' (line 367)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 367, 35), 'f', comprehension_155934)
        
        # Call to getattr(...): (line 367)
        # Processing the call arguments (line 367)
        # Getting the type of 'self' (line 367)
        self_155927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 43), 'self', False)
        # Getting the type of 'f' (line 367)
        f_155928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 49), 'f', False)
        # Processing the call keyword arguments (line 367)
        kwargs_155929 = {}
        # Getting the type of 'getattr' (line 367)
        getattr_155926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 35), 'getattr', False)
        # Calling getattr(args, kwargs) (line 367)
        getattr_call_result_155930 = invoke(stypy.reporting.localization.Localization(__file__, 367, 35), getattr_155926, *[self_155927, f_155928], **kwargs_155929)
        
        list_155935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 367, 35), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 367, 35), list_155935, getattr_call_result_155930)
        # Processing the call keyword arguments (line 367)
        kwargs_155936 = {}
        # Getting the type of 'zip' (line 367)
        zip_155925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 29), 'zip', False)
        # Calling zip(args, kwargs) (line 367)
        zip_call_result_155937 = invoke(stypy.reporting.localization.Localization(__file__, 367, 29), zip_155925, *[list_155935], **kwargs_155936)
        
        comprehension_155938 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 366, 20), zip_call_result_155937)
        # Assigning a type to the variable 's' (line 366)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 366, 20), 's', comprehension_155938)
        str_155912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 20), 'str', '(%s)')
        
        # Call to join(...): (line 366)
        # Processing the call arguments (line 366)
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 's' (line 366)
        s_155919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 55), 's', False)
        comprehension_155920 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 366, 39), s_155919)
        # Assigning a type to the variable 'i' (line 366)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 366, 39), 'i', comprehension_155920)
        
        # Call to str(...): (line 366)
        # Processing the call arguments (line 366)
        # Getting the type of 'i' (line 366)
        i_155916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 43), 'i', False)
        # Processing the call keyword arguments (line 366)
        kwargs_155917 = {}
        # Getting the type of 'str' (line 366)
        str_155915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 39), 'str', False)
        # Calling str(args, kwargs) (line 366)
        str_call_result_155918 = invoke(stypy.reporting.localization.Localization(__file__, 366, 39), str_155915, *[i_155916], **kwargs_155917)
        
        list_155921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 39), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 366, 39), list_155921, str_call_result_155918)
        # Processing the call keyword arguments (line 366)
        kwargs_155922 = {}
        str_155913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 29), 'str', ',')
        # Obtaining the member 'join' of a type (line 366)
        join_155914 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 366, 29), str_155913, 'join')
        # Calling join(args, kwargs) (line 366)
        join_call_result_155923 = invoke(stypy.reporting.localization.Localization(__file__, 366, 29), join_155914, *[list_155921], **kwargs_155922)
        
        # Applying the binary operator '%' (line 366)
        result_mod_155924 = python_operator(stypy.reporting.localization.Localization(__file__, 366, 20), '%', str_155912, join_call_result_155923)
        
        list_155939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 20), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 366, 20), list_155939, result_mod_155924)
        # Assigning a type to the variable 'mstr' (line 366)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 366, 12), 'mstr', list_155939)
        str_155940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 368, 19), 'str', '[%s]')
        
        # Call to join(...): (line 368)
        # Processing the call arguments (line 368)
        # Getting the type of 'mstr' (line 368)
        mstr_155943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 38), 'mstr', False)
        # Processing the call keyword arguments (line 368)
        kwargs_155944 = {}
        str_155941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 368, 28), 'str', ', ')
        # Obtaining the member 'join' of a type (line 368)
        join_155942 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 368, 28), str_155941, 'join')
        # Calling join(args, kwargs) (line 368)
        join_call_result_155945 = invoke(stypy.reporting.localization.Localization(__file__, 368, 28), join_155942, *[mstr_155943], **kwargs_155944)
        
        # Applying the binary operator '%' (line 368)
        result_mod_155946 = python_operator(stypy.reporting.localization.Localization(__file__, 368, 19), '%', str_155940, join_call_result_155945)
        
        # Assigning a type to the variable 'stypy_return_type' (line 368)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 368, 12), 'stypy_return_type', result_mod_155946)
        # SSA branch for the else part of an if statement (line 365)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a ListComp to a Name (line 370):
        
        # Assigning a ListComp to a Name (line 370):
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to zip(...): (line 371)
        # Processing the call arguments (line 371)
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'self' (line 371)
        self_155966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 60), 'self', False)
        # Obtaining the member 'dtype' of a type (line 371)
        dtype_155967 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 371, 60), self_155966, 'dtype')
        # Obtaining the member 'names' of a type (line 371)
        names_155968 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 371, 60), dtype_155967, 'names')
        comprehension_155969 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 371, 34), names_155968)
        # Assigning a type to the variable 'f' (line 371)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 371, 34), 'f', comprehension_155969)
        
        # Call to getattr(...): (line 371)
        # Processing the call arguments (line 371)
        # Getting the type of 'self' (line 371)
        self_155962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 42), 'self', False)
        # Getting the type of 'f' (line 371)
        f_155963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 48), 'f', False)
        # Processing the call keyword arguments (line 371)
        kwargs_155964 = {}
        # Getting the type of 'getattr' (line 371)
        getattr_155961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 34), 'getattr', False)
        # Calling getattr(args, kwargs) (line 371)
        getattr_call_result_155965 = invoke(stypy.reporting.localization.Localization(__file__, 371, 34), getattr_155961, *[self_155962, f_155963], **kwargs_155964)
        
        list_155970 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 371, 34), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 371, 34), list_155970, getattr_call_result_155965)
        # Processing the call keyword arguments (line 371)
        kwargs_155971 = {}
        # Getting the type of 'zip' (line 371)
        zip_155960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 29), 'zip', False)
        # Calling zip(args, kwargs) (line 371)
        zip_call_result_155972 = invoke(stypy.reporting.localization.Localization(__file__, 371, 29), zip_155960, *[list_155970], **kwargs_155971)
        
        comprehension_155973 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 370, 20), zip_call_result_155972)
        # Assigning a type to the variable 's' (line 370)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 20), 's', comprehension_155973)
        str_155947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 20), 'str', '%s')
        
        # Call to join(...): (line 370)
        # Processing the call arguments (line 370)
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 's' (line 370)
        s_155954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 53), 's', False)
        comprehension_155955 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 370, 37), s_155954)
        # Assigning a type to the variable 'i' (line 370)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 37), 'i', comprehension_155955)
        
        # Call to str(...): (line 370)
        # Processing the call arguments (line 370)
        # Getting the type of 'i' (line 370)
        i_155951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 41), 'i', False)
        # Processing the call keyword arguments (line 370)
        kwargs_155952 = {}
        # Getting the type of 'str' (line 370)
        str_155950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 37), 'str', False)
        # Calling str(args, kwargs) (line 370)
        str_call_result_155953 = invoke(stypy.reporting.localization.Localization(__file__, 370, 37), str_155950, *[i_155951], **kwargs_155952)
        
        list_155956 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 37), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 370, 37), list_155956, str_call_result_155953)
        # Processing the call keyword arguments (line 370)
        kwargs_155957 = {}
        str_155948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 27), 'str', ',')
        # Obtaining the member 'join' of a type (line 370)
        join_155949 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 27), str_155948, 'join')
        # Calling join(args, kwargs) (line 370)
        join_call_result_155958 = invoke(stypy.reporting.localization.Localization(__file__, 370, 27), join_155949, *[list_155956], **kwargs_155957)
        
        # Applying the binary operator '%' (line 370)
        result_mod_155959 = python_operator(stypy.reporting.localization.Localization(__file__, 370, 20), '%', str_155947, join_call_result_155958)
        
        list_155974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 20), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 370, 20), list_155974, result_mod_155959)
        # Assigning a type to the variable 'mstr' (line 370)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 12), 'mstr', list_155974)
        str_155975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 372, 19), 'str', '(%s)')
        
        # Call to join(...): (line 372)
        # Processing the call arguments (line 372)
        # Getting the type of 'mstr' (line 372)
        mstr_155978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 38), 'mstr', False)
        # Processing the call keyword arguments (line 372)
        kwargs_155979 = {}
        str_155976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 372, 28), 'str', ', ')
        # Obtaining the member 'join' of a type (line 372)
        join_155977 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 372, 28), str_155976, 'join')
        # Calling join(args, kwargs) (line 372)
        join_call_result_155980 = invoke(stypy.reporting.localization.Localization(__file__, 372, 28), join_155977, *[mstr_155978], **kwargs_155979)
        
        # Applying the binary operator '%' (line 372)
        result_mod_155981 = python_operator(stypy.reporting.localization.Localization(__file__, 372, 19), '%', str_155975, join_call_result_155980)
        
        # Assigning a type to the variable 'stypy_return_type' (line 372)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 372, 12), 'stypy_return_type', result_mod_155981)
        # SSA join for if statement (line 365)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__str__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__str__' in the type store
        # Getting the type of 'stypy_return_type' (line 360)
        stypy_return_type_155982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_155982)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__str__'
        return stypy_return_type_155982


    @norecursion
    def stypy__repr__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__repr__'
        module_type_store = module_type_store.open_function_context('__repr__', 374, 4, False)
        # Assigning a type to the variable 'self' (line 375)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 375, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MaskedRecords.stypy__repr__.__dict__.__setitem__('stypy_localization', localization)
        MaskedRecords.stypy__repr__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MaskedRecords.stypy__repr__.__dict__.__setitem__('stypy_type_store', module_type_store)
        MaskedRecords.stypy__repr__.__dict__.__setitem__('stypy_function_name', 'MaskedRecords.__repr__')
        MaskedRecords.stypy__repr__.__dict__.__setitem__('stypy_param_names_list', [])
        MaskedRecords.stypy__repr__.__dict__.__setitem__('stypy_varargs_param_name', None)
        MaskedRecords.stypy__repr__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MaskedRecords.stypy__repr__.__dict__.__setitem__('stypy_call_defaults', defaults)
        MaskedRecords.stypy__repr__.__dict__.__setitem__('stypy_call_varargs', varargs)
        MaskedRecords.stypy__repr__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MaskedRecords.stypy__repr__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MaskedRecords.__repr__', [], None, None, defaults, varargs, kwargs)

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

        str_155983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 378, (-1)), 'str', '\n        Calculates the repr representation.\n\n        ')
        
        # Assigning a Attribute to a Name (line 379):
        
        # Assigning a Attribute to a Name (line 379):
        # Getting the type of 'self' (line 379)
        self_155984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 17), 'self')
        # Obtaining the member 'dtype' of a type (line 379)
        dtype_155985 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 379, 17), self_155984, 'dtype')
        # Obtaining the member 'names' of a type (line 379)
        names_155986 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 379, 17), dtype_155985, 'names')
        # Assigning a type to the variable '_names' (line 379)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 379, 8), '_names', names_155986)
        
        # Assigning a BinOp to a Name (line 380):
        
        # Assigning a BinOp to a Name (line 380):
        str_155987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 380, 14), 'str', '%%%is : %%s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 380)
        tuple_155988 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 380, 31), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 380)
        # Adding element type (line 380)
        
        # Call to max(...): (line 380)
        # Processing the call arguments (line 380)
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of '_names' (line 380)
        _names_155994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 52), '_names', False)
        comprehension_155995 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 380, 36), _names_155994)
        # Assigning a type to the variable 'n' (line 380)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 380, 36), 'n', comprehension_155995)
        
        # Call to len(...): (line 380)
        # Processing the call arguments (line 380)
        # Getting the type of 'n' (line 380)
        n_155991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 40), 'n', False)
        # Processing the call keyword arguments (line 380)
        kwargs_155992 = {}
        # Getting the type of 'len' (line 380)
        len_155990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 36), 'len', False)
        # Calling len(args, kwargs) (line 380)
        len_call_result_155993 = invoke(stypy.reporting.localization.Localization(__file__, 380, 36), len_155990, *[n_155991], **kwargs_155992)
        
        list_155996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 380, 36), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 380, 36), list_155996, len_call_result_155993)
        # Processing the call keyword arguments (line 380)
        kwargs_155997 = {}
        # Getting the type of 'max' (line 380)
        max_155989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 31), 'max', False)
        # Calling max(args, kwargs) (line 380)
        max_call_result_155998 = invoke(stypy.reporting.localization.Localization(__file__, 380, 31), max_155989, *[list_155996], **kwargs_155997)
        
        int_155999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 380, 63), 'int')
        # Applying the binary operator '+' (line 380)
        result_add_156000 = python_operator(stypy.reporting.localization.Localization(__file__, 380, 31), '+', max_call_result_155998, int_155999)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 380, 31), tuple_155988, result_add_156000)
        
        # Applying the binary operator '%' (line 380)
        result_mod_156001 = python_operator(stypy.reporting.localization.Localization(__file__, 380, 14), '%', str_155987, tuple_155988)
        
        # Assigning a type to the variable 'fmt' (line 380)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 380, 8), 'fmt', result_mod_156001)
        
        # Assigning a ListComp to a Name (line 381):
        
        # Assigning a ListComp to a Name (line 381):
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'self' (line 381)
        self_156011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 56), 'self')
        # Obtaining the member 'dtype' of a type (line 381)
        dtype_156012 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 381, 56), self_156011, 'dtype')
        # Obtaining the member 'names' of a type (line 381)
        names_156013 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 381, 56), dtype_156012, 'names')
        comprehension_156014 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 381, 19), names_156013)
        # Assigning a type to the variable 'f' (line 381)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 381, 19), 'f', comprehension_156014)
        # Getting the type of 'fmt' (line 381)
        fmt_156002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 19), 'fmt')
        
        # Obtaining an instance of the builtin type 'tuple' (line 381)
        tuple_156003 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 381)
        # Adding element type (line 381)
        # Getting the type of 'f' (line 381)
        f_156004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 26), 'f')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 381, 26), tuple_156003, f_156004)
        # Adding element type (line 381)
        
        # Call to getattr(...): (line 381)
        # Processing the call arguments (line 381)
        # Getting the type of 'self' (line 381)
        self_156006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 37), 'self', False)
        # Getting the type of 'f' (line 381)
        f_156007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 43), 'f', False)
        # Processing the call keyword arguments (line 381)
        kwargs_156008 = {}
        # Getting the type of 'getattr' (line 381)
        getattr_156005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 29), 'getattr', False)
        # Calling getattr(args, kwargs) (line 381)
        getattr_call_result_156009 = invoke(stypy.reporting.localization.Localization(__file__, 381, 29), getattr_156005, *[self_156006, f_156007], **kwargs_156008)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 381, 26), tuple_156003, getattr_call_result_156009)
        
        # Applying the binary operator '%' (line 381)
        result_mod_156010 = python_operator(stypy.reporting.localization.Localization(__file__, 381, 19), '%', fmt_156002, tuple_156003)
        
        list_156015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 19), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 381, 19), list_156015, result_mod_156010)
        # Assigning a type to the variable 'reprstr' (line 381)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 381, 8), 'reprstr', list_156015)
        
        # Call to insert(...): (line 382)
        # Processing the call arguments (line 382)
        int_156018 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 382, 23), 'int')
        str_156019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 382, 26), 'str', 'masked_records(')
        # Processing the call keyword arguments (line 382)
        kwargs_156020 = {}
        # Getting the type of 'reprstr' (line 382)
        reprstr_156016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 8), 'reprstr', False)
        # Obtaining the member 'insert' of a type (line 382)
        insert_156017 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 382, 8), reprstr_156016, 'insert')
        # Calling insert(args, kwargs) (line 382)
        insert_call_result_156021 = invoke(stypy.reporting.localization.Localization(__file__, 382, 8), insert_156017, *[int_156018, str_156019], **kwargs_156020)
        
        
        # Call to extend(...): (line 383)
        # Processing the call arguments (line 383)
        
        # Obtaining an instance of the builtin type 'list' (line 383)
        list_156024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 383)
        # Adding element type (line 383)
        # Getting the type of 'fmt' (line 383)
        fmt_156025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 24), 'fmt', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 383)
        tuple_156026 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 31), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 383)
        # Adding element type (line 383)
        str_156027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 31), 'str', '    fill_value')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 383, 31), tuple_156026, str_156027)
        # Adding element type (line 383)
        # Getting the type of 'self' (line 383)
        self_156028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 49), 'self', False)
        # Obtaining the member 'fill_value' of a type (line 383)
        fill_value_156029 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 383, 49), self_156028, 'fill_value')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 383, 31), tuple_156026, fill_value_156029)
        
        # Applying the binary operator '%' (line 383)
        result_mod_156030 = python_operator(stypy.reporting.localization.Localization(__file__, 383, 24), '%', fmt_156025, tuple_156026)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 383, 23), list_156024, result_mod_156030)
        # Adding element type (line 383)
        str_156031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 25), 'str', '              )')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 383, 23), list_156024, str_156031)
        
        # Processing the call keyword arguments (line 383)
        kwargs_156032 = {}
        # Getting the type of 'reprstr' (line 383)
        reprstr_156022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 8), 'reprstr', False)
        # Obtaining the member 'extend' of a type (line 383)
        extend_156023 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 383, 8), reprstr_156022, 'extend')
        # Calling extend(args, kwargs) (line 383)
        extend_call_result_156033 = invoke(stypy.reporting.localization.Localization(__file__, 383, 8), extend_156023, *[list_156024], **kwargs_156032)
        
        
        # Call to str(...): (line 385)
        # Processing the call arguments (line 385)
        
        # Call to join(...): (line 385)
        # Processing the call arguments (line 385)
        # Getting the type of 'reprstr' (line 385)
        reprstr_156037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 29), 'reprstr', False)
        # Processing the call keyword arguments (line 385)
        kwargs_156038 = {}
        str_156035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 385, 19), 'str', '\n')
        # Obtaining the member 'join' of a type (line 385)
        join_156036 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 385, 19), str_156035, 'join')
        # Calling join(args, kwargs) (line 385)
        join_call_result_156039 = invoke(stypy.reporting.localization.Localization(__file__, 385, 19), join_156036, *[reprstr_156037], **kwargs_156038)
        
        # Processing the call keyword arguments (line 385)
        kwargs_156040 = {}
        # Getting the type of 'str' (line 385)
        str_156034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 15), 'str', False)
        # Calling str(args, kwargs) (line 385)
        str_call_result_156041 = invoke(stypy.reporting.localization.Localization(__file__, 385, 15), str_156034, *[join_call_result_156039], **kwargs_156040)
        
        # Assigning a type to the variable 'stypy_return_type' (line 385)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 385, 8), 'stypy_return_type', str_call_result_156041)
        
        # ################# End of '__repr__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__repr__' in the type store
        # Getting the type of 'stypy_return_type' (line 374)
        stypy_return_type_156042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_156042)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__repr__'
        return stypy_return_type_156042


    @norecursion
    def view(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 387)
        None_156043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 25), 'None')
        # Getting the type of 'None' (line 387)
        None_156044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 36), 'None')
        defaults = [None_156043, None_156044]
        # Create a new context for function 'view'
        module_type_store = module_type_store.open_function_context('view', 387, 4, False)
        # Assigning a type to the variable 'self' (line 388)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 388, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MaskedRecords.view.__dict__.__setitem__('stypy_localization', localization)
        MaskedRecords.view.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MaskedRecords.view.__dict__.__setitem__('stypy_type_store', module_type_store)
        MaskedRecords.view.__dict__.__setitem__('stypy_function_name', 'MaskedRecords.view')
        MaskedRecords.view.__dict__.__setitem__('stypy_param_names_list', ['dtype', 'type'])
        MaskedRecords.view.__dict__.__setitem__('stypy_varargs_param_name', None)
        MaskedRecords.view.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MaskedRecords.view.__dict__.__setitem__('stypy_call_defaults', defaults)
        MaskedRecords.view.__dict__.__setitem__('stypy_call_varargs', varargs)
        MaskedRecords.view.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MaskedRecords.view.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MaskedRecords.view', ['dtype', 'type'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'view', localization, ['dtype', 'type'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'view(...)' code ##################

        str_156045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 391, (-1)), 'str', '\n        Returns a view of the mrecarray.\n\n        ')
        
        # Type idiom detected: calculating its left and rigth part (line 393)
        # Getting the type of 'dtype' (line 393)
        dtype_156046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 11), 'dtype')
        # Getting the type of 'None' (line 393)
        None_156047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 20), 'None')
        
        (may_be_156048, more_types_in_union_156049) = may_be_none(dtype_156046, None_156047)

        if may_be_156048:

            if more_types_in_union_156049:
                # Runtime conditional SSA (line 393)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Type idiom detected: calculating its left and rigth part (line 394)
            # Getting the type of 'type' (line 394)
            type_156050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 15), 'type')
            # Getting the type of 'None' (line 394)
            None_156051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 23), 'None')
            
            (may_be_156052, more_types_in_union_156053) = may_be_none(type_156050, None_156051)

            if may_be_156052:

                if more_types_in_union_156053:
                    # Runtime conditional SSA (line 394)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                
                # Assigning a Call to a Name (line 395):
                
                # Assigning a Call to a Name (line 395):
                
                # Call to view(...): (line 395)
                # Processing the call arguments (line 395)
                # Getting the type of 'self' (line 395)
                self_156056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 38), 'self', False)
                # Processing the call keyword arguments (line 395)
                kwargs_156057 = {}
                # Getting the type of 'ndarray' (line 395)
                ndarray_156054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 25), 'ndarray', False)
                # Obtaining the member 'view' of a type (line 395)
                view_156055 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 395, 25), ndarray_156054, 'view')
                # Calling view(args, kwargs) (line 395)
                view_call_result_156058 = invoke(stypy.reporting.localization.Localization(__file__, 395, 25), view_156055, *[self_156056], **kwargs_156057)
                
                # Assigning a type to the variable 'output' (line 395)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 395, 16), 'output', view_call_result_156058)

                if more_types_in_union_156053:
                    # Runtime conditional SSA for else branch (line 394)
                    module_type_store.open_ssa_branch('idiom else')



            if ((not may_be_156052) or more_types_in_union_156053):
                
                # Assigning a Call to a Name (line 397):
                
                # Assigning a Call to a Name (line 397):
                
                # Call to view(...): (line 397)
                # Processing the call arguments (line 397)
                # Getting the type of 'self' (line 397)
                self_156061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 38), 'self', False)
                # Getting the type of 'type' (line 397)
                type_156062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 44), 'type', False)
                # Processing the call keyword arguments (line 397)
                kwargs_156063 = {}
                # Getting the type of 'ndarray' (line 397)
                ndarray_156059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 25), 'ndarray', False)
                # Obtaining the member 'view' of a type (line 397)
                view_156060 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 397, 25), ndarray_156059, 'view')
                # Calling view(args, kwargs) (line 397)
                view_call_result_156064 = invoke(stypy.reporting.localization.Localization(__file__, 397, 25), view_156060, *[self_156061, type_156062], **kwargs_156063)
                
                # Assigning a type to the variable 'output' (line 397)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 397, 16), 'output', view_call_result_156064)

                if (may_be_156052 and more_types_in_union_156053):
                    # SSA join for if statement (line 394)
                    module_type_store = module_type_store.join_ssa_context()


            

            if more_types_in_union_156049:
                # Runtime conditional SSA for else branch (line 393)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_156048) or more_types_in_union_156049):
            
            # Type idiom detected: calculating its left and rigth part (line 399)
            # Getting the type of 'type' (line 399)
            type_156065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 13), 'type')
            # Getting the type of 'None' (line 399)
            None_156066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 21), 'None')
            
            (may_be_156067, more_types_in_union_156068) = may_be_none(type_156065, None_156066)

            if may_be_156067:

                if more_types_in_union_156068:
                    # Runtime conditional SSA (line 399)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                
                
                # SSA begins for try-except statement (line 400)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
                
                
                # Call to issubclass(...): (line 401)
                # Processing the call arguments (line 401)
                # Getting the type of 'dtype' (line 401)
                dtype_156070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 30), 'dtype', False)
                # Getting the type of 'ndarray' (line 401)
                ndarray_156071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 37), 'ndarray', False)
                # Processing the call keyword arguments (line 401)
                kwargs_156072 = {}
                # Getting the type of 'issubclass' (line 401)
                issubclass_156069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 19), 'issubclass', False)
                # Calling issubclass(args, kwargs) (line 401)
                issubclass_call_result_156073 = invoke(stypy.reporting.localization.Localization(__file__, 401, 19), issubclass_156069, *[dtype_156070, ndarray_156071], **kwargs_156072)
                
                # Testing the type of an if condition (line 401)
                if_condition_156074 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 401, 16), issubclass_call_result_156073)
                # Assigning a type to the variable 'if_condition_156074' (line 401)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 401, 16), 'if_condition_156074', if_condition_156074)
                # SSA begins for if statement (line 401)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Call to a Name (line 402):
                
                # Assigning a Call to a Name (line 402):
                
                # Call to view(...): (line 402)
                # Processing the call arguments (line 402)
                # Getting the type of 'self' (line 402)
                self_156077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 42), 'self', False)
                # Getting the type of 'dtype' (line 402)
                dtype_156078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 48), 'dtype', False)
                # Processing the call keyword arguments (line 402)
                kwargs_156079 = {}
                # Getting the type of 'ndarray' (line 402)
                ndarray_156075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 29), 'ndarray', False)
                # Obtaining the member 'view' of a type (line 402)
                view_156076 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 402, 29), ndarray_156075, 'view')
                # Calling view(args, kwargs) (line 402)
                view_call_result_156080 = invoke(stypy.reporting.localization.Localization(__file__, 402, 29), view_156076, *[self_156077, dtype_156078], **kwargs_156079)
                
                # Assigning a type to the variable 'output' (line 402)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 402, 20), 'output', view_call_result_156080)
                
                # Assigning a Name to a Name (line 403):
                
                # Assigning a Name to a Name (line 403):
                # Getting the type of 'None' (line 403)
                None_156081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 28), 'None')
                # Assigning a type to the variable 'dtype' (line 403)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 403, 20), 'dtype', None_156081)
                # SSA branch for the else part of an if statement (line 401)
                module_type_store.open_ssa_branch('else')
                
                # Assigning a Call to a Name (line 405):
                
                # Assigning a Call to a Name (line 405):
                
                # Call to view(...): (line 405)
                # Processing the call arguments (line 405)
                # Getting the type of 'self' (line 405)
                self_156084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 42), 'self', False)
                # Getting the type of 'dtype' (line 405)
                dtype_156085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 48), 'dtype', False)
                # Processing the call keyword arguments (line 405)
                kwargs_156086 = {}
                # Getting the type of 'ndarray' (line 405)
                ndarray_156082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 29), 'ndarray', False)
                # Obtaining the member 'view' of a type (line 405)
                view_156083 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 405, 29), ndarray_156082, 'view')
                # Calling view(args, kwargs) (line 405)
                view_call_result_156087 = invoke(stypy.reporting.localization.Localization(__file__, 405, 29), view_156083, *[self_156084, dtype_156085], **kwargs_156086)
                
                # Assigning a type to the variable 'output' (line 405)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 405, 20), 'output', view_call_result_156087)
                # SSA join for if statement (line 401)
                module_type_store = module_type_store.join_ssa_context()
                
                # SSA branch for the except part of a try statement (line 400)
                # SSA branch for the except 'TypeError' branch of a try statement (line 400)
                module_type_store.open_ssa_branch('except')
                
                # Assigning a Call to a Name (line 408):
                
                # Assigning a Call to a Name (line 408):
                
                # Call to dtype(...): (line 408)
                # Processing the call arguments (line 408)
                # Getting the type of 'dtype' (line 408)
                dtype_156090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 33), 'dtype', False)
                # Processing the call keyword arguments (line 408)
                kwargs_156091 = {}
                # Getting the type of 'np' (line 408)
                np_156088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 24), 'np', False)
                # Obtaining the member 'dtype' of a type (line 408)
                dtype_156089 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 408, 24), np_156088, 'dtype')
                # Calling dtype(args, kwargs) (line 408)
                dtype_call_result_156092 = invoke(stypy.reporting.localization.Localization(__file__, 408, 24), dtype_156089, *[dtype_156090], **kwargs_156091)
                
                # Assigning a type to the variable 'dtype' (line 408)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 408, 16), 'dtype', dtype_call_result_156092)
                
                # Type idiom detected: calculating its left and rigth part (line 412)
                # Getting the type of 'dtype' (line 412)
                dtype_156093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 19), 'dtype')
                # Obtaining the member 'fields' of a type (line 412)
                fields_156094 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 412, 19), dtype_156093, 'fields')
                # Getting the type of 'None' (line 412)
                None_156095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 35), 'None')
                
                (may_be_156096, more_types_in_union_156097) = may_be_none(fields_156094, None_156095)

                if may_be_156096:

                    if more_types_in_union_156097:
                        # Runtime conditional SSA (line 412)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                    else:
                        module_type_store = module_type_store

                    
                    # Assigning a Subscript to a Name (line 413):
                    
                    # Assigning a Subscript to a Name (line 413):
                    
                    # Obtaining the type of the subscript
                    int_156098 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, 56), 'int')
                    # Getting the type of 'self' (line 413)
                    self_156099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 31), 'self')
                    # Obtaining the member '__class__' of a type (line 413)
                    class___156100 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 413, 31), self_156099, '__class__')
                    # Obtaining the member '__bases__' of a type (line 413)
                    bases___156101 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 413, 31), class___156100, '__bases__')
                    # Obtaining the member '__getitem__' of a type (line 413)
                    getitem___156102 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 413, 31), bases___156101, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 413)
                    subscript_call_result_156103 = invoke(stypy.reporting.localization.Localization(__file__, 413, 31), getitem___156102, int_156098)
                    
                    # Assigning a type to the variable 'basetype' (line 413)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 413, 20), 'basetype', subscript_call_result_156103)
                    
                    # Assigning a Call to a Name (line 414):
                    
                    # Assigning a Call to a Name (line 414):
                    
                    # Call to view(...): (line 414)
                    # Processing the call arguments (line 414)
                    # Getting the type of 'dtype' (line 414)
                    dtype_156109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 51), 'dtype', False)
                    # Getting the type of 'basetype' (line 414)
                    basetype_156110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 58), 'basetype', False)
                    # Processing the call keyword arguments (line 414)
                    kwargs_156111 = {}
                    
                    # Call to __array__(...): (line 414)
                    # Processing the call keyword arguments (line 414)
                    kwargs_156106 = {}
                    # Getting the type of 'self' (line 414)
                    self_156104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 29), 'self', False)
                    # Obtaining the member '__array__' of a type (line 414)
                    array___156105 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 414, 29), self_156104, '__array__')
                    # Calling __array__(args, kwargs) (line 414)
                    array___call_result_156107 = invoke(stypy.reporting.localization.Localization(__file__, 414, 29), array___156105, *[], **kwargs_156106)
                    
                    # Obtaining the member 'view' of a type (line 414)
                    view_156108 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 414, 29), array___call_result_156107, 'view')
                    # Calling view(args, kwargs) (line 414)
                    view_call_result_156112 = invoke(stypy.reporting.localization.Localization(__file__, 414, 29), view_156108, *[dtype_156109, basetype_156110], **kwargs_156111)
                    
                    # Assigning a type to the variable 'output' (line 414)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 414, 20), 'output', view_call_result_156112)
                    
                    # Call to _update_from(...): (line 415)
                    # Processing the call arguments (line 415)
                    # Getting the type of 'self' (line 415)
                    self_156115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 40), 'self', False)
                    # Processing the call keyword arguments (line 415)
                    kwargs_156116 = {}
                    # Getting the type of 'output' (line 415)
                    output_156113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 20), 'output', False)
                    # Obtaining the member '_update_from' of a type (line 415)
                    _update_from_156114 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 415, 20), output_156113, '_update_from')
                    # Calling _update_from(args, kwargs) (line 415)
                    _update_from_call_result_156117 = invoke(stypy.reporting.localization.Localization(__file__, 415, 20), _update_from_156114, *[self_156115], **kwargs_156116)
                    

                    if more_types_in_union_156097:
                        # Runtime conditional SSA for else branch (line 412)
                        module_type_store.open_ssa_branch('idiom else')



                if ((not may_be_156096) or more_types_in_union_156097):
                    
                    # Assigning a Call to a Name (line 417):
                    
                    # Assigning a Call to a Name (line 417):
                    
                    # Call to view(...): (line 417)
                    # Processing the call arguments (line 417)
                    # Getting the type of 'self' (line 417)
                    self_156120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 42), 'self', False)
                    # Getting the type of 'dtype' (line 417)
                    dtype_156121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 48), 'dtype', False)
                    # Processing the call keyword arguments (line 417)
                    kwargs_156122 = {}
                    # Getting the type of 'ndarray' (line 417)
                    ndarray_156118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 29), 'ndarray', False)
                    # Obtaining the member 'view' of a type (line 417)
                    view_156119 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 417, 29), ndarray_156118, 'view')
                    # Calling view(args, kwargs) (line 417)
                    view_call_result_156123 = invoke(stypy.reporting.localization.Localization(__file__, 417, 29), view_156119, *[self_156120, dtype_156121], **kwargs_156122)
                    
                    # Assigning a type to the variable 'output' (line 417)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 417, 20), 'output', view_call_result_156123)

                    if (may_be_156096 and more_types_in_union_156097):
                        # SSA join for if statement (line 412)
                        module_type_store = module_type_store.join_ssa_context()


                
                
                # Assigning a Name to a Attribute (line 418):
                
                # Assigning a Name to a Attribute (line 418):
                # Getting the type of 'None' (line 418)
                None_156124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 37), 'None')
                # Getting the type of 'output' (line 418)
                output_156125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 16), 'output')
                # Setting the type of the member '_fill_value' of a type (line 418)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 418, 16), output_156125, '_fill_value', None_156124)
                # SSA join for try-except statement (line 400)
                module_type_store = module_type_store.join_ssa_context()
                

                if more_types_in_union_156068:
                    # Runtime conditional SSA for else branch (line 399)
                    module_type_store.open_ssa_branch('idiom else')



            if ((not may_be_156067) or more_types_in_union_156068):
                
                # Assigning a Call to a Name (line 420):
                
                # Assigning a Call to a Name (line 420):
                
                # Call to view(...): (line 420)
                # Processing the call arguments (line 420)
                # Getting the type of 'self' (line 420)
                self_156128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 34), 'self', False)
                # Getting the type of 'dtype' (line 420)
                dtype_156129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 40), 'dtype', False)
                # Getting the type of 'type' (line 420)
                type_156130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 47), 'type', False)
                # Processing the call keyword arguments (line 420)
                kwargs_156131 = {}
                # Getting the type of 'ndarray' (line 420)
                ndarray_156126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 21), 'ndarray', False)
                # Obtaining the member 'view' of a type (line 420)
                view_156127 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 420, 21), ndarray_156126, 'view')
                # Calling view(args, kwargs) (line 420)
                view_call_result_156132 = invoke(stypy.reporting.localization.Localization(__file__, 420, 21), view_156127, *[self_156128, dtype_156129, type_156130], **kwargs_156131)
                
                # Assigning a type to the variable 'output' (line 420)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 420, 12), 'output', view_call_result_156132)

                if (may_be_156067 and more_types_in_union_156068):
                    # SSA join for if statement (line 399)
                    module_type_store = module_type_store.join_ssa_context()


            

            if (may_be_156048 and more_types_in_union_156049):
                # SSA join for if statement (line 393)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        
        # Call to getattr(...): (line 422)
        # Processing the call arguments (line 422)
        # Getting the type of 'output' (line 422)
        output_156134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 20), 'output', False)
        str_156135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 422, 28), 'str', '_mask')
        # Getting the type of 'nomask' (line 422)
        nomask_156136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 37), 'nomask', False)
        # Processing the call keyword arguments (line 422)
        kwargs_156137 = {}
        # Getting the type of 'getattr' (line 422)
        getattr_156133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 12), 'getattr', False)
        # Calling getattr(args, kwargs) (line 422)
        getattr_call_result_156138 = invoke(stypy.reporting.localization.Localization(__file__, 422, 12), getattr_156133, *[output_156134, str_156135, nomask_156136], **kwargs_156137)
        
        # Getting the type of 'nomask' (line 422)
        nomask_156139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 52), 'nomask')
        # Applying the binary operator 'isnot' (line 422)
        result_is_not_156140 = python_operator(stypy.reporting.localization.Localization(__file__, 422, 12), 'isnot', getattr_call_result_156138, nomask_156139)
        
        # Testing the type of an if condition (line 422)
        if_condition_156141 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 422, 8), result_is_not_156140)
        # Assigning a type to the variable 'if_condition_156141' (line 422)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 422, 8), 'if_condition_156141', if_condition_156141)
        # SSA begins for if statement (line 422)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 423):
        
        # Assigning a Call to a Name (line 423):
        
        # Call to make_mask_descr(...): (line 423)
        # Processing the call arguments (line 423)
        # Getting the type of 'output' (line 423)
        output_156144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 40), 'output', False)
        # Obtaining the member 'dtype' of a type (line 423)
        dtype_156145 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 423, 40), output_156144, 'dtype')
        # Processing the call keyword arguments (line 423)
        kwargs_156146 = {}
        # Getting the type of 'ma' (line 423)
        ma_156142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 21), 'ma', False)
        # Obtaining the member 'make_mask_descr' of a type (line 423)
        make_mask_descr_156143 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 423, 21), ma_156142, 'make_mask_descr')
        # Calling make_mask_descr(args, kwargs) (line 423)
        make_mask_descr_call_result_156147 = invoke(stypy.reporting.localization.Localization(__file__, 423, 21), make_mask_descr_156143, *[dtype_156145], **kwargs_156146)
        
        # Assigning a type to the variable 'mdtype' (line 423)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 423, 12), 'mdtype', make_mask_descr_call_result_156147)
        
        # Assigning a Call to a Attribute (line 424):
        
        # Assigning a Call to a Attribute (line 424):
        
        # Call to view(...): (line 424)
        # Processing the call arguments (line 424)
        # Getting the type of 'mdtype' (line 424)
        mdtype_156151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 43), 'mdtype', False)
        # Getting the type of 'ndarray' (line 424)
        ndarray_156152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 51), 'ndarray', False)
        # Processing the call keyword arguments (line 424)
        kwargs_156153 = {}
        # Getting the type of 'self' (line 424)
        self_156148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 27), 'self', False)
        # Obtaining the member '_mask' of a type (line 424)
        _mask_156149 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 424, 27), self_156148, '_mask')
        # Obtaining the member 'view' of a type (line 424)
        view_156150 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 424, 27), _mask_156149, 'view')
        # Calling view(args, kwargs) (line 424)
        view_call_result_156154 = invoke(stypy.reporting.localization.Localization(__file__, 424, 27), view_156150, *[mdtype_156151, ndarray_156152], **kwargs_156153)
        
        # Getting the type of 'output' (line 424)
        output_156155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 12), 'output')
        # Setting the type of the member '_mask' of a type (line 424)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 424, 12), output_156155, '_mask', view_call_result_156154)
        
        # Assigning a Attribute to a Attribute (line 425):
        
        # Assigning a Attribute to a Attribute (line 425):
        # Getting the type of 'output' (line 425)
        output_156156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 33), 'output')
        # Obtaining the member 'shape' of a type (line 425)
        shape_156157 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 425, 33), output_156156, 'shape')
        # Getting the type of 'output' (line 425)
        output_156158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 12), 'output')
        # Obtaining the member '_mask' of a type (line 425)
        _mask_156159 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 425, 12), output_156158, '_mask')
        # Setting the type of the member 'shape' of a type (line 425)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 425, 12), _mask_156159, 'shape', shape_156157)
        # SSA join for if statement (line 422)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'output' (line 426)
        output_156160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 15), 'output')
        # Assigning a type to the variable 'stypy_return_type' (line 426)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 426, 8), 'stypy_return_type', output_156160)
        
        # ################# End of 'view(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'view' in the type store
        # Getting the type of 'stypy_return_type' (line 387)
        stypy_return_type_156161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_156161)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'view'
        return stypy_return_type_156161


    @norecursion
    def harden_mask(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'harden_mask'
        module_type_store = module_type_store.open_function_context('harden_mask', 428, 4, False)
        # Assigning a type to the variable 'self' (line 429)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 429, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MaskedRecords.harden_mask.__dict__.__setitem__('stypy_localization', localization)
        MaskedRecords.harden_mask.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MaskedRecords.harden_mask.__dict__.__setitem__('stypy_type_store', module_type_store)
        MaskedRecords.harden_mask.__dict__.__setitem__('stypy_function_name', 'MaskedRecords.harden_mask')
        MaskedRecords.harden_mask.__dict__.__setitem__('stypy_param_names_list', [])
        MaskedRecords.harden_mask.__dict__.__setitem__('stypy_varargs_param_name', None)
        MaskedRecords.harden_mask.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MaskedRecords.harden_mask.__dict__.__setitem__('stypy_call_defaults', defaults)
        MaskedRecords.harden_mask.__dict__.__setitem__('stypy_call_varargs', varargs)
        MaskedRecords.harden_mask.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MaskedRecords.harden_mask.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MaskedRecords.harden_mask', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'harden_mask', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'harden_mask(...)' code ##################

        str_156162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 432, (-1)), 'str', '\n        Forces the mask to hard.\n\n        ')
        
        # Assigning a Name to a Attribute (line 433):
        
        # Assigning a Name to a Attribute (line 433):
        # Getting the type of 'True' (line 433)
        True_156163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 25), 'True')
        # Getting the type of 'self' (line 433)
        self_156164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 8), 'self')
        # Setting the type of the member '_hardmask' of a type (line 433)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 433, 8), self_156164, '_hardmask', True_156163)
        
        # ################# End of 'harden_mask(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'harden_mask' in the type store
        # Getting the type of 'stypy_return_type' (line 428)
        stypy_return_type_156165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_156165)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'harden_mask'
        return stypy_return_type_156165


    @norecursion
    def soften_mask(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'soften_mask'
        module_type_store = module_type_store.open_function_context('soften_mask', 435, 4, False)
        # Assigning a type to the variable 'self' (line 436)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 436, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MaskedRecords.soften_mask.__dict__.__setitem__('stypy_localization', localization)
        MaskedRecords.soften_mask.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MaskedRecords.soften_mask.__dict__.__setitem__('stypy_type_store', module_type_store)
        MaskedRecords.soften_mask.__dict__.__setitem__('stypy_function_name', 'MaskedRecords.soften_mask')
        MaskedRecords.soften_mask.__dict__.__setitem__('stypy_param_names_list', [])
        MaskedRecords.soften_mask.__dict__.__setitem__('stypy_varargs_param_name', None)
        MaskedRecords.soften_mask.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MaskedRecords.soften_mask.__dict__.__setitem__('stypy_call_defaults', defaults)
        MaskedRecords.soften_mask.__dict__.__setitem__('stypy_call_varargs', varargs)
        MaskedRecords.soften_mask.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MaskedRecords.soften_mask.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MaskedRecords.soften_mask', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'soften_mask', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'soften_mask(...)' code ##################

        str_156166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 439, (-1)), 'str', '\n        Forces the mask to soft\n\n        ')
        
        # Assigning a Name to a Attribute (line 440):
        
        # Assigning a Name to a Attribute (line 440):
        # Getting the type of 'False' (line 440)
        False_156167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 25), 'False')
        # Getting the type of 'self' (line 440)
        self_156168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 8), 'self')
        # Setting the type of the member '_hardmask' of a type (line 440)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 440, 8), self_156168, '_hardmask', False_156167)
        
        # ################# End of 'soften_mask(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'soften_mask' in the type store
        # Getting the type of 'stypy_return_type' (line 435)
        stypy_return_type_156169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_156169)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'soften_mask'
        return stypy_return_type_156169


    @norecursion
    def copy(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'copy'
        module_type_store = module_type_store.open_function_context('copy', 442, 4, False)
        # Assigning a type to the variable 'self' (line 443)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 443, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MaskedRecords.copy.__dict__.__setitem__('stypy_localization', localization)
        MaskedRecords.copy.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MaskedRecords.copy.__dict__.__setitem__('stypy_type_store', module_type_store)
        MaskedRecords.copy.__dict__.__setitem__('stypy_function_name', 'MaskedRecords.copy')
        MaskedRecords.copy.__dict__.__setitem__('stypy_param_names_list', [])
        MaskedRecords.copy.__dict__.__setitem__('stypy_varargs_param_name', None)
        MaskedRecords.copy.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MaskedRecords.copy.__dict__.__setitem__('stypy_call_defaults', defaults)
        MaskedRecords.copy.__dict__.__setitem__('stypy_call_varargs', varargs)
        MaskedRecords.copy.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MaskedRecords.copy.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MaskedRecords.copy', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'copy', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'copy(...)' code ##################

        str_156170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 446, (-1)), 'str', '\n        Returns a copy of the masked record.\n\n        ')
        
        # Assigning a Call to a Name (line 447):
        
        # Assigning a Call to a Name (line 447):
        
        # Call to view(...): (line 447)
        # Processing the call arguments (line 447)
        
        # Call to type(...): (line 447)
        # Processing the call arguments (line 447)
        # Getting the type of 'self' (line 447)
        self_156178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 45), 'self', False)
        # Processing the call keyword arguments (line 447)
        kwargs_156179 = {}
        # Getting the type of 'type' (line 447)
        type_156177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 40), 'type', False)
        # Calling type(args, kwargs) (line 447)
        type_call_result_156180 = invoke(stypy.reporting.localization.Localization(__file__, 447, 40), type_156177, *[self_156178], **kwargs_156179)
        
        # Processing the call keyword arguments (line 447)
        kwargs_156181 = {}
        
        # Call to copy(...): (line 447)
        # Processing the call keyword arguments (line 447)
        kwargs_156174 = {}
        # Getting the type of 'self' (line 447)
        self_156171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 17), 'self', False)
        # Obtaining the member '_data' of a type (line 447)
        _data_156172 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 447, 17), self_156171, '_data')
        # Obtaining the member 'copy' of a type (line 447)
        copy_156173 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 447, 17), _data_156172, 'copy')
        # Calling copy(args, kwargs) (line 447)
        copy_call_result_156175 = invoke(stypy.reporting.localization.Localization(__file__, 447, 17), copy_156173, *[], **kwargs_156174)
        
        # Obtaining the member 'view' of a type (line 447)
        view_156176 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 447, 17), copy_call_result_156175, 'view')
        # Calling view(args, kwargs) (line 447)
        view_call_result_156182 = invoke(stypy.reporting.localization.Localization(__file__, 447, 17), view_156176, *[type_call_result_156180], **kwargs_156181)
        
        # Assigning a type to the variable 'copied' (line 447)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 447, 8), 'copied', view_call_result_156182)
        
        # Assigning a Call to a Attribute (line 448):
        
        # Assigning a Call to a Attribute (line 448):
        
        # Call to copy(...): (line 448)
        # Processing the call keyword arguments (line 448)
        kwargs_156186 = {}
        # Getting the type of 'self' (line 448)
        self_156183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 23), 'self', False)
        # Obtaining the member '_mask' of a type (line 448)
        _mask_156184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 448, 23), self_156183, '_mask')
        # Obtaining the member 'copy' of a type (line 448)
        copy_156185 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 448, 23), _mask_156184, 'copy')
        # Calling copy(args, kwargs) (line 448)
        copy_call_result_156187 = invoke(stypy.reporting.localization.Localization(__file__, 448, 23), copy_156185, *[], **kwargs_156186)
        
        # Getting the type of 'copied' (line 448)
        copied_156188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 8), 'copied')
        # Setting the type of the member '_mask' of a type (line 448)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 448, 8), copied_156188, '_mask', copy_call_result_156187)
        # Getting the type of 'copied' (line 449)
        copied_156189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 15), 'copied')
        # Assigning a type to the variable 'stypy_return_type' (line 449)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 449, 8), 'stypy_return_type', copied_156189)
        
        # ################# End of 'copy(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'copy' in the type store
        # Getting the type of 'stypy_return_type' (line 442)
        stypy_return_type_156190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_156190)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'copy'
        return stypy_return_type_156190


    @norecursion
    def tolist(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 451)
        None_156191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 32), 'None')
        defaults = [None_156191]
        # Create a new context for function 'tolist'
        module_type_store = module_type_store.open_function_context('tolist', 451, 4, False)
        # Assigning a type to the variable 'self' (line 452)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 452, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MaskedRecords.tolist.__dict__.__setitem__('stypy_localization', localization)
        MaskedRecords.tolist.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MaskedRecords.tolist.__dict__.__setitem__('stypy_type_store', module_type_store)
        MaskedRecords.tolist.__dict__.__setitem__('stypy_function_name', 'MaskedRecords.tolist')
        MaskedRecords.tolist.__dict__.__setitem__('stypy_param_names_list', ['fill_value'])
        MaskedRecords.tolist.__dict__.__setitem__('stypy_varargs_param_name', None)
        MaskedRecords.tolist.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MaskedRecords.tolist.__dict__.__setitem__('stypy_call_defaults', defaults)
        MaskedRecords.tolist.__dict__.__setitem__('stypy_call_varargs', varargs)
        MaskedRecords.tolist.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MaskedRecords.tolist.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MaskedRecords.tolist', ['fill_value'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'tolist', localization, ['fill_value'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'tolist(...)' code ##################

        str_156192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 459, (-1)), 'str', '\n        Return the data portion of the array as a list.\n\n        Data items are converted to the nearest compatible Python type.\n        Masked values are converted to fill_value. If fill_value is None,\n        the corresponding entries in the output list will be ``None``.\n\n        ')
        
        # Type idiom detected: calculating its left and rigth part (line 460)
        # Getting the type of 'fill_value' (line 460)
        fill_value_156193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 8), 'fill_value')
        # Getting the type of 'None' (line 460)
        None_156194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 29), 'None')
        
        (may_be_156195, more_types_in_union_156196) = may_not_be_none(fill_value_156193, None_156194)

        if may_be_156195:

            if more_types_in_union_156196:
                # Runtime conditional SSA (line 460)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to tolist(...): (line 461)
            # Processing the call keyword arguments (line 461)
            kwargs_156203 = {}
            
            # Call to filled(...): (line 461)
            # Processing the call arguments (line 461)
            # Getting the type of 'fill_value' (line 461)
            fill_value_156199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 31), 'fill_value', False)
            # Processing the call keyword arguments (line 461)
            kwargs_156200 = {}
            # Getting the type of 'self' (line 461)
            self_156197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 19), 'self', False)
            # Obtaining the member 'filled' of a type (line 461)
            filled_156198 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 461, 19), self_156197, 'filled')
            # Calling filled(args, kwargs) (line 461)
            filled_call_result_156201 = invoke(stypy.reporting.localization.Localization(__file__, 461, 19), filled_156198, *[fill_value_156199], **kwargs_156200)
            
            # Obtaining the member 'tolist' of a type (line 461)
            tolist_156202 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 461, 19), filled_call_result_156201, 'tolist')
            # Calling tolist(args, kwargs) (line 461)
            tolist_call_result_156204 = invoke(stypy.reporting.localization.Localization(__file__, 461, 19), tolist_156202, *[], **kwargs_156203)
            
            # Assigning a type to the variable 'stypy_return_type' (line 461)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 461, 12), 'stypy_return_type', tolist_call_result_156204)

            if more_types_in_union_156196:
                # SSA join for if statement (line 460)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Name (line 462):
        
        # Assigning a Call to a Name (line 462):
        
        # Call to narray(...): (line 462)
        # Processing the call arguments (line 462)
        
        # Call to tolist(...): (line 462)
        # Processing the call keyword arguments (line 462)
        kwargs_156211 = {}
        
        # Call to filled(...): (line 462)
        # Processing the call keyword arguments (line 462)
        kwargs_156208 = {}
        # Getting the type of 'self' (line 462)
        self_156206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 24), 'self', False)
        # Obtaining the member 'filled' of a type (line 462)
        filled_156207 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 462, 24), self_156206, 'filled')
        # Calling filled(args, kwargs) (line 462)
        filled_call_result_156209 = invoke(stypy.reporting.localization.Localization(__file__, 462, 24), filled_156207, *[], **kwargs_156208)
        
        # Obtaining the member 'tolist' of a type (line 462)
        tolist_156210 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 462, 24), filled_call_result_156209, 'tolist')
        # Calling tolist(args, kwargs) (line 462)
        tolist_call_result_156212 = invoke(stypy.reporting.localization.Localization(__file__, 462, 24), tolist_156210, *[], **kwargs_156211)
        
        # Processing the call keyword arguments (line 462)
        # Getting the type of 'object' (line 462)
        object_156213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 54), 'object', False)
        keyword_156214 = object_156213
        kwargs_156215 = {'dtype': keyword_156214}
        # Getting the type of 'narray' (line 462)
        narray_156205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 17), 'narray', False)
        # Calling narray(args, kwargs) (line 462)
        narray_call_result_156216 = invoke(stypy.reporting.localization.Localization(__file__, 462, 17), narray_156205, *[tolist_call_result_156212], **kwargs_156215)
        
        # Assigning a type to the variable 'result' (line 462)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 462, 8), 'result', narray_call_result_156216)
        
        # Assigning a Call to a Name (line 463):
        
        # Assigning a Call to a Name (line 463):
        
        # Call to narray(...): (line 463)
        # Processing the call arguments (line 463)
        
        # Call to tolist(...): (line 463)
        # Processing the call keyword arguments (line 463)
        kwargs_156221 = {}
        # Getting the type of 'self' (line 463)
        self_156218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 22), 'self', False)
        # Obtaining the member '_mask' of a type (line 463)
        _mask_156219 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 463, 22), self_156218, '_mask')
        # Obtaining the member 'tolist' of a type (line 463)
        tolist_156220 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 463, 22), _mask_156219, 'tolist')
        # Calling tolist(args, kwargs) (line 463)
        tolist_call_result_156222 = invoke(stypy.reporting.localization.Localization(__file__, 463, 22), tolist_156220, *[], **kwargs_156221)
        
        # Processing the call keyword arguments (line 463)
        kwargs_156223 = {}
        # Getting the type of 'narray' (line 463)
        narray_156217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 15), 'narray', False)
        # Calling narray(args, kwargs) (line 463)
        narray_call_result_156224 = invoke(stypy.reporting.localization.Localization(__file__, 463, 15), narray_156217, *[tolist_call_result_156222], **kwargs_156223)
        
        # Assigning a type to the variable 'mask' (line 463)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 463, 8), 'mask', narray_call_result_156224)
        
        # Assigning a Name to a Subscript (line 464):
        
        # Assigning a Name to a Subscript (line 464):
        # Getting the type of 'None' (line 464)
        None_156225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 23), 'None')
        # Getting the type of 'result' (line 464)
        result_156226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 8), 'result')
        # Getting the type of 'mask' (line 464)
        mask_156227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 15), 'mask')
        # Storing an element on a container (line 464)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 464, 8), result_156226, (mask_156227, None_156225))
        
        # Call to tolist(...): (line 465)
        # Processing the call keyword arguments (line 465)
        kwargs_156230 = {}
        # Getting the type of 'result' (line 465)
        result_156228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 15), 'result', False)
        # Obtaining the member 'tolist' of a type (line 465)
        tolist_156229 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 465, 15), result_156228, 'tolist')
        # Calling tolist(args, kwargs) (line 465)
        tolist_call_result_156231 = invoke(stypy.reporting.localization.Localization(__file__, 465, 15), tolist_156229, *[], **kwargs_156230)
        
        # Assigning a type to the variable 'stypy_return_type' (line 465)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 465, 8), 'stypy_return_type', tolist_call_result_156231)
        
        # ################# End of 'tolist(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'tolist' in the type store
        # Getting the type of 'stypy_return_type' (line 451)
        stypy_return_type_156232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_156232)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'tolist'
        return stypy_return_type_156232


    @norecursion
    def __getstate__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__getstate__'
        module_type_store = module_type_store.open_function_context('__getstate__', 467, 4, False)
        # Assigning a type to the variable 'self' (line 468)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 468, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MaskedRecords.__getstate__.__dict__.__setitem__('stypy_localization', localization)
        MaskedRecords.__getstate__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MaskedRecords.__getstate__.__dict__.__setitem__('stypy_type_store', module_type_store)
        MaskedRecords.__getstate__.__dict__.__setitem__('stypy_function_name', 'MaskedRecords.__getstate__')
        MaskedRecords.__getstate__.__dict__.__setitem__('stypy_param_names_list', [])
        MaskedRecords.__getstate__.__dict__.__setitem__('stypy_varargs_param_name', None)
        MaskedRecords.__getstate__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MaskedRecords.__getstate__.__dict__.__setitem__('stypy_call_defaults', defaults)
        MaskedRecords.__getstate__.__dict__.__setitem__('stypy_call_varargs', varargs)
        MaskedRecords.__getstate__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MaskedRecords.__getstate__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MaskedRecords.__getstate__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__getstate__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__getstate__(...)' code ##################

        str_156233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 472, (-1)), 'str', 'Return the internal state of the masked array.\n\n        This is for pickling.\n\n        ')
        
        # Assigning a Tuple to a Name (line 473):
        
        # Assigning a Tuple to a Name (line 473):
        
        # Obtaining an instance of the builtin type 'tuple' (line 473)
        tuple_156234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 473, 17), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 473)
        # Adding element type (line 473)
        int_156235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 473, 17), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 473, 17), tuple_156234, int_156235)
        # Adding element type (line 473)
        # Getting the type of 'self' (line 474)
        self_156236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 17), 'self')
        # Obtaining the member 'shape' of a type (line 474)
        shape_156237 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 474, 17), self_156236, 'shape')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 473, 17), tuple_156234, shape_156237)
        # Adding element type (line 473)
        # Getting the type of 'self' (line 475)
        self_156238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 17), 'self')
        # Obtaining the member 'dtype' of a type (line 475)
        dtype_156239 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 475, 17), self_156238, 'dtype')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 473, 17), tuple_156234, dtype_156239)
        # Adding element type (line 473)
        # Getting the type of 'self' (line 476)
        self_156240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 17), 'self')
        # Obtaining the member 'flags' of a type (line 476)
        flags_156241 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 476, 17), self_156240, 'flags')
        # Obtaining the member 'fnc' of a type (line 476)
        fnc_156242 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 476, 17), flags_156241, 'fnc')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 473, 17), tuple_156234, fnc_156242)
        # Adding element type (line 473)
        
        # Call to tobytes(...): (line 477)
        # Processing the call keyword arguments (line 477)
        kwargs_156246 = {}
        # Getting the type of 'self' (line 477)
        self_156243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 17), 'self', False)
        # Obtaining the member '_data' of a type (line 477)
        _data_156244 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 477, 17), self_156243, '_data')
        # Obtaining the member 'tobytes' of a type (line 477)
        tobytes_156245 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 477, 17), _data_156244, 'tobytes')
        # Calling tobytes(args, kwargs) (line 477)
        tobytes_call_result_156247 = invoke(stypy.reporting.localization.Localization(__file__, 477, 17), tobytes_156245, *[], **kwargs_156246)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 473, 17), tuple_156234, tobytes_call_result_156247)
        # Adding element type (line 473)
        
        # Call to tobytes(...): (line 478)
        # Processing the call keyword arguments (line 478)
        kwargs_156251 = {}
        # Getting the type of 'self' (line 478)
        self_156248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 17), 'self', False)
        # Obtaining the member '_mask' of a type (line 478)
        _mask_156249 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 478, 17), self_156248, '_mask')
        # Obtaining the member 'tobytes' of a type (line 478)
        tobytes_156250 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 478, 17), _mask_156249, 'tobytes')
        # Calling tobytes(args, kwargs) (line 478)
        tobytes_call_result_156252 = invoke(stypy.reporting.localization.Localization(__file__, 478, 17), tobytes_156250, *[], **kwargs_156251)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 473, 17), tuple_156234, tobytes_call_result_156252)
        # Adding element type (line 473)
        # Getting the type of 'self' (line 479)
        self_156253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 17), 'self')
        # Obtaining the member '_fill_value' of a type (line 479)
        _fill_value_156254 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 479, 17), self_156253, '_fill_value')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 473, 17), tuple_156234, _fill_value_156254)
        
        # Assigning a type to the variable 'state' (line 473)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 473, 8), 'state', tuple_156234)
        # Getting the type of 'state' (line 481)
        state_156255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 15), 'state')
        # Assigning a type to the variable 'stypy_return_type' (line 481)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 481, 8), 'stypy_return_type', state_156255)
        
        # ################# End of '__getstate__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__getstate__' in the type store
        # Getting the type of 'stypy_return_type' (line 467)
        stypy_return_type_156256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_156256)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__getstate__'
        return stypy_return_type_156256


    @norecursion
    def __setstate__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__setstate__'
        module_type_store = module_type_store.open_function_context('__setstate__', 483, 4, False)
        # Assigning a type to the variable 'self' (line 484)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 484, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MaskedRecords.__setstate__.__dict__.__setitem__('stypy_localization', localization)
        MaskedRecords.__setstate__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MaskedRecords.__setstate__.__dict__.__setitem__('stypy_type_store', module_type_store)
        MaskedRecords.__setstate__.__dict__.__setitem__('stypy_function_name', 'MaskedRecords.__setstate__')
        MaskedRecords.__setstate__.__dict__.__setitem__('stypy_param_names_list', ['state'])
        MaskedRecords.__setstate__.__dict__.__setitem__('stypy_varargs_param_name', None)
        MaskedRecords.__setstate__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MaskedRecords.__setstate__.__dict__.__setitem__('stypy_call_defaults', defaults)
        MaskedRecords.__setstate__.__dict__.__setitem__('stypy_call_varargs', varargs)
        MaskedRecords.__setstate__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MaskedRecords.__setstate__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MaskedRecords.__setstate__', ['state'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__setstate__', localization, ['state'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__setstate__(...)' code ##################

        str_156257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 496, (-1)), 'str', '\n        Restore the internal state of the masked array.\n\n        This is for pickling.  ``state`` is typically the output of the\n        ``__getstate__`` output, and is a 5-tuple:\n\n        - class name\n        - a tuple giving the shape of the data\n        - a typecode for the data\n        - a binary string for the data\n        - a binary string for the mask.\n\n        ')
        
        # Assigning a Name to a Tuple (line 497):
        
        # Assigning a Subscript to a Name (line 497):
        
        # Obtaining the type of the subscript
        int_156258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 497, 8), 'int')
        # Getting the type of 'state' (line 497)
        state_156259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 46), 'state')
        # Obtaining the member '__getitem__' of a type (line 497)
        getitem___156260 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 497, 8), state_156259, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 497)
        subscript_call_result_156261 = invoke(stypy.reporting.localization.Localization(__file__, 497, 8), getitem___156260, int_156258)
        
        # Assigning a type to the variable 'tuple_var_assignment_154977' (line 497)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 497, 8), 'tuple_var_assignment_154977', subscript_call_result_156261)
        
        # Assigning a Subscript to a Name (line 497):
        
        # Obtaining the type of the subscript
        int_156262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 497, 8), 'int')
        # Getting the type of 'state' (line 497)
        state_156263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 46), 'state')
        # Obtaining the member '__getitem__' of a type (line 497)
        getitem___156264 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 497, 8), state_156263, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 497)
        subscript_call_result_156265 = invoke(stypy.reporting.localization.Localization(__file__, 497, 8), getitem___156264, int_156262)
        
        # Assigning a type to the variable 'tuple_var_assignment_154978' (line 497)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 497, 8), 'tuple_var_assignment_154978', subscript_call_result_156265)
        
        # Assigning a Subscript to a Name (line 497):
        
        # Obtaining the type of the subscript
        int_156266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 497, 8), 'int')
        # Getting the type of 'state' (line 497)
        state_156267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 46), 'state')
        # Obtaining the member '__getitem__' of a type (line 497)
        getitem___156268 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 497, 8), state_156267, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 497)
        subscript_call_result_156269 = invoke(stypy.reporting.localization.Localization(__file__, 497, 8), getitem___156268, int_156266)
        
        # Assigning a type to the variable 'tuple_var_assignment_154979' (line 497)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 497, 8), 'tuple_var_assignment_154979', subscript_call_result_156269)
        
        # Assigning a Subscript to a Name (line 497):
        
        # Obtaining the type of the subscript
        int_156270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 497, 8), 'int')
        # Getting the type of 'state' (line 497)
        state_156271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 46), 'state')
        # Obtaining the member '__getitem__' of a type (line 497)
        getitem___156272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 497, 8), state_156271, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 497)
        subscript_call_result_156273 = invoke(stypy.reporting.localization.Localization(__file__, 497, 8), getitem___156272, int_156270)
        
        # Assigning a type to the variable 'tuple_var_assignment_154980' (line 497)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 497, 8), 'tuple_var_assignment_154980', subscript_call_result_156273)
        
        # Assigning a Subscript to a Name (line 497):
        
        # Obtaining the type of the subscript
        int_156274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 497, 8), 'int')
        # Getting the type of 'state' (line 497)
        state_156275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 46), 'state')
        # Obtaining the member '__getitem__' of a type (line 497)
        getitem___156276 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 497, 8), state_156275, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 497)
        subscript_call_result_156277 = invoke(stypy.reporting.localization.Localization(__file__, 497, 8), getitem___156276, int_156274)
        
        # Assigning a type to the variable 'tuple_var_assignment_154981' (line 497)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 497, 8), 'tuple_var_assignment_154981', subscript_call_result_156277)
        
        # Assigning a Subscript to a Name (line 497):
        
        # Obtaining the type of the subscript
        int_156278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 497, 8), 'int')
        # Getting the type of 'state' (line 497)
        state_156279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 46), 'state')
        # Obtaining the member '__getitem__' of a type (line 497)
        getitem___156280 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 497, 8), state_156279, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 497)
        subscript_call_result_156281 = invoke(stypy.reporting.localization.Localization(__file__, 497, 8), getitem___156280, int_156278)
        
        # Assigning a type to the variable 'tuple_var_assignment_154982' (line 497)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 497, 8), 'tuple_var_assignment_154982', subscript_call_result_156281)
        
        # Assigning a Subscript to a Name (line 497):
        
        # Obtaining the type of the subscript
        int_156282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 497, 8), 'int')
        # Getting the type of 'state' (line 497)
        state_156283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 46), 'state')
        # Obtaining the member '__getitem__' of a type (line 497)
        getitem___156284 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 497, 8), state_156283, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 497)
        subscript_call_result_156285 = invoke(stypy.reporting.localization.Localization(__file__, 497, 8), getitem___156284, int_156282)
        
        # Assigning a type to the variable 'tuple_var_assignment_154983' (line 497)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 497, 8), 'tuple_var_assignment_154983', subscript_call_result_156285)
        
        # Assigning a Name to a Name (line 497):
        # Getting the type of 'tuple_var_assignment_154977' (line 497)
        tuple_var_assignment_154977_156286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 8), 'tuple_var_assignment_154977')
        # Assigning a type to the variable 'ver' (line 497)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 497, 9), 'ver', tuple_var_assignment_154977_156286)
        
        # Assigning a Name to a Name (line 497):
        # Getting the type of 'tuple_var_assignment_154978' (line 497)
        tuple_var_assignment_154978_156287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 8), 'tuple_var_assignment_154978')
        # Assigning a type to the variable 'shp' (line 497)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 497, 14), 'shp', tuple_var_assignment_154978_156287)
        
        # Assigning a Name to a Name (line 497):
        # Getting the type of 'tuple_var_assignment_154979' (line 497)
        tuple_var_assignment_154979_156288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 8), 'tuple_var_assignment_154979')
        # Assigning a type to the variable 'typ' (line 497)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 497, 19), 'typ', tuple_var_assignment_154979_156288)
        
        # Assigning a Name to a Name (line 497):
        # Getting the type of 'tuple_var_assignment_154980' (line 497)
        tuple_var_assignment_154980_156289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 8), 'tuple_var_assignment_154980')
        # Assigning a type to the variable 'isf' (line 497)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 497, 24), 'isf', tuple_var_assignment_154980_156289)
        
        # Assigning a Name to a Name (line 497):
        # Getting the type of 'tuple_var_assignment_154981' (line 497)
        tuple_var_assignment_154981_156290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 8), 'tuple_var_assignment_154981')
        # Assigning a type to the variable 'raw' (line 497)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 497, 29), 'raw', tuple_var_assignment_154981_156290)
        
        # Assigning a Name to a Name (line 497):
        # Getting the type of 'tuple_var_assignment_154982' (line 497)
        tuple_var_assignment_154982_156291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 8), 'tuple_var_assignment_154982')
        # Assigning a type to the variable 'msk' (line 497)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 497, 34), 'msk', tuple_var_assignment_154982_156291)
        
        # Assigning a Name to a Name (line 497):
        # Getting the type of 'tuple_var_assignment_154983' (line 497)
        tuple_var_assignment_154983_156292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 8), 'tuple_var_assignment_154983')
        # Assigning a type to the variable 'flv' (line 497)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 497, 39), 'flv', tuple_var_assignment_154983_156292)
        
        # Call to __setstate__(...): (line 498)
        # Processing the call arguments (line 498)
        # Getting the type of 'self' (line 498)
        self_156295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 29), 'self', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 498)
        tuple_156296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 498, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 498)
        # Adding element type (line 498)
        # Getting the type of 'shp' (line 498)
        shp_156297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 36), 'shp', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 498, 36), tuple_156296, shp_156297)
        # Adding element type (line 498)
        # Getting the type of 'typ' (line 498)
        typ_156298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 41), 'typ', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 498, 36), tuple_156296, typ_156298)
        # Adding element type (line 498)
        # Getting the type of 'isf' (line 498)
        isf_156299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 46), 'isf', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 498, 36), tuple_156296, isf_156299)
        # Adding element type (line 498)
        # Getting the type of 'raw' (line 498)
        raw_156300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 51), 'raw', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 498, 36), tuple_156296, raw_156300)
        
        # Processing the call keyword arguments (line 498)
        kwargs_156301 = {}
        # Getting the type of 'ndarray' (line 498)
        ndarray_156293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 8), 'ndarray', False)
        # Obtaining the member '__setstate__' of a type (line 498)
        setstate___156294 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 498, 8), ndarray_156293, '__setstate__')
        # Calling __setstate__(args, kwargs) (line 498)
        setstate___call_result_156302 = invoke(stypy.reporting.localization.Localization(__file__, 498, 8), setstate___156294, *[self_156295, tuple_156296], **kwargs_156301)
        
        
        # Assigning a Call to a Name (line 499):
        
        # Assigning a Call to a Name (line 499):
        
        # Call to dtype(...): (line 499)
        # Processing the call arguments (line 499)
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'self' (line 499)
        self_156307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 49), 'self', False)
        # Obtaining the member 'dtype' of a type (line 499)
        dtype_156308 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 499, 49), self_156307, 'dtype')
        # Obtaining the member 'descr' of a type (line 499)
        descr_156309 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 499, 49), dtype_156308, 'descr')
        comprehension_156310 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 499, 24), descr_156309)
        # Assigning a type to the variable 'k' (line 499)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 499, 24), 'k', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 499, 24), comprehension_156310))
        # Assigning a type to the variable '_' (line 499)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 499, 24), '_', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 499, 24), comprehension_156310))
        
        # Obtaining an instance of the builtin type 'tuple' (line 499)
        tuple_156304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 499, 25), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 499)
        # Adding element type (line 499)
        # Getting the type of 'k' (line 499)
        k_156305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 25), 'k', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 499, 25), tuple_156304, k_156305)
        # Adding element type (line 499)
        # Getting the type of 'bool_' (line 499)
        bool__156306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 28), 'bool_', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 499, 25), tuple_156304, bool__156306)
        
        list_156311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 499, 24), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 499, 24), list_156311, tuple_156304)
        # Processing the call keyword arguments (line 499)
        kwargs_156312 = {}
        # Getting the type of 'dtype' (line 499)
        dtype_156303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 17), 'dtype', False)
        # Calling dtype(args, kwargs) (line 499)
        dtype_call_result_156313 = invoke(stypy.reporting.localization.Localization(__file__, 499, 17), dtype_156303, *[list_156311], **kwargs_156312)
        
        # Assigning a type to the variable 'mdtype' (line 499)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 499, 8), 'mdtype', dtype_call_result_156313)
        
        # Call to __setstate__(...): (line 500)
        # Processing the call arguments (line 500)
        
        # Obtaining an instance of the builtin type 'tuple' (line 500)
        tuple_156320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 500, 45), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 500)
        # Adding element type (line 500)
        # Getting the type of 'shp' (line 500)
        shp_156321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 45), 'shp', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 500, 45), tuple_156320, shp_156321)
        # Adding element type (line 500)
        # Getting the type of 'mdtype' (line 500)
        mdtype_156322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 50), 'mdtype', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 500, 45), tuple_156320, mdtype_156322)
        # Adding element type (line 500)
        # Getting the type of 'isf' (line 500)
        isf_156323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 58), 'isf', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 500, 45), tuple_156320, isf_156323)
        # Adding element type (line 500)
        # Getting the type of 'msk' (line 500)
        msk_156324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 63), 'msk', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 500, 45), tuple_156320, msk_156324)
        
        # Processing the call keyword arguments (line 500)
        kwargs_156325 = {}
        
        # Obtaining the type of the subscript
        str_156314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 500, 22), 'str', '_mask')
        # Getting the type of 'self' (line 500)
        self_156315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 8), 'self', False)
        # Obtaining the member '__dict__' of a type (line 500)
        dict___156316 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 500, 8), self_156315, '__dict__')
        # Obtaining the member '__getitem__' of a type (line 500)
        getitem___156317 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 500, 8), dict___156316, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 500)
        subscript_call_result_156318 = invoke(stypy.reporting.localization.Localization(__file__, 500, 8), getitem___156317, str_156314)
        
        # Obtaining the member '__setstate__' of a type (line 500)
        setstate___156319 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 500, 8), subscript_call_result_156318, '__setstate__')
        # Calling __setstate__(args, kwargs) (line 500)
        setstate___call_result_156326 = invoke(stypy.reporting.localization.Localization(__file__, 500, 8), setstate___156319, *[tuple_156320], **kwargs_156325)
        
        
        # Assigning a Name to a Attribute (line 501):
        
        # Assigning a Name to a Attribute (line 501):
        # Getting the type of 'flv' (line 501)
        flv_156327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 26), 'flv')
        # Getting the type of 'self' (line 501)
        self_156328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 8), 'self')
        # Setting the type of the member 'fill_value' of a type (line 501)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 501, 8), self_156328, 'fill_value', flv_156327)
        
        # ################# End of '__setstate__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__setstate__' in the type store
        # Getting the type of 'stypy_return_type' (line 483)
        stypy_return_type_156329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_156329)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__setstate__'
        return stypy_return_type_156329


    @norecursion
    def __reduce__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__reduce__'
        module_type_store = module_type_store.open_function_context('__reduce__', 503, 4, False)
        # Assigning a type to the variable 'self' (line 504)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 504, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MaskedRecords.__reduce__.__dict__.__setitem__('stypy_localization', localization)
        MaskedRecords.__reduce__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MaskedRecords.__reduce__.__dict__.__setitem__('stypy_type_store', module_type_store)
        MaskedRecords.__reduce__.__dict__.__setitem__('stypy_function_name', 'MaskedRecords.__reduce__')
        MaskedRecords.__reduce__.__dict__.__setitem__('stypy_param_names_list', [])
        MaskedRecords.__reduce__.__dict__.__setitem__('stypy_varargs_param_name', None)
        MaskedRecords.__reduce__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MaskedRecords.__reduce__.__dict__.__setitem__('stypy_call_defaults', defaults)
        MaskedRecords.__reduce__.__dict__.__setitem__('stypy_call_varargs', varargs)
        MaskedRecords.__reduce__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MaskedRecords.__reduce__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MaskedRecords.__reduce__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__reduce__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__reduce__(...)' code ##################

        str_156330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 507, (-1)), 'str', '\n        Return a 3-tuple for pickling a MaskedArray.\n\n        ')
        
        # Obtaining an instance of the builtin type 'tuple' (line 508)
        tuple_156331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 508, 16), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 508)
        # Adding element type (line 508)
        # Getting the type of '_mrreconstruct' (line 508)
        _mrreconstruct_156332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 16), '_mrreconstruct')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 508, 16), tuple_156331, _mrreconstruct_156332)
        # Adding element type (line 508)
        
        # Obtaining an instance of the builtin type 'tuple' (line 509)
        tuple_156333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 509, 17), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 509)
        # Adding element type (line 509)
        # Getting the type of 'self' (line 509)
        self_156334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 17), 'self')
        # Obtaining the member '__class__' of a type (line 509)
        class___156335 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 509, 17), self_156334, '__class__')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 509, 17), tuple_156333, class___156335)
        # Adding element type (line 509)
        # Getting the type of 'self' (line 509)
        self_156336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 33), 'self')
        # Obtaining the member '_baseclass' of a type (line 509)
        _baseclass_156337 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 509, 33), self_156336, '_baseclass')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 509, 17), tuple_156333, _baseclass_156337)
        # Adding element type (line 509)
        
        # Obtaining an instance of the builtin type 'tuple' (line 509)
        tuple_156338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 509, 51), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 509)
        # Adding element type (line 509)
        int_156339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 509, 51), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 509, 51), tuple_156338, int_156339)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 509, 17), tuple_156333, tuple_156338)
        # Adding element type (line 509)
        str_156340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 509, 56), 'str', 'b')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 509, 17), tuple_156333, str_156340)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 508, 16), tuple_156331, tuple_156333)
        # Adding element type (line 508)
        
        # Call to __getstate__(...): (line 510)
        # Processing the call keyword arguments (line 510)
        kwargs_156343 = {}
        # Getting the type of 'self' (line 510)
        self_156341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 16), 'self', False)
        # Obtaining the member '__getstate__' of a type (line 510)
        getstate___156342 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 510, 16), self_156341, '__getstate__')
        # Calling __getstate__(args, kwargs) (line 510)
        getstate___call_result_156344 = invoke(stypy.reporting.localization.Localization(__file__, 510, 16), getstate___156342, *[], **kwargs_156343)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 508, 16), tuple_156331, getstate___call_result_156344)
        
        # Assigning a type to the variable 'stypy_return_type' (line 508)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 508, 8), 'stypy_return_type', tuple_156331)
        
        # ################# End of '__reduce__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__reduce__' in the type store
        # Getting the type of 'stypy_return_type' (line 503)
        stypy_return_type_156345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_156345)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__reduce__'
        return stypy_return_type_156345


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 110, 0, False)
        # Assigning a type to the variable 'self' (line 111)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MaskedRecords.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'MaskedRecords' (line 110)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 0), 'MaskedRecords', MaskedRecords)

# Assigning a Call to a Name (line 196):

# Call to property(...): (line 196)
# Processing the call keyword arguments (line 196)
# Getting the type of '_getdata' (line 196)
_getdata_156347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 26), '_getdata', False)
keyword_156348 = _getdata_156347
kwargs_156349 = {'fget': keyword_156348}
# Getting the type of 'property' (line 196)
property_156346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 12), 'property', False)
# Calling property(args, kwargs) (line 196)
property_call_result_156350 = invoke(stypy.reporting.localization.Localization(__file__, 196, 12), property_156346, *[], **kwargs_156349)

# Getting the type of 'MaskedRecords'
MaskedRecords_156351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'MaskedRecords')
# Setting the type of the member '_data' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), MaskedRecords_156351, '_data', property_call_result_156350)

# Assigning a Call to a Name (line 205):

# Call to property(...): (line 205)
# Processing the call keyword arguments (line 205)
# Getting the type of '_getfieldmask' (line 205)
_getfieldmask_156353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 31), '_getfieldmask', False)
keyword_156354 = _getfieldmask_156353
kwargs_156355 = {'fget': keyword_156354}
# Getting the type of 'property' (line 205)
property_156352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 17), 'property', False)
# Calling property(args, kwargs) (line 205)
property_call_result_156356 = invoke(stypy.reporting.localization.Localization(__file__, 205, 17), property_156352, *[], **kwargs_156355)

# Getting the type of 'MaskedRecords'
MaskedRecords_156357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'MaskedRecords')
# Setting the type of the member '_fieldmask' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), MaskedRecords_156357, '_fieldmask', property_call_result_156356)

@norecursion
def _mrreconstruct(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_mrreconstruct'
    module_type_store = module_type_store.open_function_context('_mrreconstruct', 512, 0, False)
    
    # Passed parameters checking function
    _mrreconstruct.stypy_localization = localization
    _mrreconstruct.stypy_type_of_self = None
    _mrreconstruct.stypy_type_store = module_type_store
    _mrreconstruct.stypy_function_name = '_mrreconstruct'
    _mrreconstruct.stypy_param_names_list = ['subtype', 'baseclass', 'baseshape', 'basetype']
    _mrreconstruct.stypy_varargs_param_name = None
    _mrreconstruct.stypy_kwargs_param_name = None
    _mrreconstruct.stypy_call_defaults = defaults
    _mrreconstruct.stypy_call_varargs = varargs
    _mrreconstruct.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_mrreconstruct', ['subtype', 'baseclass', 'baseshape', 'basetype'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_mrreconstruct', localization, ['subtype', 'baseclass', 'baseshape', 'basetype'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_mrreconstruct(...)' code ##################

    str_156358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 516, (-1)), 'str', '\n    Build a new MaskedArray from the information stored in a pickle.\n\n    ')
    
    # Assigning a Call to a Name (line 517):
    
    # Assigning a Call to a Name (line 517):
    
    # Call to view(...): (line 517)
    # Processing the call arguments (line 517)
    # Getting the type of 'subtype' (line 517)
    subtype_156367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 65), 'subtype', False)
    # Processing the call keyword arguments (line 517)
    kwargs_156368 = {}
    
    # Call to __new__(...): (line 517)
    # Processing the call arguments (line 517)
    # Getting the type of 'baseclass' (line 517)
    baseclass_156361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 28), 'baseclass', False)
    # Getting the type of 'baseshape' (line 517)
    baseshape_156362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 39), 'baseshape', False)
    # Getting the type of 'basetype' (line 517)
    basetype_156363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 50), 'basetype', False)
    # Processing the call keyword arguments (line 517)
    kwargs_156364 = {}
    # Getting the type of 'ndarray' (line 517)
    ndarray_156359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 12), 'ndarray', False)
    # Obtaining the member '__new__' of a type (line 517)
    new___156360 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 517, 12), ndarray_156359, '__new__')
    # Calling __new__(args, kwargs) (line 517)
    new___call_result_156365 = invoke(stypy.reporting.localization.Localization(__file__, 517, 12), new___156360, *[baseclass_156361, baseshape_156362, basetype_156363], **kwargs_156364)
    
    # Obtaining the member 'view' of a type (line 517)
    view_156366 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 517, 12), new___call_result_156365, 'view')
    # Calling view(args, kwargs) (line 517)
    view_call_result_156369 = invoke(stypy.reporting.localization.Localization(__file__, 517, 12), view_156366, *[subtype_156367], **kwargs_156368)
    
    # Assigning a type to the variable '_data' (line 517)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 517, 4), '_data', view_call_result_156369)
    
    # Assigning a Call to a Name (line 518):
    
    # Assigning a Call to a Name (line 518):
    
    # Call to __new__(...): (line 518)
    # Processing the call arguments (line 518)
    # Getting the type of 'ndarray' (line 518)
    ndarray_156372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 28), 'ndarray', False)
    # Getting the type of 'baseshape' (line 518)
    baseshape_156373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 37), 'baseshape', False)
    str_156374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 518, 48), 'str', 'b1')
    # Processing the call keyword arguments (line 518)
    kwargs_156375 = {}
    # Getting the type of 'ndarray' (line 518)
    ndarray_156370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 12), 'ndarray', False)
    # Obtaining the member '__new__' of a type (line 518)
    new___156371 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 518, 12), ndarray_156370, '__new__')
    # Calling __new__(args, kwargs) (line 518)
    new___call_result_156376 = invoke(stypy.reporting.localization.Localization(__file__, 518, 12), new___156371, *[ndarray_156372, baseshape_156373, str_156374], **kwargs_156375)
    
    # Assigning a type to the variable '_mask' (line 518)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 518, 4), '_mask', new___call_result_156376)
    
    # Call to __new__(...): (line 519)
    # Processing the call arguments (line 519)
    # Getting the type of 'subtype' (line 519)
    subtype_156379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 27), 'subtype', False)
    # Getting the type of '_data' (line 519)
    _data_156380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 36), '_data', False)
    # Processing the call keyword arguments (line 519)
    # Getting the type of '_mask' (line 519)
    _mask_156381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 48), '_mask', False)
    keyword_156382 = _mask_156381
    # Getting the type of 'basetype' (line 519)
    basetype_156383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 61), 'basetype', False)
    keyword_156384 = basetype_156383
    kwargs_156385 = {'dtype': keyword_156384, 'mask': keyword_156382}
    # Getting the type of 'subtype' (line 519)
    subtype_156377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 11), 'subtype', False)
    # Obtaining the member '__new__' of a type (line 519)
    new___156378 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 519, 11), subtype_156377, '__new__')
    # Calling __new__(args, kwargs) (line 519)
    new___call_result_156386 = invoke(stypy.reporting.localization.Localization(__file__, 519, 11), new___156378, *[subtype_156379, _data_156380], **kwargs_156385)
    
    # Assigning a type to the variable 'stypy_return_type' (line 519)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 519, 4), 'stypy_return_type', new___call_result_156386)
    
    # ################# End of '_mrreconstruct(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_mrreconstruct' in the type store
    # Getting the type of 'stypy_return_type' (line 512)
    stypy_return_type_156387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_156387)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_mrreconstruct'
    return stypy_return_type_156387

# Assigning a type to the variable '_mrreconstruct' (line 512)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 512, 0), '_mrreconstruct', _mrreconstruct)

# Assigning a Name to a Name (line 521):

# Assigning a Name to a Name (line 521):
# Getting the type of 'MaskedRecords' (line 521)
MaskedRecords_156388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 12), 'MaskedRecords')
# Assigning a type to the variable 'mrecarray' (line 521)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 521, 0), 'mrecarray', MaskedRecords_156388)

@norecursion
def fromarrays(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 529)
    None_156389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 32), 'None')
    # Getting the type of 'None' (line 529)
    None_156390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 44), 'None')
    # Getting the type of 'None' (line 529)
    None_156391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 58), 'None')
    # Getting the type of 'None' (line 530)
    None_156392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 21), 'None')
    # Getting the type of 'None' (line 530)
    None_156393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 34), 'None')
    # Getting the type of 'False' (line 530)
    False_156394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 48), 'False')
    # Getting the type of 'None' (line 530)
    None_156395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 65), 'None')
    # Getting the type of 'None' (line 531)
    None_156396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 26), 'None')
    defaults = [None_156389, None_156390, None_156391, None_156392, None_156393, False_156394, None_156395, None_156396]
    # Create a new context for function 'fromarrays'
    module_type_store = module_type_store.open_function_context('fromarrays', 529, 0, False)
    
    # Passed parameters checking function
    fromarrays.stypy_localization = localization
    fromarrays.stypy_type_of_self = None
    fromarrays.stypy_type_store = module_type_store
    fromarrays.stypy_function_name = 'fromarrays'
    fromarrays.stypy_param_names_list = ['arraylist', 'dtype', 'shape', 'formats', 'names', 'titles', 'aligned', 'byteorder', 'fill_value']
    fromarrays.stypy_varargs_param_name = None
    fromarrays.stypy_kwargs_param_name = None
    fromarrays.stypy_call_defaults = defaults
    fromarrays.stypy_call_varargs = varargs
    fromarrays.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'fromarrays', ['arraylist', 'dtype', 'shape', 'formats', 'names', 'titles', 'aligned', 'byteorder', 'fill_value'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'fromarrays', localization, ['arraylist', 'dtype', 'shape', 'formats', 'names', 'titles', 'aligned', 'byteorder', 'fill_value'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'fromarrays(...)' code ##################

    str_156397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 559, (-1)), 'str', '\n    Creates a mrecarray from a (flat) list of masked arrays.\n\n    Parameters\n    ----------\n    arraylist : sequence\n        A list of (masked) arrays. Each element of the sequence is first converted\n        to a masked array if needed. If a 2D array is passed as argument, it is\n        processed line by line\n    dtype : {None, dtype}, optional\n        Data type descriptor.\n    shape : {None, integer}, optional\n        Number of records. If None, shape is defined from the shape of the\n        first array in the list.\n    formats : {None, sequence}, optional\n        Sequence of formats for each individual field. If None, the formats will\n        be autodetected by inspecting the fields and selecting the highest dtype\n        possible.\n    names : {None, sequence}, optional\n        Sequence of the names of each field.\n    fill_value : {None, sequence}, optional\n        Sequence of data to be used as filling values.\n\n    Notes\n    -----\n    Lists of tuples should be preferred over lists of lists for faster processing.\n\n    ')
    
    # Assigning a ListComp to a Name (line 560):
    
    # Assigning a ListComp to a Name (line 560):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'arraylist' (line 560)
    arraylist_156402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 36), 'arraylist')
    comprehension_156403 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 560, 16), arraylist_156402)
    # Assigning a type to the variable 'x' (line 560)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 560, 16), 'x', comprehension_156403)
    
    # Call to getdata(...): (line 560)
    # Processing the call arguments (line 560)
    # Getting the type of 'x' (line 560)
    x_156399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 24), 'x', False)
    # Processing the call keyword arguments (line 560)
    kwargs_156400 = {}
    # Getting the type of 'getdata' (line 560)
    getdata_156398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 16), 'getdata', False)
    # Calling getdata(args, kwargs) (line 560)
    getdata_call_result_156401 = invoke(stypy.reporting.localization.Localization(__file__, 560, 16), getdata_156398, *[x_156399], **kwargs_156400)
    
    list_156404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 560, 16), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 560, 16), list_156404, getdata_call_result_156401)
    # Assigning a type to the variable 'datalist' (line 560)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 560, 4), 'datalist', list_156404)
    
    # Assigning a ListComp to a Name (line 561):
    
    # Assigning a ListComp to a Name (line 561):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'arraylist' (line 561)
    arraylist_156413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 56), 'arraylist')
    comprehension_156414 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 561, 16), arraylist_156413)
    # Assigning a type to the variable 'x' (line 561)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 561, 16), 'x', comprehension_156414)
    
    # Call to atleast_1d(...): (line 561)
    # Processing the call arguments (line 561)
    
    # Call to getmaskarray(...): (line 561)
    # Processing the call arguments (line 561)
    # Getting the type of 'x' (line 561)
    x_156408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 43), 'x', False)
    # Processing the call keyword arguments (line 561)
    kwargs_156409 = {}
    # Getting the type of 'getmaskarray' (line 561)
    getmaskarray_156407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 30), 'getmaskarray', False)
    # Calling getmaskarray(args, kwargs) (line 561)
    getmaskarray_call_result_156410 = invoke(stypy.reporting.localization.Localization(__file__, 561, 30), getmaskarray_156407, *[x_156408], **kwargs_156409)
    
    # Processing the call keyword arguments (line 561)
    kwargs_156411 = {}
    # Getting the type of 'np' (line 561)
    np_156405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 16), 'np', False)
    # Obtaining the member 'atleast_1d' of a type (line 561)
    atleast_1d_156406 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 561, 16), np_156405, 'atleast_1d')
    # Calling atleast_1d(args, kwargs) (line 561)
    atleast_1d_call_result_156412 = invoke(stypy.reporting.localization.Localization(__file__, 561, 16), atleast_1d_156406, *[getmaskarray_call_result_156410], **kwargs_156411)
    
    list_156415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 561, 16), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 561, 16), list_156415, atleast_1d_call_result_156412)
    # Assigning a type to the variable 'masklist' (line 561)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 561, 4), 'masklist', list_156415)
    
    # Assigning a Call to a Name (line 562):
    
    # Assigning a Call to a Name (line 562):
    
    # Call to view(...): (line 562)
    # Processing the call arguments (line 562)
    # Getting the type of 'mrecarray' (line 565)
    mrecarray_156435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 53), 'mrecarray', False)
    # Processing the call keyword arguments (line 562)
    kwargs_156436 = {}
    
    # Call to recfromarrays(...): (line 562)
    # Processing the call arguments (line 562)
    # Getting the type of 'datalist' (line 562)
    datalist_156417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 27), 'datalist', False)
    # Processing the call keyword arguments (line 562)
    # Getting the type of 'dtype' (line 563)
    dtype_156418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 33), 'dtype', False)
    keyword_156419 = dtype_156418
    # Getting the type of 'shape' (line 563)
    shape_156420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 46), 'shape', False)
    keyword_156421 = shape_156420
    # Getting the type of 'formats' (line 563)
    formats_156422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 61), 'formats', False)
    keyword_156423 = formats_156422
    # Getting the type of 'names' (line 564)
    names_156424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 33), 'names', False)
    keyword_156425 = names_156424
    # Getting the type of 'titles' (line 564)
    titles_156426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 47), 'titles', False)
    keyword_156427 = titles_156426
    # Getting the type of 'aligned' (line 564)
    aligned_156428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 63), 'aligned', False)
    keyword_156429 = aligned_156428
    # Getting the type of 'byteorder' (line 565)
    byteorder_156430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 37), 'byteorder', False)
    keyword_156431 = byteorder_156430
    kwargs_156432 = {'dtype': keyword_156419, 'shape': keyword_156421, 'titles': keyword_156427, 'names': keyword_156425, 'formats': keyword_156423, 'aligned': keyword_156429, 'byteorder': keyword_156431}
    # Getting the type of 'recfromarrays' (line 562)
    recfromarrays_156416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 13), 'recfromarrays', False)
    # Calling recfromarrays(args, kwargs) (line 562)
    recfromarrays_call_result_156433 = invoke(stypy.reporting.localization.Localization(__file__, 562, 13), recfromarrays_156416, *[datalist_156417], **kwargs_156432)
    
    # Obtaining the member 'view' of a type (line 562)
    view_156434 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 562, 13), recfromarrays_call_result_156433, 'view')
    # Calling view(args, kwargs) (line 562)
    view_call_result_156437 = invoke(stypy.reporting.localization.Localization(__file__, 562, 13), view_156434, *[mrecarray_156435], **kwargs_156436)
    
    # Assigning a type to the variable '_array' (line 562)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 562, 4), '_array', view_call_result_156437)
    
    # Assigning a Call to a Attribute (line 566):
    
    # Assigning a Call to a Attribute (line 566):
    
    # Call to list(...): (line 566)
    # Processing the call arguments (line 566)
    
    # Call to zip(...): (line 566)
    # Getting the type of 'masklist' (line 566)
    masklist_156440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 34), 'masklist', False)
    # Processing the call keyword arguments (line 566)
    kwargs_156441 = {}
    # Getting the type of 'zip' (line 566)
    zip_156439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 29), 'zip', False)
    # Calling zip(args, kwargs) (line 566)
    zip_call_result_156442 = invoke(stypy.reporting.localization.Localization(__file__, 566, 29), zip_156439, *[masklist_156440], **kwargs_156441)
    
    # Processing the call keyword arguments (line 566)
    kwargs_156443 = {}
    # Getting the type of 'list' (line 566)
    list_156438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 24), 'list', False)
    # Calling list(args, kwargs) (line 566)
    list_call_result_156444 = invoke(stypy.reporting.localization.Localization(__file__, 566, 24), list_156438, *[zip_call_result_156442], **kwargs_156443)
    
    # Getting the type of '_array' (line 566)
    _array_156445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 4), '_array')
    # Obtaining the member '_mask' of a type (line 566)
    _mask_156446 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 566, 4), _array_156445, '_mask')
    # Setting the type of the member 'flat' of a type (line 566)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 566, 4), _mask_156446, 'flat', list_call_result_156444)
    
    # Type idiom detected: calculating its left and rigth part (line 567)
    # Getting the type of 'fill_value' (line 567)
    fill_value_156447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 4), 'fill_value')
    # Getting the type of 'None' (line 567)
    None_156448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 25), 'None')
    
    (may_be_156449, more_types_in_union_156450) = may_not_be_none(fill_value_156447, None_156448)

    if may_be_156449:

        if more_types_in_union_156450:
            # Runtime conditional SSA (line 567)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Name to a Attribute (line 568):
        
        # Assigning a Name to a Attribute (line 568):
        # Getting the type of 'fill_value' (line 568)
        fill_value_156451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 28), 'fill_value')
        # Getting the type of '_array' (line 568)
        _array_156452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 8), '_array')
        # Setting the type of the member 'fill_value' of a type (line 568)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 568, 8), _array_156452, 'fill_value', fill_value_156451)

        if more_types_in_union_156450:
            # SSA join for if statement (line 567)
            module_type_store = module_type_store.join_ssa_context()


    
    # Getting the type of '_array' (line 569)
    _array_156453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 11), '_array')
    # Assigning a type to the variable 'stypy_return_type' (line 569)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 569, 4), 'stypy_return_type', _array_156453)
    
    # ################# End of 'fromarrays(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'fromarrays' in the type store
    # Getting the type of 'stypy_return_type' (line 529)
    stypy_return_type_156454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_156454)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'fromarrays'
    return stypy_return_type_156454

# Assigning a type to the variable 'fromarrays' (line 529)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 529, 0), 'fromarrays', fromarrays)

@norecursion
def fromrecords(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 572)
    None_156455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 31), 'None')
    # Getting the type of 'None' (line 572)
    None_156456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 43), 'None')
    # Getting the type of 'None' (line 572)
    None_156457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 57), 'None')
    # Getting the type of 'None' (line 572)
    None_156458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 69), 'None')
    # Getting the type of 'None' (line 573)
    None_156459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 23), 'None')
    # Getting the type of 'False' (line 573)
    False_156460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 37), 'False')
    # Getting the type of 'None' (line 573)
    None_156461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 54), 'None')
    # Getting the type of 'None' (line 574)
    None_156462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 27), 'None')
    # Getting the type of 'nomask' (line 574)
    nomask_156463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 38), 'nomask')
    defaults = [None_156455, None_156456, None_156457, None_156458, None_156459, False_156460, None_156461, None_156462, nomask_156463]
    # Create a new context for function 'fromrecords'
    module_type_store = module_type_store.open_function_context('fromrecords', 572, 0, False)
    
    # Passed parameters checking function
    fromrecords.stypy_localization = localization
    fromrecords.stypy_type_of_self = None
    fromrecords.stypy_type_store = module_type_store
    fromrecords.stypy_function_name = 'fromrecords'
    fromrecords.stypy_param_names_list = ['reclist', 'dtype', 'shape', 'formats', 'names', 'titles', 'aligned', 'byteorder', 'fill_value', 'mask']
    fromrecords.stypy_varargs_param_name = None
    fromrecords.stypy_kwargs_param_name = None
    fromrecords.stypy_call_defaults = defaults
    fromrecords.stypy_call_varargs = varargs
    fromrecords.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'fromrecords', ['reclist', 'dtype', 'shape', 'formats', 'names', 'titles', 'aligned', 'byteorder', 'fill_value', 'mask'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'fromrecords', localization, ['reclist', 'dtype', 'shape', 'formats', 'names', 'titles', 'aligned', 'byteorder', 'fill_value', 'mask'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'fromrecords(...)' code ##################

    str_156464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 604, (-1)), 'str', '\n    Creates a MaskedRecords from a list of records.\n\n    Parameters\n    ----------\n    reclist : sequence\n        A list of records. Each element of the sequence is first converted\n        to a masked array if needed. If a 2D array is passed as argument, it is\n        processed line by line\n    dtype : {None, dtype}, optional\n        Data type descriptor.\n    shape : {None,int}, optional\n        Number of records. If None, ``shape`` is defined from the shape of the\n        first array in the list.\n    formats : {None, sequence}, optional\n        Sequence of formats for each individual field. If None, the formats will\n        be autodetected by inspecting the fields and selecting the highest dtype\n        possible.\n    names : {None, sequence}, optional\n        Sequence of the names of each field.\n    fill_value : {None, sequence}, optional\n        Sequence of data to be used as filling values.\n    mask : {nomask, sequence}, optional.\n        External mask to apply on the data.\n\n    Notes\n    -----\n    Lists of tuples should be preferred over lists of lists for faster processing.\n\n    ')
    
    # Assigning a Call to a Name (line 606):
    
    # Assigning a Call to a Name (line 606):
    
    # Call to getattr(...): (line 606)
    # Processing the call arguments (line 606)
    # Getting the type of 'reclist' (line 606)
    reclist_156466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 606, 20), 'reclist', False)
    str_156467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 606, 29), 'str', '_mask')
    # Getting the type of 'None' (line 606)
    None_156468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 606, 38), 'None', False)
    # Processing the call keyword arguments (line 606)
    kwargs_156469 = {}
    # Getting the type of 'getattr' (line 606)
    getattr_156465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 606, 12), 'getattr', False)
    # Calling getattr(args, kwargs) (line 606)
    getattr_call_result_156470 = invoke(stypy.reporting.localization.Localization(__file__, 606, 12), getattr_156465, *[reclist_156466, str_156467, None_156468], **kwargs_156469)
    
    # Assigning a type to the variable '_mask' (line 606)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 606, 4), '_mask', getattr_call_result_156470)
    
    
    # Call to isinstance(...): (line 608)
    # Processing the call arguments (line 608)
    # Getting the type of 'reclist' (line 608)
    reclist_156472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 608, 18), 'reclist', False)
    # Getting the type of 'ndarray' (line 608)
    ndarray_156473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 608, 27), 'ndarray', False)
    # Processing the call keyword arguments (line 608)
    kwargs_156474 = {}
    # Getting the type of 'isinstance' (line 608)
    isinstance_156471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 608, 7), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 608)
    isinstance_call_result_156475 = invoke(stypy.reporting.localization.Localization(__file__, 608, 7), isinstance_156471, *[reclist_156472, ndarray_156473], **kwargs_156474)
    
    # Testing the type of an if condition (line 608)
    if_condition_156476 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 608, 4), isinstance_call_result_156475)
    # Assigning a type to the variable 'if_condition_156476' (line 608)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 608, 4), 'if_condition_156476', if_condition_156476)
    # SSA begins for if statement (line 608)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Call to isinstance(...): (line 610)
    # Processing the call arguments (line 610)
    # Getting the type of 'reclist' (line 610)
    reclist_156478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 22), 'reclist', False)
    # Getting the type of 'MaskedArray' (line 610)
    MaskedArray_156479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 31), 'MaskedArray', False)
    # Processing the call keyword arguments (line 610)
    kwargs_156480 = {}
    # Getting the type of 'isinstance' (line 610)
    isinstance_156477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 11), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 610)
    isinstance_call_result_156481 = invoke(stypy.reporting.localization.Localization(__file__, 610, 11), isinstance_156477, *[reclist_156478, MaskedArray_156479], **kwargs_156480)
    
    # Testing the type of an if condition (line 610)
    if_condition_156482 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 610, 8), isinstance_call_result_156481)
    # Assigning a type to the variable 'if_condition_156482' (line 610)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 610, 8), 'if_condition_156482', if_condition_156482)
    # SSA begins for if statement (line 610)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 611):
    
    # Assigning a Call to a Name (line 611):
    
    # Call to view(...): (line 611)
    # Processing the call arguments (line 611)
    # Getting the type of 'ndarray' (line 611)
    ndarray_156488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 611, 44), 'ndarray', False)
    # Processing the call keyword arguments (line 611)
    kwargs_156489 = {}
    
    # Call to filled(...): (line 611)
    # Processing the call keyword arguments (line 611)
    kwargs_156485 = {}
    # Getting the type of 'reclist' (line 611)
    reclist_156483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 611, 22), 'reclist', False)
    # Obtaining the member 'filled' of a type (line 611)
    filled_156484 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 611, 22), reclist_156483, 'filled')
    # Calling filled(args, kwargs) (line 611)
    filled_call_result_156486 = invoke(stypy.reporting.localization.Localization(__file__, 611, 22), filled_156484, *[], **kwargs_156485)
    
    # Obtaining the member 'view' of a type (line 611)
    view_156487 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 611, 22), filled_call_result_156486, 'view')
    # Calling view(args, kwargs) (line 611)
    view_call_result_156490 = invoke(stypy.reporting.localization.Localization(__file__, 611, 22), view_156487, *[ndarray_156488], **kwargs_156489)
    
    # Assigning a type to the variable 'reclist' (line 611)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 611, 12), 'reclist', view_call_result_156490)
    # SSA join for if statement (line 610)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 613)
    # Getting the type of 'dtype' (line 613)
    dtype_156491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 11), 'dtype')
    # Getting the type of 'None' (line 613)
    None_156492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 20), 'None')
    
    (may_be_156493, more_types_in_union_156494) = may_be_none(dtype_156491, None_156492)

    if may_be_156493:

        if more_types_in_union_156494:
            # Runtime conditional SSA (line 613)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Attribute to a Name (line 614):
        
        # Assigning a Attribute to a Name (line 614):
        # Getting the type of 'reclist' (line 614)
        reclist_156495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 614, 20), 'reclist')
        # Obtaining the member 'dtype' of a type (line 614)
        dtype_156496 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 614, 20), reclist_156495, 'dtype')
        # Assigning a type to the variable 'dtype' (line 614)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 614, 12), 'dtype', dtype_156496)

        if more_types_in_union_156494:
            # SSA join for if statement (line 613)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 615):
    
    # Assigning a Call to a Name (line 615):
    
    # Call to tolist(...): (line 615)
    # Processing the call keyword arguments (line 615)
    kwargs_156499 = {}
    # Getting the type of 'reclist' (line 615)
    reclist_156497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 18), 'reclist', False)
    # Obtaining the member 'tolist' of a type (line 615)
    tolist_156498 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 615, 18), reclist_156497, 'tolist')
    # Calling tolist(args, kwargs) (line 615)
    tolist_call_result_156500 = invoke(stypy.reporting.localization.Localization(__file__, 615, 18), tolist_156498, *[], **kwargs_156499)
    
    # Assigning a type to the variable 'reclist' (line 615)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 615, 8), 'reclist', tolist_call_result_156500)
    # SSA join for if statement (line 608)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 616):
    
    # Assigning a Call to a Name (line 616):
    
    # Call to view(...): (line 616)
    # Processing the call arguments (line 616)
    # Getting the type of 'mrecarray' (line 618)
    mrecarray_156520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 69), 'mrecarray', False)
    # Processing the call keyword arguments (line 616)
    kwargs_156521 = {}
    
    # Call to recfromrecords(...): (line 616)
    # Processing the call arguments (line 616)
    # Getting the type of 'reclist' (line 616)
    reclist_156502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 26), 'reclist', False)
    # Processing the call keyword arguments (line 616)
    # Getting the type of 'dtype' (line 616)
    dtype_156503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 41), 'dtype', False)
    keyword_156504 = dtype_156503
    # Getting the type of 'shape' (line 616)
    shape_156505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 54), 'shape', False)
    keyword_156506 = shape_156505
    # Getting the type of 'formats' (line 616)
    formats_156507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 69), 'formats', False)
    keyword_156508 = formats_156507
    # Getting the type of 'names' (line 617)
    names_156509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 32), 'names', False)
    keyword_156510 = names_156509
    # Getting the type of 'titles' (line 617)
    titles_156511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 46), 'titles', False)
    keyword_156512 = titles_156511
    # Getting the type of 'aligned' (line 618)
    aligned_156513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 34), 'aligned', False)
    keyword_156514 = aligned_156513
    # Getting the type of 'byteorder' (line 618)
    byteorder_156515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 53), 'byteorder', False)
    keyword_156516 = byteorder_156515
    kwargs_156517 = {'dtype': keyword_156504, 'shape': keyword_156506, 'titles': keyword_156512, 'names': keyword_156510, 'formats': keyword_156508, 'aligned': keyword_156514, 'byteorder': keyword_156516}
    # Getting the type of 'recfromrecords' (line 616)
    recfromrecords_156501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 11), 'recfromrecords', False)
    # Calling recfromrecords(args, kwargs) (line 616)
    recfromrecords_call_result_156518 = invoke(stypy.reporting.localization.Localization(__file__, 616, 11), recfromrecords_156501, *[reclist_156502], **kwargs_156517)
    
    # Obtaining the member 'view' of a type (line 616)
    view_156519 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 616, 11), recfromrecords_call_result_156518, 'view')
    # Calling view(args, kwargs) (line 616)
    view_call_result_156522 = invoke(stypy.reporting.localization.Localization(__file__, 616, 11), view_156519, *[mrecarray_156520], **kwargs_156521)
    
    # Assigning a type to the variable 'mrec' (line 616)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 616, 4), 'mrec', view_call_result_156522)
    
    # Type idiom detected: calculating its left and rigth part (line 620)
    # Getting the type of 'fill_value' (line 620)
    fill_value_156523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 4), 'fill_value')
    # Getting the type of 'None' (line 620)
    None_156524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 25), 'None')
    
    (may_be_156525, more_types_in_union_156526) = may_not_be_none(fill_value_156523, None_156524)

    if may_be_156525:

        if more_types_in_union_156526:
            # Runtime conditional SSA (line 620)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Name to a Attribute (line 621):
        
        # Assigning a Name to a Attribute (line 621):
        # Getting the type of 'fill_value' (line 621)
        fill_value_156527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 26), 'fill_value')
        # Getting the type of 'mrec' (line 621)
        mrec_156528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 8), 'mrec')
        # Setting the type of the member 'fill_value' of a type (line 621)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 621, 8), mrec_156528, 'fill_value', fill_value_156527)

        if more_types_in_union_156526:
            # SSA join for if statement (line 620)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Getting the type of 'mask' (line 623)
    mask_156529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 7), 'mask')
    # Getting the type of 'nomask' (line 623)
    nomask_156530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 19), 'nomask')
    # Applying the binary operator 'isnot' (line 623)
    result_is_not_156531 = python_operator(stypy.reporting.localization.Localization(__file__, 623, 7), 'isnot', mask_156529, nomask_156530)
    
    # Testing the type of an if condition (line 623)
    if_condition_156532 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 623, 4), result_is_not_156531)
    # Assigning a type to the variable 'if_condition_156532' (line 623)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 623, 4), 'if_condition_156532', if_condition_156532)
    # SSA begins for if statement (line 623)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 624):
    
    # Assigning a Call to a Name (line 624):
    
    # Call to array(...): (line 624)
    # Processing the call arguments (line 624)
    # Getting the type of 'mask' (line 624)
    mask_156535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 624, 24), 'mask', False)
    # Processing the call keyword arguments (line 624)
    # Getting the type of 'False' (line 624)
    False_156536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 624, 35), 'False', False)
    keyword_156537 = False_156536
    kwargs_156538 = {'copy': keyword_156537}
    # Getting the type of 'np' (line 624)
    np_156533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 624, 15), 'np', False)
    # Obtaining the member 'array' of a type (line 624)
    array_156534 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 624, 15), np_156533, 'array')
    # Calling array(args, kwargs) (line 624)
    array_call_result_156539 = invoke(stypy.reporting.localization.Localization(__file__, 624, 15), array_156534, *[mask_156535], **kwargs_156538)
    
    # Assigning a type to the variable 'mask' (line 624)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 624, 8), 'mask', array_call_result_156539)
    
    # Assigning a Call to a Name (line 625):
    
    # Assigning a Call to a Name (line 625):
    
    # Call to len(...): (line 625)
    # Processing the call arguments (line 625)
    # Getting the type of 'mask' (line 625)
    mask_156541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 31), 'mask', False)
    # Obtaining the member 'dtype' of a type (line 625)
    dtype_156542 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 625, 31), mask_156541, 'dtype')
    # Processing the call keyword arguments (line 625)
    kwargs_156543 = {}
    # Getting the type of 'len' (line 625)
    len_156540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 27), 'len', False)
    # Calling len(args, kwargs) (line 625)
    len_call_result_156544 = invoke(stypy.reporting.localization.Localization(__file__, 625, 27), len_156540, *[dtype_156542], **kwargs_156543)
    
    # Assigning a type to the variable 'maskrecordlength' (line 625)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 625, 8), 'maskrecordlength', len_call_result_156544)
    
    # Getting the type of 'maskrecordlength' (line 626)
    maskrecordlength_156545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 626, 11), 'maskrecordlength')
    # Testing the type of an if condition (line 626)
    if_condition_156546 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 626, 8), maskrecordlength_156545)
    # Assigning a type to the variable 'if_condition_156546' (line 626)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 626, 8), 'if_condition_156546', if_condition_156546)
    # SSA begins for if statement (line 626)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Attribute (line 627):
    
    # Assigning a Name to a Attribute (line 627):
    # Getting the type of 'mask' (line 627)
    mask_156547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 627, 30), 'mask')
    # Getting the type of 'mrec' (line 627)
    mrec_156548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 627, 12), 'mrec')
    # Obtaining the member '_mask' of a type (line 627)
    _mask_156549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 627, 12), mrec_156548, '_mask')
    # Setting the type of the member 'flat' of a type (line 627)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 627, 12), _mask_156549, 'flat', mask_156547)
    # SSA branch for the else part of an if statement (line 626)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Call to len(...): (line 628)
    # Processing the call arguments (line 628)
    # Getting the type of 'mask' (line 628)
    mask_156551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 628, 17), 'mask', False)
    # Obtaining the member 'shape' of a type (line 628)
    shape_156552 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 628, 17), mask_156551, 'shape')
    # Processing the call keyword arguments (line 628)
    kwargs_156553 = {}
    # Getting the type of 'len' (line 628)
    len_156550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 628, 13), 'len', False)
    # Calling len(args, kwargs) (line 628)
    len_call_result_156554 = invoke(stypy.reporting.localization.Localization(__file__, 628, 13), len_156550, *[shape_156552], **kwargs_156553)
    
    int_156555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 628, 32), 'int')
    # Applying the binary operator '==' (line 628)
    result_eq_156556 = python_operator(stypy.reporting.localization.Localization(__file__, 628, 13), '==', len_call_result_156554, int_156555)
    
    # Testing the type of an if condition (line 628)
    if_condition_156557 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 628, 13), result_eq_156556)
    # Assigning a type to the variable 'if_condition_156557' (line 628)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 628, 13), 'if_condition_156557', if_condition_156557)
    # SSA begins for if statement (line 628)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a ListComp to a Attribute (line 629):
    
    # Assigning a ListComp to a Attribute (line 629):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'mask' (line 629)
    mask_156562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 629, 49), 'mask')
    comprehension_156563 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 629, 31), mask_156562)
    # Assigning a type to the variable 'm' (line 629)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 629, 31), 'm', comprehension_156563)
    
    # Call to tuple(...): (line 629)
    # Processing the call arguments (line 629)
    # Getting the type of 'm' (line 629)
    m_156559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 629, 37), 'm', False)
    # Processing the call keyword arguments (line 629)
    kwargs_156560 = {}
    # Getting the type of 'tuple' (line 629)
    tuple_156558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 629, 31), 'tuple', False)
    # Calling tuple(args, kwargs) (line 629)
    tuple_call_result_156561 = invoke(stypy.reporting.localization.Localization(__file__, 629, 31), tuple_156558, *[m_156559], **kwargs_156560)
    
    list_156564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 629, 31), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 629, 31), list_156564, tuple_call_result_156561)
    # Getting the type of 'mrec' (line 629)
    mrec_156565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 629, 12), 'mrec')
    # Obtaining the member '_mask' of a type (line 629)
    _mask_156566 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 629, 12), mrec_156565, '_mask')
    # Setting the type of the member 'flat' of a type (line 629)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 629, 12), _mask_156566, 'flat', list_156564)
    # SSA branch for the else part of an if statement (line 628)
    module_type_store.open_ssa_branch('else')
    
    # Call to __setmask__(...): (line 631)
    # Processing the call arguments (line 631)
    # Getting the type of 'mask' (line 631)
    mask_156569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 631, 29), 'mask', False)
    # Processing the call keyword arguments (line 631)
    kwargs_156570 = {}
    # Getting the type of 'mrec' (line 631)
    mrec_156567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 631, 12), 'mrec', False)
    # Obtaining the member '__setmask__' of a type (line 631)
    setmask___156568 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 631, 12), mrec_156567, '__setmask__')
    # Calling __setmask__(args, kwargs) (line 631)
    setmask___call_result_156571 = invoke(stypy.reporting.localization.Localization(__file__, 631, 12), setmask___156568, *[mask_156569], **kwargs_156570)
    
    # SSA join for if statement (line 628)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 626)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 623)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 632)
    # Getting the type of '_mask' (line 632)
    _mask_156572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 632, 4), '_mask')
    # Getting the type of 'None' (line 632)
    None_156573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 632, 20), 'None')
    
    (may_be_156574, more_types_in_union_156575) = may_not_be_none(_mask_156572, None_156573)

    if may_be_156574:

        if more_types_in_union_156575:
            # Runtime conditional SSA (line 632)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Name to a Subscript (line 633):
        
        # Assigning a Name to a Subscript (line 633):
        # Getting the type of '_mask' (line 633)
        _mask_156576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 633, 24), '_mask')
        # Getting the type of 'mrec' (line 633)
        mrec_156577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 633, 8), 'mrec')
        # Obtaining the member '_mask' of a type (line 633)
        _mask_156578 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 633, 8), mrec_156577, '_mask')
        slice_156579 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 633, 8), None, None, None)
        # Storing an element on a container (line 633)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 633, 8), _mask_156578, (slice_156579, _mask_156576))

        if more_types_in_union_156575:
            # SSA join for if statement (line 632)
            module_type_store = module_type_store.join_ssa_context()


    
    # Getting the type of 'mrec' (line 634)
    mrec_156580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 634, 11), 'mrec')
    # Assigning a type to the variable 'stypy_return_type' (line 634)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 634, 4), 'stypy_return_type', mrec_156580)
    
    # ################# End of 'fromrecords(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'fromrecords' in the type store
    # Getting the type of 'stypy_return_type' (line 572)
    stypy_return_type_156581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_156581)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'fromrecords'
    return stypy_return_type_156581

# Assigning a type to the variable 'fromrecords' (line 572)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 572, 0), 'fromrecords', fromrecords)

@norecursion
def _guessvartypes(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_guessvartypes'
    module_type_store = module_type_store.open_function_context('_guessvartypes', 637, 0, False)
    
    # Passed parameters checking function
    _guessvartypes.stypy_localization = localization
    _guessvartypes.stypy_type_of_self = None
    _guessvartypes.stypy_type_store = module_type_store
    _guessvartypes.stypy_function_name = '_guessvartypes'
    _guessvartypes.stypy_param_names_list = ['arr']
    _guessvartypes.stypy_varargs_param_name = None
    _guessvartypes.stypy_kwargs_param_name = None
    _guessvartypes.stypy_call_defaults = defaults
    _guessvartypes.stypy_call_varargs = varargs
    _guessvartypes.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_guessvartypes', ['arr'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_guessvartypes', localization, ['arr'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_guessvartypes(...)' code ##################

    str_156582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 646, (-1)), 'str', '\n    Tries to guess the dtypes of the str_ ndarray `arr`.\n\n    Guesses by testing element-wise conversion. Returns a list of dtypes.\n    The array is first converted to ndarray. If the array is 2D, the test\n    is performed on the first line. An exception is raised if the file is\n    3D or more.\n\n    ')
    
    # Assigning a List to a Name (line 647):
    
    # Assigning a List to a Name (line 647):
    
    # Obtaining an instance of the builtin type 'list' (line 647)
    list_156583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 647, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 647)
    
    # Assigning a type to the variable 'vartypes' (line 647)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 647, 4), 'vartypes', list_156583)
    
    # Assigning a Call to a Name (line 648):
    
    # Assigning a Call to a Name (line 648):
    
    # Call to asarray(...): (line 648)
    # Processing the call arguments (line 648)
    # Getting the type of 'arr' (line 648)
    arr_156586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 648, 21), 'arr', False)
    # Processing the call keyword arguments (line 648)
    kwargs_156587 = {}
    # Getting the type of 'np' (line 648)
    np_156584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 648, 10), 'np', False)
    # Obtaining the member 'asarray' of a type (line 648)
    asarray_156585 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 648, 10), np_156584, 'asarray')
    # Calling asarray(args, kwargs) (line 648)
    asarray_call_result_156588 = invoke(stypy.reporting.localization.Localization(__file__, 648, 10), asarray_156585, *[arr_156586], **kwargs_156587)
    
    # Assigning a type to the variable 'arr' (line 648)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 648, 4), 'arr', asarray_call_result_156588)
    
    
    
    # Call to len(...): (line 649)
    # Processing the call arguments (line 649)
    # Getting the type of 'arr' (line 649)
    arr_156590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 11), 'arr', False)
    # Obtaining the member 'shape' of a type (line 649)
    shape_156591 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 649, 11), arr_156590, 'shape')
    # Processing the call keyword arguments (line 649)
    kwargs_156592 = {}
    # Getting the type of 'len' (line 649)
    len_156589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 7), 'len', False)
    # Calling len(args, kwargs) (line 649)
    len_call_result_156593 = invoke(stypy.reporting.localization.Localization(__file__, 649, 7), len_156589, *[shape_156591], **kwargs_156592)
    
    int_156594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 649, 25), 'int')
    # Applying the binary operator '==' (line 649)
    result_eq_156595 = python_operator(stypy.reporting.localization.Localization(__file__, 649, 7), '==', len_call_result_156593, int_156594)
    
    # Testing the type of an if condition (line 649)
    if_condition_156596 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 649, 4), result_eq_156595)
    # Assigning a type to the variable 'if_condition_156596' (line 649)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 649, 4), 'if_condition_156596', if_condition_156596)
    # SSA begins for if statement (line 649)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 650):
    
    # Assigning a Subscript to a Name (line 650):
    
    # Obtaining the type of the subscript
    int_156597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 650, 18), 'int')
    # Getting the type of 'arr' (line 650)
    arr_156598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 14), 'arr')
    # Obtaining the member '__getitem__' of a type (line 650)
    getitem___156599 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 650, 14), arr_156598, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 650)
    subscript_call_result_156600 = invoke(stypy.reporting.localization.Localization(__file__, 650, 14), getitem___156599, int_156597)
    
    # Assigning a type to the variable 'arr' (line 650)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 650, 8), 'arr', subscript_call_result_156600)
    # SSA branch for the else part of an if statement (line 649)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Call to len(...): (line 651)
    # Processing the call arguments (line 651)
    # Getting the type of 'arr' (line 651)
    arr_156602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 13), 'arr', False)
    # Obtaining the member 'shape' of a type (line 651)
    shape_156603 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 651, 13), arr_156602, 'shape')
    # Processing the call keyword arguments (line 651)
    kwargs_156604 = {}
    # Getting the type of 'len' (line 651)
    len_156601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 9), 'len', False)
    # Calling len(args, kwargs) (line 651)
    len_call_result_156605 = invoke(stypy.reporting.localization.Localization(__file__, 651, 9), len_156601, *[shape_156603], **kwargs_156604)
    
    int_156606 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 651, 26), 'int')
    # Applying the binary operator '>' (line 651)
    result_gt_156607 = python_operator(stypy.reporting.localization.Localization(__file__, 651, 9), '>', len_call_result_156605, int_156606)
    
    # Testing the type of an if condition (line 651)
    if_condition_156608 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 651, 9), result_gt_156607)
    # Assigning a type to the variable 'if_condition_156608' (line 651)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 651, 9), 'if_condition_156608', if_condition_156608)
    # SSA begins for if statement (line 651)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 652)
    # Processing the call arguments (line 652)
    str_156610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 652, 25), 'str', 'The array should be 2D at most!')
    # Processing the call keyword arguments (line 652)
    kwargs_156611 = {}
    # Getting the type of 'ValueError' (line 652)
    ValueError_156609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 652, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 652)
    ValueError_call_result_156612 = invoke(stypy.reporting.localization.Localization(__file__, 652, 14), ValueError_156609, *[str_156610], **kwargs_156611)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 652, 8), ValueError_call_result_156612, 'raise parameter', BaseException)
    # SSA join for if statement (line 651)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 649)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'arr' (line 654)
    arr_156613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 654, 13), 'arr')
    # Testing the type of a for loop iterable (line 654)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 654, 4), arr_156613)
    # Getting the type of the for loop variable (line 654)
    for_loop_var_156614 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 654, 4), arr_156613)
    # Assigning a type to the variable 'f' (line 654)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 654, 4), 'f', for_loop_var_156614)
    # SSA begins for a for statement (line 654)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # SSA begins for try-except statement (line 655)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Call to int(...): (line 656)
    # Processing the call arguments (line 656)
    # Getting the type of 'f' (line 656)
    f_156616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 16), 'f', False)
    # Processing the call keyword arguments (line 656)
    kwargs_156617 = {}
    # Getting the type of 'int' (line 656)
    int_156615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 12), 'int', False)
    # Calling int(args, kwargs) (line 656)
    int_call_result_156618 = invoke(stypy.reporting.localization.Localization(__file__, 656, 12), int_156615, *[f_156616], **kwargs_156617)
    
    # SSA branch for the except part of a try statement (line 655)
    # SSA branch for the except 'ValueError' branch of a try statement (line 655)
    module_type_store.open_ssa_branch('except')
    
    
    # SSA begins for try-except statement (line 658)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Call to float(...): (line 659)
    # Processing the call arguments (line 659)
    # Getting the type of 'f' (line 659)
    f_156620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 22), 'f', False)
    # Processing the call keyword arguments (line 659)
    kwargs_156621 = {}
    # Getting the type of 'float' (line 659)
    float_156619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 16), 'float', False)
    # Calling float(args, kwargs) (line 659)
    float_call_result_156622 = invoke(stypy.reporting.localization.Localization(__file__, 659, 16), float_156619, *[f_156620], **kwargs_156621)
    
    # SSA branch for the except part of a try statement (line 658)
    # SSA branch for the except 'ValueError' branch of a try statement (line 658)
    module_type_store.open_ssa_branch('except')
    
    
    # SSA begins for try-except statement (line 661)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Call to complex(...): (line 662)
    # Processing the call arguments (line 662)
    # Getting the type of 'f' (line 662)
    f_156624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 28), 'f', False)
    # Processing the call keyword arguments (line 662)
    kwargs_156625 = {}
    # Getting the type of 'complex' (line 662)
    complex_156623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 20), 'complex', False)
    # Calling complex(args, kwargs) (line 662)
    complex_call_result_156626 = invoke(stypy.reporting.localization.Localization(__file__, 662, 20), complex_156623, *[f_156624], **kwargs_156625)
    
    # SSA branch for the except part of a try statement (line 661)
    # SSA branch for the except 'ValueError' branch of a try statement (line 661)
    module_type_store.open_ssa_branch('except')
    
    # Call to append(...): (line 664)
    # Processing the call arguments (line 664)
    # Getting the type of 'arr' (line 664)
    arr_156629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 664, 36), 'arr', False)
    # Obtaining the member 'dtype' of a type (line 664)
    dtype_156630 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 664, 36), arr_156629, 'dtype')
    # Processing the call keyword arguments (line 664)
    kwargs_156631 = {}
    # Getting the type of 'vartypes' (line 664)
    vartypes_156627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 664, 20), 'vartypes', False)
    # Obtaining the member 'append' of a type (line 664)
    append_156628 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 664, 20), vartypes_156627, 'append')
    # Calling append(args, kwargs) (line 664)
    append_call_result_156632 = invoke(stypy.reporting.localization.Localization(__file__, 664, 20), append_156628, *[dtype_156630], **kwargs_156631)
    
    # SSA branch for the else branch of a try statement (line 661)
    module_type_store.open_ssa_branch('except else')
    
    # Call to append(...): (line 666)
    # Processing the call arguments (line 666)
    
    # Call to dtype(...): (line 666)
    # Processing the call arguments (line 666)
    # Getting the type of 'complex' (line 666)
    complex_156637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 45), 'complex', False)
    # Processing the call keyword arguments (line 666)
    kwargs_156638 = {}
    # Getting the type of 'np' (line 666)
    np_156635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 36), 'np', False)
    # Obtaining the member 'dtype' of a type (line 666)
    dtype_156636 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 666, 36), np_156635, 'dtype')
    # Calling dtype(args, kwargs) (line 666)
    dtype_call_result_156639 = invoke(stypy.reporting.localization.Localization(__file__, 666, 36), dtype_156636, *[complex_156637], **kwargs_156638)
    
    # Processing the call keyword arguments (line 666)
    kwargs_156640 = {}
    # Getting the type of 'vartypes' (line 666)
    vartypes_156633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 20), 'vartypes', False)
    # Obtaining the member 'append' of a type (line 666)
    append_156634 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 666, 20), vartypes_156633, 'append')
    # Calling append(args, kwargs) (line 666)
    append_call_result_156641 = invoke(stypy.reporting.localization.Localization(__file__, 666, 20), append_156634, *[dtype_call_result_156639], **kwargs_156640)
    
    # SSA join for try-except statement (line 661)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else branch of a try statement (line 658)
    module_type_store.open_ssa_branch('except else')
    
    # Call to append(...): (line 668)
    # Processing the call arguments (line 668)
    
    # Call to dtype(...): (line 668)
    # Processing the call arguments (line 668)
    # Getting the type of 'float' (line 668)
    float_156646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 41), 'float', False)
    # Processing the call keyword arguments (line 668)
    kwargs_156647 = {}
    # Getting the type of 'np' (line 668)
    np_156644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 32), 'np', False)
    # Obtaining the member 'dtype' of a type (line 668)
    dtype_156645 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 668, 32), np_156644, 'dtype')
    # Calling dtype(args, kwargs) (line 668)
    dtype_call_result_156648 = invoke(stypy.reporting.localization.Localization(__file__, 668, 32), dtype_156645, *[float_156646], **kwargs_156647)
    
    # Processing the call keyword arguments (line 668)
    kwargs_156649 = {}
    # Getting the type of 'vartypes' (line 668)
    vartypes_156642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 16), 'vartypes', False)
    # Obtaining the member 'append' of a type (line 668)
    append_156643 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 668, 16), vartypes_156642, 'append')
    # Calling append(args, kwargs) (line 668)
    append_call_result_156650 = invoke(stypy.reporting.localization.Localization(__file__, 668, 16), append_156643, *[dtype_call_result_156648], **kwargs_156649)
    
    # SSA join for try-except statement (line 658)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else branch of a try statement (line 655)
    module_type_store.open_ssa_branch('except else')
    
    # Call to append(...): (line 670)
    # Processing the call arguments (line 670)
    
    # Call to dtype(...): (line 670)
    # Processing the call arguments (line 670)
    # Getting the type of 'int' (line 670)
    int_156655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 670, 37), 'int', False)
    # Processing the call keyword arguments (line 670)
    kwargs_156656 = {}
    # Getting the type of 'np' (line 670)
    np_156653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 670, 28), 'np', False)
    # Obtaining the member 'dtype' of a type (line 670)
    dtype_156654 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 670, 28), np_156653, 'dtype')
    # Calling dtype(args, kwargs) (line 670)
    dtype_call_result_156657 = invoke(stypy.reporting.localization.Localization(__file__, 670, 28), dtype_156654, *[int_156655], **kwargs_156656)
    
    # Processing the call keyword arguments (line 670)
    kwargs_156658 = {}
    # Getting the type of 'vartypes' (line 670)
    vartypes_156651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 670, 12), 'vartypes', False)
    # Obtaining the member 'append' of a type (line 670)
    append_156652 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 670, 12), vartypes_156651, 'append')
    # Calling append(args, kwargs) (line 670)
    append_call_result_156659 = invoke(stypy.reporting.localization.Localization(__file__, 670, 12), append_156652, *[dtype_call_result_156657], **kwargs_156658)
    
    # SSA join for try-except statement (line 655)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'vartypes' (line 671)
    vartypes_156660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 671, 11), 'vartypes')
    # Assigning a type to the variable 'stypy_return_type' (line 671)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 671, 4), 'stypy_return_type', vartypes_156660)
    
    # ################# End of '_guessvartypes(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_guessvartypes' in the type store
    # Getting the type of 'stypy_return_type' (line 637)
    stypy_return_type_156661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_156661)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_guessvartypes'
    return stypy_return_type_156661

# Assigning a type to the variable '_guessvartypes' (line 637)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 637, 0), '_guessvartypes', _guessvartypes)

@norecursion
def openfile(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'openfile'
    module_type_store = module_type_store.open_function_context('openfile', 674, 0, False)
    
    # Passed parameters checking function
    openfile.stypy_localization = localization
    openfile.stypy_type_of_self = None
    openfile.stypy_type_store = module_type_store
    openfile.stypy_function_name = 'openfile'
    openfile.stypy_param_names_list = ['fname']
    openfile.stypy_varargs_param_name = None
    openfile.stypy_kwargs_param_name = None
    openfile.stypy_call_defaults = defaults
    openfile.stypy_call_varargs = varargs
    openfile.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'openfile', ['fname'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'openfile', localization, ['fname'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'openfile(...)' code ##################

    str_156662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 678, (-1)), 'str', '\n    Opens the file handle of file `fname`.\n\n    ')
    
    # Type idiom detected: calculating its left and rigth part (line 680)
    str_156663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 680, 22), 'str', 'readline')
    # Getting the type of 'fname' (line 680)
    fname_156664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 680, 15), 'fname')
    
    (may_be_156665, more_types_in_union_156666) = may_provide_member(str_156663, fname_156664)

    if may_be_156665:

        if more_types_in_union_156666:
            # Runtime conditional SSA (line 680)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'fname' (line 680)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 680, 4), 'fname', remove_not_member_provider_from_union(fname_156664, 'readline'))
        # Getting the type of 'fname' (line 681)
        fname_156667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 15), 'fname')
        # Assigning a type to the variable 'stypy_return_type' (line 681)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 681, 8), 'stypy_return_type', fname_156667)

        if more_types_in_union_156666:
            # SSA join for if statement (line 680)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # SSA begins for try-except statement (line 683)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Name (line 684):
    
    # Assigning a Call to a Name (line 684):
    
    # Call to open(...): (line 684)
    # Processing the call arguments (line 684)
    # Getting the type of 'fname' (line 684)
    fname_156669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 684, 17), 'fname', False)
    # Processing the call keyword arguments (line 684)
    kwargs_156670 = {}
    # Getting the type of 'open' (line 684)
    open_156668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 684, 12), 'open', False)
    # Calling open(args, kwargs) (line 684)
    open_call_result_156671 = invoke(stypy.reporting.localization.Localization(__file__, 684, 12), open_156668, *[fname_156669], **kwargs_156670)
    
    # Assigning a type to the variable 'f' (line 684)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 684, 8), 'f', open_call_result_156671)
    # SSA branch for the except part of a try statement (line 683)
    # SSA branch for the except 'IOError' branch of a try statement (line 683)
    module_type_store.open_ssa_branch('except')
    
    # Call to IOError(...): (line 686)
    # Processing the call arguments (line 686)
    str_156673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 686, 22), 'str', "No such file: '%s'")
    # Getting the type of 'fname' (line 686)
    fname_156674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 686, 45), 'fname', False)
    # Applying the binary operator '%' (line 686)
    result_mod_156675 = python_operator(stypy.reporting.localization.Localization(__file__, 686, 22), '%', str_156673, fname_156674)
    
    # Processing the call keyword arguments (line 686)
    kwargs_156676 = {}
    # Getting the type of 'IOError' (line 686)
    IOError_156672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 686, 14), 'IOError', False)
    # Calling IOError(args, kwargs) (line 686)
    IOError_call_result_156677 = invoke(stypy.reporting.localization.Localization(__file__, 686, 14), IOError_156672, *[result_mod_156675], **kwargs_156676)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 686, 8), IOError_call_result_156677, 'raise parameter', BaseException)
    # SSA join for try-except statement (line 683)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Obtaining the type of the subscript
    int_156678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 687, 21), 'int')
    slice_156679 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 687, 7), None, int_156678, None)
    
    # Call to readline(...): (line 687)
    # Processing the call keyword arguments (line 687)
    kwargs_156682 = {}
    # Getting the type of 'f' (line 687)
    f_156680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 687, 7), 'f', False)
    # Obtaining the member 'readline' of a type (line 687)
    readline_156681 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 687, 7), f_156680, 'readline')
    # Calling readline(args, kwargs) (line 687)
    readline_call_result_156683 = invoke(stypy.reporting.localization.Localization(__file__, 687, 7), readline_156681, *[], **kwargs_156682)
    
    # Obtaining the member '__getitem__' of a type (line 687)
    getitem___156684 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 687, 7), readline_call_result_156683, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 687)
    subscript_call_result_156685 = invoke(stypy.reporting.localization.Localization(__file__, 687, 7), getitem___156684, slice_156679)
    
    str_156686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 687, 27), 'str', '\\x')
    # Applying the binary operator '!=' (line 687)
    result_ne_156687 = python_operator(stypy.reporting.localization.Localization(__file__, 687, 7), '!=', subscript_call_result_156685, str_156686)
    
    # Testing the type of an if condition (line 687)
    if_condition_156688 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 687, 4), result_ne_156687)
    # Assigning a type to the variable 'if_condition_156688' (line 687)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 687, 4), 'if_condition_156688', if_condition_156688)
    # SSA begins for if statement (line 687)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to seek(...): (line 688)
    # Processing the call arguments (line 688)
    int_156691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 688, 15), 'int')
    int_156692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 688, 18), 'int')
    # Processing the call keyword arguments (line 688)
    kwargs_156693 = {}
    # Getting the type of 'f' (line 688)
    f_156689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 688, 8), 'f', False)
    # Obtaining the member 'seek' of a type (line 688)
    seek_156690 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 688, 8), f_156689, 'seek')
    # Calling seek(args, kwargs) (line 688)
    seek_call_result_156694 = invoke(stypy.reporting.localization.Localization(__file__, 688, 8), seek_156690, *[int_156691, int_156692], **kwargs_156693)
    
    # Getting the type of 'f' (line 689)
    f_156695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 15), 'f')
    # Assigning a type to the variable 'stypy_return_type' (line 689)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 689, 8), 'stypy_return_type', f_156695)
    # SSA join for if statement (line 687)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to close(...): (line 690)
    # Processing the call keyword arguments (line 690)
    kwargs_156698 = {}
    # Getting the type of 'f' (line 690)
    f_156696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 4), 'f', False)
    # Obtaining the member 'close' of a type (line 690)
    close_156697 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 690, 4), f_156696, 'close')
    # Calling close(args, kwargs) (line 690)
    close_call_result_156699 = invoke(stypy.reporting.localization.Localization(__file__, 690, 4), close_156697, *[], **kwargs_156698)
    
    
    # Call to NotImplementedError(...): (line 691)
    # Processing the call arguments (line 691)
    str_156701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 691, 30), 'str', 'Wow, binary file')
    # Processing the call keyword arguments (line 691)
    kwargs_156702 = {}
    # Getting the type of 'NotImplementedError' (line 691)
    NotImplementedError_156700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 691, 10), 'NotImplementedError', False)
    # Calling NotImplementedError(args, kwargs) (line 691)
    NotImplementedError_call_result_156703 = invoke(stypy.reporting.localization.Localization(__file__, 691, 10), NotImplementedError_156700, *[str_156701], **kwargs_156702)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 691, 4), NotImplementedError_call_result_156703, 'raise parameter', BaseException)
    
    # ################# End of 'openfile(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'openfile' in the type store
    # Getting the type of 'stypy_return_type' (line 674)
    stypy_return_type_156704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 674, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_156704)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'openfile'
    return stypy_return_type_156704

# Assigning a type to the variable 'openfile' (line 674)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 674, 0), 'openfile', openfile)

@norecursion
def fromtextfile(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 694)
    None_156705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 34), 'None')
    str_156706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 694, 52), 'str', '#')
    str_156707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 694, 69), 'str', '')
    # Getting the type of 'None' (line 695)
    None_156708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 26), 'None')
    # Getting the type of 'None' (line 695)
    None_156709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 41), 'None')
    defaults = [None_156705, str_156706, str_156707, None_156708, None_156709]
    # Create a new context for function 'fromtextfile'
    module_type_store = module_type_store.open_function_context('fromtextfile', 694, 0, False)
    
    # Passed parameters checking function
    fromtextfile.stypy_localization = localization
    fromtextfile.stypy_type_of_self = None
    fromtextfile.stypy_type_store = module_type_store
    fromtextfile.stypy_function_name = 'fromtextfile'
    fromtextfile.stypy_param_names_list = ['fname', 'delimitor', 'commentchar', 'missingchar', 'varnames', 'vartypes']
    fromtextfile.stypy_varargs_param_name = None
    fromtextfile.stypy_kwargs_param_name = None
    fromtextfile.stypy_call_defaults = defaults
    fromtextfile.stypy_call_varargs = varargs
    fromtextfile.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'fromtextfile', ['fname', 'delimitor', 'commentchar', 'missingchar', 'varnames', 'vartypes'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'fromtextfile', localization, ['fname', 'delimitor', 'commentchar', 'missingchar', 'varnames', 'vartypes'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'fromtextfile(...)' code ##################

    str_156710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 718, (-1)), 'str', "\n    Creates a mrecarray from data stored in the file `filename`.\n\n    Parameters\n    ----------\n    fname : {file name/handle}\n        Handle of an opened file.\n    delimitor : {None, string}, optional\n        Alphanumeric character used to separate columns in the file.\n        If None, any (group of) white spacestring(s) will be used.\n    commentchar : {'#', string}, optional\n        Alphanumeric character used to mark the start of a comment.\n    missingchar : {'', string}, optional\n        String indicating missing data, and used to create the masks.\n    varnames : {None, sequence}, optional\n        Sequence of the variable names. If None, a list will be created from\n        the first non empty line of the file.\n    vartypes : {None, sequence}, optional\n        Sequence of the variables dtypes. If None, it will be estimated from\n        the first non-commented line.\n\n\n    Ultra simple: the varnames are in the header, one line")
    
    # Assigning a Call to a Name (line 720):
    
    # Assigning a Call to a Name (line 720):
    
    # Call to openfile(...): (line 720)
    # Processing the call arguments (line 720)
    # Getting the type of 'fname' (line 720)
    fname_156712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 720, 21), 'fname', False)
    # Processing the call keyword arguments (line 720)
    kwargs_156713 = {}
    # Getting the type of 'openfile' (line 720)
    openfile_156711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 720, 12), 'openfile', False)
    # Calling openfile(args, kwargs) (line 720)
    openfile_call_result_156714 = invoke(stypy.reporting.localization.Localization(__file__, 720, 12), openfile_156711, *[fname_156712], **kwargs_156713)
    
    # Assigning a type to the variable 'ftext' (line 720)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 720, 4), 'ftext', openfile_call_result_156714)
    
    # Getting the type of 'True' (line 723)
    True_156715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 723, 10), 'True')
    # Testing the type of an if condition (line 723)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 723, 4), True_156715)
    # SSA begins for while statement (line 723)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Assigning a Call to a Name (line 724):
    
    # Assigning a Call to a Name (line 724):
    
    # Call to readline(...): (line 724)
    # Processing the call keyword arguments (line 724)
    kwargs_156718 = {}
    # Getting the type of 'ftext' (line 724)
    ftext_156716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 724, 15), 'ftext', False)
    # Obtaining the member 'readline' of a type (line 724)
    readline_156717 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 724, 15), ftext_156716, 'readline')
    # Calling readline(args, kwargs) (line 724)
    readline_call_result_156719 = invoke(stypy.reporting.localization.Localization(__file__, 724, 15), readline_156717, *[], **kwargs_156718)
    
    # Assigning a type to the variable 'line' (line 724)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 724, 8), 'line', readline_call_result_156719)
    
    # Assigning a Call to a Name (line 725):
    
    # Assigning a Call to a Name (line 725):
    
    # Call to strip(...): (line 725)
    # Processing the call keyword arguments (line 725)
    kwargs_156730 = {}
    
    # Obtaining the type of the subscript
    
    # Call to find(...): (line 725)
    # Processing the call arguments (line 725)
    # Getting the type of 'commentchar' (line 725)
    commentchar_156722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 725, 36), 'commentchar', False)
    # Processing the call keyword arguments (line 725)
    kwargs_156723 = {}
    # Getting the type of 'line' (line 725)
    line_156720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 725, 26), 'line', False)
    # Obtaining the member 'find' of a type (line 725)
    find_156721 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 725, 26), line_156720, 'find')
    # Calling find(args, kwargs) (line 725)
    find_call_result_156724 = invoke(stypy.reporting.localization.Localization(__file__, 725, 26), find_156721, *[commentchar_156722], **kwargs_156723)
    
    slice_156725 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 725, 20), None, find_call_result_156724, None)
    # Getting the type of 'line' (line 725)
    line_156726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 725, 20), 'line', False)
    # Obtaining the member '__getitem__' of a type (line 725)
    getitem___156727 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 725, 20), line_156726, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 725)
    subscript_call_result_156728 = invoke(stypy.reporting.localization.Localization(__file__, 725, 20), getitem___156727, slice_156725)
    
    # Obtaining the member 'strip' of a type (line 725)
    strip_156729 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 725, 20), subscript_call_result_156728, 'strip')
    # Calling strip(args, kwargs) (line 725)
    strip_call_result_156731 = invoke(stypy.reporting.localization.Localization(__file__, 725, 20), strip_156729, *[], **kwargs_156730)
    
    # Assigning a type to the variable 'firstline' (line 725)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 725, 8), 'firstline', strip_call_result_156731)
    
    # Assigning a Call to a Name (line 726):
    
    # Assigning a Call to a Name (line 726):
    
    # Call to split(...): (line 726)
    # Processing the call arguments (line 726)
    # Getting the type of 'delimitor' (line 726)
    delimitor_156734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 726, 36), 'delimitor', False)
    # Processing the call keyword arguments (line 726)
    kwargs_156735 = {}
    # Getting the type of 'firstline' (line 726)
    firstline_156732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 726, 20), 'firstline', False)
    # Obtaining the member 'split' of a type (line 726)
    split_156733 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 726, 20), firstline_156732, 'split')
    # Calling split(args, kwargs) (line 726)
    split_call_result_156736 = invoke(stypy.reporting.localization.Localization(__file__, 726, 20), split_156733, *[delimitor_156734], **kwargs_156735)
    
    # Assigning a type to the variable '_varnames' (line 726)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 726, 8), '_varnames', split_call_result_156736)
    
    
    
    # Call to len(...): (line 727)
    # Processing the call arguments (line 727)
    # Getting the type of '_varnames' (line 727)
    _varnames_156738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 727, 15), '_varnames', False)
    # Processing the call keyword arguments (line 727)
    kwargs_156739 = {}
    # Getting the type of 'len' (line 727)
    len_156737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 727, 11), 'len', False)
    # Calling len(args, kwargs) (line 727)
    len_call_result_156740 = invoke(stypy.reporting.localization.Localization(__file__, 727, 11), len_156737, *[_varnames_156738], **kwargs_156739)
    
    int_156741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 727, 28), 'int')
    # Applying the binary operator '>' (line 727)
    result_gt_156742 = python_operator(stypy.reporting.localization.Localization(__file__, 727, 11), '>', len_call_result_156740, int_156741)
    
    # Testing the type of an if condition (line 727)
    if_condition_156743 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 727, 8), result_gt_156742)
    # Assigning a type to the variable 'if_condition_156743' (line 727)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 727, 8), 'if_condition_156743', if_condition_156743)
    # SSA begins for if statement (line 727)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 727)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for while statement (line 723)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 729)
    # Getting the type of 'varnames' (line 729)
    varnames_156744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 729, 7), 'varnames')
    # Getting the type of 'None' (line 729)
    None_156745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 729, 19), 'None')
    
    (may_be_156746, more_types_in_union_156747) = may_be_none(varnames_156744, None_156745)

    if may_be_156746:

        if more_types_in_union_156747:
            # Runtime conditional SSA (line 729)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Name to a Name (line 730):
        
        # Assigning a Name to a Name (line 730):
        # Getting the type of '_varnames' (line 730)
        _varnames_156748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 730, 19), '_varnames')
        # Assigning a type to the variable 'varnames' (line 730)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 730, 8), 'varnames', _varnames_156748)

        if more_types_in_union_156747:
            # SSA join for if statement (line 729)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 733):
    
    # Assigning a Call to a Name (line 733):
    
    # Call to masked_array(...): (line 733)
    # Processing the call arguments (line 733)
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'ftext' (line 733)
    ftext_156771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 733, 73), 'ftext', False)
    comprehension_156772 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 733, 31), ftext_156771)
    # Assigning a type to the variable 'line' (line 733)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 733, 31), 'line', comprehension_156772)
    
    # Evaluating a boolean operation
    
    
    # Obtaining the type of the subscript
    int_156758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 734, 39), 'int')
    # Getting the type of 'line' (line 734)
    line_156759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 734, 34), 'line', False)
    # Obtaining the member '__getitem__' of a type (line 734)
    getitem___156760 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 734, 34), line_156759, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 734)
    subscript_call_result_156761 = invoke(stypy.reporting.localization.Localization(__file__, 734, 34), getitem___156760, int_156758)
    
    # Getting the type of 'commentchar' (line 734)
    commentchar_156762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 734, 45), 'commentchar', False)
    # Applying the binary operator '!=' (line 734)
    result_ne_156763 = python_operator(stypy.reporting.localization.Localization(__file__, 734, 34), '!=', subscript_call_result_156761, commentchar_156762)
    
    
    
    # Call to len(...): (line 734)
    # Processing the call arguments (line 734)
    # Getting the type of 'line' (line 734)
    line_156765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 734, 65), 'line', False)
    # Processing the call keyword arguments (line 734)
    kwargs_156766 = {}
    # Getting the type of 'len' (line 734)
    len_156764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 734, 61), 'len', False)
    # Calling len(args, kwargs) (line 734)
    len_call_result_156767 = invoke(stypy.reporting.localization.Localization(__file__, 734, 61), len_156764, *[line_156765], **kwargs_156766)
    
    int_156768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 734, 73), 'int')
    # Applying the binary operator '>' (line 734)
    result_gt_156769 = python_operator(stypy.reporting.localization.Localization(__file__, 734, 61), '>', len_call_result_156767, int_156768)
    
    # Applying the binary operator 'and' (line 734)
    result_and_keyword_156770 = python_operator(stypy.reporting.localization.Localization(__file__, 734, 34), 'and', result_ne_156763, result_gt_156769)
    
    
    # Call to split(...): (line 733)
    # Processing the call arguments (line 733)
    # Getting the type of 'delimitor' (line 733)
    delimitor_156755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 733, 50), 'delimitor', False)
    # Processing the call keyword arguments (line 733)
    kwargs_156756 = {}
    
    # Call to strip(...): (line 733)
    # Processing the call keyword arguments (line 733)
    kwargs_156752 = {}
    # Getting the type of 'line' (line 733)
    line_156750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 733, 31), 'line', False)
    # Obtaining the member 'strip' of a type (line 733)
    strip_156751 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 733, 31), line_156750, 'strip')
    # Calling strip(args, kwargs) (line 733)
    strip_call_result_156753 = invoke(stypy.reporting.localization.Localization(__file__, 733, 31), strip_156751, *[], **kwargs_156752)
    
    # Obtaining the member 'split' of a type (line 733)
    split_156754 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 733, 31), strip_call_result_156753, 'split')
    # Calling split(args, kwargs) (line 733)
    split_call_result_156757 = invoke(stypy.reporting.localization.Localization(__file__, 733, 31), split_156754, *[delimitor_156755], **kwargs_156756)
    
    list_156773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 733, 31), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 733, 31), list_156773, split_call_result_156757)
    # Processing the call keyword arguments (line 733)
    kwargs_156774 = {}
    # Getting the type of 'masked_array' (line 733)
    masked_array_156749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 733, 17), 'masked_array', False)
    # Calling masked_array(args, kwargs) (line 733)
    masked_array_call_result_156775 = invoke(stypy.reporting.localization.Localization(__file__, 733, 17), masked_array_156749, *[list_156773], **kwargs_156774)
    
    # Assigning a type to the variable '_variables' (line 733)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 733, 4), '_variables', masked_array_call_result_156775)
    
    # Assigning a Attribute to a Tuple (line 735):
    
    # Assigning a Subscript to a Name (line 735):
    
    # Obtaining the type of the subscript
    int_156776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 735, 4), 'int')
    # Getting the type of '_variables' (line 735)
    _variables_156777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 735, 19), '_variables')
    # Obtaining the member 'shape' of a type (line 735)
    shape_156778 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 735, 19), _variables_156777, 'shape')
    # Obtaining the member '__getitem__' of a type (line 735)
    getitem___156779 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 735, 4), shape_156778, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 735)
    subscript_call_result_156780 = invoke(stypy.reporting.localization.Localization(__file__, 735, 4), getitem___156779, int_156776)
    
    # Assigning a type to the variable 'tuple_var_assignment_154984' (line 735)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 735, 4), 'tuple_var_assignment_154984', subscript_call_result_156780)
    
    # Assigning a Subscript to a Name (line 735):
    
    # Obtaining the type of the subscript
    int_156781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 735, 4), 'int')
    # Getting the type of '_variables' (line 735)
    _variables_156782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 735, 19), '_variables')
    # Obtaining the member 'shape' of a type (line 735)
    shape_156783 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 735, 19), _variables_156782, 'shape')
    # Obtaining the member '__getitem__' of a type (line 735)
    getitem___156784 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 735, 4), shape_156783, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 735)
    subscript_call_result_156785 = invoke(stypy.reporting.localization.Localization(__file__, 735, 4), getitem___156784, int_156781)
    
    # Assigning a type to the variable 'tuple_var_assignment_154985' (line 735)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 735, 4), 'tuple_var_assignment_154985', subscript_call_result_156785)
    
    # Assigning a Name to a Name (line 735):
    # Getting the type of 'tuple_var_assignment_154984' (line 735)
    tuple_var_assignment_154984_156786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 735, 4), 'tuple_var_assignment_154984')
    # Assigning a type to the variable '_' (line 735)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 735, 5), '_', tuple_var_assignment_154984_156786)
    
    # Assigning a Name to a Name (line 735):
    # Getting the type of 'tuple_var_assignment_154985' (line 735)
    tuple_var_assignment_154985_156787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 735, 4), 'tuple_var_assignment_154985')
    # Assigning a type to the variable 'nfields' (line 735)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 735, 8), 'nfields', tuple_var_assignment_154985_156787)
    
    # Call to close(...): (line 736)
    # Processing the call keyword arguments (line 736)
    kwargs_156790 = {}
    # Getting the type of 'ftext' (line 736)
    ftext_156788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 736, 4), 'ftext', False)
    # Obtaining the member 'close' of a type (line 736)
    close_156789 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 736, 4), ftext_156788, 'close')
    # Calling close(args, kwargs) (line 736)
    close_call_result_156791 = invoke(stypy.reporting.localization.Localization(__file__, 736, 4), close_156789, *[], **kwargs_156790)
    
    
    # Type idiom detected: calculating its left and rigth part (line 739)
    # Getting the type of 'vartypes' (line 739)
    vartypes_156792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 739, 7), 'vartypes')
    # Getting the type of 'None' (line 739)
    None_156793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 739, 19), 'None')
    
    (may_be_156794, more_types_in_union_156795) = may_be_none(vartypes_156792, None_156793)

    if may_be_156794:

        if more_types_in_union_156795:
            # Runtime conditional SSA (line 739)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 740):
        
        # Assigning a Call to a Name (line 740):
        
        # Call to _guessvartypes(...): (line 740)
        # Processing the call arguments (line 740)
        
        # Obtaining the type of the subscript
        int_156797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 740, 45), 'int')
        # Getting the type of '_variables' (line 740)
        _variables_156798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 740, 34), '_variables', False)
        # Obtaining the member '__getitem__' of a type (line 740)
        getitem___156799 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 740, 34), _variables_156798, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 740)
        subscript_call_result_156800 = invoke(stypy.reporting.localization.Localization(__file__, 740, 34), getitem___156799, int_156797)
        
        # Processing the call keyword arguments (line 740)
        kwargs_156801 = {}
        # Getting the type of '_guessvartypes' (line 740)
        _guessvartypes_156796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 740, 19), '_guessvartypes', False)
        # Calling _guessvartypes(args, kwargs) (line 740)
        _guessvartypes_call_result_156802 = invoke(stypy.reporting.localization.Localization(__file__, 740, 19), _guessvartypes_156796, *[subscript_call_result_156800], **kwargs_156801)
        
        # Assigning a type to the variable 'vartypes' (line 740)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 740, 8), 'vartypes', _guessvartypes_call_result_156802)

        if more_types_in_union_156795:
            # Runtime conditional SSA for else branch (line 739)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_156794) or more_types_in_union_156795):
        
        # Assigning a ListComp to a Name (line 742):
        
        # Assigning a ListComp to a Name (line 742):
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'vartypes' (line 742)
        vartypes_156808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 41), 'vartypes')
        comprehension_156809 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 742, 20), vartypes_156808)
        # Assigning a type to the variable 'v' (line 742)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 742, 20), 'v', comprehension_156809)
        
        # Call to dtype(...): (line 742)
        # Processing the call arguments (line 742)
        # Getting the type of 'v' (line 742)
        v_156805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 29), 'v', False)
        # Processing the call keyword arguments (line 742)
        kwargs_156806 = {}
        # Getting the type of 'np' (line 742)
        np_156803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 20), 'np', False)
        # Obtaining the member 'dtype' of a type (line 742)
        dtype_156804 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 742, 20), np_156803, 'dtype')
        # Calling dtype(args, kwargs) (line 742)
        dtype_call_result_156807 = invoke(stypy.reporting.localization.Localization(__file__, 742, 20), dtype_156804, *[v_156805], **kwargs_156806)
        
        list_156810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 742, 20), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 742, 20), list_156810, dtype_call_result_156807)
        # Assigning a type to the variable 'vartypes' (line 742)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 742, 8), 'vartypes', list_156810)
        
        
        
        # Call to len(...): (line 743)
        # Processing the call arguments (line 743)
        # Getting the type of 'vartypes' (line 743)
        vartypes_156812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 743, 15), 'vartypes', False)
        # Processing the call keyword arguments (line 743)
        kwargs_156813 = {}
        # Getting the type of 'len' (line 743)
        len_156811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 743, 11), 'len', False)
        # Calling len(args, kwargs) (line 743)
        len_call_result_156814 = invoke(stypy.reporting.localization.Localization(__file__, 743, 11), len_156811, *[vartypes_156812], **kwargs_156813)
        
        # Getting the type of 'nfields' (line 743)
        nfields_156815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 743, 28), 'nfields')
        # Applying the binary operator '!=' (line 743)
        result_ne_156816 = python_operator(stypy.reporting.localization.Localization(__file__, 743, 11), '!=', len_call_result_156814, nfields_156815)
        
        # Testing the type of an if condition (line 743)
        if_condition_156817 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 743, 8), result_ne_156816)
        # Assigning a type to the variable 'if_condition_156817' (line 743)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 743, 8), 'if_condition_156817', if_condition_156817)
        # SSA begins for if statement (line 743)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Str to a Name (line 744):
        
        # Assigning a Str to a Name (line 744):
        str_156818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 744, 18), 'str', 'Attempting to %i dtypes for %i fields!')
        # Assigning a type to the variable 'msg' (line 744)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 744, 12), 'msg', str_156818)
        
        # Getting the type of 'msg' (line 745)
        msg_156819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 745, 12), 'msg')
        str_156820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 745, 19), 'str', ' Reverting to default.')
        # Applying the binary operator '+=' (line 745)
        result_iadd_156821 = python_operator(stypy.reporting.localization.Localization(__file__, 745, 12), '+=', msg_156819, str_156820)
        # Assigning a type to the variable 'msg' (line 745)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 745, 12), 'msg', result_iadd_156821)
        
        
        # Call to warn(...): (line 746)
        # Processing the call arguments (line 746)
        # Getting the type of 'msg' (line 746)
        msg_156824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 746, 26), 'msg', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 746)
        tuple_156825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 746, 33), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 746)
        # Adding element type (line 746)
        
        # Call to len(...): (line 746)
        # Processing the call arguments (line 746)
        # Getting the type of 'vartypes' (line 746)
        vartypes_156827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 746, 37), 'vartypes', False)
        # Processing the call keyword arguments (line 746)
        kwargs_156828 = {}
        # Getting the type of 'len' (line 746)
        len_156826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 746, 33), 'len', False)
        # Calling len(args, kwargs) (line 746)
        len_call_result_156829 = invoke(stypy.reporting.localization.Localization(__file__, 746, 33), len_156826, *[vartypes_156827], **kwargs_156828)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 746, 33), tuple_156825, len_call_result_156829)
        # Adding element type (line 746)
        # Getting the type of 'nfields' (line 746)
        nfields_156830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 746, 48), 'nfields', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 746, 33), tuple_156825, nfields_156830)
        
        # Applying the binary operator '%' (line 746)
        result_mod_156831 = python_operator(stypy.reporting.localization.Localization(__file__, 746, 26), '%', msg_156824, tuple_156825)
        
        # Processing the call keyword arguments (line 746)
        kwargs_156832 = {}
        # Getting the type of 'warnings' (line 746)
        warnings_156822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 746, 12), 'warnings', False)
        # Obtaining the member 'warn' of a type (line 746)
        warn_156823 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 746, 12), warnings_156822, 'warn')
        # Calling warn(args, kwargs) (line 746)
        warn_call_result_156833 = invoke(stypy.reporting.localization.Localization(__file__, 746, 12), warn_156823, *[result_mod_156831], **kwargs_156832)
        
        
        # Assigning a Call to a Name (line 747):
        
        # Assigning a Call to a Name (line 747):
        
        # Call to _guessvartypes(...): (line 747)
        # Processing the call arguments (line 747)
        
        # Obtaining the type of the subscript
        int_156835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 747, 49), 'int')
        # Getting the type of '_variables' (line 747)
        _variables_156836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 747, 38), '_variables', False)
        # Obtaining the member '__getitem__' of a type (line 747)
        getitem___156837 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 747, 38), _variables_156836, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 747)
        subscript_call_result_156838 = invoke(stypy.reporting.localization.Localization(__file__, 747, 38), getitem___156837, int_156835)
        
        # Processing the call keyword arguments (line 747)
        kwargs_156839 = {}
        # Getting the type of '_guessvartypes' (line 747)
        _guessvartypes_156834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 747, 23), '_guessvartypes', False)
        # Calling _guessvartypes(args, kwargs) (line 747)
        _guessvartypes_call_result_156840 = invoke(stypy.reporting.localization.Localization(__file__, 747, 23), _guessvartypes_156834, *[subscript_call_result_156838], **kwargs_156839)
        
        # Assigning a type to the variable 'vartypes' (line 747)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 747, 12), 'vartypes', _guessvartypes_call_result_156840)
        # SSA join for if statement (line 743)
        module_type_store = module_type_store.join_ssa_context()
        

        if (may_be_156794 and more_types_in_union_156795):
            # SSA join for if statement (line 739)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a ListComp to a Name (line 750):
    
    # Assigning a ListComp to a Name (line 750):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to zip(...): (line 750)
    # Processing the call arguments (line 750)
    # Getting the type of 'varnames' (line 750)
    varnames_156845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 750, 39), 'varnames', False)
    # Getting the type of 'vartypes' (line 750)
    vartypes_156846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 750, 49), 'vartypes', False)
    # Processing the call keyword arguments (line 750)
    kwargs_156847 = {}
    # Getting the type of 'zip' (line 750)
    zip_156844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 750, 35), 'zip', False)
    # Calling zip(args, kwargs) (line 750)
    zip_call_result_156848 = invoke(stypy.reporting.localization.Localization(__file__, 750, 35), zip_156844, *[varnames_156845, vartypes_156846], **kwargs_156847)
    
    comprehension_156849 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 750, 14), zip_call_result_156848)
    # Assigning a type to the variable 'n' (line 750)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 750, 14), 'n', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 750, 14), comprehension_156849))
    # Assigning a type to the variable 'f' (line 750)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 750, 14), 'f', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 750, 14), comprehension_156849))
    
    # Obtaining an instance of the builtin type 'tuple' (line 750)
    tuple_156841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 750, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 750)
    # Adding element type (line 750)
    # Getting the type of 'n' (line 750)
    n_156842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 750, 15), 'n')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 750, 15), tuple_156841, n_156842)
    # Adding element type (line 750)
    # Getting the type of 'f' (line 750)
    f_156843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 750, 18), 'f')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 750, 15), tuple_156841, f_156843)
    
    list_156850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 750, 14), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 750, 14), list_156850, tuple_156841)
    # Assigning a type to the variable 'mdescr' (line 750)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 750, 4), 'mdescr', list_156850)
    
    # Assigning a ListComp to a Name (line 751):
    
    # Assigning a ListComp to a Name (line 751):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'vartypes' (line 751)
    vartypes_156856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 751, 48), 'vartypes')
    comprehension_156857 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 751, 14), vartypes_156856)
    # Assigning a type to the variable 'f' (line 751)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 751, 14), 'f', comprehension_156857)
    
    # Call to default_fill_value(...): (line 751)
    # Processing the call arguments (line 751)
    # Getting the type of 'f' (line 751)
    f_156853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 751, 36), 'f', False)
    # Processing the call keyword arguments (line 751)
    kwargs_156854 = {}
    # Getting the type of 'ma' (line 751)
    ma_156851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 751, 14), 'ma', False)
    # Obtaining the member 'default_fill_value' of a type (line 751)
    default_fill_value_156852 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 751, 14), ma_156851, 'default_fill_value')
    # Calling default_fill_value(args, kwargs) (line 751)
    default_fill_value_call_result_156855 = invoke(stypy.reporting.localization.Localization(__file__, 751, 14), default_fill_value_156852, *[f_156853], **kwargs_156854)
    
    list_156858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 751, 14), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 751, 14), list_156858, default_fill_value_call_result_156855)
    # Assigning a type to the variable 'mfillv' (line 751)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 751, 4), 'mfillv', list_156858)
    
    # Assigning a Compare to a Name (line 755):
    
    # Assigning a Compare to a Name (line 755):
    
    # Getting the type of '_variables' (line 755)
    _variables_156859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 755, 13), '_variables')
    # Obtaining the member 'T' of a type (line 755)
    T_156860 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 755, 13), _variables_156859, 'T')
    # Getting the type of 'missingchar' (line 755)
    missingchar_156861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 755, 29), 'missingchar')
    # Applying the binary operator '==' (line 755)
    result_eq_156862 = python_operator(stypy.reporting.localization.Localization(__file__, 755, 13), '==', T_156860, missingchar_156861)
    
    # Assigning a type to the variable '_mask' (line 755)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 755, 4), '_mask', result_eq_156862)
    
    # Assigning a ListComp to a Name (line 756):
    
    # Assigning a ListComp to a Name (line 756):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to zip(...): (line 757)
    # Processing the call arguments (line 757)
    # Getting the type of '_variables' (line 757)
    _variables_156874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 757, 41), '_variables', False)
    # Obtaining the member 'T' of a type (line 757)
    T_156875 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 757, 41), _variables_156874, 'T')
    # Getting the type of '_mask' (line 757)
    _mask_156876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 757, 55), '_mask', False)
    # Getting the type of 'vartypes' (line 757)
    vartypes_156877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 757, 62), 'vartypes', False)
    # Getting the type of 'mfillv' (line 757)
    mfillv_156878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 757, 72), 'mfillv', False)
    # Processing the call keyword arguments (line 757)
    kwargs_156879 = {}
    # Getting the type of 'zip' (line 757)
    zip_156873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 757, 37), 'zip', False)
    # Calling zip(args, kwargs) (line 757)
    zip_call_result_156880 = invoke(stypy.reporting.localization.Localization(__file__, 757, 37), zip_156873, *[T_156875, _mask_156876, vartypes_156877, mfillv_156878], **kwargs_156879)
    
    comprehension_156881 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 756, 17), zip_call_result_156880)
    # Assigning a type to the variable 'a' (line 756)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 756, 17), 'a', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 756, 17), comprehension_156881))
    # Assigning a type to the variable 'm' (line 756)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 756, 17), 'm', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 756, 17), comprehension_156881))
    # Assigning a type to the variable 't' (line 756)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 756, 17), 't', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 756, 17), comprehension_156881))
    # Assigning a type to the variable 'f' (line 756)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 756, 17), 'f', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 756, 17), comprehension_156881))
    
    # Call to masked_array(...): (line 756)
    # Processing the call arguments (line 756)
    # Getting the type of 'a' (line 756)
    a_156864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 756, 30), 'a', False)
    # Processing the call keyword arguments (line 756)
    # Getting the type of 'm' (line 756)
    m_156865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 756, 38), 'm', False)
    keyword_156866 = m_156865
    # Getting the type of 't' (line 756)
    t_156867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 756, 47), 't', False)
    keyword_156868 = t_156867
    # Getting the type of 'f' (line 756)
    f_156869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 756, 61), 'f', False)
    keyword_156870 = f_156869
    kwargs_156871 = {'dtype': keyword_156868, 'fill_value': keyword_156870, 'mask': keyword_156866}
    # Getting the type of 'masked_array' (line 756)
    masked_array_156863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 756, 17), 'masked_array', False)
    # Calling masked_array(args, kwargs) (line 756)
    masked_array_call_result_156872 = invoke(stypy.reporting.localization.Localization(__file__, 756, 17), masked_array_156863, *[a_156864], **kwargs_156871)
    
    list_156882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 756, 17), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 756, 17), list_156882, masked_array_call_result_156872)
    # Assigning a type to the variable '_datalist' (line 756)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 756, 4), '_datalist', list_156882)
    
    # Call to fromarrays(...): (line 759)
    # Processing the call arguments (line 759)
    # Getting the type of '_datalist' (line 759)
    _datalist_156884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 759, 22), '_datalist', False)
    # Processing the call keyword arguments (line 759)
    # Getting the type of 'mdescr' (line 759)
    mdescr_156885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 759, 39), 'mdescr', False)
    keyword_156886 = mdescr_156885
    kwargs_156887 = {'dtype': keyword_156886}
    # Getting the type of 'fromarrays' (line 759)
    fromarrays_156883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 759, 11), 'fromarrays', False)
    # Calling fromarrays(args, kwargs) (line 759)
    fromarrays_call_result_156888 = invoke(stypy.reporting.localization.Localization(__file__, 759, 11), fromarrays_156883, *[_datalist_156884], **kwargs_156887)
    
    # Assigning a type to the variable 'stypy_return_type' (line 759)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 759, 4), 'stypy_return_type', fromarrays_call_result_156888)
    
    # ################# End of 'fromtextfile(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'fromtextfile' in the type store
    # Getting the type of 'stypy_return_type' (line 694)
    stypy_return_type_156889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_156889)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'fromtextfile'
    return stypy_return_type_156889

# Assigning a type to the variable 'fromtextfile' (line 694)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 694, 0), 'fromtextfile', fromtextfile)

@norecursion
def addfield(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 762)
    None_156890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 762, 45), 'None')
    defaults = [None_156890]
    # Create a new context for function 'addfield'
    module_type_store = module_type_store.open_function_context('addfield', 762, 0, False)
    
    # Passed parameters checking function
    addfield.stypy_localization = localization
    addfield.stypy_type_of_self = None
    addfield.stypy_type_store = module_type_store
    addfield.stypy_function_name = 'addfield'
    addfield.stypy_param_names_list = ['mrecord', 'newfield', 'newfieldname']
    addfield.stypy_varargs_param_name = None
    addfield.stypy_kwargs_param_name = None
    addfield.stypy_call_defaults = defaults
    addfield.stypy_call_varargs = varargs
    addfield.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'addfield', ['mrecord', 'newfield', 'newfieldname'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'addfield', localization, ['mrecord', 'newfield', 'newfieldname'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'addfield(...)' code ##################

    str_156891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 769, (-1)), 'str', "Adds a new field to the masked record array\n\n    Uses `newfield` as data and `newfieldname` as name. If `newfieldname`\n    is None, the new field name is set to 'fi', where `i` is the number of\n    existing fields.\n\n    ")
    
    # Assigning a Attribute to a Name (line 770):
    
    # Assigning a Attribute to a Name (line 770):
    # Getting the type of 'mrecord' (line 770)
    mrecord_156892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 770, 12), 'mrecord')
    # Obtaining the member '_data' of a type (line 770)
    _data_156893 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 770, 12), mrecord_156892, '_data')
    # Assigning a type to the variable '_data' (line 770)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 770, 4), '_data', _data_156893)
    
    # Assigning a Attribute to a Name (line 771):
    
    # Assigning a Attribute to a Name (line 771):
    # Getting the type of 'mrecord' (line 771)
    mrecord_156894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 771, 12), 'mrecord')
    # Obtaining the member '_mask' of a type (line 771)
    _mask_156895 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 771, 12), mrecord_156894, '_mask')
    # Assigning a type to the variable '_mask' (line 771)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 771, 4), '_mask', _mask_156895)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'newfieldname' (line 772)
    newfieldname_156896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 772, 7), 'newfieldname')
    # Getting the type of 'None' (line 772)
    None_156897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 772, 23), 'None')
    # Applying the binary operator 'is' (line 772)
    result_is__156898 = python_operator(stypy.reporting.localization.Localization(__file__, 772, 7), 'is', newfieldname_156896, None_156897)
    
    
    # Getting the type of 'newfieldname' (line 772)
    newfieldname_156899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 772, 31), 'newfieldname')
    # Getting the type of 'reserved_fields' (line 772)
    reserved_fields_156900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 772, 47), 'reserved_fields')
    # Applying the binary operator 'in' (line 772)
    result_contains_156901 = python_operator(stypy.reporting.localization.Localization(__file__, 772, 31), 'in', newfieldname_156899, reserved_fields_156900)
    
    # Applying the binary operator 'or' (line 772)
    result_or_keyword_156902 = python_operator(stypy.reporting.localization.Localization(__file__, 772, 7), 'or', result_is__156898, result_contains_156901)
    
    # Testing the type of an if condition (line 772)
    if_condition_156903 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 772, 4), result_or_keyword_156902)
    # Assigning a type to the variable 'if_condition_156903' (line 772)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 772, 4), 'if_condition_156903', if_condition_156903)
    # SSA begins for if statement (line 772)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 773):
    
    # Assigning a BinOp to a Name (line 773):
    str_156904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 773, 23), 'str', 'f%i')
    
    # Call to len(...): (line 773)
    # Processing the call arguments (line 773)
    # Getting the type of '_data' (line 773)
    _data_156906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 773, 35), '_data', False)
    # Obtaining the member 'dtype' of a type (line 773)
    dtype_156907 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 773, 35), _data_156906, 'dtype')
    # Processing the call keyword arguments (line 773)
    kwargs_156908 = {}
    # Getting the type of 'len' (line 773)
    len_156905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 773, 31), 'len', False)
    # Calling len(args, kwargs) (line 773)
    len_call_result_156909 = invoke(stypy.reporting.localization.Localization(__file__, 773, 31), len_156905, *[dtype_156907], **kwargs_156908)
    
    # Applying the binary operator '%' (line 773)
    result_mod_156910 = python_operator(stypy.reporting.localization.Localization(__file__, 773, 23), '%', str_156904, len_call_result_156909)
    
    # Assigning a type to the variable 'newfieldname' (line 773)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 773, 8), 'newfieldname', result_mod_156910)
    # SSA join for if statement (line 772)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 774):
    
    # Assigning a Call to a Name (line 774):
    
    # Call to array(...): (line 774)
    # Processing the call arguments (line 774)
    # Getting the type of 'newfield' (line 774)
    newfield_156913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 774, 24), 'newfield', False)
    # Processing the call keyword arguments (line 774)
    kwargs_156914 = {}
    # Getting the type of 'ma' (line 774)
    ma_156911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 774, 15), 'ma', False)
    # Obtaining the member 'array' of a type (line 774)
    array_156912 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 774, 15), ma_156911, 'array')
    # Calling array(args, kwargs) (line 774)
    array_call_result_156915 = invoke(stypy.reporting.localization.Localization(__file__, 774, 15), array_156912, *[newfield_156913], **kwargs_156914)
    
    # Assigning a type to the variable 'newfield' (line 774)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 774, 4), 'newfield', array_call_result_156915)
    
    # Assigning a Call to a Name (line 777):
    
    # Assigning a Call to a Name (line 777):
    
    # Call to dtype(...): (line 777)
    # Processing the call arguments (line 777)
    # Getting the type of '_data' (line 777)
    _data_156918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 777, 24), '_data', False)
    # Obtaining the member 'dtype' of a type (line 777)
    dtype_156919 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 777, 24), _data_156918, 'dtype')
    # Obtaining the member 'descr' of a type (line 777)
    descr_156920 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 777, 24), dtype_156919, 'descr')
    
    # Obtaining an instance of the builtin type 'list' (line 777)
    list_156921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 777, 44), 'list')
    # Adding type elements to the builtin type 'list' instance (line 777)
    # Adding element type (line 777)
    
    # Obtaining an instance of the builtin type 'tuple' (line 777)
    tuple_156922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 777, 46), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 777)
    # Adding element type (line 777)
    # Getting the type of 'newfieldname' (line 777)
    newfieldname_156923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 777, 46), 'newfieldname', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 777, 46), tuple_156922, newfieldname_156923)
    # Adding element type (line 777)
    # Getting the type of 'newfield' (line 777)
    newfield_156924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 777, 60), 'newfield', False)
    # Obtaining the member 'dtype' of a type (line 777)
    dtype_156925 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 777, 60), newfield_156924, 'dtype')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 777, 46), tuple_156922, dtype_156925)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 777, 44), list_156921, tuple_156922)
    
    # Applying the binary operator '+' (line 777)
    result_add_156926 = python_operator(stypy.reporting.localization.Localization(__file__, 777, 24), '+', descr_156920, list_156921)
    
    # Processing the call keyword arguments (line 777)
    kwargs_156927 = {}
    # Getting the type of 'np' (line 777)
    np_156916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 777, 15), 'np', False)
    # Obtaining the member 'dtype' of a type (line 777)
    dtype_156917 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 777, 15), np_156916, 'dtype')
    # Calling dtype(args, kwargs) (line 777)
    dtype_call_result_156928 = invoke(stypy.reporting.localization.Localization(__file__, 777, 15), dtype_156917, *[result_add_156926], **kwargs_156927)
    
    # Assigning a type to the variable 'newdtype' (line 777)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 777, 4), 'newdtype', dtype_call_result_156928)
    
    # Assigning a Call to a Name (line 778):
    
    # Assigning a Call to a Name (line 778):
    
    # Call to recarray(...): (line 778)
    # Processing the call arguments (line 778)
    # Getting the type of '_data' (line 778)
    _data_156930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 778, 23), '_data', False)
    # Obtaining the member 'shape' of a type (line 778)
    shape_156931 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 778, 23), _data_156930, 'shape')
    # Getting the type of 'newdtype' (line 778)
    newdtype_156932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 778, 36), 'newdtype', False)
    # Processing the call keyword arguments (line 778)
    kwargs_156933 = {}
    # Getting the type of 'recarray' (line 778)
    recarray_156929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 778, 14), 'recarray', False)
    # Calling recarray(args, kwargs) (line 778)
    recarray_call_result_156934 = invoke(stypy.reporting.localization.Localization(__file__, 778, 14), recarray_156929, *[shape_156931, newdtype_156932], **kwargs_156933)
    
    # Assigning a type to the variable 'newdata' (line 778)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 778, 4), 'newdata', recarray_call_result_156934)
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to values(...): (line 781)
    # Processing the call keyword arguments (line 781)
    kwargs_156949 = {}
    # Getting the type of '_data' (line 781)
    _data_156945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 781, 18), '_data', False)
    # Obtaining the member 'dtype' of a type (line 781)
    dtype_156946 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 781, 18), _data_156945, 'dtype')
    # Obtaining the member 'fields' of a type (line 781)
    fields_156947 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 781, 18), dtype_156946, 'fields')
    # Obtaining the member 'values' of a type (line 781)
    values_156948 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 781, 18), fields_156947, 'values')
    # Calling values(args, kwargs) (line 781)
    values_call_result_156950 = invoke(stypy.reporting.localization.Localization(__file__, 781, 18), values_156948, *[], **kwargs_156949)
    
    comprehension_156951 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 780, 5), values_call_result_156950)
    # Assigning a type to the variable 'f' (line 780)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 780, 5), 'f', comprehension_156951)
    
    # Call to setfield(...): (line 780)
    # Processing the call arguments (line 780)
    
    # Call to getfield(...): (line 780)
    # Getting the type of 'f' (line 780)
    f_156939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 780, 38), 'f', False)
    # Processing the call keyword arguments (line 780)
    kwargs_156940 = {}
    # Getting the type of '_data' (line 780)
    _data_156937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 780, 22), '_data', False)
    # Obtaining the member 'getfield' of a type (line 780)
    getfield_156938 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 780, 22), _data_156937, 'getfield')
    # Calling getfield(args, kwargs) (line 780)
    getfield_call_result_156941 = invoke(stypy.reporting.localization.Localization(__file__, 780, 22), getfield_156938, *[f_156939], **kwargs_156940)
    
    # Getting the type of 'f' (line 780)
    f_156942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 780, 43), 'f', False)
    # Processing the call keyword arguments (line 780)
    kwargs_156943 = {}
    # Getting the type of 'newdata' (line 780)
    newdata_156935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 780, 5), 'newdata', False)
    # Obtaining the member 'setfield' of a type (line 780)
    setfield_156936 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 780, 5), newdata_156935, 'setfield')
    # Calling setfield(args, kwargs) (line 780)
    setfield_call_result_156944 = invoke(stypy.reporting.localization.Localization(__file__, 780, 5), setfield_156936, *[getfield_call_result_156941, f_156942], **kwargs_156943)
    
    list_156952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 780, 5), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 780, 5), list_156952, setfield_call_result_156944)
    
    # Call to setfield(...): (line 783)
    # Processing the call arguments (line 783)
    # Getting the type of 'newfield' (line 783)
    newfield_156955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 783, 21), 'newfield', False)
    # Obtaining the member '_data' of a type (line 783)
    _data_156956 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 783, 21), newfield_156955, '_data')
    
    # Obtaining the type of the subscript
    # Getting the type of 'newfieldname' (line 783)
    newfieldname_156957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 783, 59), 'newfieldname', False)
    # Getting the type of 'newdata' (line 783)
    newdata_156958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 783, 38), 'newdata', False)
    # Obtaining the member 'dtype' of a type (line 783)
    dtype_156959 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 783, 38), newdata_156958, 'dtype')
    # Obtaining the member 'fields' of a type (line 783)
    fields_156960 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 783, 38), dtype_156959, 'fields')
    # Obtaining the member '__getitem__' of a type (line 783)
    getitem___156961 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 783, 38), fields_156960, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 783)
    subscript_call_result_156962 = invoke(stypy.reporting.localization.Localization(__file__, 783, 38), getitem___156961, newfieldname_156957)
    
    # Processing the call keyword arguments (line 783)
    kwargs_156963 = {}
    # Getting the type of 'newdata' (line 783)
    newdata_156953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 783, 4), 'newdata', False)
    # Obtaining the member 'setfield' of a type (line 783)
    setfield_156954 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 783, 4), newdata_156953, 'setfield')
    # Calling setfield(args, kwargs) (line 783)
    setfield_call_result_156964 = invoke(stypy.reporting.localization.Localization(__file__, 783, 4), setfield_156954, *[_data_156956, subscript_call_result_156962], **kwargs_156963)
    
    
    # Assigning a Call to a Name (line 784):
    
    # Assigning a Call to a Name (line 784):
    
    # Call to view(...): (line 784)
    # Processing the call arguments (line 784)
    # Getting the type of 'MaskedRecords' (line 784)
    MaskedRecords_156967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 784, 27), 'MaskedRecords', False)
    # Processing the call keyword arguments (line 784)
    kwargs_156968 = {}
    # Getting the type of 'newdata' (line 784)
    newdata_156965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 784, 14), 'newdata', False)
    # Obtaining the member 'view' of a type (line 784)
    view_156966 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 784, 14), newdata_156965, 'view')
    # Calling view(args, kwargs) (line 784)
    view_call_result_156969 = invoke(stypy.reporting.localization.Localization(__file__, 784, 14), view_156966, *[MaskedRecords_156967], **kwargs_156968)
    
    # Assigning a type to the variable 'newdata' (line 784)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 784, 4), 'newdata', view_call_result_156969)
    
    # Assigning a Call to a Name (line 787):
    
    # Assigning a Call to a Name (line 787):
    
    # Call to dtype(...): (line 787)
    # Processing the call arguments (line 787)
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'newdtype' (line 787)
    newdtype_156975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 787, 46), 'newdtype', False)
    # Obtaining the member 'names' of a type (line 787)
    names_156976 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 787, 46), newdtype_156975, 'names')
    comprehension_156977 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 787, 26), names_156976)
    # Assigning a type to the variable 'n' (line 787)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 787, 26), 'n', comprehension_156977)
    
    # Obtaining an instance of the builtin type 'tuple' (line 787)
    tuple_156972 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 787, 27), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 787)
    # Adding element type (line 787)
    # Getting the type of 'n' (line 787)
    n_156973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 787, 27), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 787, 27), tuple_156972, n_156973)
    # Adding element type (line 787)
    # Getting the type of 'bool_' (line 787)
    bool__156974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 787, 30), 'bool_', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 787, 27), tuple_156972, bool__156974)
    
    list_156978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 787, 26), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 787, 26), list_156978, tuple_156972)
    # Processing the call keyword arguments (line 787)
    kwargs_156979 = {}
    # Getting the type of 'np' (line 787)
    np_156970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 787, 16), 'np', False)
    # Obtaining the member 'dtype' of a type (line 787)
    dtype_156971 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 787, 16), np_156970, 'dtype')
    # Calling dtype(args, kwargs) (line 787)
    dtype_call_result_156980 = invoke(stypy.reporting.localization.Localization(__file__, 787, 16), dtype_156971, *[list_156978], **kwargs_156979)
    
    # Assigning a type to the variable 'newmdtype' (line 787)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 787, 4), 'newmdtype', dtype_call_result_156980)
    
    # Assigning a Call to a Name (line 788):
    
    # Assigning a Call to a Name (line 788):
    
    # Call to recarray(...): (line 788)
    # Processing the call arguments (line 788)
    # Getting the type of '_data' (line 788)
    _data_156982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 788, 23), '_data', False)
    # Obtaining the member 'shape' of a type (line 788)
    shape_156983 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 788, 23), _data_156982, 'shape')
    # Getting the type of 'newmdtype' (line 788)
    newmdtype_156984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 788, 36), 'newmdtype', False)
    # Processing the call keyword arguments (line 788)
    kwargs_156985 = {}
    # Getting the type of 'recarray' (line 788)
    recarray_156981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 788, 14), 'recarray', False)
    # Calling recarray(args, kwargs) (line 788)
    recarray_call_result_156986 = invoke(stypy.reporting.localization.Localization(__file__, 788, 14), recarray_156981, *[shape_156983, newmdtype_156984], **kwargs_156985)
    
    # Assigning a type to the variable 'newmask' (line 788)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 788, 4), 'newmask', recarray_call_result_156986)
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to values(...): (line 791)
    # Processing the call keyword arguments (line 791)
    kwargs_157001 = {}
    # Getting the type of '_mask' (line 791)
    _mask_156997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 791, 18), '_mask', False)
    # Obtaining the member 'dtype' of a type (line 791)
    dtype_156998 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 791, 18), _mask_156997, 'dtype')
    # Obtaining the member 'fields' of a type (line 791)
    fields_156999 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 791, 18), dtype_156998, 'fields')
    # Obtaining the member 'values' of a type (line 791)
    values_157000 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 791, 18), fields_156999, 'values')
    # Calling values(args, kwargs) (line 791)
    values_call_result_157002 = invoke(stypy.reporting.localization.Localization(__file__, 791, 18), values_157000, *[], **kwargs_157001)
    
    comprehension_157003 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 790, 5), values_call_result_157002)
    # Assigning a type to the variable 'f' (line 790)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 790, 5), 'f', comprehension_157003)
    
    # Call to setfield(...): (line 790)
    # Processing the call arguments (line 790)
    
    # Call to getfield(...): (line 790)
    # Getting the type of 'f' (line 790)
    f_156991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 790, 38), 'f', False)
    # Processing the call keyword arguments (line 790)
    kwargs_156992 = {}
    # Getting the type of '_mask' (line 790)
    _mask_156989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 790, 22), '_mask', False)
    # Obtaining the member 'getfield' of a type (line 790)
    getfield_156990 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 790, 22), _mask_156989, 'getfield')
    # Calling getfield(args, kwargs) (line 790)
    getfield_call_result_156993 = invoke(stypy.reporting.localization.Localization(__file__, 790, 22), getfield_156990, *[f_156991], **kwargs_156992)
    
    # Getting the type of 'f' (line 790)
    f_156994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 790, 43), 'f', False)
    # Processing the call keyword arguments (line 790)
    kwargs_156995 = {}
    # Getting the type of 'newmask' (line 790)
    newmask_156987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 790, 5), 'newmask', False)
    # Obtaining the member 'setfield' of a type (line 790)
    setfield_156988 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 790, 5), newmask_156987, 'setfield')
    # Calling setfield(args, kwargs) (line 790)
    setfield_call_result_156996 = invoke(stypy.reporting.localization.Localization(__file__, 790, 5), setfield_156988, *[getfield_call_result_156993, f_156994], **kwargs_156995)
    
    list_157004 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 790, 5), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 790, 5), list_157004, setfield_call_result_156996)
    
    # Call to setfield(...): (line 793)
    # Processing the call arguments (line 793)
    
    # Call to getmaskarray(...): (line 793)
    # Processing the call arguments (line 793)
    # Getting the type of 'newfield' (line 793)
    newfield_157008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 793, 34), 'newfield', False)
    # Processing the call keyword arguments (line 793)
    kwargs_157009 = {}
    # Getting the type of 'getmaskarray' (line 793)
    getmaskarray_157007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 793, 21), 'getmaskarray', False)
    # Calling getmaskarray(args, kwargs) (line 793)
    getmaskarray_call_result_157010 = invoke(stypy.reporting.localization.Localization(__file__, 793, 21), getmaskarray_157007, *[newfield_157008], **kwargs_157009)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'newfieldname' (line 794)
    newfieldname_157011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 794, 43), 'newfieldname', False)
    # Getting the type of 'newmask' (line 794)
    newmask_157012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 794, 22), 'newmask', False)
    # Obtaining the member 'dtype' of a type (line 794)
    dtype_157013 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 794, 22), newmask_157012, 'dtype')
    # Obtaining the member 'fields' of a type (line 794)
    fields_157014 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 794, 22), dtype_157013, 'fields')
    # Obtaining the member '__getitem__' of a type (line 794)
    getitem___157015 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 794, 22), fields_157014, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 794)
    subscript_call_result_157016 = invoke(stypy.reporting.localization.Localization(__file__, 794, 22), getitem___157015, newfieldname_157011)
    
    # Processing the call keyword arguments (line 793)
    kwargs_157017 = {}
    # Getting the type of 'newmask' (line 793)
    newmask_157005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 793, 4), 'newmask', False)
    # Obtaining the member 'setfield' of a type (line 793)
    setfield_157006 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 793, 4), newmask_157005, 'setfield')
    # Calling setfield(args, kwargs) (line 793)
    setfield_call_result_157018 = invoke(stypy.reporting.localization.Localization(__file__, 793, 4), setfield_157006, *[getmaskarray_call_result_157010, subscript_call_result_157016], **kwargs_157017)
    
    
    # Assigning a Name to a Attribute (line 795):
    
    # Assigning a Name to a Attribute (line 795):
    # Getting the type of 'newmask' (line 795)
    newmask_157019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 795, 20), 'newmask')
    # Getting the type of 'newdata' (line 795)
    newdata_157020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 795, 4), 'newdata')
    # Setting the type of the member '_mask' of a type (line 795)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 795, 4), newdata_157020, '_mask', newmask_157019)
    # Getting the type of 'newdata' (line 796)
    newdata_157021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 796, 11), 'newdata')
    # Assigning a type to the variable 'stypy_return_type' (line 796)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 796, 4), 'stypy_return_type', newdata_157021)
    
    # ################# End of 'addfield(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'addfield' in the type store
    # Getting the type of 'stypy_return_type' (line 762)
    stypy_return_type_157022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 762, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_157022)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'addfield'
    return stypy_return_type_157022

# Assigning a type to the variable 'addfield' (line 762)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 762, 0), 'addfield', addfield)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
