
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: numerictypes: Define the numeric type objects
3: 
4: This module is designed so "from numerictypes import \\*" is safe.
5: Exported symbols include:
6: 
7:   Dictionary with all registered number types (including aliases):
8:     typeDict
9: 
10:   Type objects (not all will be available, depends on platform):
11:       see variable sctypes for which ones you have
12: 
13:     Bit-width names
14: 
15:     int8 int16 int32 int64 int128
16:     uint8 uint16 uint32 uint64 uint128
17:     float16 float32 float64 float96 float128 float256
18:     complex32 complex64 complex128 complex192 complex256 complex512
19:     datetime64 timedelta64
20: 
21:     c-based names
22: 
23:     bool_
24: 
25:     object_
26: 
27:     void, str_, unicode_
28: 
29:     byte, ubyte,
30:     short, ushort
31:     intc, uintc,
32:     intp, uintp,
33:     int_, uint,
34:     longlong, ulonglong,
35: 
36:     single, csingle,
37:     float_, complex_,
38:     longfloat, clongfloat,
39: 
40:    As part of the type-hierarchy:    xx -- is bit-width
41: 
42:    generic
43:      +-> bool_                                  (kind=b)
44:      +-> number                                 (kind=i)
45:      |     integer
46:      |     signedinteger   (intxx)
47:      |     byte
48:      |     short
49:      |     intc
50:      |     intp           int0
51:      |     int_
52:      |     longlong
53:      +-> unsignedinteger  (uintxx)              (kind=u)
54:      |     ubyte
55:      |     ushort
56:      |     uintc
57:      |     uintp          uint0
58:      |     uint_
59:      |     ulonglong
60:      +-> inexact
61:      |   +-> floating           (floatxx)       (kind=f)
62:      |   |     half
63:      |   |     single
64:      |   |     float_  (double)
65:      |   |     longfloat
66:      |   \\-> complexfloating    (complexxx)     (kind=c)
67:      |         csingle  (singlecomplex)
68:      |         complex_ (cfloat, cdouble)
69:      |         clongfloat (longcomplex)
70:      +-> flexible
71:      |     character
72:      |     void                                 (kind=V)
73:      |
74:      |     str_     (string_, bytes_)           (kind=S)    [Python 2]
75:      |     unicode_                             (kind=U)    [Python 2]
76:      |
77:      |     bytes_   (string_)                   (kind=S)    [Python 3]
78:      |     str_     (unicode_)                  (kind=U)    [Python 3]
79:      |
80:      \\-> object_ (not used much)                (kind=O)
81: 
82: '''
83: from __future__ import division, absolute_import, print_function
84: 
85: import types as _types
86: import sys
87: import numbers
88: 
89: from numpy.compat import bytes, long
90: from numpy.core.multiarray import (
91:         typeinfo, ndarray, array, empty, dtype, datetime_data,
92:         datetime_as_string, busday_offset, busday_count, is_busday,
93:         busdaycalendar
94:         )
95: 
96: 
97: # we add more at the bottom
98: __all__ = ['sctypeDict', 'sctypeNA', 'typeDict', 'typeNA', 'sctypes',
99:            'ScalarType', 'obj2sctype', 'cast', 'nbytes', 'sctype2char',
100:            'maximum_sctype', 'issctype', 'typecodes', 'find_common_type',
101:            'issubdtype', 'datetime_data', 'datetime_as_string',
102:            'busday_offset', 'busday_count', 'is_busday', 'busdaycalendar',
103:            ]
104: 
105: 
106: # we don't export these for import *, but we do want them accessible
107: # as numerictypes.bool, etc.
108: if sys.version_info[0] >= 3:
109:     from builtins import bool, int, float, complex, object, str
110:     unicode = str
111: else:
112:     from __builtin__ import bool, int, float, complex, object, unicode, str
113: 
114: 
115: # String-handling utilities to avoid locale-dependence.
116: 
117: # "import string" is costly to import!
118: # Construct the translation tables directly
119: #   "A" = chr(65), "a" = chr(97)
120: _all_chars = [chr(_m) for _m in range(256)]
121: _ascii_upper = _all_chars[65:65+26]
122: _ascii_lower = _all_chars[97:97+26]
123: LOWER_TABLE = "".join(_all_chars[:65] + _ascii_lower + _all_chars[65+26:])
124: UPPER_TABLE = "".join(_all_chars[:97] + _ascii_upper + _all_chars[97+26:])
125: 
126: 
127: def english_lower(s):
128:     ''' Apply English case rules to convert ASCII strings to all lower case.
129: 
130:     This is an internal utility function to replace calls to str.lower() such
131:     that we can avoid changing behavior with changing locales. In particular,
132:     Turkish has distinct dotted and dotless variants of the Latin letter "I" in
133:     both lowercase and uppercase. Thus, "I".lower() != "i" in a "tr" locale.
134: 
135:     Parameters
136:     ----------
137:     s : str
138: 
139:     Returns
140:     -------
141:     lowered : str
142: 
143:     Examples
144:     --------
145:     >>> from numpy.core.numerictypes import english_lower
146:     >>> english_lower('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_')
147:     'abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz0123456789_'
148:     >>> english_lower('')
149:     ''
150:     '''
151:     lowered = s.translate(LOWER_TABLE)
152:     return lowered
153: 
154: def english_upper(s):
155:     ''' Apply English case rules to convert ASCII strings to all upper case.
156: 
157:     This is an internal utility function to replace calls to str.upper() such
158:     that we can avoid changing behavior with changing locales. In particular,
159:     Turkish has distinct dotted and dotless variants of the Latin letter "I" in
160:     both lowercase and uppercase. Thus, "i".upper() != "I" in a "tr" locale.
161: 
162:     Parameters
163:     ----------
164:     s : str
165: 
166:     Returns
167:     -------
168:     uppered : str
169: 
170:     Examples
171:     --------
172:     >>> from numpy.core.numerictypes import english_upper
173:     >>> english_upper('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_')
174:     'ABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_'
175:     >>> english_upper('')
176:     ''
177:     '''
178:     uppered = s.translate(UPPER_TABLE)
179:     return uppered
180: 
181: def english_capitalize(s):
182:     ''' Apply English case rules to convert the first character of an ASCII
183:     string to upper case.
184: 
185:     This is an internal utility function to replace calls to str.capitalize()
186:     such that we can avoid changing behavior with changing locales.
187: 
188:     Parameters
189:     ----------
190:     s : str
191: 
192:     Returns
193:     -------
194:     capitalized : str
195: 
196:     Examples
197:     --------
198:     >>> from numpy.core.numerictypes import english_capitalize
199:     >>> english_capitalize('int8')
200:     'Int8'
201:     >>> english_capitalize('Int8')
202:     'Int8'
203:     >>> english_capitalize('')
204:     ''
205:     '''
206:     if s:
207:         return english_upper(s[0]) + s[1:]
208:     else:
209:         return s
210: 
211: 
212: sctypeDict = {}      # Contains all leaf-node scalar types with aliases
213: sctypeNA = {}        # Contails all leaf-node types -> numarray type equivalences
214: allTypes = {}      # Collect the types we will add to the module here
215: 
216: def _evalname(name):
217:     k = 0
218:     for ch in name:
219:         if ch in '0123456789':
220:             break
221:         k += 1
222:     try:
223:         bits = int(name[k:])
224:     except ValueError:
225:         bits = 0
226:     base = name[:k]
227:     return base, bits
228: 
229: def bitname(obj):
230:     '''Return a bit-width name for a given type object'''
231:     name = obj.__name__
232:     base = ''
233:     char = ''
234:     try:
235:         if name[-1] == '_':
236:             newname = name[:-1]
237:         else:
238:             newname = name
239:         info = typeinfo[english_upper(newname)]
240:         assert(info[-1] == obj)  # sanity check
241:         bits = info[2]
242: 
243:     except KeyError:     # bit-width name
244:         base, bits = _evalname(name)
245:         char = base[0]
246: 
247:     if name == 'bool_':
248:         char = 'b'
249:         base = 'bool'
250:     elif name == 'void':
251:         char = 'V'
252:         base = 'void'
253:     elif name == 'object_':
254:         char = 'O'
255:         base = 'object'
256:         bits = 0
257:     elif name == 'datetime64':
258:         char = 'M'
259:     elif name == 'timedelta64':
260:         char = 'm'
261: 
262:     if sys.version_info[0] >= 3:
263:         if name == 'bytes_':
264:             char = 'S'
265:             base = 'bytes'
266:         elif name == 'str_':
267:             char = 'U'
268:             base = 'str'
269:     else:
270:         if name == 'string_':
271:             char = 'S'
272:             base = 'string'
273:         elif name == 'unicode_':
274:             char = 'U'
275:             base = 'unicode'
276: 
277:     bytes = bits // 8
278: 
279:     if char != '' and bytes != 0:
280:         char = "%s%d" % (char, bytes)
281: 
282:     return base, bits, char
283: 
284: 
285: def _add_types():
286:     for a in typeinfo.keys():
287:         name = english_lower(a)
288:         if isinstance(typeinfo[a], tuple):
289:             typeobj = typeinfo[a][-1]
290: 
291:             # define C-name and insert typenum and typechar references also
292:             allTypes[name] = typeobj
293:             sctypeDict[name] = typeobj
294:             sctypeDict[typeinfo[a][0]] = typeobj
295:             sctypeDict[typeinfo[a][1]] = typeobj
296: 
297:         else:  # generic class
298:             allTypes[name] = typeinfo[a]
299: _add_types()
300: 
301: def _add_aliases():
302:     for a in typeinfo.keys():
303:         name = english_lower(a)
304:         if not isinstance(typeinfo[a], tuple):
305:             continue
306:         typeobj = typeinfo[a][-1]
307:         # insert bit-width version for this class (if relevant)
308:         base, bit, char = bitname(typeobj)
309:         if base[-3:] == 'int' or char[0] in 'ui':
310:             continue
311:         if base != '':
312:             myname = "%s%d" % (base, bit)
313:             if ((name != 'longdouble' and name != 'clongdouble') or
314:                    myname not in allTypes.keys()):
315:                 allTypes[myname] = typeobj
316:                 sctypeDict[myname] = typeobj
317:                 if base == 'complex':
318:                     na_name = '%s%d' % (english_capitalize(base), bit//2)
319:                 elif base == 'bool':
320:                     na_name = english_capitalize(base)
321:                     sctypeDict[na_name] = typeobj
322:                 else:
323:                     na_name = "%s%d" % (english_capitalize(base), bit)
324:                     sctypeDict[na_name] = typeobj
325:                 sctypeNA[na_name] = typeobj
326:                 sctypeDict[na_name] = typeobj
327:                 sctypeNA[typeobj] = na_name
328:                 sctypeNA[typeinfo[a][0]] = na_name
329:         if char != '':
330:             sctypeDict[char] = typeobj
331:             sctypeNA[char] = na_name
332: _add_aliases()
333: 
334: # Integers are handled so that the int32 and int64 types should agree
335: # exactly with NPY_INT32, NPY_INT64. We need to enforce the same checking
336: # as is done in arrayobject.h where the order of getting a bit-width match
337: # is long, longlong, int, short, char.
338: def _add_integer_aliases():
339:     _ctypes = ['LONG', 'LONGLONG', 'INT', 'SHORT', 'BYTE']
340:     for ctype in _ctypes:
341:         val = typeinfo[ctype]
342:         bits = val[2]
343:         charname = 'i%d' % (bits//8,)
344:         ucharname = 'u%d' % (bits//8,)
345:         intname = 'int%d' % bits
346:         UIntname = 'UInt%d' % bits
347:         Intname = 'Int%d' % bits
348:         uval = typeinfo['U'+ctype]
349:         typeobj = val[-1]
350:         utypeobj = uval[-1]
351:         if intname not in allTypes.keys():
352:             uintname = 'uint%d' % bits
353:             allTypes[intname] = typeobj
354:             allTypes[uintname] = utypeobj
355:             sctypeDict[intname] = typeobj
356:             sctypeDict[uintname] = utypeobj
357:             sctypeDict[Intname] = typeobj
358:             sctypeDict[UIntname] = utypeobj
359:             sctypeDict[charname] = typeobj
360:             sctypeDict[ucharname] = utypeobj
361:             sctypeNA[Intname] = typeobj
362:             sctypeNA[UIntname] = utypeobj
363:             sctypeNA[charname] = typeobj
364:             sctypeNA[ucharname] = utypeobj
365:         sctypeNA[typeobj] = Intname
366:         sctypeNA[utypeobj] = UIntname
367:         sctypeNA[val[0]] = Intname
368:         sctypeNA[uval[0]] = UIntname
369: _add_integer_aliases()
370: 
371: # We use these later
372: void = allTypes['void']
373: generic = allTypes['generic']
374: 
375: #
376: # Rework the Python names (so that float and complex and int are consistent
377: #                            with Python usage)
378: #
379: def _set_up_aliases():
380:     type_pairs = [('complex_', 'cdouble'),
381:                   ('int0', 'intp'),
382:                   ('uint0', 'uintp'),
383:                   ('single', 'float'),
384:                   ('csingle', 'cfloat'),
385:                   ('singlecomplex', 'cfloat'),
386:                   ('float_', 'double'),
387:                   ('intc', 'int'),
388:                   ('uintc', 'uint'),
389:                   ('int_', 'long'),
390:                   ('uint', 'ulong'),
391:                   ('cfloat', 'cdouble'),
392:                   ('longfloat', 'longdouble'),
393:                   ('clongfloat', 'clongdouble'),
394:                   ('longcomplex', 'clongdouble'),
395:                   ('bool_', 'bool'),
396:                   ('unicode_', 'unicode'),
397:                   ('object_', 'object')]
398:     if sys.version_info[0] >= 3:
399:         type_pairs.extend([('bytes_', 'string'),
400:                            ('str_', 'unicode'),
401:                            ('string_', 'string')])
402:     else:
403:         type_pairs.extend([('str_', 'string'),
404:                            ('string_', 'string'),
405:                            ('bytes_', 'string')])
406:     for alias, t in type_pairs:
407:         allTypes[alias] = allTypes[t]
408:         sctypeDict[alias] = sctypeDict[t]
409:     # Remove aliases overriding python types and modules
410:     to_remove = ['ulong', 'object', 'unicode', 'int', 'long', 'float',
411:                  'complex', 'bool', 'string', 'datetime', 'timedelta']
412:     if sys.version_info[0] >= 3:
413:         # Py3K
414:         to_remove.append('bytes')
415:         to_remove.append('str')
416:         to_remove.remove('unicode')
417:         to_remove.remove('long')
418:     for t in to_remove:
419:         try:
420:             del allTypes[t]
421:             del sctypeDict[t]
422:         except KeyError:
423:             pass
424: _set_up_aliases()
425: 
426: # Now, construct dictionary to lookup character codes from types
427: _sctype2char_dict = {}
428: def _construct_char_code_lookup():
429:     for name in typeinfo.keys():
430:         tup = typeinfo[name]
431:         if isinstance(tup, tuple):
432:             if tup[0] not in ['p', 'P']:
433:                 _sctype2char_dict[tup[-1]] = tup[0]
434: _construct_char_code_lookup()
435: 
436: 
437: sctypes = {'int': [],
438:            'uint':[],
439:            'float':[],
440:            'complex':[],
441:            'others':[bool, object, str, unicode, void]}
442: 
443: def _add_array_type(typename, bits):
444:     try:
445:         t = allTypes['%s%d' % (typename, bits)]
446:     except KeyError:
447:         pass
448:     else:
449:         sctypes[typename].append(t)
450: 
451: def _set_array_types():
452:     ibytes = [1, 2, 4, 8, 16, 32, 64]
453:     fbytes = [2, 4, 8, 10, 12, 16, 32, 64]
454:     for bytes in ibytes:
455:         bits = 8*bytes
456:         _add_array_type('int', bits)
457:         _add_array_type('uint', bits)
458:     for bytes in fbytes:
459:         bits = 8*bytes
460:         _add_array_type('float', bits)
461:         _add_array_type('complex', 2*bits)
462:     _gi = dtype('p')
463:     if _gi.type not in sctypes['int']:
464:         indx = 0
465:         sz = _gi.itemsize
466:         _lst = sctypes['int']
467:         while (indx < len(_lst) and sz >= _lst[indx](0).itemsize):
468:             indx += 1
469:         sctypes['int'].insert(indx, _gi.type)
470:         sctypes['uint'].insert(indx, dtype('P').type)
471: _set_array_types()
472: 
473: 
474: genericTypeRank = ['bool', 'int8', 'uint8', 'int16', 'uint16',
475:                    'int32', 'uint32', 'int64', 'uint64', 'int128',
476:                    'uint128', 'float16',
477:                    'float32', 'float64', 'float80', 'float96', 'float128',
478:                    'float256',
479:                    'complex32', 'complex64', 'complex128', 'complex160',
480:                    'complex192', 'complex256', 'complex512', 'object']
481: 
482: def maximum_sctype(t):
483:     '''
484:     Return the scalar type of highest precision of the same kind as the input.
485: 
486:     Parameters
487:     ----------
488:     t : dtype or dtype specifier
489:         The input data type. This can be a `dtype` object or an object that
490:         is convertible to a `dtype`.
491: 
492:     Returns
493:     -------
494:     out : dtype
495:         The highest precision data type of the same kind (`dtype.kind`) as `t`.
496: 
497:     See Also
498:     --------
499:     obj2sctype, mintypecode, sctype2char
500:     dtype
501: 
502:     Examples
503:     --------
504:     >>> np.maximum_sctype(np.int)
505:     <type 'numpy.int64'>
506:     >>> np.maximum_sctype(np.uint8)
507:     <type 'numpy.uint64'>
508:     >>> np.maximum_sctype(np.complex)
509:     <type 'numpy.complex192'>
510: 
511:     >>> np.maximum_sctype(str)
512:     <type 'numpy.string_'>
513: 
514:     >>> np.maximum_sctype('i2')
515:     <type 'numpy.int64'>
516:     >>> np.maximum_sctype('f4')
517:     <type 'numpy.float96'>
518: 
519:     '''
520:     g = obj2sctype(t)
521:     if g is None:
522:         return t
523:     t = g
524:     name = t.__name__
525:     base, bits = _evalname(name)
526:     if bits == 0:
527:         return t
528:     else:
529:         return sctypes[base][-1]
530: 
531: try:
532:     buffer_type = _types.BufferType
533: except AttributeError:
534:     # Py3K
535:     buffer_type = memoryview
536: 
537: _python_types = {int: 'int_',
538:                  float: 'float_',
539:                  complex: 'complex_',
540:                  bool: 'bool_',
541:                  bytes: 'bytes_',
542:                  unicode: 'unicode_',
543:                  buffer_type: 'void',
544:                  }
545: 
546: if sys.version_info[0] >= 3:
547:     def _python_type(t):
548:         '''returns the type corresponding to a certain Python type'''
549:         if not isinstance(t, type):
550:             t = type(t)
551:         return allTypes[_python_types.get(t, 'object_')]
552: else:
553:     def _python_type(t):
554:         '''returns the type corresponding to a certain Python type'''
555:         if not isinstance(t, _types.TypeType):
556:             t = type(t)
557:         return allTypes[_python_types.get(t, 'object_')]
558: 
559: def issctype(rep):
560:     '''
561:     Determines whether the given object represents a scalar data-type.
562: 
563:     Parameters
564:     ----------
565:     rep : any
566:         If `rep` is an instance of a scalar dtype, True is returned. If not,
567:         False is returned.
568: 
569:     Returns
570:     -------
571:     out : bool
572:         Boolean result of check whether `rep` is a scalar dtype.
573: 
574:     See Also
575:     --------
576:     issubsctype, issubdtype, obj2sctype, sctype2char
577: 
578:     Examples
579:     --------
580:     >>> np.issctype(np.int32)
581:     True
582:     >>> np.issctype(list)
583:     False
584:     >>> np.issctype(1.1)
585:     False
586: 
587:     Strings are also a scalar type:
588: 
589:     >>> np.issctype(np.dtype('str'))
590:     True
591: 
592:     '''
593:     if not isinstance(rep, (type, dtype)):
594:         return False
595:     try:
596:         res = obj2sctype(rep)
597:         if res and res != object_:
598:             return True
599:         return False
600:     except:
601:         return False
602: 
603: def obj2sctype(rep, default=None):
604:     '''
605:     Return the scalar dtype or NumPy equivalent of Python type of an object.
606: 
607:     Parameters
608:     ----------
609:     rep : any
610:         The object of which the type is returned.
611:     default : any, optional
612:         If given, this is returned for objects whose types can not be
613:         determined. If not given, None is returned for those objects.
614: 
615:     Returns
616:     -------
617:     dtype : dtype or Python type
618:         The data type of `rep`.
619: 
620:     See Also
621:     --------
622:     sctype2char, issctype, issubsctype, issubdtype, maximum_sctype
623: 
624:     Examples
625:     --------
626:     >>> np.obj2sctype(np.int32)
627:     <type 'numpy.int32'>
628:     >>> np.obj2sctype(np.array([1., 2.]))
629:     <type 'numpy.float64'>
630:     >>> np.obj2sctype(np.array([1.j]))
631:     <type 'numpy.complex128'>
632: 
633:     >>> np.obj2sctype(dict)
634:     <type 'numpy.object_'>
635:     >>> np.obj2sctype('string')
636:     <type 'numpy.string_'>
637: 
638:     >>> np.obj2sctype(1, default=list)
639:     <type 'list'>
640: 
641:     '''
642:     try:
643:         if issubclass(rep, generic):
644:             return rep
645:     except TypeError:
646:         pass
647:     if isinstance(rep, dtype):
648:         return rep.type
649:     if isinstance(rep, type):
650:         return _python_type(rep)
651:     if isinstance(rep, ndarray):
652:         return rep.dtype.type
653:     try:
654:         res = dtype(rep)
655:     except:
656:         return default
657:     return res.type
658: 
659: 
660: def issubclass_(arg1, arg2):
661:     '''
662:     Determine if a class is a subclass of a second class.
663: 
664:     `issubclass_` is equivalent to the Python built-in ``issubclass``,
665:     except that it returns False instead of raising a TypeError if one
666:     of the arguments is not a class.
667: 
668:     Parameters
669:     ----------
670:     arg1 : class
671:         Input class. True is returned if `arg1` is a subclass of `arg2`.
672:     arg2 : class or tuple of classes.
673:         Input class. If a tuple of classes, True is returned if `arg1` is a
674:         subclass of any of the tuple elements.
675: 
676:     Returns
677:     -------
678:     out : bool
679:         Whether `arg1` is a subclass of `arg2` or not.
680: 
681:     See Also
682:     --------
683:     issubsctype, issubdtype, issctype
684: 
685:     Examples
686:     --------
687:     >>> np.issubclass_(np.int32, np.int)
688:     True
689:     >>> np.issubclass_(np.int32, np.float)
690:     False
691: 
692:     '''
693:     try:
694:         return issubclass(arg1, arg2)
695:     except TypeError:
696:         return False
697: 
698: def issubsctype(arg1, arg2):
699:     '''
700:     Determine if the first argument is a subclass of the second argument.
701: 
702:     Parameters
703:     ----------
704:     arg1, arg2 : dtype or dtype specifier
705:         Data-types.
706: 
707:     Returns
708:     -------
709:     out : bool
710:         The result.
711: 
712:     See Also
713:     --------
714:     issctype, issubdtype,obj2sctype
715: 
716:     Examples
717:     --------
718:     >>> np.issubsctype('S8', str)
719:     True
720:     >>> np.issubsctype(np.array([1]), np.int)
721:     True
722:     >>> np.issubsctype(np.array([1]), np.float)
723:     False
724: 
725:     '''
726:     return issubclass(obj2sctype(arg1), obj2sctype(arg2))
727: 
728: def issubdtype(arg1, arg2):
729:     '''
730:     Returns True if first argument is a typecode lower/equal in type hierarchy.
731: 
732:     Parameters
733:     ----------
734:     arg1, arg2 : dtype_like
735:         dtype or string representing a typecode.
736: 
737:     Returns
738:     -------
739:     out : bool
740: 
741:     See Also
742:     --------
743:     issubsctype, issubclass_
744:     numpy.core.numerictypes : Overview of numpy type hierarchy.
745: 
746:     Examples
747:     --------
748:     >>> np.issubdtype('S1', str)
749:     True
750:     >>> np.issubdtype(np.float64, np.float32)
751:     False
752: 
753:     '''
754:     if issubclass_(arg2, generic):
755:         return issubclass(dtype(arg1).type, arg2)
756:     mro = dtype(arg2).type.mro()
757:     if len(mro) > 1:
758:         val = mro[1]
759:     else:
760:         val = mro[0]
761:     return issubclass(dtype(arg1).type, val)
762: 
763: 
764: # This dictionary allows look up based on any alias for an array data-type
765: class _typedict(dict):
766:     '''
767:     Base object for a dictionary for look-up with any alias for an array dtype.
768: 
769:     Instances of `_typedict` can not be used as dictionaries directly,
770:     first they have to be populated.
771: 
772:     '''
773: 
774:     def __getitem__(self, obj):
775:         return dict.__getitem__(self, obj2sctype(obj))
776: 
777: nbytes = _typedict()
778: _alignment = _typedict()
779: _maxvals = _typedict()
780: _minvals = _typedict()
781: def _construct_lookups():
782:     for name, val in typeinfo.items():
783:         if not isinstance(val, tuple):
784:             continue
785:         obj = val[-1]
786:         nbytes[obj] = val[2] // 8
787:         _alignment[obj] = val[3]
788:         if (len(val) > 5):
789:             _maxvals[obj] = val[4]
790:             _minvals[obj] = val[5]
791:         else:
792:             _maxvals[obj] = None
793:             _minvals[obj] = None
794: 
795: _construct_lookups()
796: 
797: def sctype2char(sctype):
798:     '''
799:     Return the string representation of a scalar dtype.
800: 
801:     Parameters
802:     ----------
803:     sctype : scalar dtype or object
804:         If a scalar dtype, the corresponding string character is
805:         returned. If an object, `sctype2char` tries to infer its scalar type
806:         and then return the corresponding string character.
807: 
808:     Returns
809:     -------
810:     typechar : str
811:         The string character corresponding to the scalar type.
812: 
813:     Raises
814:     ------
815:     ValueError
816:         If `sctype` is an object for which the type can not be inferred.
817: 
818:     See Also
819:     --------
820:     obj2sctype, issctype, issubsctype, mintypecode
821: 
822:     Examples
823:     --------
824:     >>> for sctype in [np.int32, np.float, np.complex, np.string_, np.ndarray]:
825:     ...     print(np.sctype2char(sctype))
826:     l
827:     d
828:     D
829:     S
830:     O
831: 
832:     >>> x = np.array([1., 2-1.j])
833:     >>> np.sctype2char(x)
834:     'D'
835:     >>> np.sctype2char(list)
836:     'O'
837: 
838:     '''
839:     sctype = obj2sctype(sctype)
840:     if sctype is None:
841:         raise ValueError("unrecognized type")
842:     return _sctype2char_dict[sctype]
843: 
844: # Create dictionary of casting functions that wrap sequences
845: # indexed by type or type character
846: 
847: 
848: cast = _typedict()
849: try:
850:     ScalarType = [_types.IntType, _types.FloatType, _types.ComplexType,
851:                   _types.LongType, _types.BooleanType,
852:                    _types.StringType, _types.UnicodeType, _types.BufferType]
853: except AttributeError:
854:     # Py3K
855:     ScalarType = [int, float, complex, int, bool, bytes, str, memoryview]
856: 
857: ScalarType.extend(_sctype2char_dict.keys())
858: ScalarType = tuple(ScalarType)
859: for key in _sctype2char_dict.keys():
860:     cast[key] = lambda x, k=key: array(x, copy=False).astype(k)
861: 
862: # Create the typestring lookup dictionary
863: _typestr = _typedict()
864: for key in _sctype2char_dict.keys():
865:     if issubclass(key, allTypes['flexible']):
866:         _typestr[key] = _sctype2char_dict[key]
867:     else:
868:         _typestr[key] = empty((1,), key).dtype.str[1:]
869: 
870: # Make sure all typestrings are in sctypeDict
871: for key, val in _typestr.items():
872:     if val not in sctypeDict:
873:         sctypeDict[val] = key
874: 
875: # Add additional strings to the sctypeDict
876: 
877: if sys.version_info[0] >= 3:
878:     _toadd = ['int', 'float', 'complex', 'bool', 'object',
879:               'str', 'bytes', 'object', ('a', allTypes['bytes_'])]
880: else:
881:     _toadd = ['int', 'float', 'complex', 'bool', 'object', 'string',
882:               ('str', allTypes['string_']),
883:               'unicode', 'object', ('a', allTypes['string_'])]
884: 
885: for name in _toadd:
886:     if isinstance(name, tuple):
887:         sctypeDict[name[0]] = name[1]
888:     else:
889:         sctypeDict[name] = allTypes['%s_' % name]
890: 
891: del _toadd, name
892: 
893: # Now add the types we've determined to this module
894: for key in allTypes:
895:     globals()[key] = allTypes[key]
896:     __all__.append(key)
897: 
898: del key
899: 
900: typecodes = {'Character':'c',
901:              'Integer':'bhilqp',
902:              'UnsignedInteger':'BHILQP',
903:              'Float':'efdg',
904:              'Complex':'FDG',
905:              'AllInteger':'bBhHiIlLqQpP',
906:              'AllFloat':'efdgFDG',
907:              'Datetime': 'Mm',
908:              'All':'?bhilqpBHILQPefdgFDGSUVOMm'}
909: 
910: # backwards compatibility --- deprecated name
911: typeDict = sctypeDict
912: typeNA = sctypeNA
913: 
914: # b -> boolean
915: # u -> unsigned integer
916: # i -> signed integer
917: # f -> floating point
918: # c -> complex
919: # M -> datetime
920: # m -> timedelta
921: # S -> string
922: # U -> Unicode string
923: # V -> record
924: # O -> Python object
925: _kind_list = ['b', 'u', 'i', 'f', 'c', 'S', 'U', 'V', 'O', 'M', 'm']
926: 
927: __test_types = '?'+typecodes['AllInteger'][:-2]+typecodes['AllFloat']+'O'
928: __len_test_types = len(__test_types)
929: 
930: # Keep incrementing until a common type both can be coerced to
931: #  is found.  Otherwise, return None
932: def _find_common_coerce(a, b):
933:     if a > b:
934:         return a
935:     try:
936:         thisind = __test_types.index(a.char)
937:     except ValueError:
938:         return None
939:     return _can_coerce_all([a, b], start=thisind)
940: 
941: # Find a data-type that all data-types in a list can be coerced to
942: def _can_coerce_all(dtypelist, start=0):
943:     N = len(dtypelist)
944:     if N == 0:
945:         return None
946:     if N == 1:
947:         return dtypelist[0]
948:     thisind = start
949:     while thisind < __len_test_types:
950:         newdtype = dtype(__test_types[thisind])
951:         numcoerce = len([x for x in dtypelist if newdtype >= x])
952:         if numcoerce == N:
953:             return newdtype
954:         thisind += 1
955:     return None
956: 
957: def _register_types():
958:     numbers.Integral.register(integer)
959:     numbers.Complex.register(inexact)
960:     numbers.Real.register(floating)
961: 
962: _register_types()
963: 
964: def find_common_type(array_types, scalar_types):
965:     '''
966:     Determine common type following standard coercion rules.
967: 
968:     Parameters
969:     ----------
970:     array_types : sequence
971:         A list of dtypes or dtype convertible objects representing arrays.
972:     scalar_types : sequence
973:         A list of dtypes or dtype convertible objects representing scalars.
974: 
975:     Returns
976:     -------
977:     datatype : dtype
978:         The common data type, which is the maximum of `array_types` ignoring
979:         `scalar_types`, unless the maximum of `scalar_types` is of a
980:         different kind (`dtype.kind`). If the kind is not understood, then
981:         None is returned.
982: 
983:     See Also
984:     --------
985:     dtype, common_type, can_cast, mintypecode
986: 
987:     Examples
988:     --------
989:     >>> np.find_common_type([], [np.int64, np.float32, np.complex])
990:     dtype('complex128')
991:     >>> np.find_common_type([np.int64, np.float32], [])
992:     dtype('float64')
993: 
994:     The standard casting rules ensure that a scalar cannot up-cast an
995:     array unless the scalar is of a fundamentally different kind of data
996:     (i.e. under a different hierarchy in the data type hierarchy) then
997:     the array:
998: 
999:     >>> np.find_common_type([np.float32], [np.int64, np.float64])
1000:     dtype('float32')
1001: 
1002:     Complex is of a different type, so it up-casts the float in the
1003:     `array_types` argument:
1004: 
1005:     >>> np.find_common_type([np.float32], [np.complex])
1006:     dtype('complex128')
1007: 
1008:     Type specifier strings are convertible to dtypes and can therefore
1009:     be used instead of dtypes:
1010: 
1011:     >>> np.find_common_type(['f4', 'f4', 'i4'], ['c8'])
1012:     dtype('complex128')
1013: 
1014:     '''
1015:     array_types = [dtype(x) for x in array_types]
1016:     scalar_types = [dtype(x) for x in scalar_types]
1017: 
1018:     maxa = _can_coerce_all(array_types)
1019:     maxsc = _can_coerce_all(scalar_types)
1020: 
1021:     if maxa is None:
1022:         return maxsc
1023: 
1024:     if maxsc is None:
1025:         return maxa
1026: 
1027:     try:
1028:         index_a = _kind_list.index(maxa.kind)
1029:         index_sc = _kind_list.index(maxsc.kind)
1030:     except ValueError:
1031:         return None
1032: 
1033:     if index_sc > index_a:
1034:         return _find_common_coerce(maxsc, maxa)
1035:     else:
1036:         return maxa
1037: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# **************************************************************************************************
# THIS FILE DO NOT CONTAIN TYPE INFERENCE CODE. ITS TYPES ARE MANUALLY INSERTED INTO ITS TYPE STORE
# **************************************************************************************************


update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/stypy/sgmc/sgmc_cache/site_packages/numpy/core')
import numpy.core.numerictypes as stypy_origin_module
remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/stypy/sgmc/sgmc_cache/site_packages/numpy/core')

from stypy.invokation.handlers.instance_to_type import turn_to_type
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'allTypes', turn_to_type(getattr(stypy_origin_module, 'allTypes')))
	module_type_store.add_manual_type_var('allTypes')
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'string_', turn_to_type(getattr(stypy_origin_module, 'string_')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'typeinfo', turn_to_type(getattr(stypy_origin_module, 'typeinfo')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'longfloat', turn_to_type(getattr(stypy_origin_module, 'longfloat')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'half', turn_to_type(getattr(stypy_origin_module, 'half')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'absolute_import', turn_to_type(getattr(stypy_origin_module, 'absolute_import')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'typeDict', turn_to_type(getattr(stypy_origin_module, 'typeDict')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), '__len_test_types', turn_to_type(getattr(stypy_origin_module, '__len_test_types')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'unicode', turn_to_type(getattr(stypy_origin_module, 'unicode')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'integer', turn_to_type(getattr(stypy_origin_module, 'integer')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), '_typedict', turn_to_type(getattr(stypy_origin_module, '_typedict')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'cfloat', turn_to_type(getattr(stypy_origin_module, 'cfloat')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'unicode_', turn_to_type(getattr(stypy_origin_module, 'unicode_')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), '_add_integer_aliases', turn_to_type(getattr(stypy_origin_module, '_add_integer_aliases')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'character', turn_to_type(getattr(stypy_origin_module, 'character')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'timedelta64', turn_to_type(getattr(stypy_origin_module, 'timedelta64')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), '_m', turn_to_type(getattr(stypy_origin_module, '_m')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'uint16', turn_to_type(getattr(stypy_origin_module, 'uint16')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'maximum_sctype', turn_to_type(getattr(stypy_origin_module, 'maximum_sctype')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'bitname', turn_to_type(getattr(stypy_origin_module, 'bitname')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'longdouble', turn_to_type(getattr(stypy_origin_module, 'longdouble')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'float32', turn_to_type(getattr(stypy_origin_module, 'float32')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'division', turn_to_type(getattr(stypy_origin_module, 'division')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'issubsctype', turn_to_type(getattr(stypy_origin_module, 'issubsctype')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'busdaycalendar', turn_to_type(getattr(stypy_origin_module, 'busdaycalendar')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), '__file__', turn_to_type(getattr(stypy_origin_module, '__file__')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'complex_', turn_to_type(getattr(stypy_origin_module, 'complex_')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'unicode0', turn_to_type(getattr(stypy_origin_module, 'unicode0')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'datetime64', turn_to_type(getattr(stypy_origin_module, 'datetime64')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'complexfloating', turn_to_type(getattr(stypy_origin_module, 'complexfloating')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'void0', turn_to_type(getattr(stypy_origin_module, 'void0')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'genericTypeRank', turn_to_type(getattr(stypy_origin_module, 'genericTypeRank')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'uint8', turn_to_type(getattr(stypy_origin_module, 'uint8')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'bytes', turn_to_type(getattr(stypy_origin_module, 'bytes')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), '_evalname', turn_to_type(getattr(stypy_origin_module, '_evalname')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), '_python_types', turn_to_type(getattr(stypy_origin_module, '_python_types')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'english_upper', turn_to_type(getattr(stypy_origin_module, 'english_upper')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'english_capitalize', turn_to_type(getattr(stypy_origin_module, 'english_capitalize')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), '_kind_list', turn_to_type(getattr(stypy_origin_module, '_kind_list')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'object0', turn_to_type(getattr(stypy_origin_module, 'object0')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), '_can_coerce_all', turn_to_type(getattr(stypy_origin_module, '_can_coerce_all')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), '_alignment', turn_to_type(getattr(stypy_origin_module, '_alignment')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), '_maxvals', turn_to_type(getattr(stypy_origin_module, '_maxvals')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'float16', turn_to_type(getattr(stypy_origin_module, 'float16')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ulonglong', turn_to_type(getattr(stypy_origin_module, 'ulonglong')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), '__all__', turn_to_type(getattr(stypy_origin_module, '__all__')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), '_find_common_coerce', turn_to_type(getattr(stypy_origin_module, '_find_common_coerce')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), '_typestr', turn_to_type(getattr(stypy_origin_module, '_typestr')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'sctypeDict', turn_to_type(getattr(stypy_origin_module, 'sctypeDict')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), '_sctype2char_dict', turn_to_type(getattr(stypy_origin_module, '_sctype2char_dict')))
	module_type_store.add_manual_type_var('_sctype2char_dict')
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'uint64', turn_to_type(getattr(stypy_origin_module, 'uint64')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'datetime_as_string', turn_to_type(getattr(stypy_origin_module, 'datetime_as_string')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'uint32', turn_to_type(getattr(stypy_origin_module, 'uint32')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'typeNA', turn_to_type(getattr(stypy_origin_module, 'typeNA')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'complex64', turn_to_type(getattr(stypy_origin_module, 'complex64')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'sctypes', turn_to_type(getattr(stypy_origin_module, 'sctypes')))
	module_type_store.add_manual_type_var('sctypes')
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), '__name__', turn_to_type(getattr(stypy_origin_module, '__name__')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'byte', turn_to_type(getattr(stypy_origin_module, 'byte')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'print_function', turn_to_type(getattr(stypy_origin_module, 'print_function')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'buffer_type', turn_to_type(getattr(stypy_origin_module, 'buffer_type')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'float64', turn_to_type(getattr(stypy_origin_module, 'float64')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ushort', turn_to_type(getattr(stypy_origin_module, 'ushort')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), '_minvals', turn_to_type(getattr(stypy_origin_module, '_minvals')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'float_', turn_to_type(getattr(stypy_origin_module, 'float_')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'signedinteger', turn_to_type(getattr(stypy_origin_module, 'signedinteger')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'object_', turn_to_type(getattr(stypy_origin_module, 'object_')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'is_busday', turn_to_type(getattr(stypy_origin_module, 'is_busday')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'uintp', turn_to_type(getattr(stypy_origin_module, 'uintp')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'intc', turn_to_type(getattr(stypy_origin_module, 'intc')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'csingle', turn_to_type(getattr(stypy_origin_module, 'csingle')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'busday_count', turn_to_type(getattr(stypy_origin_module, 'busday_count')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'unsignedinteger', turn_to_type(getattr(stypy_origin_module, 'unsignedinteger')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'float', turn_to_type(getattr(stypy_origin_module, 'float')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'number', turn_to_type(getattr(stypy_origin_module, 'number')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'issubclass_', turn_to_type(getattr(stypy_origin_module, 'issubclass_')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'bool8', turn_to_type(getattr(stypy_origin_module, 'bool8')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), '_ascii_upper', turn_to_type(getattr(stypy_origin_module, '_ascii_upper')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'intp', turn_to_type(getattr(stypy_origin_module, 'intp')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'uintc', turn_to_type(getattr(stypy_origin_module, 'uintc')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'array', turn_to_type(getattr(stypy_origin_module, 'array')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'bytes_', turn_to_type(getattr(stypy_origin_module, 'bytes_')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'val', turn_to_type(getattr(stypy_origin_module, 'val')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'cdouble', turn_to_type(getattr(stypy_origin_module, 'cdouble')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'complex128', turn_to_type(getattr(stypy_origin_module, 'complex128')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'long', turn_to_type(getattr(stypy_origin_module, 'long')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'busday_offset', turn_to_type(getattr(stypy_origin_module, 'busday_offset')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ubyte', turn_to_type(getattr(stypy_origin_module, 'ubyte')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'flexible', turn_to_type(getattr(stypy_origin_module, 'flexible')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'int_', turn_to_type(getattr(stypy_origin_module, 'int_')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'LOWER_TABLE', turn_to_type(getattr(stypy_origin_module, 'LOWER_TABLE')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'empty', turn_to_type(getattr(stypy_origin_module, 'empty')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'find_common_type', turn_to_type(getattr(stypy_origin_module, 'find_common_type')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), '_construct_char_code_lookup', turn_to_type(getattr(stypy_origin_module, '_construct_char_code_lookup')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'uint0', turn_to_type(getattr(stypy_origin_module, 'uint0')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), '_construct_lookups', turn_to_type(getattr(stypy_origin_module, '_construct_lookups')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), '_set_array_types', turn_to_type(getattr(stypy_origin_module, '_set_array_types')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'bool_', turn_to_type(getattr(stypy_origin_module, 'bool_')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'sctype2char', turn_to_type(getattr(stypy_origin_module, 'sctype2char')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'inexact', turn_to_type(getattr(stypy_origin_module, 'inexact')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), '_python_type', turn_to_type(getattr(stypy_origin_module, '_python_type')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), '_set_up_aliases', turn_to_type(getattr(stypy_origin_module, '_set_up_aliases')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'typecodes', turn_to_type(getattr(stypy_origin_module, 'typecodes')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), '_add_types', turn_to_type(getattr(stypy_origin_module, '_add_types')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'dtype', turn_to_type(getattr(stypy_origin_module, 'dtype')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'int8', turn_to_type(getattr(stypy_origin_module, 'int8')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), '_types', turn_to_type(getattr(stypy_origin_module, '_types')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), '__test_types', turn_to_type(getattr(stypy_origin_module, '__test_types')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'double', turn_to_type(getattr(stypy_origin_module, 'double')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), '__doc__', turn_to_type(getattr(stypy_origin_module, '__doc__')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'cast', turn_to_type(getattr(stypy_origin_module, 'cast')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'obj2sctype', turn_to_type(getattr(stypy_origin_module, 'obj2sctype')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'clongdouble', turn_to_type(getattr(stypy_origin_module, 'clongdouble')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'str', turn_to_type(getattr(stypy_origin_module, 'str')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'issctype', turn_to_type(getattr(stypy_origin_module, 'issctype')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), '_add_aliases', turn_to_type(getattr(stypy_origin_module, '_add_aliases')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'int32', turn_to_type(getattr(stypy_origin_module, 'int32')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'int', turn_to_type(getattr(stypy_origin_module, 'int')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), '_add_array_type', turn_to_type(getattr(stypy_origin_module, '_add_array_type')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'english_lower', turn_to_type(getattr(stypy_origin_module, 'english_lower')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'single', turn_to_type(getattr(stypy_origin_module, 'single')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'numbers', turn_to_type(getattr(stypy_origin_module, 'numbers')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ScalarType', turn_to_type(getattr(stypy_origin_module, 'ScalarType')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'UPPER_TABLE', turn_to_type(getattr(stypy_origin_module, 'UPPER_TABLE')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'floating', turn_to_type(getattr(stypy_origin_module, 'floating')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'generic', turn_to_type(getattr(stypy_origin_module, 'generic')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), '__package__', turn_to_type(getattr(stypy_origin_module, '__package__')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'longcomplex', turn_to_type(getattr(stypy_origin_module, 'longcomplex')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'complex', turn_to_type(getattr(stypy_origin_module, 'complex')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'bool', turn_to_type(getattr(stypy_origin_module, 'bool')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'sctypeNA', turn_to_type(getattr(stypy_origin_module, 'sctypeNA')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'void', turn_to_type(getattr(stypy_origin_module, 'void')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'string0', turn_to_type(getattr(stypy_origin_module, 'string0')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'longlong', turn_to_type(getattr(stypy_origin_module, 'longlong')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), '_ascii_lower', turn_to_type(getattr(stypy_origin_module, '_ascii_lower')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'int16', turn_to_type(getattr(stypy_origin_module, 'int16')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'str_', turn_to_type(getattr(stypy_origin_module, 'str_')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'object', turn_to_type(getattr(stypy_origin_module, 'object')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'singlecomplex', turn_to_type(getattr(stypy_origin_module, 'singlecomplex')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'sys', turn_to_type(getattr(stypy_origin_module, 'sys')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), '_register_types', turn_to_type(getattr(stypy_origin_module, '_register_types')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'uint', turn_to_type(getattr(stypy_origin_module, 'uint')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ndarray', turn_to_type(getattr(stypy_origin_module, 'ndarray')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'short', turn_to_type(getattr(stypy_origin_module, 'short')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'datetime_data', turn_to_type(getattr(stypy_origin_module, 'datetime_data')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'clongfloat', turn_to_type(getattr(stypy_origin_module, 'clongfloat')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'int64', turn_to_type(getattr(stypy_origin_module, 'int64')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'nbytes', turn_to_type(getattr(stypy_origin_module, 'nbytes')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), '_all_chars', turn_to_type(getattr(stypy_origin_module, '_all_chars')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'issubdtype', turn_to_type(getattr(stypy_origin_module, 'issubdtype')))
except:
	pass
try:
	module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'int0', turn_to_type(getattr(stypy_origin_module, 'int0')))
except:
	pass

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
