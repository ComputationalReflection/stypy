
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''A collection of functions designed to help I/O with ascii files.
2: 
3: '''
4: from __future__ import division, absolute_import, print_function
5: 
6: __docformat__ = "restructuredtext en"
7: 
8: import sys
9: import numpy as np
10: import numpy.core.numeric as nx
11: from numpy.compat import asbytes, bytes, asbytes_nested, basestring
12: 
13: if sys.version_info[0] >= 3:
14:     from builtins import bool, int, float, complex, object, str
15:     unicode = str
16: else:
17:     from __builtin__ import bool, int, float, complex, object, unicode, str
18: 
19: 
20: if sys.version_info[0] >= 3:
21:     def _bytes_to_complex(s):
22:         return complex(s.decode('ascii'))
23: 
24:     def _bytes_to_name(s):
25:         return s.decode('ascii')
26: else:
27:     _bytes_to_complex = complex
28:     _bytes_to_name = str
29: 
30: 
31: def _is_string_like(obj):
32:     '''
33:     Check whether obj behaves like a string.
34:     '''
35:     try:
36:         obj + ''
37:     except (TypeError, ValueError):
38:         return False
39:     return True
40: 
41: 
42: def _is_bytes_like(obj):
43:     '''
44:     Check whether obj behaves like a bytes object.
45:     '''
46:     try:
47:         obj + asbytes('')
48:     except (TypeError, ValueError):
49:         return False
50:     return True
51: 
52: 
53: def _to_filehandle(fname, flag='r', return_opened=False):
54:     '''
55:     Returns the filehandle corresponding to a string or a file.
56:     If the string ends in '.gz', the file is automatically unzipped.
57: 
58:     Parameters
59:     ----------
60:     fname : string, filehandle
61:         Name of the file whose filehandle must be returned.
62:     flag : string, optional
63:         Flag indicating the status of the file ('r' for read, 'w' for write).
64:     return_opened : boolean, optional
65:         Whether to return the opening status of the file.
66:     '''
67:     if _is_string_like(fname):
68:         if fname.endswith('.gz'):
69:             import gzip
70:             fhd = gzip.open(fname, flag)
71:         elif fname.endswith('.bz2'):
72:             import bz2
73:             fhd = bz2.BZ2File(fname)
74:         else:
75:             fhd = file(fname, flag)
76:         opened = True
77:     elif hasattr(fname, 'seek'):
78:         fhd = fname
79:         opened = False
80:     else:
81:         raise ValueError('fname must be a string or file handle')
82:     if return_opened:
83:         return fhd, opened
84:     return fhd
85: 
86: 
87: def has_nested_fields(ndtype):
88:     '''
89:     Returns whether one or several fields of a dtype are nested.
90: 
91:     Parameters
92:     ----------
93:     ndtype : dtype
94:         Data-type of a structured array.
95: 
96:     Raises
97:     ------
98:     AttributeError
99:         If `ndtype` does not have a `names` attribute.
100: 
101:     Examples
102:     --------
103:     >>> dt = np.dtype([('name', 'S4'), ('x', float), ('y', float)])
104:     >>> np.lib._iotools.has_nested_fields(dt)
105:     False
106: 
107:     '''
108:     for name in ndtype.names or ():
109:         if ndtype[name].names:
110:             return True
111:     return False
112: 
113: 
114: def flatten_dtype(ndtype, flatten_base=False):
115:     '''
116:     Unpack a structured data-type by collapsing nested fields and/or fields
117:     with a shape.
118: 
119:     Note that the field names are lost.
120: 
121:     Parameters
122:     ----------
123:     ndtype : dtype
124:         The datatype to collapse
125:     flatten_base : {False, True}, optional
126:         Whether to transform a field with a shape into several fields or not.
127: 
128:     Examples
129:     --------
130:     >>> dt = np.dtype([('name', 'S4'), ('x', float), ('y', float),
131:     ...                ('block', int, (2, 3))])
132:     >>> np.lib._iotools.flatten_dtype(dt)
133:     [dtype('|S4'), dtype('float64'), dtype('float64'), dtype('int32')]
134:     >>> np.lib._iotools.flatten_dtype(dt, flatten_base=True)
135:     [dtype('|S4'), dtype('float64'), dtype('float64'), dtype('int32'),
136:      dtype('int32'), dtype('int32'), dtype('int32'), dtype('int32'),
137:      dtype('int32')]
138: 
139:     '''
140:     names = ndtype.names
141:     if names is None:
142:         if flatten_base:
143:             return [ndtype.base] * int(np.prod(ndtype.shape))
144:         return [ndtype.base]
145:     else:
146:         types = []
147:         for field in names:
148:             info = ndtype.fields[field]
149:             flat_dt = flatten_dtype(info[0], flatten_base)
150:             types.extend(flat_dt)
151:         return types
152: 
153: 
154: class LineSplitter(object):
155:     '''
156:     Object to split a string at a given delimiter or at given places.
157: 
158:     Parameters
159:     ----------
160:     delimiter : str, int, or sequence of ints, optional
161:         If a string, character used to delimit consecutive fields.
162:         If an integer or a sequence of integers, width(s) of each field.
163:     comments : str, optional
164:         Character used to mark the beginning of a comment. Default is '#'.
165:     autostrip : bool, optional
166:         Whether to strip each individual field. Default is True.
167: 
168:     '''
169: 
170:     def autostrip(self, method):
171:         '''
172:         Wrapper to strip each member of the output of `method`.
173: 
174:         Parameters
175:         ----------
176:         method : function
177:             Function that takes a single argument and returns a sequence of
178:             strings.
179: 
180:         Returns
181:         -------
182:         wrapped : function
183:             The result of wrapping `method`. `wrapped` takes a single input
184:             argument and returns a list of strings that are stripped of
185:             white-space.
186: 
187:         '''
188:         return lambda input: [_.strip() for _ in method(input)]
189:     #
190: 
191:     def __init__(self, delimiter=None, comments=asbytes('#'), autostrip=True):
192:         self.comments = comments
193:         # Delimiter is a character
194:         if isinstance(delimiter, unicode):
195:             delimiter = delimiter.encode('ascii')
196:         if (delimiter is None) or _is_bytes_like(delimiter):
197:             delimiter = delimiter or None
198:             _handyman = self._delimited_splitter
199:         # Delimiter is a list of field widths
200:         elif hasattr(delimiter, '__iter__'):
201:             _handyman = self._variablewidth_splitter
202:             idx = np.cumsum([0] + list(delimiter))
203:             delimiter = [slice(i, j) for (i, j) in zip(idx[:-1], idx[1:])]
204:         # Delimiter is a single integer
205:         elif int(delimiter):
206:             (_handyman, delimiter) = (
207:                     self._fixedwidth_splitter, int(delimiter))
208:         else:
209:             (_handyman, delimiter) = (self._delimited_splitter, None)
210:         self.delimiter = delimiter
211:         if autostrip:
212:             self._handyman = self.autostrip(_handyman)
213:         else:
214:             self._handyman = _handyman
215:     #
216: 
217:     def _delimited_splitter(self, line):
218:         if self.comments is not None:
219:             line = line.split(self.comments)[0]
220:         line = line.strip(asbytes(" \r\n"))
221:         if not line:
222:             return []
223:         return line.split(self.delimiter)
224:     #
225: 
226:     def _fixedwidth_splitter(self, line):
227:         if self.comments is not None:
228:             line = line.split(self.comments)[0]
229:         line = line.strip(asbytes("\r\n"))
230:         if not line:
231:             return []
232:         fixed = self.delimiter
233:         slices = [slice(i, i + fixed) for i in range(0, len(line), fixed)]
234:         return [line[s] for s in slices]
235:     #
236: 
237:     def _variablewidth_splitter(self, line):
238:         if self.comments is not None:
239:             line = line.split(self.comments)[0]
240:         if not line:
241:             return []
242:         slices = self.delimiter
243:         return [line[s] for s in slices]
244:     #
245: 
246:     def __call__(self, line):
247:         return self._handyman(line)
248: 
249: 
250: class NameValidator(object):
251:     '''
252:     Object to validate a list of strings to use as field names.
253: 
254:     The strings are stripped of any non alphanumeric character, and spaces
255:     are replaced by '_'. During instantiation, the user can define a list
256:     of names to exclude, as well as a list of invalid characters. Names in
257:     the exclusion list are appended a '_' character.
258: 
259:     Once an instance has been created, it can be called with a list of
260:     names, and a list of valid names will be created.  The `__call__`
261:     method accepts an optional keyword "default" that sets the default name
262:     in case of ambiguity. By default this is 'f', so that names will
263:     default to `f0`, `f1`, etc.
264: 
265:     Parameters
266:     ----------
267:     excludelist : sequence, optional
268:         A list of names to exclude. This list is appended to the default
269:         list ['return', 'file', 'print']. Excluded names are appended an
270:         underscore: for example, `file` becomes `file_` if supplied.
271:     deletechars : str, optional
272:         A string combining invalid characters that must be deleted from the
273:         names.
274:     case_sensitive : {True, False, 'upper', 'lower'}, optional
275:         * If True, field names are case-sensitive.
276:         * If False or 'upper', field names are converted to upper case.
277:         * If 'lower', field names are converted to lower case.
278: 
279:         The default value is True.
280:     replace_space : '_', optional
281:         Character(s) used in replacement of white spaces.
282: 
283:     Notes
284:     -----
285:     Calling an instance of `NameValidator` is the same as calling its
286:     method `validate`.
287: 
288:     Examples
289:     --------
290:     >>> validator = np.lib._iotools.NameValidator()
291:     >>> validator(['file', 'field2', 'with space', 'CaSe'])
292:     ['file_', 'field2', 'with_space', 'CaSe']
293: 
294:     >>> validator = np.lib._iotools.NameValidator(excludelist=['excl'],
295:                                                   deletechars='q',
296:                                                   case_sensitive='False')
297:     >>> validator(['excl', 'field2', 'no_q', 'with space', 'CaSe'])
298:     ['excl_', 'field2', 'no_', 'with_space', 'case']
299: 
300:     '''
301:     #
302:     defaultexcludelist = ['return', 'file', 'print']
303:     defaultdeletechars = set('''~!@#$%^&*()-=+~\|]}[{';: /?.>,<''')
304:     #
305: 
306:     def __init__(self, excludelist=None, deletechars=None,
307:                  case_sensitive=None, replace_space='_'):
308:         # Process the exclusion list ..
309:         if excludelist is None:
310:             excludelist = []
311:         excludelist.extend(self.defaultexcludelist)
312:         self.excludelist = excludelist
313:         # Process the list of characters to delete
314:         if deletechars is None:
315:             delete = self.defaultdeletechars
316:         else:
317:             delete = set(deletechars)
318:         delete.add('"')
319:         self.deletechars = delete
320:         # Process the case option .....
321:         if (case_sensitive is None) or (case_sensitive is True):
322:             self.case_converter = lambda x: x
323:         elif (case_sensitive is False) or case_sensitive.startswith('u'):
324:             self.case_converter = lambda x: x.upper()
325:         elif case_sensitive.startswith('l'):
326:             self.case_converter = lambda x: x.lower()
327:         else:
328:             msg = 'unrecognized case_sensitive value %s.' % case_sensitive
329:             raise ValueError(msg)
330:         #
331:         self.replace_space = replace_space
332: 
333:     def validate(self, names, defaultfmt="f%i", nbfields=None):
334:         '''
335:         Validate a list of strings as field names for a structured array.
336: 
337:         Parameters
338:         ----------
339:         names : sequence of str
340:             Strings to be validated.
341:         defaultfmt : str, optional
342:             Default format string, used if validating a given string
343:             reduces its length to zero.
344:         nbfields : integer, optional
345:             Final number of validated names, used to expand or shrink the
346:             initial list of names.
347: 
348:         Returns
349:         -------
350:         validatednames : list of str
351:             The list of validated field names.
352: 
353:         Notes
354:         -----
355:         A `NameValidator` instance can be called directly, which is the
356:         same as calling `validate`. For examples, see `NameValidator`.
357: 
358:         '''
359:         # Initial checks ..............
360:         if (names is None):
361:             if (nbfields is None):
362:                 return None
363:             names = []
364:         if isinstance(names, basestring):
365:             names = [names, ]
366:         if nbfields is not None:
367:             nbnames = len(names)
368:             if (nbnames < nbfields):
369:                 names = list(names) + [''] * (nbfields - nbnames)
370:             elif (nbnames > nbfields):
371:                 names = names[:nbfields]
372:         # Set some shortcuts ...........
373:         deletechars = self.deletechars
374:         excludelist = self.excludelist
375:         case_converter = self.case_converter
376:         replace_space = self.replace_space
377:         # Initializes some variables ...
378:         validatednames = []
379:         seen = dict()
380:         nbempty = 0
381:         #
382:         for item in names:
383:             item = case_converter(item).strip()
384:             if replace_space:
385:                 item = item.replace(' ', replace_space)
386:             item = ''.join([c for c in item if c not in deletechars])
387:             if item == '':
388:                 item = defaultfmt % nbempty
389:                 while item in names:
390:                     nbempty += 1
391:                     item = defaultfmt % nbempty
392:                 nbempty += 1
393:             elif item in excludelist:
394:                 item += '_'
395:             cnt = seen.get(item, 0)
396:             if cnt > 0:
397:                 validatednames.append(item + '_%d' % cnt)
398:             else:
399:                 validatednames.append(item)
400:             seen[item] = cnt + 1
401:         return tuple(validatednames)
402:     #
403: 
404:     def __call__(self, names, defaultfmt="f%i", nbfields=None):
405:         return self.validate(names, defaultfmt=defaultfmt, nbfields=nbfields)
406: 
407: 
408: def str2bool(value):
409:     '''
410:     Tries to transform a string supposed to represent a boolean to a boolean.
411: 
412:     Parameters
413:     ----------
414:     value : str
415:         The string that is transformed to a boolean.
416: 
417:     Returns
418:     -------
419:     boolval : bool
420:         The boolean representation of `value`.
421: 
422:     Raises
423:     ------
424:     ValueError
425:         If the string is not 'True' or 'False' (case independent)
426: 
427:     Examples
428:     --------
429:     >>> np.lib._iotools.str2bool('TRUE')
430:     True
431:     >>> np.lib._iotools.str2bool('false')
432:     False
433: 
434:     '''
435:     value = value.upper()
436:     if value == asbytes('TRUE'):
437:         return True
438:     elif value == asbytes('FALSE'):
439:         return False
440:     else:
441:         raise ValueError("Invalid boolean")
442: 
443: 
444: class ConverterError(Exception):
445:     '''
446:     Exception raised when an error occurs in a converter for string values.
447: 
448:     '''
449:     pass
450: 
451: 
452: class ConverterLockError(ConverterError):
453:     '''
454:     Exception raised when an attempt is made to upgrade a locked converter.
455: 
456:     '''
457:     pass
458: 
459: 
460: class ConversionWarning(UserWarning):
461:     '''
462:     Warning issued when a string converter has a problem.
463: 
464:     Notes
465:     -----
466:     In `genfromtxt` a `ConversionWarning` is issued if raising exceptions
467:     is explicitly suppressed with the "invalid_raise" keyword.
468: 
469:     '''
470:     pass
471: 
472: 
473: class StringConverter(object):
474:     '''
475:     Factory class for function transforming a string into another object
476:     (int, float).
477: 
478:     After initialization, an instance can be called to transform a string
479:     into another object. If the string is recognized as representing a
480:     missing value, a default value is returned.
481: 
482:     Attributes
483:     ----------
484:     func : function
485:         Function used for the conversion.
486:     default : any
487:         Default value to return when the input corresponds to a missing
488:         value.
489:     type : type
490:         Type of the output.
491:     _status : int
492:         Integer representing the order of the conversion.
493:     _mapper : sequence of tuples
494:         Sequence of tuples (dtype, function, default value) to evaluate in
495:         order.
496:     _locked : bool
497:         Holds `locked` parameter.
498: 
499:     Parameters
500:     ----------
501:     dtype_or_func : {None, dtype, function}, optional
502:         If a `dtype`, specifies the input data type, used to define a basic
503:         function and a default value for missing data. For example, when
504:         `dtype` is float, the `func` attribute is set to `float` and the
505:         default value to `np.nan`.  If a function, this function is used to
506:         convert a string to another object. In this case, it is recommended
507:         to give an associated default value as input.
508:     default : any, optional
509:         Value to return by default, that is, when the string to be
510:         converted is flagged as missing. If not given, `StringConverter`
511:         tries to supply a reasonable default value.
512:     missing_values : sequence of str, optional
513:         Sequence of strings indicating a missing value.
514:     locked : bool, optional
515:         Whether the StringConverter should be locked to prevent automatic
516:         upgrade or not. Default is False.
517: 
518:     '''
519:     #
520:     _mapper = [(nx.bool_, str2bool, False),
521:                (nx.integer, int, -1)]
522: 
523:     # On 32-bit systems, we need to make sure that we explicitly include
524:     # nx.int64 since ns.integer is nx.int32.
525:     if nx.dtype(nx.integer).itemsize < nx.dtype(nx.int64).itemsize:
526:         _mapper.append((nx.int64, int, -1))
527: 
528:     _mapper.extend([(nx.floating, float, nx.nan),
529:                     (complex, _bytes_to_complex, nx.nan + 0j),
530:                     (nx.longdouble, nx.longdouble, nx.nan),
531:                     (nx.string_, bytes, asbytes('???'))])
532: 
533:     (_defaulttype, _defaultfunc, _defaultfill) = zip(*_mapper)
534: 
535:     @classmethod
536:     def _getdtype(cls, val):
537:         '''Returns the dtype of the input variable.'''
538:         return np.array(val).dtype
539:     #
540: 
541:     @classmethod
542:     def _getsubdtype(cls, val):
543:         '''Returns the type of the dtype of the input variable.'''
544:         return np.array(val).dtype.type
545:     #
546:     # This is a bit annoying. We want to return the "general" type in most
547:     # cases (ie. "string" rather than "S10"), but we want to return the
548:     # specific type for datetime64 (ie. "datetime64[us]" rather than
549:     # "datetime64").
550: 
551:     @classmethod
552:     def _dtypeortype(cls, dtype):
553:         '''Returns dtype for datetime64 and type of dtype otherwise.'''
554:         if dtype.type == np.datetime64:
555:             return dtype
556:         return dtype.type
557:     #
558: 
559:     @classmethod
560:     def upgrade_mapper(cls, func, default=None):
561:         '''
562:     Upgrade the mapper of a StringConverter by adding a new function and
563:     its corresponding default.
564: 
565:     The input function (or sequence of functions) and its associated
566:     default value (if any) is inserted in penultimate position of the
567:     mapper.  The corresponding type is estimated from the dtype of the
568:     default value.
569: 
570:     Parameters
571:     ----------
572:     func : var
573:         Function, or sequence of functions
574: 
575:     Examples
576:     --------
577:     >>> import dateutil.parser
578:     >>> import datetime
579:     >>> dateparser = datetustil.parser.parse
580:     >>> defaultdate = datetime.date(2000, 1, 1)
581:     >>> StringConverter.upgrade_mapper(dateparser, default=defaultdate)
582:         '''
583:         # Func is a single functions
584:         if hasattr(func, '__call__'):
585:             cls._mapper.insert(-1, (cls._getsubdtype(default), func, default))
586:             return
587:         elif hasattr(func, '__iter__'):
588:             if isinstance(func[0], (tuple, list)):
589:                 for _ in func:
590:                     cls._mapper.insert(-1, _)
591:                 return
592:             if default is None:
593:                 default = [None] * len(func)
594:             else:
595:                 default = list(default)
596:                 default.append([None] * (len(func) - len(default)))
597:             for (fct, dft) in zip(func, default):
598:                 cls._mapper.insert(-1, (cls._getsubdtype(dft), fct, dft))
599:     #
600: 
601:     def __init__(self, dtype_or_func=None, default=None, missing_values=None,
602:                  locked=False):
603:         # Convert unicode (for Py3)
604:         if isinstance(missing_values, unicode):
605:             missing_values = asbytes(missing_values)
606:         elif isinstance(missing_values, (list, tuple)):
607:             missing_values = asbytes_nested(missing_values)
608:         # Defines a lock for upgrade
609:         self._locked = bool(locked)
610:         # No input dtype: minimal initialization
611:         if dtype_or_func is None:
612:             self.func = str2bool
613:             self._status = 0
614:             self.default = default or False
615:             dtype = np.dtype('bool')
616:         else:
617:             # Is the input a np.dtype ?
618:             try:
619:                 self.func = None
620:                 dtype = np.dtype(dtype_or_func)
621:             except TypeError:
622:                 # dtype_or_func must be a function, then
623:                 if not hasattr(dtype_or_func, '__call__'):
624:                     errmsg = ("The input argument `dtype` is neither a"
625:                               " function nor a dtype (got '%s' instead)")
626:                     raise TypeError(errmsg % type(dtype_or_func))
627:                 # Set the function
628:                 self.func = dtype_or_func
629:                 # If we don't have a default, try to guess it or set it to
630:                 # None
631:                 if default is None:
632:                     try:
633:                         default = self.func(asbytes('0'))
634:                     except ValueError:
635:                         default = None
636:                 dtype = self._getdtype(default)
637:             # Set the status according to the dtype
638:             _status = -1
639:             for (i, (deftype, func, default_def)) in enumerate(self._mapper):
640:                 if np.issubdtype(dtype.type, deftype):
641:                     _status = i
642:                     if default is None:
643:                         self.default = default_def
644:                     else:
645:                         self.default = default
646:                     break
647:             # if a converter for the specific dtype is available use that
648:             last_func = func
649:             for (i, (deftype, func, default_def)) in enumerate(self._mapper):
650:                 if dtype.type == deftype:
651:                     _status = i
652:                     last_func = func
653:                     if default is None:
654:                         self.default = default_def
655:                     else:
656:                         self.default = default
657:                     break
658:             func = last_func
659:             if _status == -1:
660:                 # We never found a match in the _mapper...
661:                 _status = 0
662:                 self.default = default
663:             self._status = _status
664:             # If the input was a dtype, set the function to the last we saw
665:             if self.func is None:
666:                 self.func = func
667:             # If the status is 1 (int), change the function to
668:             # something more robust.
669:             if self.func == self._mapper[1][1]:
670:                 if issubclass(dtype.type, np.uint64):
671:                     self.func = np.uint64
672:                 elif issubclass(dtype.type, np.int64):
673:                     self.func = np.int64
674:                 else:
675:                     self.func = lambda x: int(float(x))
676:         # Store the list of strings corresponding to missing values.
677:         if missing_values is None:
678:             self.missing_values = set([asbytes('')])
679:         else:
680:             if isinstance(missing_values, bytes):
681:                 missing_values = missing_values.split(asbytes(","))
682:             self.missing_values = set(list(missing_values) + [asbytes('')])
683:         #
684:         self._callingfunction = self._strict_call
685:         self.type = self._dtypeortype(dtype)
686:         self._checked = False
687:         self._initial_default = default
688:     #
689: 
690:     def _loose_call(self, value):
691:         try:
692:             return self.func(value)
693:         except ValueError:
694:             return self.default
695:     #
696: 
697:     def _strict_call(self, value):
698:         try:
699: 
700:             # We check if we can convert the value using the current function
701:             new_value = self.func(value)
702: 
703:             # In addition to having to check whether func can convert the
704:             # value, we also have to make sure that we don't get overflow
705:             # errors for integers.
706:             if self.func is int:
707:                 try:
708:                     np.array(value, dtype=self.type)
709:                 except OverflowError:
710:                     raise ValueError
711: 
712:             # We're still here so we can now return the new value
713:             return new_value
714: 
715:         except ValueError:
716:             if value.strip() in self.missing_values:
717:                 if not self._status:
718:                     self._checked = False
719:                 return self.default
720:             raise ValueError("Cannot convert string '%s'" % value)
721:     #
722: 
723:     def __call__(self, value):
724:         return self._callingfunction(value)
725:     #
726: 
727:     def upgrade(self, value):
728:         '''
729:         Find the best converter for a given string, and return the result.
730: 
731:         The supplied string `value` is converted by testing different
732:         converters in order. First the `func` method of the
733:         `StringConverter` instance is tried, if this fails other available
734:         converters are tried.  The order in which these other converters
735:         are tried is determined by the `_status` attribute of the instance.
736: 
737:         Parameters
738:         ----------
739:         value : str
740:             The string to convert.
741: 
742:         Returns
743:         -------
744:         out : any
745:             The result of converting `value` with the appropriate converter.
746: 
747:         '''
748:         self._checked = True
749:         try:
750:             return self._strict_call(value)
751:         except ValueError:
752:             # Raise an exception if we locked the converter...
753:             if self._locked:
754:                 errmsg = "Converter is locked and cannot be upgraded"
755:                 raise ConverterLockError(errmsg)
756:             _statusmax = len(self._mapper)
757:             # Complains if we try to upgrade by the maximum
758:             _status = self._status
759:             if _status == _statusmax:
760:                 errmsg = "Could not find a valid conversion function"
761:                 raise ConverterError(errmsg)
762:             elif _status < _statusmax - 1:
763:                 _status += 1
764:             (self.type, self.func, default) = self._mapper[_status]
765:             self._status = _status
766:             if self._initial_default is not None:
767:                 self.default = self._initial_default
768:             else:
769:                 self.default = default
770:             return self.upgrade(value)
771: 
772:     def iterupgrade(self, value):
773:         self._checked = True
774:         if not hasattr(value, '__iter__'):
775:             value = (value,)
776:         _strict_call = self._strict_call
777:         try:
778:             for _m in value:
779:                 _strict_call(_m)
780:         except ValueError:
781:             # Raise an exception if we locked the converter...
782:             if self._locked:
783:                 errmsg = "Converter is locked and cannot be upgraded"
784:                 raise ConverterLockError(errmsg)
785:             _statusmax = len(self._mapper)
786:             # Complains if we try to upgrade by the maximum
787:             _status = self._status
788:             if _status == _statusmax:
789:                 raise ConverterError(
790:                     "Could not find a valid conversion function"
791:                     )
792:             elif _status < _statusmax - 1:
793:                 _status += 1
794:             (self.type, self.func, default) = self._mapper[_status]
795:             if self._initial_default is not None:
796:                 self.default = self._initial_default
797:             else:
798:                 self.default = default
799:             self._status = _status
800:             self.iterupgrade(value)
801: 
802:     def update(self, func, default=None, testing_value=None,
803:                missing_values=asbytes(''), locked=False):
804:         '''
805:         Set StringConverter attributes directly.
806: 
807:         Parameters
808:         ----------
809:         func : function
810:             Conversion function.
811:         default : any, optional
812:             Value to return by default, that is, when the string to be
813:             converted is flagged as missing. If not given,
814:             `StringConverter` tries to supply a reasonable default value.
815:         testing_value : str, optional
816:             A string representing a standard input value of the converter.
817:             This string is used to help defining a reasonable default
818:             value.
819:         missing_values : sequence of str, optional
820:             Sequence of strings indicating a missing value.
821:         locked : bool, optional
822:             Whether the StringConverter should be locked to prevent
823:             automatic upgrade or not. Default is False.
824: 
825:         Notes
826:         -----
827:         `update` takes the same parameters as the constructor of
828:         `StringConverter`, except that `func` does not accept a `dtype`
829:         whereas `dtype_or_func` in the constructor does.
830: 
831:         '''
832:         self.func = func
833:         self._locked = locked
834:         # Don't reset the default to None if we can avoid it
835:         if default is not None:
836:             self.default = default
837:             self.type = self._dtypeortype(self._getdtype(default))
838:         else:
839:             try:
840:                 tester = func(testing_value or asbytes('1'))
841:             except (TypeError, ValueError):
842:                 tester = None
843:             self.type = self._dtypeortype(self._getdtype(tester))
844:         # Add the missing values to the existing set
845:         if missing_values is not None:
846:             if _is_bytes_like(missing_values):
847:                 self.missing_values.add(missing_values)
848:             elif hasattr(missing_values, '__iter__'):
849:                 for val in missing_values:
850:                     self.missing_values.add(val)
851:         else:
852:             self.missing_values = []
853: 
854: 
855: def easy_dtype(ndtype, names=None, defaultfmt="f%i", **validationargs):
856:     '''
857:     Convenience function to create a `np.dtype` object.
858: 
859:     The function processes the input `dtype` and matches it with the given
860:     names.
861: 
862:     Parameters
863:     ----------
864:     ndtype : var
865:         Definition of the dtype. Can be any string or dictionary recognized
866:         by the `np.dtype` function, or a sequence of types.
867:     names : str or sequence, optional
868:         Sequence of strings to use as field names for a structured dtype.
869:         For convenience, `names` can be a string of a comma-separated list
870:         of names.
871:     defaultfmt : str, optional
872:         Format string used to define missing names, such as ``"f%i"``
873:         (default) or ``"fields_%02i"``.
874:     validationargs : optional
875:         A series of optional arguments used to initialize a
876:         `NameValidator`.
877: 
878:     Examples
879:     --------
880:     >>> np.lib._iotools.easy_dtype(float)
881:     dtype('float64')
882:     >>> np.lib._iotools.easy_dtype("i4, f8")
883:     dtype([('f0', '<i4'), ('f1', '<f8')])
884:     >>> np.lib._iotools.easy_dtype("i4, f8", defaultfmt="field_%03i")
885:     dtype([('field_000', '<i4'), ('field_001', '<f8')])
886: 
887:     >>> np.lib._iotools.easy_dtype((int, float, float), names="a,b,c")
888:     dtype([('a', '<i8'), ('b', '<f8'), ('c', '<f8')])
889:     >>> np.lib._iotools.easy_dtype(float, names="a,b,c")
890:     dtype([('a', '<f8'), ('b', '<f8'), ('c', '<f8')])
891: 
892:     '''
893:     try:
894:         ndtype = np.dtype(ndtype)
895:     except TypeError:
896:         validate = NameValidator(**validationargs)
897:         nbfields = len(ndtype)
898:         if names is None:
899:             names = [''] * len(ndtype)
900:         elif isinstance(names, basestring):
901:             names = names.split(",")
902:         names = validate(names, nbfields=nbfields, defaultfmt=defaultfmt)
903:         ndtype = np.dtype(dict(formats=ndtype, names=names))
904:     else:
905:         nbtypes = len(ndtype)
906:         # Explicit names
907:         if names is not None:
908:             validate = NameValidator(**validationargs)
909:             if isinstance(names, basestring):
910:                 names = names.split(",")
911:             # Simple dtype: repeat to match the nb of names
912:             if nbtypes == 0:
913:                 formats = tuple([ndtype.type] * len(names))
914:                 names = validate(names, defaultfmt=defaultfmt)
915:                 ndtype = np.dtype(list(zip(names, formats)))
916:             # Structured dtype: just validate the names as needed
917:             else:
918:                 ndtype.names = validate(names, nbfields=nbtypes,
919:                                         defaultfmt=defaultfmt)
920:         # No implicit names
921:         elif (nbtypes > 0):
922:             validate = NameValidator(**validationargs)
923:             # Default initial names : should we change the format ?
924:             if ((ndtype.names == tuple("f%i" % i for i in range(nbtypes))) and
925:                     (defaultfmt != "f%i")):
926:                 ndtype.names = validate([''] * nbtypes, defaultfmt=defaultfmt)
927:             # Explicit initial names : just validate
928:             else:
929:                 ndtype.names = validate(ndtype.names, defaultfmt=defaultfmt)
930:     return ndtype
931: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_132144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, (-1)), 'str', 'A collection of functions designed to help I/O with ascii files.\n\n')

# Assigning a Str to a Name (line 6):

# Assigning a Str to a Name (line 6):
str_132145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 16), 'str', 'restructuredtext en')
# Assigning a type to the variable '__docformat__' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), '__docformat__', str_132145)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'import sys' statement (line 8)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'import numpy' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/lib/')
import_132146 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy')

if (type(import_132146) is not StypyTypeError):

    if (import_132146 != 'pyd_module'):
        __import__(import_132146)
        sys_modules_132147 = sys.modules[import_132146]
        import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'np', sys_modules_132147.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy', import_132146)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/lib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'import numpy.core.numeric' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/lib/')
import_132148 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy.core.numeric')

if (type(import_132148) is not StypyTypeError):

    if (import_132148 != 'pyd_module'):
        __import__(import_132148)
        sys_modules_132149 = sys.modules[import_132148]
        import_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'nx', sys_modules_132149.module_type_store, module_type_store)
    else:
        import numpy.core.numeric as nx

        import_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'nx', numpy.core.numeric, module_type_store)

else:
    # Assigning a type to the variable 'numpy.core.numeric' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy.core.numeric', import_132148)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/lib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from numpy.compat import asbytes, bytes, asbytes_nested, basestring' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/lib/')
import_132150 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'numpy.compat')

if (type(import_132150) is not StypyTypeError):

    if (import_132150 != 'pyd_module'):
        __import__(import_132150)
        sys_modules_132151 = sys.modules[import_132150]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'numpy.compat', sys_modules_132151.module_type_store, module_type_store, ['asbytes', 'bytes', 'asbytes_nested', 'basestring'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_132151, sys_modules_132151.module_type_store, module_type_store)
    else:
        from numpy.compat import asbytes, bytes, asbytes_nested, basestring

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'numpy.compat', None, module_type_store, ['asbytes', 'bytes', 'asbytes_nested', 'basestring'], [asbytes, bytes, asbytes_nested, basestring])

else:
    # Assigning a type to the variable 'numpy.compat' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'numpy.compat', import_132150)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/lib/')




# Obtaining the type of the subscript
int_132152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 20), 'int')
# Getting the type of 'sys' (line 13)
sys_132153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 3), 'sys')
# Obtaining the member 'version_info' of a type (line 13)
version_info_132154 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 3), sys_132153, 'version_info')
# Obtaining the member '__getitem__' of a type (line 13)
getitem___132155 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 3), version_info_132154, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 13)
subscript_call_result_132156 = invoke(stypy.reporting.localization.Localization(__file__, 13, 3), getitem___132155, int_132152)

int_132157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 26), 'int')
# Applying the binary operator '>=' (line 13)
result_ge_132158 = python_operator(stypy.reporting.localization.Localization(__file__, 13, 3), '>=', subscript_call_result_132156, int_132157)

# Testing the type of an if condition (line 13)
if_condition_132159 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 13, 0), result_ge_132158)
# Assigning a type to the variable 'if_condition_132159' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'if_condition_132159', if_condition_132159)
# SSA begins for if statement (line 13)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 4))

# 'from builtins import bool, int, float, complex, object, str' statement (line 14)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/lib/')
import_132160 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 14, 4), 'builtins')

if (type(import_132160) is not StypyTypeError):

    if (import_132160 != 'pyd_module'):
        __import__(import_132160)
        sys_modules_132161 = sys.modules[import_132160]
        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 4), 'builtins', sys_modules_132161.module_type_store, module_type_store, ['bool', 'int', 'float', 'complex', 'object', 'str'])
        nest_module(stypy.reporting.localization.Localization(__file__, 14, 4), __file__, sys_modules_132161, sys_modules_132161.module_type_store, module_type_store)
    else:
        from builtins import bool, int, float, complex, object, str

        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 4), 'builtins', None, module_type_store, ['bool', 'int', 'float', 'complex', 'object', 'str'], [bool, int, float, complex, object, str])

else:
    # Assigning a type to the variable 'builtins' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'builtins', import_132160)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/lib/')


# Assigning a Name to a Name (line 15):

# Assigning a Name to a Name (line 15):
# Getting the type of 'str' (line 15)
str_132162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 14), 'str')
# Assigning a type to the variable 'unicode' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'unicode', str_132162)
# SSA branch for the else part of an if statement (line 13)
module_type_store.open_ssa_branch('else')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 17, 4))

# 'from __builtin__ import bool, int, float, complex, object, unicode, str' statement (line 17)
from __builtin__ import bool, int, float, complex, object, unicode, str

import_from_module(stypy.reporting.localization.Localization(__file__, 17, 4), '__builtin__', None, module_type_store, ['bool', 'int', 'float', 'complex', 'object', 'unicode', 'str'], [bool, int, float, complex, object, unicode, str])

# SSA join for if statement (line 13)
module_type_store = module_type_store.join_ssa_context()




# Obtaining the type of the subscript
int_132163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 20), 'int')
# Getting the type of 'sys' (line 20)
sys_132164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 3), 'sys')
# Obtaining the member 'version_info' of a type (line 20)
version_info_132165 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 3), sys_132164, 'version_info')
# Obtaining the member '__getitem__' of a type (line 20)
getitem___132166 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 3), version_info_132165, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 20)
subscript_call_result_132167 = invoke(stypy.reporting.localization.Localization(__file__, 20, 3), getitem___132166, int_132163)

int_132168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 26), 'int')
# Applying the binary operator '>=' (line 20)
result_ge_132169 = python_operator(stypy.reporting.localization.Localization(__file__, 20, 3), '>=', subscript_call_result_132167, int_132168)

# Testing the type of an if condition (line 20)
if_condition_132170 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 20, 0), result_ge_132169)
# Assigning a type to the variable 'if_condition_132170' (line 20)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'if_condition_132170', if_condition_132170)
# SSA begins for if statement (line 20)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

@norecursion
def _bytes_to_complex(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_bytes_to_complex'
    module_type_store = module_type_store.open_function_context('_bytes_to_complex', 21, 4, False)
    
    # Passed parameters checking function
    _bytes_to_complex.stypy_localization = localization
    _bytes_to_complex.stypy_type_of_self = None
    _bytes_to_complex.stypy_type_store = module_type_store
    _bytes_to_complex.stypy_function_name = '_bytes_to_complex'
    _bytes_to_complex.stypy_param_names_list = ['s']
    _bytes_to_complex.stypy_varargs_param_name = None
    _bytes_to_complex.stypy_kwargs_param_name = None
    _bytes_to_complex.stypy_call_defaults = defaults
    _bytes_to_complex.stypy_call_varargs = varargs
    _bytes_to_complex.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_bytes_to_complex', ['s'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_bytes_to_complex', localization, ['s'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_bytes_to_complex(...)' code ##################

    
    # Call to complex(...): (line 22)
    # Processing the call arguments (line 22)
    
    # Call to decode(...): (line 22)
    # Processing the call arguments (line 22)
    str_132174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 32), 'str', 'ascii')
    # Processing the call keyword arguments (line 22)
    kwargs_132175 = {}
    # Getting the type of 's' (line 22)
    s_132172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 23), 's', False)
    # Obtaining the member 'decode' of a type (line 22)
    decode_132173 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 23), s_132172, 'decode')
    # Calling decode(args, kwargs) (line 22)
    decode_call_result_132176 = invoke(stypy.reporting.localization.Localization(__file__, 22, 23), decode_132173, *[str_132174], **kwargs_132175)
    
    # Processing the call keyword arguments (line 22)
    kwargs_132177 = {}
    # Getting the type of 'complex' (line 22)
    complex_132171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 15), 'complex', False)
    # Calling complex(args, kwargs) (line 22)
    complex_call_result_132178 = invoke(stypy.reporting.localization.Localization(__file__, 22, 15), complex_132171, *[decode_call_result_132176], **kwargs_132177)
    
    # Assigning a type to the variable 'stypy_return_type' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'stypy_return_type', complex_call_result_132178)
    
    # ################# End of '_bytes_to_complex(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_bytes_to_complex' in the type store
    # Getting the type of 'stypy_return_type' (line 21)
    stypy_return_type_132179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_132179)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_bytes_to_complex'
    return stypy_return_type_132179

# Assigning a type to the variable '_bytes_to_complex' (line 21)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), '_bytes_to_complex', _bytes_to_complex)

@norecursion
def _bytes_to_name(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_bytes_to_name'
    module_type_store = module_type_store.open_function_context('_bytes_to_name', 24, 4, False)
    
    # Passed parameters checking function
    _bytes_to_name.stypy_localization = localization
    _bytes_to_name.stypy_type_of_self = None
    _bytes_to_name.stypy_type_store = module_type_store
    _bytes_to_name.stypy_function_name = '_bytes_to_name'
    _bytes_to_name.stypy_param_names_list = ['s']
    _bytes_to_name.stypy_varargs_param_name = None
    _bytes_to_name.stypy_kwargs_param_name = None
    _bytes_to_name.stypy_call_defaults = defaults
    _bytes_to_name.stypy_call_varargs = varargs
    _bytes_to_name.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_bytes_to_name', ['s'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_bytes_to_name', localization, ['s'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_bytes_to_name(...)' code ##################

    
    # Call to decode(...): (line 25)
    # Processing the call arguments (line 25)
    str_132182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 24), 'str', 'ascii')
    # Processing the call keyword arguments (line 25)
    kwargs_132183 = {}
    # Getting the type of 's' (line 25)
    s_132180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 15), 's', False)
    # Obtaining the member 'decode' of a type (line 25)
    decode_132181 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 15), s_132180, 'decode')
    # Calling decode(args, kwargs) (line 25)
    decode_call_result_132184 = invoke(stypy.reporting.localization.Localization(__file__, 25, 15), decode_132181, *[str_132182], **kwargs_132183)
    
    # Assigning a type to the variable 'stypy_return_type' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'stypy_return_type', decode_call_result_132184)
    
    # ################# End of '_bytes_to_name(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_bytes_to_name' in the type store
    # Getting the type of 'stypy_return_type' (line 24)
    stypy_return_type_132185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_132185)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_bytes_to_name'
    return stypy_return_type_132185

# Assigning a type to the variable '_bytes_to_name' (line 24)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), '_bytes_to_name', _bytes_to_name)
# SSA branch for the else part of an if statement (line 20)
module_type_store.open_ssa_branch('else')

# Assigning a Name to a Name (line 27):

# Assigning a Name to a Name (line 27):
# Getting the type of 'complex' (line 27)
complex_132186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 24), 'complex')
# Assigning a type to the variable '_bytes_to_complex' (line 27)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), '_bytes_to_complex', complex_132186)

# Assigning a Name to a Name (line 28):

# Assigning a Name to a Name (line 28):
# Getting the type of 'str' (line 28)
str_132187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 21), 'str')
# Assigning a type to the variable '_bytes_to_name' (line 28)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), '_bytes_to_name', str_132187)
# SSA join for if statement (line 20)
module_type_store = module_type_store.join_ssa_context()


@norecursion
def _is_string_like(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_is_string_like'
    module_type_store = module_type_store.open_function_context('_is_string_like', 31, 0, False)
    
    # Passed parameters checking function
    _is_string_like.stypy_localization = localization
    _is_string_like.stypy_type_of_self = None
    _is_string_like.stypy_type_store = module_type_store
    _is_string_like.stypy_function_name = '_is_string_like'
    _is_string_like.stypy_param_names_list = ['obj']
    _is_string_like.stypy_varargs_param_name = None
    _is_string_like.stypy_kwargs_param_name = None
    _is_string_like.stypy_call_defaults = defaults
    _is_string_like.stypy_call_varargs = varargs
    _is_string_like.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_is_string_like', ['obj'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_is_string_like', localization, ['obj'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_is_string_like(...)' code ##################

    str_132188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, (-1)), 'str', '\n    Check whether obj behaves like a string.\n    ')
    
    
    # SSA begins for try-except statement (line 35)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    # Getting the type of 'obj' (line 36)
    obj_132189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'obj')
    str_132190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 14), 'str', '')
    # Applying the binary operator '+' (line 36)
    result_add_132191 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 8), '+', obj_132189, str_132190)
    
    # SSA branch for the except part of a try statement (line 35)
    # SSA branch for the except 'Tuple' branch of a try statement (line 35)
    module_type_store.open_ssa_branch('except')
    # Getting the type of 'False' (line 38)
    False_132192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 15), 'False')
    # Assigning a type to the variable 'stypy_return_type' (line 38)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'stypy_return_type', False_132192)
    # SSA join for try-except statement (line 35)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'True' (line 39)
    True_132193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 11), 'True')
    # Assigning a type to the variable 'stypy_return_type' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'stypy_return_type', True_132193)
    
    # ################# End of '_is_string_like(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_is_string_like' in the type store
    # Getting the type of 'stypy_return_type' (line 31)
    stypy_return_type_132194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_132194)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_is_string_like'
    return stypy_return_type_132194

# Assigning a type to the variable '_is_string_like' (line 31)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 0), '_is_string_like', _is_string_like)

@norecursion
def _is_bytes_like(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_is_bytes_like'
    module_type_store = module_type_store.open_function_context('_is_bytes_like', 42, 0, False)
    
    # Passed parameters checking function
    _is_bytes_like.stypy_localization = localization
    _is_bytes_like.stypy_type_of_self = None
    _is_bytes_like.stypy_type_store = module_type_store
    _is_bytes_like.stypy_function_name = '_is_bytes_like'
    _is_bytes_like.stypy_param_names_list = ['obj']
    _is_bytes_like.stypy_varargs_param_name = None
    _is_bytes_like.stypy_kwargs_param_name = None
    _is_bytes_like.stypy_call_defaults = defaults
    _is_bytes_like.stypy_call_varargs = varargs
    _is_bytes_like.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_is_bytes_like', ['obj'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_is_bytes_like', localization, ['obj'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_is_bytes_like(...)' code ##################

    str_132195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, (-1)), 'str', '\n    Check whether obj behaves like a bytes object.\n    ')
    
    
    # SSA begins for try-except statement (line 46)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    # Getting the type of 'obj' (line 47)
    obj_132196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'obj')
    
    # Call to asbytes(...): (line 47)
    # Processing the call arguments (line 47)
    str_132198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 22), 'str', '')
    # Processing the call keyword arguments (line 47)
    kwargs_132199 = {}
    # Getting the type of 'asbytes' (line 47)
    asbytes_132197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 14), 'asbytes', False)
    # Calling asbytes(args, kwargs) (line 47)
    asbytes_call_result_132200 = invoke(stypy.reporting.localization.Localization(__file__, 47, 14), asbytes_132197, *[str_132198], **kwargs_132199)
    
    # Applying the binary operator '+' (line 47)
    result_add_132201 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 8), '+', obj_132196, asbytes_call_result_132200)
    
    # SSA branch for the except part of a try statement (line 46)
    # SSA branch for the except 'Tuple' branch of a try statement (line 46)
    module_type_store.open_ssa_branch('except')
    # Getting the type of 'False' (line 49)
    False_132202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 15), 'False')
    # Assigning a type to the variable 'stypy_return_type' (line 49)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'stypy_return_type', False_132202)
    # SSA join for try-except statement (line 46)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'True' (line 50)
    True_132203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 11), 'True')
    # Assigning a type to the variable 'stypy_return_type' (line 50)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'stypy_return_type', True_132203)
    
    # ################# End of '_is_bytes_like(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_is_bytes_like' in the type store
    # Getting the type of 'stypy_return_type' (line 42)
    stypy_return_type_132204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_132204)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_is_bytes_like'
    return stypy_return_type_132204

# Assigning a type to the variable '_is_bytes_like' (line 42)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 0), '_is_bytes_like', _is_bytes_like)

@norecursion
def _to_filehandle(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_132205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 31), 'str', 'r')
    # Getting the type of 'False' (line 53)
    False_132206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 50), 'False')
    defaults = [str_132205, False_132206]
    # Create a new context for function '_to_filehandle'
    module_type_store = module_type_store.open_function_context('_to_filehandle', 53, 0, False)
    
    # Passed parameters checking function
    _to_filehandle.stypy_localization = localization
    _to_filehandle.stypy_type_of_self = None
    _to_filehandle.stypy_type_store = module_type_store
    _to_filehandle.stypy_function_name = '_to_filehandle'
    _to_filehandle.stypy_param_names_list = ['fname', 'flag', 'return_opened']
    _to_filehandle.stypy_varargs_param_name = None
    _to_filehandle.stypy_kwargs_param_name = None
    _to_filehandle.stypy_call_defaults = defaults
    _to_filehandle.stypy_call_varargs = varargs
    _to_filehandle.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_to_filehandle', ['fname', 'flag', 'return_opened'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_to_filehandle', localization, ['fname', 'flag', 'return_opened'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_to_filehandle(...)' code ##################

    str_132207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, (-1)), 'str', "\n    Returns the filehandle corresponding to a string or a file.\n    If the string ends in '.gz', the file is automatically unzipped.\n\n    Parameters\n    ----------\n    fname : string, filehandle\n        Name of the file whose filehandle must be returned.\n    flag : string, optional\n        Flag indicating the status of the file ('r' for read, 'w' for write).\n    return_opened : boolean, optional\n        Whether to return the opening status of the file.\n    ")
    
    
    # Call to _is_string_like(...): (line 67)
    # Processing the call arguments (line 67)
    # Getting the type of 'fname' (line 67)
    fname_132209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 23), 'fname', False)
    # Processing the call keyword arguments (line 67)
    kwargs_132210 = {}
    # Getting the type of '_is_string_like' (line 67)
    _is_string_like_132208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 7), '_is_string_like', False)
    # Calling _is_string_like(args, kwargs) (line 67)
    _is_string_like_call_result_132211 = invoke(stypy.reporting.localization.Localization(__file__, 67, 7), _is_string_like_132208, *[fname_132209], **kwargs_132210)
    
    # Testing the type of an if condition (line 67)
    if_condition_132212 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 67, 4), _is_string_like_call_result_132211)
    # Assigning a type to the variable 'if_condition_132212' (line 67)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'if_condition_132212', if_condition_132212)
    # SSA begins for if statement (line 67)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Call to endswith(...): (line 68)
    # Processing the call arguments (line 68)
    str_132215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 26), 'str', '.gz')
    # Processing the call keyword arguments (line 68)
    kwargs_132216 = {}
    # Getting the type of 'fname' (line 68)
    fname_132213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 11), 'fname', False)
    # Obtaining the member 'endswith' of a type (line 68)
    endswith_132214 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 11), fname_132213, 'endswith')
    # Calling endswith(args, kwargs) (line 68)
    endswith_call_result_132217 = invoke(stypy.reporting.localization.Localization(__file__, 68, 11), endswith_132214, *[str_132215], **kwargs_132216)
    
    # Testing the type of an if condition (line 68)
    if_condition_132218 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 68, 8), endswith_call_result_132217)
    # Assigning a type to the variable 'if_condition_132218' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'if_condition_132218', if_condition_132218)
    # SSA begins for if statement (line 68)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 69, 12))
    
    # 'import gzip' statement (line 69)
    import gzip

    import_module(stypy.reporting.localization.Localization(__file__, 69, 12), 'gzip', gzip, module_type_store)
    
    
    # Assigning a Call to a Name (line 70):
    
    # Assigning a Call to a Name (line 70):
    
    # Call to open(...): (line 70)
    # Processing the call arguments (line 70)
    # Getting the type of 'fname' (line 70)
    fname_132221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 28), 'fname', False)
    # Getting the type of 'flag' (line 70)
    flag_132222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 35), 'flag', False)
    # Processing the call keyword arguments (line 70)
    kwargs_132223 = {}
    # Getting the type of 'gzip' (line 70)
    gzip_132219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 18), 'gzip', False)
    # Obtaining the member 'open' of a type (line 70)
    open_132220 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 18), gzip_132219, 'open')
    # Calling open(args, kwargs) (line 70)
    open_call_result_132224 = invoke(stypy.reporting.localization.Localization(__file__, 70, 18), open_132220, *[fname_132221, flag_132222], **kwargs_132223)
    
    # Assigning a type to the variable 'fhd' (line 70)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 12), 'fhd', open_call_result_132224)
    # SSA branch for the else part of an if statement (line 68)
    module_type_store.open_ssa_branch('else')
    
    
    # Call to endswith(...): (line 71)
    # Processing the call arguments (line 71)
    str_132227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 28), 'str', '.bz2')
    # Processing the call keyword arguments (line 71)
    kwargs_132228 = {}
    # Getting the type of 'fname' (line 71)
    fname_132225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 13), 'fname', False)
    # Obtaining the member 'endswith' of a type (line 71)
    endswith_132226 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 13), fname_132225, 'endswith')
    # Calling endswith(args, kwargs) (line 71)
    endswith_call_result_132229 = invoke(stypy.reporting.localization.Localization(__file__, 71, 13), endswith_132226, *[str_132227], **kwargs_132228)
    
    # Testing the type of an if condition (line 71)
    if_condition_132230 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 71, 13), endswith_call_result_132229)
    # Assigning a type to the variable 'if_condition_132230' (line 71)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 13), 'if_condition_132230', if_condition_132230)
    # SSA begins for if statement (line 71)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 72, 12))
    
    # 'import bz2' statement (line 72)
    import bz2

    import_module(stypy.reporting.localization.Localization(__file__, 72, 12), 'bz2', bz2, module_type_store)
    
    
    # Assigning a Call to a Name (line 73):
    
    # Assigning a Call to a Name (line 73):
    
    # Call to BZ2File(...): (line 73)
    # Processing the call arguments (line 73)
    # Getting the type of 'fname' (line 73)
    fname_132233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 30), 'fname', False)
    # Processing the call keyword arguments (line 73)
    kwargs_132234 = {}
    # Getting the type of 'bz2' (line 73)
    bz2_132231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 18), 'bz2', False)
    # Obtaining the member 'BZ2File' of a type (line 73)
    BZ2File_132232 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 18), bz2_132231, 'BZ2File')
    # Calling BZ2File(args, kwargs) (line 73)
    BZ2File_call_result_132235 = invoke(stypy.reporting.localization.Localization(__file__, 73, 18), BZ2File_132232, *[fname_132233], **kwargs_132234)
    
    # Assigning a type to the variable 'fhd' (line 73)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 12), 'fhd', BZ2File_call_result_132235)
    # SSA branch for the else part of an if statement (line 71)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 75):
    
    # Assigning a Call to a Name (line 75):
    
    # Call to file(...): (line 75)
    # Processing the call arguments (line 75)
    # Getting the type of 'fname' (line 75)
    fname_132237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 23), 'fname', False)
    # Getting the type of 'flag' (line 75)
    flag_132238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 30), 'flag', False)
    # Processing the call keyword arguments (line 75)
    kwargs_132239 = {}
    # Getting the type of 'file' (line 75)
    file_132236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 18), 'file', False)
    # Calling file(args, kwargs) (line 75)
    file_call_result_132240 = invoke(stypy.reporting.localization.Localization(__file__, 75, 18), file_132236, *[fname_132237, flag_132238], **kwargs_132239)
    
    # Assigning a type to the variable 'fhd' (line 75)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 12), 'fhd', file_call_result_132240)
    # SSA join for if statement (line 71)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 68)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Name (line 76):
    
    # Assigning a Name to a Name (line 76):
    # Getting the type of 'True' (line 76)
    True_132241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 17), 'True')
    # Assigning a type to the variable 'opened' (line 76)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'opened', True_132241)
    # SSA branch for the else part of an if statement (line 67)
    module_type_store.open_ssa_branch('else')
    
    # Type idiom detected: calculating its left and rigth part (line 77)
    str_132242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 24), 'str', 'seek')
    # Getting the type of 'fname' (line 77)
    fname_132243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 17), 'fname')
    
    (may_be_132244, more_types_in_union_132245) = may_provide_member(str_132242, fname_132243)

    if may_be_132244:

        if more_types_in_union_132245:
            # Runtime conditional SSA (line 77)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'fname' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 9), 'fname', remove_not_member_provider_from_union(fname_132243, 'seek'))
        
        # Assigning a Name to a Name (line 78):
        
        # Assigning a Name to a Name (line 78):
        # Getting the type of 'fname' (line 78)
        fname_132246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 14), 'fname')
        # Assigning a type to the variable 'fhd' (line 78)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'fhd', fname_132246)
        
        # Assigning a Name to a Name (line 79):
        
        # Assigning a Name to a Name (line 79):
        # Getting the type of 'False' (line 79)
        False_132247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 17), 'False')
        # Assigning a type to the variable 'opened' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'opened', False_132247)

        if more_types_in_union_132245:
            # Runtime conditional SSA for else branch (line 77)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_132244) or more_types_in_union_132245):
        # Assigning a type to the variable 'fname' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 9), 'fname', remove_member_provider_from_union(fname_132243, 'seek'))
        
        # Call to ValueError(...): (line 81)
        # Processing the call arguments (line 81)
        str_132249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 25), 'str', 'fname must be a string or file handle')
        # Processing the call keyword arguments (line 81)
        kwargs_132250 = {}
        # Getting the type of 'ValueError' (line 81)
        ValueError_132248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 14), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 81)
        ValueError_call_result_132251 = invoke(stypy.reporting.localization.Localization(__file__, 81, 14), ValueError_132248, *[str_132249], **kwargs_132250)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 81, 8), ValueError_call_result_132251, 'raise parameter', BaseException)

        if (may_be_132244 and more_types_in_union_132245):
            # SSA join for if statement (line 77)
            module_type_store = module_type_store.join_ssa_context()


    
    # SSA join for if statement (line 67)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'return_opened' (line 82)
    return_opened_132252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 7), 'return_opened')
    # Testing the type of an if condition (line 82)
    if_condition_132253 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 82, 4), return_opened_132252)
    # Assigning a type to the variable 'if_condition_132253' (line 82)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'if_condition_132253', if_condition_132253)
    # SSA begins for if statement (line 82)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'tuple' (line 83)
    tuple_132254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 83)
    # Adding element type (line 83)
    # Getting the type of 'fhd' (line 83)
    fhd_132255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 15), 'fhd')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 15), tuple_132254, fhd_132255)
    # Adding element type (line 83)
    # Getting the type of 'opened' (line 83)
    opened_132256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 20), 'opened')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 15), tuple_132254, opened_132256)
    
    # Assigning a type to the variable 'stypy_return_type' (line 83)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'stypy_return_type', tuple_132254)
    # SSA join for if statement (line 82)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'fhd' (line 84)
    fhd_132257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 11), 'fhd')
    # Assigning a type to the variable 'stypy_return_type' (line 84)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 4), 'stypy_return_type', fhd_132257)
    
    # ################# End of '_to_filehandle(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_to_filehandle' in the type store
    # Getting the type of 'stypy_return_type' (line 53)
    stypy_return_type_132258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_132258)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_to_filehandle'
    return stypy_return_type_132258

# Assigning a type to the variable '_to_filehandle' (line 53)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 0), '_to_filehandle', _to_filehandle)

@norecursion
def has_nested_fields(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'has_nested_fields'
    module_type_store = module_type_store.open_function_context('has_nested_fields', 87, 0, False)
    
    # Passed parameters checking function
    has_nested_fields.stypy_localization = localization
    has_nested_fields.stypy_type_of_self = None
    has_nested_fields.stypy_type_store = module_type_store
    has_nested_fields.stypy_function_name = 'has_nested_fields'
    has_nested_fields.stypy_param_names_list = ['ndtype']
    has_nested_fields.stypy_varargs_param_name = None
    has_nested_fields.stypy_kwargs_param_name = None
    has_nested_fields.stypy_call_defaults = defaults
    has_nested_fields.stypy_call_varargs = varargs
    has_nested_fields.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'has_nested_fields', ['ndtype'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'has_nested_fields', localization, ['ndtype'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'has_nested_fields(...)' code ##################

    str_132259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, (-1)), 'str', "\n    Returns whether one or several fields of a dtype are nested.\n\n    Parameters\n    ----------\n    ndtype : dtype\n        Data-type of a structured array.\n\n    Raises\n    ------\n    AttributeError\n        If `ndtype` does not have a `names` attribute.\n\n    Examples\n    --------\n    >>> dt = np.dtype([('name', 'S4'), ('x', float), ('y', float)])\n    >>> np.lib._iotools.has_nested_fields(dt)\n    False\n\n    ")
    
    
    # Evaluating a boolean operation
    # Getting the type of 'ndtype' (line 108)
    ndtype_132260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 16), 'ndtype')
    # Obtaining the member 'names' of a type (line 108)
    names_132261 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 16), ndtype_132260, 'names')
    
    # Obtaining an instance of the builtin type 'tuple' (line 108)
    tuple_132262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 32), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 108)
    
    # Applying the binary operator 'or' (line 108)
    result_or_keyword_132263 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 16), 'or', names_132261, tuple_132262)
    
    # Testing the type of a for loop iterable (line 108)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 108, 4), result_or_keyword_132263)
    # Getting the type of the for loop variable (line 108)
    for_loop_var_132264 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 108, 4), result_or_keyword_132263)
    # Assigning a type to the variable 'name' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), 'name', for_loop_var_132264)
    # SSA begins for a for statement (line 108)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'name' (line 109)
    name_132265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 18), 'name')
    # Getting the type of 'ndtype' (line 109)
    ndtype_132266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 11), 'ndtype')
    # Obtaining the member '__getitem__' of a type (line 109)
    getitem___132267 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 11), ndtype_132266, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 109)
    subscript_call_result_132268 = invoke(stypy.reporting.localization.Localization(__file__, 109, 11), getitem___132267, name_132265)
    
    # Obtaining the member 'names' of a type (line 109)
    names_132269 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 11), subscript_call_result_132268, 'names')
    # Testing the type of an if condition (line 109)
    if_condition_132270 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 109, 8), names_132269)
    # Assigning a type to the variable 'if_condition_132270' (line 109)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'if_condition_132270', if_condition_132270)
    # SSA begins for if statement (line 109)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'True' (line 110)
    True_132271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 19), 'True')
    # Assigning a type to the variable 'stypy_return_type' (line 110)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 12), 'stypy_return_type', True_132271)
    # SSA join for if statement (line 109)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'False' (line 111)
    False_132272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 11), 'False')
    # Assigning a type to the variable 'stypy_return_type' (line 111)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 4), 'stypy_return_type', False_132272)
    
    # ################# End of 'has_nested_fields(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'has_nested_fields' in the type store
    # Getting the type of 'stypy_return_type' (line 87)
    stypy_return_type_132273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_132273)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'has_nested_fields'
    return stypy_return_type_132273

# Assigning a type to the variable 'has_nested_fields' (line 87)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 0), 'has_nested_fields', has_nested_fields)

@norecursion
def flatten_dtype(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'False' (line 114)
    False_132274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 39), 'False')
    defaults = [False_132274]
    # Create a new context for function 'flatten_dtype'
    module_type_store = module_type_store.open_function_context('flatten_dtype', 114, 0, False)
    
    # Passed parameters checking function
    flatten_dtype.stypy_localization = localization
    flatten_dtype.stypy_type_of_self = None
    flatten_dtype.stypy_type_store = module_type_store
    flatten_dtype.stypy_function_name = 'flatten_dtype'
    flatten_dtype.stypy_param_names_list = ['ndtype', 'flatten_base']
    flatten_dtype.stypy_varargs_param_name = None
    flatten_dtype.stypy_kwargs_param_name = None
    flatten_dtype.stypy_call_defaults = defaults
    flatten_dtype.stypy_call_varargs = varargs
    flatten_dtype.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'flatten_dtype', ['ndtype', 'flatten_base'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'flatten_dtype', localization, ['ndtype', 'flatten_base'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'flatten_dtype(...)' code ##################

    str_132275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, (-1)), 'str', "\n    Unpack a structured data-type by collapsing nested fields and/or fields\n    with a shape.\n\n    Note that the field names are lost.\n\n    Parameters\n    ----------\n    ndtype : dtype\n        The datatype to collapse\n    flatten_base : {False, True}, optional\n        Whether to transform a field with a shape into several fields or not.\n\n    Examples\n    --------\n    >>> dt = np.dtype([('name', 'S4'), ('x', float), ('y', float),\n    ...                ('block', int, (2, 3))])\n    >>> np.lib._iotools.flatten_dtype(dt)\n    [dtype('|S4'), dtype('float64'), dtype('float64'), dtype('int32')]\n    >>> np.lib._iotools.flatten_dtype(dt, flatten_base=True)\n    [dtype('|S4'), dtype('float64'), dtype('float64'), dtype('int32'),\n     dtype('int32'), dtype('int32'), dtype('int32'), dtype('int32'),\n     dtype('int32')]\n\n    ")
    
    # Assigning a Attribute to a Name (line 140):
    
    # Assigning a Attribute to a Name (line 140):
    # Getting the type of 'ndtype' (line 140)
    ndtype_132276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 12), 'ndtype')
    # Obtaining the member 'names' of a type (line 140)
    names_132277 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 12), ndtype_132276, 'names')
    # Assigning a type to the variable 'names' (line 140)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 4), 'names', names_132277)
    
    # Type idiom detected: calculating its left and rigth part (line 141)
    # Getting the type of 'names' (line 141)
    names_132278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 7), 'names')
    # Getting the type of 'None' (line 141)
    None_132279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 16), 'None')
    
    (may_be_132280, more_types_in_union_132281) = may_be_none(names_132278, None_132279)

    if may_be_132280:

        if more_types_in_union_132281:
            # Runtime conditional SSA (line 141)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Getting the type of 'flatten_base' (line 142)
        flatten_base_132282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 11), 'flatten_base')
        # Testing the type of an if condition (line 142)
        if_condition_132283 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 142, 8), flatten_base_132282)
        # Assigning a type to the variable 'if_condition_132283' (line 142)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'if_condition_132283', if_condition_132283)
        # SSA begins for if statement (line 142)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Obtaining an instance of the builtin type 'list' (line 143)
        list_132284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 143)
        # Adding element type (line 143)
        # Getting the type of 'ndtype' (line 143)
        ndtype_132285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 20), 'ndtype')
        # Obtaining the member 'base' of a type (line 143)
        base_132286 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 20), ndtype_132285, 'base')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 143, 19), list_132284, base_132286)
        
        
        # Call to int(...): (line 143)
        # Processing the call arguments (line 143)
        
        # Call to prod(...): (line 143)
        # Processing the call arguments (line 143)
        # Getting the type of 'ndtype' (line 143)
        ndtype_132290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 47), 'ndtype', False)
        # Obtaining the member 'shape' of a type (line 143)
        shape_132291 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 47), ndtype_132290, 'shape')
        # Processing the call keyword arguments (line 143)
        kwargs_132292 = {}
        # Getting the type of 'np' (line 143)
        np_132288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 39), 'np', False)
        # Obtaining the member 'prod' of a type (line 143)
        prod_132289 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 39), np_132288, 'prod')
        # Calling prod(args, kwargs) (line 143)
        prod_call_result_132293 = invoke(stypy.reporting.localization.Localization(__file__, 143, 39), prod_132289, *[shape_132291], **kwargs_132292)
        
        # Processing the call keyword arguments (line 143)
        kwargs_132294 = {}
        # Getting the type of 'int' (line 143)
        int_132287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 35), 'int', False)
        # Calling int(args, kwargs) (line 143)
        int_call_result_132295 = invoke(stypy.reporting.localization.Localization(__file__, 143, 35), int_132287, *[prod_call_result_132293], **kwargs_132294)
        
        # Applying the binary operator '*' (line 143)
        result_mul_132296 = python_operator(stypy.reporting.localization.Localization(__file__, 143, 19), '*', list_132284, int_call_result_132295)
        
        # Assigning a type to the variable 'stypy_return_type' (line 143)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 12), 'stypy_return_type', result_mul_132296)
        # SSA join for if statement (line 142)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Obtaining an instance of the builtin type 'list' (line 144)
        list_132297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 144)
        # Adding element type (line 144)
        # Getting the type of 'ndtype' (line 144)
        ndtype_132298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 16), 'ndtype')
        # Obtaining the member 'base' of a type (line 144)
        base_132299 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 16), ndtype_132298, 'base')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 15), list_132297, base_132299)
        
        # Assigning a type to the variable 'stypy_return_type' (line 144)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'stypy_return_type', list_132297)

        if more_types_in_union_132281:
            # Runtime conditional SSA for else branch (line 141)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_132280) or more_types_in_union_132281):
        
        # Assigning a List to a Name (line 146):
        
        # Assigning a List to a Name (line 146):
        
        # Obtaining an instance of the builtin type 'list' (line 146)
        list_132300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 146)
        
        # Assigning a type to the variable 'types' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'types', list_132300)
        
        # Getting the type of 'names' (line 147)
        names_132301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 21), 'names')
        # Testing the type of a for loop iterable (line 147)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 147, 8), names_132301)
        # Getting the type of the for loop variable (line 147)
        for_loop_var_132302 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 147, 8), names_132301)
        # Assigning a type to the variable 'field' (line 147)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'field', for_loop_var_132302)
        # SSA begins for a for statement (line 147)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Subscript to a Name (line 148):
        
        # Assigning a Subscript to a Name (line 148):
        
        # Obtaining the type of the subscript
        # Getting the type of 'field' (line 148)
        field_132303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 33), 'field')
        # Getting the type of 'ndtype' (line 148)
        ndtype_132304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 19), 'ndtype')
        # Obtaining the member 'fields' of a type (line 148)
        fields_132305 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 19), ndtype_132304, 'fields')
        # Obtaining the member '__getitem__' of a type (line 148)
        getitem___132306 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 19), fields_132305, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 148)
        subscript_call_result_132307 = invoke(stypy.reporting.localization.Localization(__file__, 148, 19), getitem___132306, field_132303)
        
        # Assigning a type to the variable 'info' (line 148)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 12), 'info', subscript_call_result_132307)
        
        # Assigning a Call to a Name (line 149):
        
        # Assigning a Call to a Name (line 149):
        
        # Call to flatten_dtype(...): (line 149)
        # Processing the call arguments (line 149)
        
        # Obtaining the type of the subscript
        int_132309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 41), 'int')
        # Getting the type of 'info' (line 149)
        info_132310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 36), 'info', False)
        # Obtaining the member '__getitem__' of a type (line 149)
        getitem___132311 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 36), info_132310, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 149)
        subscript_call_result_132312 = invoke(stypy.reporting.localization.Localization(__file__, 149, 36), getitem___132311, int_132309)
        
        # Getting the type of 'flatten_base' (line 149)
        flatten_base_132313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 45), 'flatten_base', False)
        # Processing the call keyword arguments (line 149)
        kwargs_132314 = {}
        # Getting the type of 'flatten_dtype' (line 149)
        flatten_dtype_132308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 22), 'flatten_dtype', False)
        # Calling flatten_dtype(args, kwargs) (line 149)
        flatten_dtype_call_result_132315 = invoke(stypy.reporting.localization.Localization(__file__, 149, 22), flatten_dtype_132308, *[subscript_call_result_132312, flatten_base_132313], **kwargs_132314)
        
        # Assigning a type to the variable 'flat_dt' (line 149)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 12), 'flat_dt', flatten_dtype_call_result_132315)
        
        # Call to extend(...): (line 150)
        # Processing the call arguments (line 150)
        # Getting the type of 'flat_dt' (line 150)
        flat_dt_132318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 25), 'flat_dt', False)
        # Processing the call keyword arguments (line 150)
        kwargs_132319 = {}
        # Getting the type of 'types' (line 150)
        types_132316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 12), 'types', False)
        # Obtaining the member 'extend' of a type (line 150)
        extend_132317 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 12), types_132316, 'extend')
        # Calling extend(args, kwargs) (line 150)
        extend_call_result_132320 = invoke(stypy.reporting.localization.Localization(__file__, 150, 12), extend_132317, *[flat_dt_132318], **kwargs_132319)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'types' (line 151)
        types_132321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 15), 'types')
        # Assigning a type to the variable 'stypy_return_type' (line 151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'stypy_return_type', types_132321)

        if (may_be_132280 and more_types_in_union_132281):
            # SSA join for if statement (line 141)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # ################# End of 'flatten_dtype(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'flatten_dtype' in the type store
    # Getting the type of 'stypy_return_type' (line 114)
    stypy_return_type_132322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_132322)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'flatten_dtype'
    return stypy_return_type_132322

# Assigning a type to the variable 'flatten_dtype' (line 114)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 0), 'flatten_dtype', flatten_dtype)
# Declaration of the 'LineSplitter' class

class LineSplitter(object, ):
    str_132323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, (-1)), 'str', "\n    Object to split a string at a given delimiter or at given places.\n\n    Parameters\n    ----------\n    delimiter : str, int, or sequence of ints, optional\n        If a string, character used to delimit consecutive fields.\n        If an integer or a sequence of integers, width(s) of each field.\n    comments : str, optional\n        Character used to mark the beginning of a comment. Default is '#'.\n    autostrip : bool, optional\n        Whether to strip each individual field. Default is True.\n\n    ")

    @norecursion
    def autostrip(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'autostrip'
        module_type_store = module_type_store.open_function_context('autostrip', 170, 4, False)
        # Assigning a type to the variable 'self' (line 171)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        LineSplitter.autostrip.__dict__.__setitem__('stypy_localization', localization)
        LineSplitter.autostrip.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        LineSplitter.autostrip.__dict__.__setitem__('stypy_type_store', module_type_store)
        LineSplitter.autostrip.__dict__.__setitem__('stypy_function_name', 'LineSplitter.autostrip')
        LineSplitter.autostrip.__dict__.__setitem__('stypy_param_names_list', ['method'])
        LineSplitter.autostrip.__dict__.__setitem__('stypy_varargs_param_name', None)
        LineSplitter.autostrip.__dict__.__setitem__('stypy_kwargs_param_name', None)
        LineSplitter.autostrip.__dict__.__setitem__('stypy_call_defaults', defaults)
        LineSplitter.autostrip.__dict__.__setitem__('stypy_call_varargs', varargs)
        LineSplitter.autostrip.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        LineSplitter.autostrip.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LineSplitter.autostrip', ['method'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'autostrip', localization, ['method'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'autostrip(...)' code ##################

        str_132324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, (-1)), 'str', '\n        Wrapper to strip each member of the output of `method`.\n\n        Parameters\n        ----------\n        method : function\n            Function that takes a single argument and returns a sequence of\n            strings.\n\n        Returns\n        -------\n        wrapped : function\n            The result of wrapping `method`. `wrapped` takes a single input\n            argument and returns a list of strings that are stripped of\n            white-space.\n\n        ')

        @norecursion
        def _stypy_temp_lambda_32(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_32'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_32', 188, 15, True)
            # Passed parameters checking function
            _stypy_temp_lambda_32.stypy_localization = localization
            _stypy_temp_lambda_32.stypy_type_of_self = None
            _stypy_temp_lambda_32.stypy_type_store = module_type_store
            _stypy_temp_lambda_32.stypy_function_name = '_stypy_temp_lambda_32'
            _stypy_temp_lambda_32.stypy_param_names_list = ['input']
            _stypy_temp_lambda_32.stypy_varargs_param_name = None
            _stypy_temp_lambda_32.stypy_kwargs_param_name = None
            _stypy_temp_lambda_32.stypy_call_defaults = defaults
            _stypy_temp_lambda_32.stypy_call_varargs = varargs
            _stypy_temp_lambda_32.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_32', ['input'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_32', ['input'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            # Calculating list comprehension
            # Calculating comprehension expression
            
            # Call to method(...): (line 188)
            # Processing the call arguments (line 188)
            # Getting the type of 'input' (line 188)
            input_132330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 56), 'input', False)
            # Processing the call keyword arguments (line 188)
            kwargs_132331 = {}
            # Getting the type of 'method' (line 188)
            method_132329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 49), 'method', False)
            # Calling method(args, kwargs) (line 188)
            method_call_result_132332 = invoke(stypy.reporting.localization.Localization(__file__, 188, 49), method_132329, *[input_132330], **kwargs_132331)
            
            comprehension_132333 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 188, 30), method_call_result_132332)
            # Assigning a type to the variable '_' (line 188)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 30), '_', comprehension_132333)
            
            # Call to strip(...): (line 188)
            # Processing the call keyword arguments (line 188)
            kwargs_132327 = {}
            # Getting the type of '_' (line 188)
            __132325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 30), '_', False)
            # Obtaining the member 'strip' of a type (line 188)
            strip_132326 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 30), __132325, 'strip')
            # Calling strip(args, kwargs) (line 188)
            strip_call_result_132328 = invoke(stypy.reporting.localization.Localization(__file__, 188, 30), strip_132326, *[], **kwargs_132327)
            
            list_132334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 30), 'list')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 188, 30), list_132334, strip_call_result_132328)
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 188)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 15), 'stypy_return_type', list_132334)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_32' in the type store
            # Getting the type of 'stypy_return_type' (line 188)
            stypy_return_type_132335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 15), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_132335)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_32'
            return stypy_return_type_132335

        # Assigning a type to the variable '_stypy_temp_lambda_32' (line 188)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 15), '_stypy_temp_lambda_32', _stypy_temp_lambda_32)
        # Getting the type of '_stypy_temp_lambda_32' (line 188)
        _stypy_temp_lambda_32_132336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 15), '_stypy_temp_lambda_32')
        # Assigning a type to the variable 'stypy_return_type' (line 188)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 8), 'stypy_return_type', _stypy_temp_lambda_32_132336)
        
        # ################# End of 'autostrip(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'autostrip' in the type store
        # Getting the type of 'stypy_return_type' (line 170)
        stypy_return_type_132337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_132337)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'autostrip'
        return stypy_return_type_132337


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 191)
        None_132338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 33), 'None')
        
        # Call to asbytes(...): (line 191)
        # Processing the call arguments (line 191)
        str_132340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 56), 'str', '#')
        # Processing the call keyword arguments (line 191)
        kwargs_132341 = {}
        # Getting the type of 'asbytes' (line 191)
        asbytes_132339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 48), 'asbytes', False)
        # Calling asbytes(args, kwargs) (line 191)
        asbytes_call_result_132342 = invoke(stypy.reporting.localization.Localization(__file__, 191, 48), asbytes_132339, *[str_132340], **kwargs_132341)
        
        # Getting the type of 'True' (line 191)
        True_132343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 72), 'True')
        defaults = [None_132338, asbytes_call_result_132342, True_132343]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 191, 4, False)
        # Assigning a type to the variable 'self' (line 192)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LineSplitter.__init__', ['delimiter', 'comments', 'autostrip'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['delimiter', 'comments', 'autostrip'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 192):
        
        # Assigning a Name to a Attribute (line 192):
        # Getting the type of 'comments' (line 192)
        comments_132344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 24), 'comments')
        # Getting the type of 'self' (line 192)
        self_132345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 8), 'self')
        # Setting the type of the member 'comments' of a type (line 192)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 8), self_132345, 'comments', comments_132344)
        
        # Type idiom detected: calculating its left and rigth part (line 194)
        # Getting the type of 'unicode' (line 194)
        unicode_132346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 33), 'unicode')
        # Getting the type of 'delimiter' (line 194)
        delimiter_132347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 22), 'delimiter')
        
        (may_be_132348, more_types_in_union_132349) = may_be_subtype(unicode_132346, delimiter_132347)

        if may_be_132348:

            if more_types_in_union_132349:
                # Runtime conditional SSA (line 194)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'delimiter' (line 194)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 8), 'delimiter', remove_not_subtype_from_union(delimiter_132347, unicode))
            
            # Assigning a Call to a Name (line 195):
            
            # Assigning a Call to a Name (line 195):
            
            # Call to encode(...): (line 195)
            # Processing the call arguments (line 195)
            str_132352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 41), 'str', 'ascii')
            # Processing the call keyword arguments (line 195)
            kwargs_132353 = {}
            # Getting the type of 'delimiter' (line 195)
            delimiter_132350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 24), 'delimiter', False)
            # Obtaining the member 'encode' of a type (line 195)
            encode_132351 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 24), delimiter_132350, 'encode')
            # Calling encode(args, kwargs) (line 195)
            encode_call_result_132354 = invoke(stypy.reporting.localization.Localization(__file__, 195, 24), encode_132351, *[str_132352], **kwargs_132353)
            
            # Assigning a type to the variable 'delimiter' (line 195)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 12), 'delimiter', encode_call_result_132354)

            if more_types_in_union_132349:
                # SSA join for if statement (line 194)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'delimiter' (line 196)
        delimiter_132355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 12), 'delimiter')
        # Getting the type of 'None' (line 196)
        None_132356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 25), 'None')
        # Applying the binary operator 'is' (line 196)
        result_is__132357 = python_operator(stypy.reporting.localization.Localization(__file__, 196, 12), 'is', delimiter_132355, None_132356)
        
        
        # Call to _is_bytes_like(...): (line 196)
        # Processing the call arguments (line 196)
        # Getting the type of 'delimiter' (line 196)
        delimiter_132359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 49), 'delimiter', False)
        # Processing the call keyword arguments (line 196)
        kwargs_132360 = {}
        # Getting the type of '_is_bytes_like' (line 196)
        _is_bytes_like_132358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 34), '_is_bytes_like', False)
        # Calling _is_bytes_like(args, kwargs) (line 196)
        _is_bytes_like_call_result_132361 = invoke(stypy.reporting.localization.Localization(__file__, 196, 34), _is_bytes_like_132358, *[delimiter_132359], **kwargs_132360)
        
        # Applying the binary operator 'or' (line 196)
        result_or_keyword_132362 = python_operator(stypy.reporting.localization.Localization(__file__, 196, 11), 'or', result_is__132357, _is_bytes_like_call_result_132361)
        
        # Testing the type of an if condition (line 196)
        if_condition_132363 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 196, 8), result_or_keyword_132362)
        # Assigning a type to the variable 'if_condition_132363' (line 196)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'if_condition_132363', if_condition_132363)
        # SSA begins for if statement (line 196)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BoolOp to a Name (line 197):
        
        # Assigning a BoolOp to a Name (line 197):
        
        # Evaluating a boolean operation
        # Getting the type of 'delimiter' (line 197)
        delimiter_132364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 24), 'delimiter')
        # Getting the type of 'None' (line 197)
        None_132365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 37), 'None')
        # Applying the binary operator 'or' (line 197)
        result_or_keyword_132366 = python_operator(stypy.reporting.localization.Localization(__file__, 197, 24), 'or', delimiter_132364, None_132365)
        
        # Assigning a type to the variable 'delimiter' (line 197)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 12), 'delimiter', result_or_keyword_132366)
        
        # Assigning a Attribute to a Name (line 198):
        
        # Assigning a Attribute to a Name (line 198):
        # Getting the type of 'self' (line 198)
        self_132367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 24), 'self')
        # Obtaining the member '_delimited_splitter' of a type (line 198)
        _delimited_splitter_132368 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 24), self_132367, '_delimited_splitter')
        # Assigning a type to the variable '_handyman' (line 198)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 12), '_handyman', _delimited_splitter_132368)
        # SSA branch for the else part of an if statement (line 196)
        module_type_store.open_ssa_branch('else')
        
        # Type idiom detected: calculating its left and rigth part (line 200)
        str_132369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 32), 'str', '__iter__')
        # Getting the type of 'delimiter' (line 200)
        delimiter_132370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 21), 'delimiter')
        
        (may_be_132371, more_types_in_union_132372) = may_provide_member(str_132369, delimiter_132370)

        if may_be_132371:

            if more_types_in_union_132372:
                # Runtime conditional SSA (line 200)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'delimiter' (line 200)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 13), 'delimiter', remove_not_member_provider_from_union(delimiter_132370, '__iter__'))
            
            # Assigning a Attribute to a Name (line 201):
            
            # Assigning a Attribute to a Name (line 201):
            # Getting the type of 'self' (line 201)
            self_132373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 24), 'self')
            # Obtaining the member '_variablewidth_splitter' of a type (line 201)
            _variablewidth_splitter_132374 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 24), self_132373, '_variablewidth_splitter')
            # Assigning a type to the variable '_handyman' (line 201)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 12), '_handyman', _variablewidth_splitter_132374)
            
            # Assigning a Call to a Name (line 202):
            
            # Assigning a Call to a Name (line 202):
            
            # Call to cumsum(...): (line 202)
            # Processing the call arguments (line 202)
            
            # Obtaining an instance of the builtin type 'list' (line 202)
            list_132377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 28), 'list')
            # Adding type elements to the builtin type 'list' instance (line 202)
            # Adding element type (line 202)
            int_132378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 29), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 28), list_132377, int_132378)
            
            
            # Call to list(...): (line 202)
            # Processing the call arguments (line 202)
            # Getting the type of 'delimiter' (line 202)
            delimiter_132380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 39), 'delimiter', False)
            # Processing the call keyword arguments (line 202)
            kwargs_132381 = {}
            # Getting the type of 'list' (line 202)
            list_132379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 34), 'list', False)
            # Calling list(args, kwargs) (line 202)
            list_call_result_132382 = invoke(stypy.reporting.localization.Localization(__file__, 202, 34), list_132379, *[delimiter_132380], **kwargs_132381)
            
            # Applying the binary operator '+' (line 202)
            result_add_132383 = python_operator(stypy.reporting.localization.Localization(__file__, 202, 28), '+', list_132377, list_call_result_132382)
            
            # Processing the call keyword arguments (line 202)
            kwargs_132384 = {}
            # Getting the type of 'np' (line 202)
            np_132375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 18), 'np', False)
            # Obtaining the member 'cumsum' of a type (line 202)
            cumsum_132376 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 18), np_132375, 'cumsum')
            # Calling cumsum(args, kwargs) (line 202)
            cumsum_call_result_132385 = invoke(stypy.reporting.localization.Localization(__file__, 202, 18), cumsum_132376, *[result_add_132383], **kwargs_132384)
            
            # Assigning a type to the variable 'idx' (line 202)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 12), 'idx', cumsum_call_result_132385)
            
            # Assigning a ListComp to a Name (line 203):
            
            # Assigning a ListComp to a Name (line 203):
            # Calculating list comprehension
            # Calculating comprehension expression
            
            # Call to zip(...): (line 203)
            # Processing the call arguments (line 203)
            
            # Obtaining the type of the subscript
            int_132392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 60), 'int')
            slice_132393 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 203, 55), None, int_132392, None)
            # Getting the type of 'idx' (line 203)
            idx_132394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 55), 'idx', False)
            # Obtaining the member '__getitem__' of a type (line 203)
            getitem___132395 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 55), idx_132394, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 203)
            subscript_call_result_132396 = invoke(stypy.reporting.localization.Localization(__file__, 203, 55), getitem___132395, slice_132393)
            
            
            # Obtaining the type of the subscript
            int_132397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 69), 'int')
            slice_132398 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 203, 65), int_132397, None, None)
            # Getting the type of 'idx' (line 203)
            idx_132399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 65), 'idx', False)
            # Obtaining the member '__getitem__' of a type (line 203)
            getitem___132400 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 65), idx_132399, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 203)
            subscript_call_result_132401 = invoke(stypy.reporting.localization.Localization(__file__, 203, 65), getitem___132400, slice_132398)
            
            # Processing the call keyword arguments (line 203)
            kwargs_132402 = {}
            # Getting the type of 'zip' (line 203)
            zip_132391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 51), 'zip', False)
            # Calling zip(args, kwargs) (line 203)
            zip_call_result_132403 = invoke(stypy.reporting.localization.Localization(__file__, 203, 51), zip_132391, *[subscript_call_result_132396, subscript_call_result_132401], **kwargs_132402)
            
            comprehension_132404 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 203, 25), zip_call_result_132403)
            # Assigning a type to the variable 'i' (line 203)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 25), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 203, 25), comprehension_132404))
            # Assigning a type to the variable 'j' (line 203)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 25), 'j', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 203, 25), comprehension_132404))
            
            # Call to slice(...): (line 203)
            # Processing the call arguments (line 203)
            # Getting the type of 'i' (line 203)
            i_132387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 31), 'i', False)
            # Getting the type of 'j' (line 203)
            j_132388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 34), 'j', False)
            # Processing the call keyword arguments (line 203)
            kwargs_132389 = {}
            # Getting the type of 'slice' (line 203)
            slice_132386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 25), 'slice', False)
            # Calling slice(args, kwargs) (line 203)
            slice_call_result_132390 = invoke(stypy.reporting.localization.Localization(__file__, 203, 25), slice_132386, *[i_132387, j_132388], **kwargs_132389)
            
            list_132405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 25), 'list')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 203, 25), list_132405, slice_call_result_132390)
            # Assigning a type to the variable 'delimiter' (line 203)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 12), 'delimiter', list_132405)

            if more_types_in_union_132372:
                # Runtime conditional SSA for else branch (line 200)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_132371) or more_types_in_union_132372):
            # Assigning a type to the variable 'delimiter' (line 200)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 13), 'delimiter', remove_member_provider_from_union(delimiter_132370, '__iter__'))
            
            
            # Call to int(...): (line 205)
            # Processing the call arguments (line 205)
            # Getting the type of 'delimiter' (line 205)
            delimiter_132407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 17), 'delimiter', False)
            # Processing the call keyword arguments (line 205)
            kwargs_132408 = {}
            # Getting the type of 'int' (line 205)
            int_132406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 13), 'int', False)
            # Calling int(args, kwargs) (line 205)
            int_call_result_132409 = invoke(stypy.reporting.localization.Localization(__file__, 205, 13), int_132406, *[delimiter_132407], **kwargs_132408)
            
            # Testing the type of an if condition (line 205)
            if_condition_132410 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 205, 13), int_call_result_132409)
            # Assigning a type to the variable 'if_condition_132410' (line 205)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 13), 'if_condition_132410', if_condition_132410)
            # SSA begins for if statement (line 205)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Tuple to a Tuple (line 206):
            
            # Assigning a Attribute to a Name (line 206):
            # Getting the type of 'self' (line 207)
            self_132411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 20), 'self')
            # Obtaining the member '_fixedwidth_splitter' of a type (line 207)
            _fixedwidth_splitter_132412 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 20), self_132411, '_fixedwidth_splitter')
            # Assigning a type to the variable 'tuple_assignment_132130' (line 206)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 12), 'tuple_assignment_132130', _fixedwidth_splitter_132412)
            
            # Assigning a Call to a Name (line 206):
            
            # Call to int(...): (line 207)
            # Processing the call arguments (line 207)
            # Getting the type of 'delimiter' (line 207)
            delimiter_132414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 51), 'delimiter', False)
            # Processing the call keyword arguments (line 207)
            kwargs_132415 = {}
            # Getting the type of 'int' (line 207)
            int_132413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 47), 'int', False)
            # Calling int(args, kwargs) (line 207)
            int_call_result_132416 = invoke(stypy.reporting.localization.Localization(__file__, 207, 47), int_132413, *[delimiter_132414], **kwargs_132415)
            
            # Assigning a type to the variable 'tuple_assignment_132131' (line 206)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 12), 'tuple_assignment_132131', int_call_result_132416)
            
            # Assigning a Name to a Name (line 206):
            # Getting the type of 'tuple_assignment_132130' (line 206)
            tuple_assignment_132130_132417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 12), 'tuple_assignment_132130')
            # Assigning a type to the variable '_handyman' (line 206)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 13), '_handyman', tuple_assignment_132130_132417)
            
            # Assigning a Name to a Name (line 206):
            # Getting the type of 'tuple_assignment_132131' (line 206)
            tuple_assignment_132131_132418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 12), 'tuple_assignment_132131')
            # Assigning a type to the variable 'delimiter' (line 206)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 24), 'delimiter', tuple_assignment_132131_132418)
            # SSA branch for the else part of an if statement (line 205)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Tuple to a Tuple (line 209):
            
            # Assigning a Attribute to a Name (line 209):
            # Getting the type of 'self' (line 209)
            self_132419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 38), 'self')
            # Obtaining the member '_delimited_splitter' of a type (line 209)
            _delimited_splitter_132420 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 38), self_132419, '_delimited_splitter')
            # Assigning a type to the variable 'tuple_assignment_132132' (line 209)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 12), 'tuple_assignment_132132', _delimited_splitter_132420)
            
            # Assigning a Name to a Name (line 209):
            # Getting the type of 'None' (line 209)
            None_132421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 64), 'None')
            # Assigning a type to the variable 'tuple_assignment_132133' (line 209)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 12), 'tuple_assignment_132133', None_132421)
            
            # Assigning a Name to a Name (line 209):
            # Getting the type of 'tuple_assignment_132132' (line 209)
            tuple_assignment_132132_132422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 12), 'tuple_assignment_132132')
            # Assigning a type to the variable '_handyman' (line 209)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 13), '_handyman', tuple_assignment_132132_132422)
            
            # Assigning a Name to a Name (line 209):
            # Getting the type of 'tuple_assignment_132133' (line 209)
            tuple_assignment_132133_132423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 12), 'tuple_assignment_132133')
            # Assigning a type to the variable 'delimiter' (line 209)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 24), 'delimiter', tuple_assignment_132133_132423)
            # SSA join for if statement (line 205)
            module_type_store = module_type_store.join_ssa_context()
            

            if (may_be_132371 and more_types_in_union_132372):
                # SSA join for if statement (line 200)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for if statement (line 196)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 210):
        
        # Assigning a Name to a Attribute (line 210):
        # Getting the type of 'delimiter' (line 210)
        delimiter_132424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 25), 'delimiter')
        # Getting the type of 'self' (line 210)
        self_132425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 8), 'self')
        # Setting the type of the member 'delimiter' of a type (line 210)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 8), self_132425, 'delimiter', delimiter_132424)
        
        # Getting the type of 'autostrip' (line 211)
        autostrip_132426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 11), 'autostrip')
        # Testing the type of an if condition (line 211)
        if_condition_132427 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 211, 8), autostrip_132426)
        # Assigning a type to the variable 'if_condition_132427' (line 211)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 8), 'if_condition_132427', if_condition_132427)
        # SSA begins for if statement (line 211)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Attribute (line 212):
        
        # Assigning a Call to a Attribute (line 212):
        
        # Call to autostrip(...): (line 212)
        # Processing the call arguments (line 212)
        # Getting the type of '_handyman' (line 212)
        _handyman_132430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 44), '_handyman', False)
        # Processing the call keyword arguments (line 212)
        kwargs_132431 = {}
        # Getting the type of 'self' (line 212)
        self_132428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 29), 'self', False)
        # Obtaining the member 'autostrip' of a type (line 212)
        autostrip_132429 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 29), self_132428, 'autostrip')
        # Calling autostrip(args, kwargs) (line 212)
        autostrip_call_result_132432 = invoke(stypy.reporting.localization.Localization(__file__, 212, 29), autostrip_132429, *[_handyman_132430], **kwargs_132431)
        
        # Getting the type of 'self' (line 212)
        self_132433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 12), 'self')
        # Setting the type of the member '_handyman' of a type (line 212)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 12), self_132433, '_handyman', autostrip_call_result_132432)
        # SSA branch for the else part of an if statement (line 211)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Name to a Attribute (line 214):
        
        # Assigning a Name to a Attribute (line 214):
        # Getting the type of '_handyman' (line 214)
        _handyman_132434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 29), '_handyman')
        # Getting the type of 'self' (line 214)
        self_132435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 12), 'self')
        # Setting the type of the member '_handyman' of a type (line 214)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 12), self_132435, '_handyman', _handyman_132434)
        # SSA join for if statement (line 211)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def _delimited_splitter(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_delimited_splitter'
        module_type_store = module_type_store.open_function_context('_delimited_splitter', 217, 4, False)
        # Assigning a type to the variable 'self' (line 218)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        LineSplitter._delimited_splitter.__dict__.__setitem__('stypy_localization', localization)
        LineSplitter._delimited_splitter.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        LineSplitter._delimited_splitter.__dict__.__setitem__('stypy_type_store', module_type_store)
        LineSplitter._delimited_splitter.__dict__.__setitem__('stypy_function_name', 'LineSplitter._delimited_splitter')
        LineSplitter._delimited_splitter.__dict__.__setitem__('stypy_param_names_list', ['line'])
        LineSplitter._delimited_splitter.__dict__.__setitem__('stypy_varargs_param_name', None)
        LineSplitter._delimited_splitter.__dict__.__setitem__('stypy_kwargs_param_name', None)
        LineSplitter._delimited_splitter.__dict__.__setitem__('stypy_call_defaults', defaults)
        LineSplitter._delimited_splitter.__dict__.__setitem__('stypy_call_varargs', varargs)
        LineSplitter._delimited_splitter.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        LineSplitter._delimited_splitter.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LineSplitter._delimited_splitter', ['line'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_delimited_splitter', localization, ['line'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_delimited_splitter(...)' code ##################

        
        
        # Getting the type of 'self' (line 218)
        self_132436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 11), 'self')
        # Obtaining the member 'comments' of a type (line 218)
        comments_132437 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 11), self_132436, 'comments')
        # Getting the type of 'None' (line 218)
        None_132438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 32), 'None')
        # Applying the binary operator 'isnot' (line 218)
        result_is_not_132439 = python_operator(stypy.reporting.localization.Localization(__file__, 218, 11), 'isnot', comments_132437, None_132438)
        
        # Testing the type of an if condition (line 218)
        if_condition_132440 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 218, 8), result_is_not_132439)
        # Assigning a type to the variable 'if_condition_132440' (line 218)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 8), 'if_condition_132440', if_condition_132440)
        # SSA begins for if statement (line 218)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Name (line 219):
        
        # Assigning a Subscript to a Name (line 219):
        
        # Obtaining the type of the subscript
        int_132441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 45), 'int')
        
        # Call to split(...): (line 219)
        # Processing the call arguments (line 219)
        # Getting the type of 'self' (line 219)
        self_132444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 30), 'self', False)
        # Obtaining the member 'comments' of a type (line 219)
        comments_132445 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 30), self_132444, 'comments')
        # Processing the call keyword arguments (line 219)
        kwargs_132446 = {}
        # Getting the type of 'line' (line 219)
        line_132442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 19), 'line', False)
        # Obtaining the member 'split' of a type (line 219)
        split_132443 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 19), line_132442, 'split')
        # Calling split(args, kwargs) (line 219)
        split_call_result_132447 = invoke(stypy.reporting.localization.Localization(__file__, 219, 19), split_132443, *[comments_132445], **kwargs_132446)
        
        # Obtaining the member '__getitem__' of a type (line 219)
        getitem___132448 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 19), split_call_result_132447, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 219)
        subscript_call_result_132449 = invoke(stypy.reporting.localization.Localization(__file__, 219, 19), getitem___132448, int_132441)
        
        # Assigning a type to the variable 'line' (line 219)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 12), 'line', subscript_call_result_132449)
        # SSA join for if statement (line 218)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 220):
        
        # Assigning a Call to a Name (line 220):
        
        # Call to strip(...): (line 220)
        # Processing the call arguments (line 220)
        
        # Call to asbytes(...): (line 220)
        # Processing the call arguments (line 220)
        str_132453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 34), 'str', ' \r\n')
        # Processing the call keyword arguments (line 220)
        kwargs_132454 = {}
        # Getting the type of 'asbytes' (line 220)
        asbytes_132452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 26), 'asbytes', False)
        # Calling asbytes(args, kwargs) (line 220)
        asbytes_call_result_132455 = invoke(stypy.reporting.localization.Localization(__file__, 220, 26), asbytes_132452, *[str_132453], **kwargs_132454)
        
        # Processing the call keyword arguments (line 220)
        kwargs_132456 = {}
        # Getting the type of 'line' (line 220)
        line_132450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 15), 'line', False)
        # Obtaining the member 'strip' of a type (line 220)
        strip_132451 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 15), line_132450, 'strip')
        # Calling strip(args, kwargs) (line 220)
        strip_call_result_132457 = invoke(stypy.reporting.localization.Localization(__file__, 220, 15), strip_132451, *[asbytes_call_result_132455], **kwargs_132456)
        
        # Assigning a type to the variable 'line' (line 220)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 8), 'line', strip_call_result_132457)
        
        
        # Getting the type of 'line' (line 221)
        line_132458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 15), 'line')
        # Applying the 'not' unary operator (line 221)
        result_not__132459 = python_operator(stypy.reporting.localization.Localization(__file__, 221, 11), 'not', line_132458)
        
        # Testing the type of an if condition (line 221)
        if_condition_132460 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 221, 8), result_not__132459)
        # Assigning a type to the variable 'if_condition_132460' (line 221)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 8), 'if_condition_132460', if_condition_132460)
        # SSA begins for if statement (line 221)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Obtaining an instance of the builtin type 'list' (line 222)
        list_132461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 222)
        
        # Assigning a type to the variable 'stypy_return_type' (line 222)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 12), 'stypy_return_type', list_132461)
        # SSA join for if statement (line 221)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to split(...): (line 223)
        # Processing the call arguments (line 223)
        # Getting the type of 'self' (line 223)
        self_132464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 26), 'self', False)
        # Obtaining the member 'delimiter' of a type (line 223)
        delimiter_132465 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 26), self_132464, 'delimiter')
        # Processing the call keyword arguments (line 223)
        kwargs_132466 = {}
        # Getting the type of 'line' (line 223)
        line_132462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 15), 'line', False)
        # Obtaining the member 'split' of a type (line 223)
        split_132463 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 15), line_132462, 'split')
        # Calling split(args, kwargs) (line 223)
        split_call_result_132467 = invoke(stypy.reporting.localization.Localization(__file__, 223, 15), split_132463, *[delimiter_132465], **kwargs_132466)
        
        # Assigning a type to the variable 'stypy_return_type' (line 223)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'stypy_return_type', split_call_result_132467)
        
        # ################# End of '_delimited_splitter(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_delimited_splitter' in the type store
        # Getting the type of 'stypy_return_type' (line 217)
        stypy_return_type_132468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_132468)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_delimited_splitter'
        return stypy_return_type_132468


    @norecursion
    def _fixedwidth_splitter(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_fixedwidth_splitter'
        module_type_store = module_type_store.open_function_context('_fixedwidth_splitter', 226, 4, False)
        # Assigning a type to the variable 'self' (line 227)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        LineSplitter._fixedwidth_splitter.__dict__.__setitem__('stypy_localization', localization)
        LineSplitter._fixedwidth_splitter.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        LineSplitter._fixedwidth_splitter.__dict__.__setitem__('stypy_type_store', module_type_store)
        LineSplitter._fixedwidth_splitter.__dict__.__setitem__('stypy_function_name', 'LineSplitter._fixedwidth_splitter')
        LineSplitter._fixedwidth_splitter.__dict__.__setitem__('stypy_param_names_list', ['line'])
        LineSplitter._fixedwidth_splitter.__dict__.__setitem__('stypy_varargs_param_name', None)
        LineSplitter._fixedwidth_splitter.__dict__.__setitem__('stypy_kwargs_param_name', None)
        LineSplitter._fixedwidth_splitter.__dict__.__setitem__('stypy_call_defaults', defaults)
        LineSplitter._fixedwidth_splitter.__dict__.__setitem__('stypy_call_varargs', varargs)
        LineSplitter._fixedwidth_splitter.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        LineSplitter._fixedwidth_splitter.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LineSplitter._fixedwidth_splitter', ['line'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_fixedwidth_splitter', localization, ['line'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_fixedwidth_splitter(...)' code ##################

        
        
        # Getting the type of 'self' (line 227)
        self_132469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 11), 'self')
        # Obtaining the member 'comments' of a type (line 227)
        comments_132470 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 11), self_132469, 'comments')
        # Getting the type of 'None' (line 227)
        None_132471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 32), 'None')
        # Applying the binary operator 'isnot' (line 227)
        result_is_not_132472 = python_operator(stypy.reporting.localization.Localization(__file__, 227, 11), 'isnot', comments_132470, None_132471)
        
        # Testing the type of an if condition (line 227)
        if_condition_132473 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 227, 8), result_is_not_132472)
        # Assigning a type to the variable 'if_condition_132473' (line 227)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'if_condition_132473', if_condition_132473)
        # SSA begins for if statement (line 227)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Name (line 228):
        
        # Assigning a Subscript to a Name (line 228):
        
        # Obtaining the type of the subscript
        int_132474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 45), 'int')
        
        # Call to split(...): (line 228)
        # Processing the call arguments (line 228)
        # Getting the type of 'self' (line 228)
        self_132477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 30), 'self', False)
        # Obtaining the member 'comments' of a type (line 228)
        comments_132478 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 30), self_132477, 'comments')
        # Processing the call keyword arguments (line 228)
        kwargs_132479 = {}
        # Getting the type of 'line' (line 228)
        line_132475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 19), 'line', False)
        # Obtaining the member 'split' of a type (line 228)
        split_132476 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 19), line_132475, 'split')
        # Calling split(args, kwargs) (line 228)
        split_call_result_132480 = invoke(stypy.reporting.localization.Localization(__file__, 228, 19), split_132476, *[comments_132478], **kwargs_132479)
        
        # Obtaining the member '__getitem__' of a type (line 228)
        getitem___132481 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 19), split_call_result_132480, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 228)
        subscript_call_result_132482 = invoke(stypy.reporting.localization.Localization(__file__, 228, 19), getitem___132481, int_132474)
        
        # Assigning a type to the variable 'line' (line 228)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 12), 'line', subscript_call_result_132482)
        # SSA join for if statement (line 227)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 229):
        
        # Assigning a Call to a Name (line 229):
        
        # Call to strip(...): (line 229)
        # Processing the call arguments (line 229)
        
        # Call to asbytes(...): (line 229)
        # Processing the call arguments (line 229)
        str_132486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 34), 'str', '\r\n')
        # Processing the call keyword arguments (line 229)
        kwargs_132487 = {}
        # Getting the type of 'asbytes' (line 229)
        asbytes_132485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 26), 'asbytes', False)
        # Calling asbytes(args, kwargs) (line 229)
        asbytes_call_result_132488 = invoke(stypy.reporting.localization.Localization(__file__, 229, 26), asbytes_132485, *[str_132486], **kwargs_132487)
        
        # Processing the call keyword arguments (line 229)
        kwargs_132489 = {}
        # Getting the type of 'line' (line 229)
        line_132483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 15), 'line', False)
        # Obtaining the member 'strip' of a type (line 229)
        strip_132484 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 15), line_132483, 'strip')
        # Calling strip(args, kwargs) (line 229)
        strip_call_result_132490 = invoke(stypy.reporting.localization.Localization(__file__, 229, 15), strip_132484, *[asbytes_call_result_132488], **kwargs_132489)
        
        # Assigning a type to the variable 'line' (line 229)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 8), 'line', strip_call_result_132490)
        
        
        # Getting the type of 'line' (line 230)
        line_132491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 15), 'line')
        # Applying the 'not' unary operator (line 230)
        result_not__132492 = python_operator(stypy.reporting.localization.Localization(__file__, 230, 11), 'not', line_132491)
        
        # Testing the type of an if condition (line 230)
        if_condition_132493 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 230, 8), result_not__132492)
        # Assigning a type to the variable 'if_condition_132493' (line 230)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 8), 'if_condition_132493', if_condition_132493)
        # SSA begins for if statement (line 230)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Obtaining an instance of the builtin type 'list' (line 231)
        list_132494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 231)
        
        # Assigning a type to the variable 'stypy_return_type' (line 231)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 12), 'stypy_return_type', list_132494)
        # SSA join for if statement (line 230)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Attribute to a Name (line 232):
        
        # Assigning a Attribute to a Name (line 232):
        # Getting the type of 'self' (line 232)
        self_132495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 16), 'self')
        # Obtaining the member 'delimiter' of a type (line 232)
        delimiter_132496 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 16), self_132495, 'delimiter')
        # Assigning a type to the variable 'fixed' (line 232)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 8), 'fixed', delimiter_132496)
        
        # Assigning a ListComp to a Name (line 233):
        
        # Assigning a ListComp to a Name (line 233):
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to range(...): (line 233)
        # Processing the call arguments (line 233)
        int_132505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 53), 'int')
        
        # Call to len(...): (line 233)
        # Processing the call arguments (line 233)
        # Getting the type of 'line' (line 233)
        line_132507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 60), 'line', False)
        # Processing the call keyword arguments (line 233)
        kwargs_132508 = {}
        # Getting the type of 'len' (line 233)
        len_132506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 56), 'len', False)
        # Calling len(args, kwargs) (line 233)
        len_call_result_132509 = invoke(stypy.reporting.localization.Localization(__file__, 233, 56), len_132506, *[line_132507], **kwargs_132508)
        
        # Getting the type of 'fixed' (line 233)
        fixed_132510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 67), 'fixed', False)
        # Processing the call keyword arguments (line 233)
        kwargs_132511 = {}
        # Getting the type of 'range' (line 233)
        range_132504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 47), 'range', False)
        # Calling range(args, kwargs) (line 233)
        range_call_result_132512 = invoke(stypy.reporting.localization.Localization(__file__, 233, 47), range_132504, *[int_132505, len_call_result_132509, fixed_132510], **kwargs_132511)
        
        comprehension_132513 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 233, 18), range_call_result_132512)
        # Assigning a type to the variable 'i' (line 233)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 18), 'i', comprehension_132513)
        
        # Call to slice(...): (line 233)
        # Processing the call arguments (line 233)
        # Getting the type of 'i' (line 233)
        i_132498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 24), 'i', False)
        # Getting the type of 'i' (line 233)
        i_132499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 27), 'i', False)
        # Getting the type of 'fixed' (line 233)
        fixed_132500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 31), 'fixed', False)
        # Applying the binary operator '+' (line 233)
        result_add_132501 = python_operator(stypy.reporting.localization.Localization(__file__, 233, 27), '+', i_132499, fixed_132500)
        
        # Processing the call keyword arguments (line 233)
        kwargs_132502 = {}
        # Getting the type of 'slice' (line 233)
        slice_132497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 18), 'slice', False)
        # Calling slice(args, kwargs) (line 233)
        slice_call_result_132503 = invoke(stypy.reporting.localization.Localization(__file__, 233, 18), slice_132497, *[i_132498, result_add_132501], **kwargs_132502)
        
        list_132514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 18), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 233, 18), list_132514, slice_call_result_132503)
        # Assigning a type to the variable 'slices' (line 233)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 8), 'slices', list_132514)
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'slices' (line 234)
        slices_132519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 33), 'slices')
        comprehension_132520 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 234, 16), slices_132519)
        # Assigning a type to the variable 's' (line 234)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 16), 's', comprehension_132520)
        
        # Obtaining the type of the subscript
        # Getting the type of 's' (line 234)
        s_132515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 21), 's')
        # Getting the type of 'line' (line 234)
        line_132516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 16), 'line')
        # Obtaining the member '__getitem__' of a type (line 234)
        getitem___132517 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 234, 16), line_132516, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 234)
        subscript_call_result_132518 = invoke(stypy.reporting.localization.Localization(__file__, 234, 16), getitem___132517, s_132515)
        
        list_132521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 16), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 234, 16), list_132521, subscript_call_result_132518)
        # Assigning a type to the variable 'stypy_return_type' (line 234)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 8), 'stypy_return_type', list_132521)
        
        # ################# End of '_fixedwidth_splitter(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_fixedwidth_splitter' in the type store
        # Getting the type of 'stypy_return_type' (line 226)
        stypy_return_type_132522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_132522)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_fixedwidth_splitter'
        return stypy_return_type_132522


    @norecursion
    def _variablewidth_splitter(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_variablewidth_splitter'
        module_type_store = module_type_store.open_function_context('_variablewidth_splitter', 237, 4, False)
        # Assigning a type to the variable 'self' (line 238)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        LineSplitter._variablewidth_splitter.__dict__.__setitem__('stypy_localization', localization)
        LineSplitter._variablewidth_splitter.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        LineSplitter._variablewidth_splitter.__dict__.__setitem__('stypy_type_store', module_type_store)
        LineSplitter._variablewidth_splitter.__dict__.__setitem__('stypy_function_name', 'LineSplitter._variablewidth_splitter')
        LineSplitter._variablewidth_splitter.__dict__.__setitem__('stypy_param_names_list', ['line'])
        LineSplitter._variablewidth_splitter.__dict__.__setitem__('stypy_varargs_param_name', None)
        LineSplitter._variablewidth_splitter.__dict__.__setitem__('stypy_kwargs_param_name', None)
        LineSplitter._variablewidth_splitter.__dict__.__setitem__('stypy_call_defaults', defaults)
        LineSplitter._variablewidth_splitter.__dict__.__setitem__('stypy_call_varargs', varargs)
        LineSplitter._variablewidth_splitter.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        LineSplitter._variablewidth_splitter.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LineSplitter._variablewidth_splitter', ['line'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_variablewidth_splitter', localization, ['line'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_variablewidth_splitter(...)' code ##################

        
        
        # Getting the type of 'self' (line 238)
        self_132523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 11), 'self')
        # Obtaining the member 'comments' of a type (line 238)
        comments_132524 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 11), self_132523, 'comments')
        # Getting the type of 'None' (line 238)
        None_132525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 32), 'None')
        # Applying the binary operator 'isnot' (line 238)
        result_is_not_132526 = python_operator(stypy.reporting.localization.Localization(__file__, 238, 11), 'isnot', comments_132524, None_132525)
        
        # Testing the type of an if condition (line 238)
        if_condition_132527 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 238, 8), result_is_not_132526)
        # Assigning a type to the variable 'if_condition_132527' (line 238)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 8), 'if_condition_132527', if_condition_132527)
        # SSA begins for if statement (line 238)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Name (line 239):
        
        # Assigning a Subscript to a Name (line 239):
        
        # Obtaining the type of the subscript
        int_132528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 45), 'int')
        
        # Call to split(...): (line 239)
        # Processing the call arguments (line 239)
        # Getting the type of 'self' (line 239)
        self_132531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 30), 'self', False)
        # Obtaining the member 'comments' of a type (line 239)
        comments_132532 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 30), self_132531, 'comments')
        # Processing the call keyword arguments (line 239)
        kwargs_132533 = {}
        # Getting the type of 'line' (line 239)
        line_132529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 19), 'line', False)
        # Obtaining the member 'split' of a type (line 239)
        split_132530 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 19), line_132529, 'split')
        # Calling split(args, kwargs) (line 239)
        split_call_result_132534 = invoke(stypy.reporting.localization.Localization(__file__, 239, 19), split_132530, *[comments_132532], **kwargs_132533)
        
        # Obtaining the member '__getitem__' of a type (line 239)
        getitem___132535 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 19), split_call_result_132534, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 239)
        subscript_call_result_132536 = invoke(stypy.reporting.localization.Localization(__file__, 239, 19), getitem___132535, int_132528)
        
        # Assigning a type to the variable 'line' (line 239)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 12), 'line', subscript_call_result_132536)
        # SSA join for if statement (line 238)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'line' (line 240)
        line_132537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 15), 'line')
        # Applying the 'not' unary operator (line 240)
        result_not__132538 = python_operator(stypy.reporting.localization.Localization(__file__, 240, 11), 'not', line_132537)
        
        # Testing the type of an if condition (line 240)
        if_condition_132539 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 240, 8), result_not__132538)
        # Assigning a type to the variable 'if_condition_132539' (line 240)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'if_condition_132539', if_condition_132539)
        # SSA begins for if statement (line 240)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Obtaining an instance of the builtin type 'list' (line 241)
        list_132540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 241)
        
        # Assigning a type to the variable 'stypy_return_type' (line 241)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 12), 'stypy_return_type', list_132540)
        # SSA join for if statement (line 240)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Attribute to a Name (line 242):
        
        # Assigning a Attribute to a Name (line 242):
        # Getting the type of 'self' (line 242)
        self_132541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 17), 'self')
        # Obtaining the member 'delimiter' of a type (line 242)
        delimiter_132542 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 17), self_132541, 'delimiter')
        # Assigning a type to the variable 'slices' (line 242)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 8), 'slices', delimiter_132542)
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'slices' (line 243)
        slices_132547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 33), 'slices')
        comprehension_132548 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 243, 16), slices_132547)
        # Assigning a type to the variable 's' (line 243)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 16), 's', comprehension_132548)
        
        # Obtaining the type of the subscript
        # Getting the type of 's' (line 243)
        s_132543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 21), 's')
        # Getting the type of 'line' (line 243)
        line_132544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 16), 'line')
        # Obtaining the member '__getitem__' of a type (line 243)
        getitem___132545 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 16), line_132544, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 243)
        subscript_call_result_132546 = invoke(stypy.reporting.localization.Localization(__file__, 243, 16), getitem___132545, s_132543)
        
        list_132549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 16), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 243, 16), list_132549, subscript_call_result_132546)
        # Assigning a type to the variable 'stypy_return_type' (line 243)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 8), 'stypy_return_type', list_132549)
        
        # ################# End of '_variablewidth_splitter(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_variablewidth_splitter' in the type store
        # Getting the type of 'stypy_return_type' (line 237)
        stypy_return_type_132550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_132550)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_variablewidth_splitter'
        return stypy_return_type_132550


    @norecursion
    def __call__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__call__'
        module_type_store = module_type_store.open_function_context('__call__', 246, 4, False)
        # Assigning a type to the variable 'self' (line 247)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        LineSplitter.__call__.__dict__.__setitem__('stypy_localization', localization)
        LineSplitter.__call__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        LineSplitter.__call__.__dict__.__setitem__('stypy_type_store', module_type_store)
        LineSplitter.__call__.__dict__.__setitem__('stypy_function_name', 'LineSplitter.__call__')
        LineSplitter.__call__.__dict__.__setitem__('stypy_param_names_list', ['line'])
        LineSplitter.__call__.__dict__.__setitem__('stypy_varargs_param_name', None)
        LineSplitter.__call__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        LineSplitter.__call__.__dict__.__setitem__('stypy_call_defaults', defaults)
        LineSplitter.__call__.__dict__.__setitem__('stypy_call_varargs', varargs)
        LineSplitter.__call__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        LineSplitter.__call__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LineSplitter.__call__', ['line'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__call__', localization, ['line'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__call__(...)' code ##################

        
        # Call to _handyman(...): (line 247)
        # Processing the call arguments (line 247)
        # Getting the type of 'line' (line 247)
        line_132553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 30), 'line', False)
        # Processing the call keyword arguments (line 247)
        kwargs_132554 = {}
        # Getting the type of 'self' (line 247)
        self_132551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 15), 'self', False)
        # Obtaining the member '_handyman' of a type (line 247)
        _handyman_132552 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 15), self_132551, '_handyman')
        # Calling _handyman(args, kwargs) (line 247)
        _handyman_call_result_132555 = invoke(stypy.reporting.localization.Localization(__file__, 247, 15), _handyman_132552, *[line_132553], **kwargs_132554)
        
        # Assigning a type to the variable 'stypy_return_type' (line 247)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 8), 'stypy_return_type', _handyman_call_result_132555)
        
        # ################# End of '__call__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__call__' in the type store
        # Getting the type of 'stypy_return_type' (line 246)
        stypy_return_type_132556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_132556)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__call__'
        return stypy_return_type_132556


# Assigning a type to the variable 'LineSplitter' (line 154)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 0), 'LineSplitter', LineSplitter)
# Declaration of the 'NameValidator' class

class NameValidator(object, ):
    str_132557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, (-1)), 'str', '\n    Object to validate a list of strings to use as field names.\n\n    The strings are stripped of any non alphanumeric character, and spaces\n    are replaced by \'_\'. During instantiation, the user can define a list\n    of names to exclude, as well as a list of invalid characters. Names in\n    the exclusion list are appended a \'_\' character.\n\n    Once an instance has been created, it can be called with a list of\n    names, and a list of valid names will be created.  The `__call__`\n    method accepts an optional keyword "default" that sets the default name\n    in case of ambiguity. By default this is \'f\', so that names will\n    default to `f0`, `f1`, etc.\n\n    Parameters\n    ----------\n    excludelist : sequence, optional\n        A list of names to exclude. This list is appended to the default\n        list [\'return\', \'file\', \'print\']. Excluded names are appended an\n        underscore: for example, `file` becomes `file_` if supplied.\n    deletechars : str, optional\n        A string combining invalid characters that must be deleted from the\n        names.\n    case_sensitive : {True, False, \'upper\', \'lower\'}, optional\n        * If True, field names are case-sensitive.\n        * If False or \'upper\', field names are converted to upper case.\n        * If \'lower\', field names are converted to lower case.\n\n        The default value is True.\n    replace_space : \'_\', optional\n        Character(s) used in replacement of white spaces.\n\n    Notes\n    -----\n    Calling an instance of `NameValidator` is the same as calling its\n    method `validate`.\n\n    Examples\n    --------\n    >>> validator = np.lib._iotools.NameValidator()\n    >>> validator([\'file\', \'field2\', \'with space\', \'CaSe\'])\n    [\'file_\', \'field2\', \'with_space\', \'CaSe\']\n\n    >>> validator = np.lib._iotools.NameValidator(excludelist=[\'excl\'],\n                                                  deletechars=\'q\',\n                                                  case_sensitive=\'False\')\n    >>> validator([\'excl\', \'field2\', \'no_q\', \'with space\', \'CaSe\'])\n    [\'excl_\', \'field2\', \'no_\', \'with_space\', \'case\']\n\n    ')
    
    # Assigning a List to a Name (line 302):
    
    # Assigning a Call to a Name (line 303):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 306)
        None_132558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 35), 'None')
        # Getting the type of 'None' (line 306)
        None_132559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 53), 'None')
        # Getting the type of 'None' (line 307)
        None_132560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 32), 'None')
        str_132561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 52), 'str', '_')
        defaults = [None_132558, None_132559, None_132560, str_132561]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 306, 4, False)
        # Assigning a type to the variable 'self' (line 307)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 307, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NameValidator.__init__', ['excludelist', 'deletechars', 'case_sensitive', 'replace_space'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['excludelist', 'deletechars', 'case_sensitive', 'replace_space'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Type idiom detected: calculating its left and rigth part (line 309)
        # Getting the type of 'excludelist' (line 309)
        excludelist_132562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 11), 'excludelist')
        # Getting the type of 'None' (line 309)
        None_132563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 26), 'None')
        
        (may_be_132564, more_types_in_union_132565) = may_be_none(excludelist_132562, None_132563)

        if may_be_132564:

            if more_types_in_union_132565:
                # Runtime conditional SSA (line 309)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a List to a Name (line 310):
            
            # Assigning a List to a Name (line 310):
            
            # Obtaining an instance of the builtin type 'list' (line 310)
            list_132566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 26), 'list')
            # Adding type elements to the builtin type 'list' instance (line 310)
            
            # Assigning a type to the variable 'excludelist' (line 310)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 12), 'excludelist', list_132566)

            if more_types_in_union_132565:
                # SSA join for if statement (line 309)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Call to extend(...): (line 311)
        # Processing the call arguments (line 311)
        # Getting the type of 'self' (line 311)
        self_132569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 27), 'self', False)
        # Obtaining the member 'defaultexcludelist' of a type (line 311)
        defaultexcludelist_132570 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 27), self_132569, 'defaultexcludelist')
        # Processing the call keyword arguments (line 311)
        kwargs_132571 = {}
        # Getting the type of 'excludelist' (line 311)
        excludelist_132567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 8), 'excludelist', False)
        # Obtaining the member 'extend' of a type (line 311)
        extend_132568 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 8), excludelist_132567, 'extend')
        # Calling extend(args, kwargs) (line 311)
        extend_call_result_132572 = invoke(stypy.reporting.localization.Localization(__file__, 311, 8), extend_132568, *[defaultexcludelist_132570], **kwargs_132571)
        
        
        # Assigning a Name to a Attribute (line 312):
        
        # Assigning a Name to a Attribute (line 312):
        # Getting the type of 'excludelist' (line 312)
        excludelist_132573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 27), 'excludelist')
        # Getting the type of 'self' (line 312)
        self_132574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 8), 'self')
        # Setting the type of the member 'excludelist' of a type (line 312)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 8), self_132574, 'excludelist', excludelist_132573)
        
        # Type idiom detected: calculating its left and rigth part (line 314)
        # Getting the type of 'deletechars' (line 314)
        deletechars_132575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 11), 'deletechars')
        # Getting the type of 'None' (line 314)
        None_132576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 26), 'None')
        
        (may_be_132577, more_types_in_union_132578) = may_be_none(deletechars_132575, None_132576)

        if may_be_132577:

            if more_types_in_union_132578:
                # Runtime conditional SSA (line 314)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Attribute to a Name (line 315):
            
            # Assigning a Attribute to a Name (line 315):
            # Getting the type of 'self' (line 315)
            self_132579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 21), 'self')
            # Obtaining the member 'defaultdeletechars' of a type (line 315)
            defaultdeletechars_132580 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 315, 21), self_132579, 'defaultdeletechars')
            # Assigning a type to the variable 'delete' (line 315)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 12), 'delete', defaultdeletechars_132580)

            if more_types_in_union_132578:
                # Runtime conditional SSA for else branch (line 314)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_132577) or more_types_in_union_132578):
            
            # Assigning a Call to a Name (line 317):
            
            # Assigning a Call to a Name (line 317):
            
            # Call to set(...): (line 317)
            # Processing the call arguments (line 317)
            # Getting the type of 'deletechars' (line 317)
            deletechars_132582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 25), 'deletechars', False)
            # Processing the call keyword arguments (line 317)
            kwargs_132583 = {}
            # Getting the type of 'set' (line 317)
            set_132581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 21), 'set', False)
            # Calling set(args, kwargs) (line 317)
            set_call_result_132584 = invoke(stypy.reporting.localization.Localization(__file__, 317, 21), set_132581, *[deletechars_132582], **kwargs_132583)
            
            # Assigning a type to the variable 'delete' (line 317)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 317, 12), 'delete', set_call_result_132584)

            if (may_be_132577 and more_types_in_union_132578):
                # SSA join for if statement (line 314)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Call to add(...): (line 318)
        # Processing the call arguments (line 318)
        str_132587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 318, 19), 'str', '"')
        # Processing the call keyword arguments (line 318)
        kwargs_132588 = {}
        # Getting the type of 'delete' (line 318)
        delete_132585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 8), 'delete', False)
        # Obtaining the member 'add' of a type (line 318)
        add_132586 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 8), delete_132585, 'add')
        # Calling add(args, kwargs) (line 318)
        add_call_result_132589 = invoke(stypy.reporting.localization.Localization(__file__, 318, 8), add_132586, *[str_132587], **kwargs_132588)
        
        
        # Assigning a Name to a Attribute (line 319):
        
        # Assigning a Name to a Attribute (line 319):
        # Getting the type of 'delete' (line 319)
        delete_132590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 27), 'delete')
        # Getting the type of 'self' (line 319)
        self_132591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 8), 'self')
        # Setting the type of the member 'deletechars' of a type (line 319)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 319, 8), self_132591, 'deletechars', delete_132590)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'case_sensitive' (line 321)
        case_sensitive_132592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 12), 'case_sensitive')
        # Getting the type of 'None' (line 321)
        None_132593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 30), 'None')
        # Applying the binary operator 'is' (line 321)
        result_is__132594 = python_operator(stypy.reporting.localization.Localization(__file__, 321, 12), 'is', case_sensitive_132592, None_132593)
        
        
        # Getting the type of 'case_sensitive' (line 321)
        case_sensitive_132595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 40), 'case_sensitive')
        # Getting the type of 'True' (line 321)
        True_132596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 58), 'True')
        # Applying the binary operator 'is' (line 321)
        result_is__132597 = python_operator(stypy.reporting.localization.Localization(__file__, 321, 40), 'is', case_sensitive_132595, True_132596)
        
        # Applying the binary operator 'or' (line 321)
        result_or_keyword_132598 = python_operator(stypy.reporting.localization.Localization(__file__, 321, 11), 'or', result_is__132594, result_is__132597)
        
        # Testing the type of an if condition (line 321)
        if_condition_132599 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 321, 8), result_or_keyword_132598)
        # Assigning a type to the variable 'if_condition_132599' (line 321)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 321, 8), 'if_condition_132599', if_condition_132599)
        # SSA begins for if statement (line 321)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Lambda to a Attribute (line 322):
        
        # Assigning a Lambda to a Attribute (line 322):

        @norecursion
        def _stypy_temp_lambda_33(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_33'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_33', 322, 34, True)
            # Passed parameters checking function
            _stypy_temp_lambda_33.stypy_localization = localization
            _stypy_temp_lambda_33.stypy_type_of_self = None
            _stypy_temp_lambda_33.stypy_type_store = module_type_store
            _stypy_temp_lambda_33.stypy_function_name = '_stypy_temp_lambda_33'
            _stypy_temp_lambda_33.stypy_param_names_list = ['x']
            _stypy_temp_lambda_33.stypy_varargs_param_name = None
            _stypy_temp_lambda_33.stypy_kwargs_param_name = None
            _stypy_temp_lambda_33.stypy_call_defaults = defaults
            _stypy_temp_lambda_33.stypy_call_varargs = varargs
            _stypy_temp_lambda_33.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_33', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_33', ['x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            # Getting the type of 'x' (line 322)
            x_132600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 44), 'x')
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 322)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 322, 34), 'stypy_return_type', x_132600)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_33' in the type store
            # Getting the type of 'stypy_return_type' (line 322)
            stypy_return_type_132601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 34), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_132601)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_33'
            return stypy_return_type_132601

        # Assigning a type to the variable '_stypy_temp_lambda_33' (line 322)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 322, 34), '_stypy_temp_lambda_33', _stypy_temp_lambda_33)
        # Getting the type of '_stypy_temp_lambda_33' (line 322)
        _stypy_temp_lambda_33_132602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 34), '_stypy_temp_lambda_33')
        # Getting the type of 'self' (line 322)
        self_132603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 12), 'self')
        # Setting the type of the member 'case_converter' of a type (line 322)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 322, 12), self_132603, 'case_converter', _stypy_temp_lambda_33_132602)
        # SSA branch for the else part of an if statement (line 321)
        module_type_store.open_ssa_branch('else')
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'case_sensitive' (line 323)
        case_sensitive_132604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 14), 'case_sensitive')
        # Getting the type of 'False' (line 323)
        False_132605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 32), 'False')
        # Applying the binary operator 'is' (line 323)
        result_is__132606 = python_operator(stypy.reporting.localization.Localization(__file__, 323, 14), 'is', case_sensitive_132604, False_132605)
        
        
        # Call to startswith(...): (line 323)
        # Processing the call arguments (line 323)
        str_132609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 68), 'str', 'u')
        # Processing the call keyword arguments (line 323)
        kwargs_132610 = {}
        # Getting the type of 'case_sensitive' (line 323)
        case_sensitive_132607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 42), 'case_sensitive', False)
        # Obtaining the member 'startswith' of a type (line 323)
        startswith_132608 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 323, 42), case_sensitive_132607, 'startswith')
        # Calling startswith(args, kwargs) (line 323)
        startswith_call_result_132611 = invoke(stypy.reporting.localization.Localization(__file__, 323, 42), startswith_132608, *[str_132609], **kwargs_132610)
        
        # Applying the binary operator 'or' (line 323)
        result_or_keyword_132612 = python_operator(stypy.reporting.localization.Localization(__file__, 323, 13), 'or', result_is__132606, startswith_call_result_132611)
        
        # Testing the type of an if condition (line 323)
        if_condition_132613 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 323, 13), result_or_keyword_132612)
        # Assigning a type to the variable 'if_condition_132613' (line 323)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 13), 'if_condition_132613', if_condition_132613)
        # SSA begins for if statement (line 323)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Lambda to a Attribute (line 324):
        
        # Assigning a Lambda to a Attribute (line 324):

        @norecursion
        def _stypy_temp_lambda_34(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_34'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_34', 324, 34, True)
            # Passed parameters checking function
            _stypy_temp_lambda_34.stypy_localization = localization
            _stypy_temp_lambda_34.stypy_type_of_self = None
            _stypy_temp_lambda_34.stypy_type_store = module_type_store
            _stypy_temp_lambda_34.stypy_function_name = '_stypy_temp_lambda_34'
            _stypy_temp_lambda_34.stypy_param_names_list = ['x']
            _stypy_temp_lambda_34.stypy_varargs_param_name = None
            _stypy_temp_lambda_34.stypy_kwargs_param_name = None
            _stypy_temp_lambda_34.stypy_call_defaults = defaults
            _stypy_temp_lambda_34.stypy_call_varargs = varargs
            _stypy_temp_lambda_34.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_34', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_34', ['x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Call to upper(...): (line 324)
            # Processing the call keyword arguments (line 324)
            kwargs_132616 = {}
            # Getting the type of 'x' (line 324)
            x_132614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 44), 'x', False)
            # Obtaining the member 'upper' of a type (line 324)
            upper_132615 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 324, 44), x_132614, 'upper')
            # Calling upper(args, kwargs) (line 324)
            upper_call_result_132617 = invoke(stypy.reporting.localization.Localization(__file__, 324, 44), upper_132615, *[], **kwargs_132616)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 324)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 324, 34), 'stypy_return_type', upper_call_result_132617)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_34' in the type store
            # Getting the type of 'stypy_return_type' (line 324)
            stypy_return_type_132618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 34), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_132618)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_34'
            return stypy_return_type_132618

        # Assigning a type to the variable '_stypy_temp_lambda_34' (line 324)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 324, 34), '_stypy_temp_lambda_34', _stypy_temp_lambda_34)
        # Getting the type of '_stypy_temp_lambda_34' (line 324)
        _stypy_temp_lambda_34_132619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 34), '_stypy_temp_lambda_34')
        # Getting the type of 'self' (line 324)
        self_132620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 12), 'self')
        # Setting the type of the member 'case_converter' of a type (line 324)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 324, 12), self_132620, 'case_converter', _stypy_temp_lambda_34_132619)
        # SSA branch for the else part of an if statement (line 323)
        module_type_store.open_ssa_branch('else')
        
        
        # Call to startswith(...): (line 325)
        # Processing the call arguments (line 325)
        str_132623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 39), 'str', 'l')
        # Processing the call keyword arguments (line 325)
        kwargs_132624 = {}
        # Getting the type of 'case_sensitive' (line 325)
        case_sensitive_132621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 13), 'case_sensitive', False)
        # Obtaining the member 'startswith' of a type (line 325)
        startswith_132622 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 325, 13), case_sensitive_132621, 'startswith')
        # Calling startswith(args, kwargs) (line 325)
        startswith_call_result_132625 = invoke(stypy.reporting.localization.Localization(__file__, 325, 13), startswith_132622, *[str_132623], **kwargs_132624)
        
        # Testing the type of an if condition (line 325)
        if_condition_132626 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 325, 13), startswith_call_result_132625)
        # Assigning a type to the variable 'if_condition_132626' (line 325)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 13), 'if_condition_132626', if_condition_132626)
        # SSA begins for if statement (line 325)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Lambda to a Attribute (line 326):
        
        # Assigning a Lambda to a Attribute (line 326):

        @norecursion
        def _stypy_temp_lambda_35(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_35'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_35', 326, 34, True)
            # Passed parameters checking function
            _stypy_temp_lambda_35.stypy_localization = localization
            _stypy_temp_lambda_35.stypy_type_of_self = None
            _stypy_temp_lambda_35.stypy_type_store = module_type_store
            _stypy_temp_lambda_35.stypy_function_name = '_stypy_temp_lambda_35'
            _stypy_temp_lambda_35.stypy_param_names_list = ['x']
            _stypy_temp_lambda_35.stypy_varargs_param_name = None
            _stypy_temp_lambda_35.stypy_kwargs_param_name = None
            _stypy_temp_lambda_35.stypy_call_defaults = defaults
            _stypy_temp_lambda_35.stypy_call_varargs = varargs
            _stypy_temp_lambda_35.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_35', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_35', ['x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Call to lower(...): (line 326)
            # Processing the call keyword arguments (line 326)
            kwargs_132629 = {}
            # Getting the type of 'x' (line 326)
            x_132627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 44), 'x', False)
            # Obtaining the member 'lower' of a type (line 326)
            lower_132628 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 44), x_132627, 'lower')
            # Calling lower(args, kwargs) (line 326)
            lower_call_result_132630 = invoke(stypy.reporting.localization.Localization(__file__, 326, 44), lower_132628, *[], **kwargs_132629)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 326)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 34), 'stypy_return_type', lower_call_result_132630)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_35' in the type store
            # Getting the type of 'stypy_return_type' (line 326)
            stypy_return_type_132631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 34), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_132631)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_35'
            return stypy_return_type_132631

        # Assigning a type to the variable '_stypy_temp_lambda_35' (line 326)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 34), '_stypy_temp_lambda_35', _stypy_temp_lambda_35)
        # Getting the type of '_stypy_temp_lambda_35' (line 326)
        _stypy_temp_lambda_35_132632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 34), '_stypy_temp_lambda_35')
        # Getting the type of 'self' (line 326)
        self_132633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 12), 'self')
        # Setting the type of the member 'case_converter' of a type (line 326)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 12), self_132633, 'case_converter', _stypy_temp_lambda_35_132632)
        # SSA branch for the else part of an if statement (line 325)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a BinOp to a Name (line 328):
        
        # Assigning a BinOp to a Name (line 328):
        str_132634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 18), 'str', 'unrecognized case_sensitive value %s.')
        # Getting the type of 'case_sensitive' (line 328)
        case_sensitive_132635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 60), 'case_sensitive')
        # Applying the binary operator '%' (line 328)
        result_mod_132636 = python_operator(stypy.reporting.localization.Localization(__file__, 328, 18), '%', str_132634, case_sensitive_132635)
        
        # Assigning a type to the variable 'msg' (line 328)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 328, 12), 'msg', result_mod_132636)
        
        # Call to ValueError(...): (line 329)
        # Processing the call arguments (line 329)
        # Getting the type of 'msg' (line 329)
        msg_132638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 29), 'msg', False)
        # Processing the call keyword arguments (line 329)
        kwargs_132639 = {}
        # Getting the type of 'ValueError' (line 329)
        ValueError_132637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 329)
        ValueError_call_result_132640 = invoke(stypy.reporting.localization.Localization(__file__, 329, 18), ValueError_132637, *[msg_132638], **kwargs_132639)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 329, 12), ValueError_call_result_132640, 'raise parameter', BaseException)
        # SSA join for if statement (line 325)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 323)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 321)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 331):
        
        # Assigning a Name to a Attribute (line 331):
        # Getting the type of 'replace_space' (line 331)
        replace_space_132641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 29), 'replace_space')
        # Getting the type of 'self' (line 331)
        self_132642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 8), 'self')
        # Setting the type of the member 'replace_space' of a type (line 331)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 8), self_132642, 'replace_space', replace_space_132641)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def validate(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        str_132643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 41), 'str', 'f%i')
        # Getting the type of 'None' (line 333)
        None_132644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 57), 'None')
        defaults = [str_132643, None_132644]
        # Create a new context for function 'validate'
        module_type_store = module_type_store.open_function_context('validate', 333, 4, False)
        # Assigning a type to the variable 'self' (line 334)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        NameValidator.validate.__dict__.__setitem__('stypy_localization', localization)
        NameValidator.validate.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        NameValidator.validate.__dict__.__setitem__('stypy_type_store', module_type_store)
        NameValidator.validate.__dict__.__setitem__('stypy_function_name', 'NameValidator.validate')
        NameValidator.validate.__dict__.__setitem__('stypy_param_names_list', ['names', 'defaultfmt', 'nbfields'])
        NameValidator.validate.__dict__.__setitem__('stypy_varargs_param_name', None)
        NameValidator.validate.__dict__.__setitem__('stypy_kwargs_param_name', None)
        NameValidator.validate.__dict__.__setitem__('stypy_call_defaults', defaults)
        NameValidator.validate.__dict__.__setitem__('stypy_call_varargs', varargs)
        NameValidator.validate.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        NameValidator.validate.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NameValidator.validate', ['names', 'defaultfmt', 'nbfields'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'validate', localization, ['names', 'defaultfmt', 'nbfields'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'validate(...)' code ##################

        str_132645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 358, (-1)), 'str', '\n        Validate a list of strings as field names for a structured array.\n\n        Parameters\n        ----------\n        names : sequence of str\n            Strings to be validated.\n        defaultfmt : str, optional\n            Default format string, used if validating a given string\n            reduces its length to zero.\n        nbfields : integer, optional\n            Final number of validated names, used to expand or shrink the\n            initial list of names.\n\n        Returns\n        -------\n        validatednames : list of str\n            The list of validated field names.\n\n        Notes\n        -----\n        A `NameValidator` instance can be called directly, which is the\n        same as calling `validate`. For examples, see `NameValidator`.\n\n        ')
        
        # Type idiom detected: calculating its left and rigth part (line 360)
        # Getting the type of 'names' (line 360)
        names_132646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 12), 'names')
        # Getting the type of 'None' (line 360)
        None_132647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 21), 'None')
        
        (may_be_132648, more_types_in_union_132649) = may_be_none(names_132646, None_132647)

        if may_be_132648:

            if more_types_in_union_132649:
                # Runtime conditional SSA (line 360)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Type idiom detected: calculating its left and rigth part (line 361)
            # Getting the type of 'nbfields' (line 361)
            nbfields_132650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 16), 'nbfields')
            # Getting the type of 'None' (line 361)
            None_132651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 28), 'None')
            
            (may_be_132652, more_types_in_union_132653) = may_be_none(nbfields_132650, None_132651)

            if may_be_132652:

                if more_types_in_union_132653:
                    # Runtime conditional SSA (line 361)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                # Getting the type of 'None' (line 362)
                None_132654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 23), 'None')
                # Assigning a type to the variable 'stypy_return_type' (line 362)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 16), 'stypy_return_type', None_132654)

                if more_types_in_union_132653:
                    # SSA join for if statement (line 361)
                    module_type_store = module_type_store.join_ssa_context()


            
            
            # Assigning a List to a Name (line 363):
            
            # Assigning a List to a Name (line 363):
            
            # Obtaining an instance of the builtin type 'list' (line 363)
            list_132655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 363, 20), 'list')
            # Adding type elements to the builtin type 'list' instance (line 363)
            
            # Assigning a type to the variable 'names' (line 363)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 12), 'names', list_132655)

            if more_types_in_union_132649:
                # SSA join for if statement (line 360)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 364)
        # Getting the type of 'basestring' (line 364)
        basestring_132656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 29), 'basestring')
        # Getting the type of 'names' (line 364)
        names_132657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 22), 'names')
        
        (may_be_132658, more_types_in_union_132659) = may_be_subtype(basestring_132656, names_132657)

        if may_be_132658:

            if more_types_in_union_132659:
                # Runtime conditional SSA (line 364)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'names' (line 364)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 364, 8), 'names', remove_not_subtype_from_union(names_132657, basestring))
            
            # Assigning a List to a Name (line 365):
            
            # Assigning a List to a Name (line 365):
            
            # Obtaining an instance of the builtin type 'list' (line 365)
            list_132660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 365, 20), 'list')
            # Adding type elements to the builtin type 'list' instance (line 365)
            # Adding element type (line 365)
            # Getting the type of 'names' (line 365)
            names_132661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 21), 'names')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 365, 20), list_132660, names_132661)
            
            # Assigning a type to the variable 'names' (line 365)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 365, 12), 'names', list_132660)

            if more_types_in_union_132659:
                # SSA join for if statement (line 364)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 366)
        # Getting the type of 'nbfields' (line 366)
        nbfields_132662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 8), 'nbfields')
        # Getting the type of 'None' (line 366)
        None_132663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 27), 'None')
        
        (may_be_132664, more_types_in_union_132665) = may_not_be_none(nbfields_132662, None_132663)

        if may_be_132664:

            if more_types_in_union_132665:
                # Runtime conditional SSA (line 366)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 367):
            
            # Assigning a Call to a Name (line 367):
            
            # Call to len(...): (line 367)
            # Processing the call arguments (line 367)
            # Getting the type of 'names' (line 367)
            names_132667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 26), 'names', False)
            # Processing the call keyword arguments (line 367)
            kwargs_132668 = {}
            # Getting the type of 'len' (line 367)
            len_132666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 22), 'len', False)
            # Calling len(args, kwargs) (line 367)
            len_call_result_132669 = invoke(stypy.reporting.localization.Localization(__file__, 367, 22), len_132666, *[names_132667], **kwargs_132668)
            
            # Assigning a type to the variable 'nbnames' (line 367)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 367, 12), 'nbnames', len_call_result_132669)
            
            
            # Getting the type of 'nbnames' (line 368)
            nbnames_132670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 16), 'nbnames')
            # Getting the type of 'nbfields' (line 368)
            nbfields_132671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 26), 'nbfields')
            # Applying the binary operator '<' (line 368)
            result_lt_132672 = python_operator(stypy.reporting.localization.Localization(__file__, 368, 16), '<', nbnames_132670, nbfields_132671)
            
            # Testing the type of an if condition (line 368)
            if_condition_132673 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 368, 12), result_lt_132672)
            # Assigning a type to the variable 'if_condition_132673' (line 368)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 368, 12), 'if_condition_132673', if_condition_132673)
            # SSA begins for if statement (line 368)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a BinOp to a Name (line 369):
            
            # Assigning a BinOp to a Name (line 369):
            
            # Call to list(...): (line 369)
            # Processing the call arguments (line 369)
            # Getting the type of 'names' (line 369)
            names_132675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 29), 'names', False)
            # Processing the call keyword arguments (line 369)
            kwargs_132676 = {}
            # Getting the type of 'list' (line 369)
            list_132674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 24), 'list', False)
            # Calling list(args, kwargs) (line 369)
            list_call_result_132677 = invoke(stypy.reporting.localization.Localization(__file__, 369, 24), list_132674, *[names_132675], **kwargs_132676)
            
            
            # Obtaining an instance of the builtin type 'list' (line 369)
            list_132678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 369, 38), 'list')
            # Adding type elements to the builtin type 'list' instance (line 369)
            # Adding element type (line 369)
            str_132679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 369, 39), 'str', '')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 369, 38), list_132678, str_132679)
            
            # Getting the type of 'nbfields' (line 369)
            nbfields_132680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 46), 'nbfields')
            # Getting the type of 'nbnames' (line 369)
            nbnames_132681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 57), 'nbnames')
            # Applying the binary operator '-' (line 369)
            result_sub_132682 = python_operator(stypy.reporting.localization.Localization(__file__, 369, 46), '-', nbfields_132680, nbnames_132681)
            
            # Applying the binary operator '*' (line 369)
            result_mul_132683 = python_operator(stypy.reporting.localization.Localization(__file__, 369, 38), '*', list_132678, result_sub_132682)
            
            # Applying the binary operator '+' (line 369)
            result_add_132684 = python_operator(stypy.reporting.localization.Localization(__file__, 369, 24), '+', list_call_result_132677, result_mul_132683)
            
            # Assigning a type to the variable 'names' (line 369)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 369, 16), 'names', result_add_132684)
            # SSA branch for the else part of an if statement (line 368)
            module_type_store.open_ssa_branch('else')
            
            
            # Getting the type of 'nbnames' (line 370)
            nbnames_132685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 18), 'nbnames')
            # Getting the type of 'nbfields' (line 370)
            nbfields_132686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 28), 'nbfields')
            # Applying the binary operator '>' (line 370)
            result_gt_132687 = python_operator(stypy.reporting.localization.Localization(__file__, 370, 18), '>', nbnames_132685, nbfields_132686)
            
            # Testing the type of an if condition (line 370)
            if_condition_132688 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 370, 17), result_gt_132687)
            # Assigning a type to the variable 'if_condition_132688' (line 370)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 17), 'if_condition_132688', if_condition_132688)
            # SSA begins for if statement (line 370)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Subscript to a Name (line 371):
            
            # Assigning a Subscript to a Name (line 371):
            
            # Obtaining the type of the subscript
            # Getting the type of 'nbfields' (line 371)
            nbfields_132689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 31), 'nbfields')
            slice_132690 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 371, 24), None, nbfields_132689, None)
            # Getting the type of 'names' (line 371)
            names_132691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 24), 'names')
            # Obtaining the member '__getitem__' of a type (line 371)
            getitem___132692 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 371, 24), names_132691, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 371)
            subscript_call_result_132693 = invoke(stypy.reporting.localization.Localization(__file__, 371, 24), getitem___132692, slice_132690)
            
            # Assigning a type to the variable 'names' (line 371)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 371, 16), 'names', subscript_call_result_132693)
            # SSA join for if statement (line 370)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 368)
            module_type_store = module_type_store.join_ssa_context()
            

            if more_types_in_union_132665:
                # SSA join for if statement (line 366)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Attribute to a Name (line 373):
        
        # Assigning a Attribute to a Name (line 373):
        # Getting the type of 'self' (line 373)
        self_132694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 22), 'self')
        # Obtaining the member 'deletechars' of a type (line 373)
        deletechars_132695 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 22), self_132694, 'deletechars')
        # Assigning a type to the variable 'deletechars' (line 373)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 373, 8), 'deletechars', deletechars_132695)
        
        # Assigning a Attribute to a Name (line 374):
        
        # Assigning a Attribute to a Name (line 374):
        # Getting the type of 'self' (line 374)
        self_132696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 22), 'self')
        # Obtaining the member 'excludelist' of a type (line 374)
        excludelist_132697 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 374, 22), self_132696, 'excludelist')
        # Assigning a type to the variable 'excludelist' (line 374)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 374, 8), 'excludelist', excludelist_132697)
        
        # Assigning a Attribute to a Name (line 375):
        
        # Assigning a Attribute to a Name (line 375):
        # Getting the type of 'self' (line 375)
        self_132698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 25), 'self')
        # Obtaining the member 'case_converter' of a type (line 375)
        case_converter_132699 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 375, 25), self_132698, 'case_converter')
        # Assigning a type to the variable 'case_converter' (line 375)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 375, 8), 'case_converter', case_converter_132699)
        
        # Assigning a Attribute to a Name (line 376):
        
        # Assigning a Attribute to a Name (line 376):
        # Getting the type of 'self' (line 376)
        self_132700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 24), 'self')
        # Obtaining the member 'replace_space' of a type (line 376)
        replace_space_132701 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 376, 24), self_132700, 'replace_space')
        # Assigning a type to the variable 'replace_space' (line 376)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 376, 8), 'replace_space', replace_space_132701)
        
        # Assigning a List to a Name (line 378):
        
        # Assigning a List to a Name (line 378):
        
        # Obtaining an instance of the builtin type 'list' (line 378)
        list_132702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 378, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 378)
        
        # Assigning a type to the variable 'validatednames' (line 378)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 378, 8), 'validatednames', list_132702)
        
        # Assigning a Call to a Name (line 379):
        
        # Assigning a Call to a Name (line 379):
        
        # Call to dict(...): (line 379)
        # Processing the call keyword arguments (line 379)
        kwargs_132704 = {}
        # Getting the type of 'dict' (line 379)
        dict_132703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 15), 'dict', False)
        # Calling dict(args, kwargs) (line 379)
        dict_call_result_132705 = invoke(stypy.reporting.localization.Localization(__file__, 379, 15), dict_132703, *[], **kwargs_132704)
        
        # Assigning a type to the variable 'seen' (line 379)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 379, 8), 'seen', dict_call_result_132705)
        
        # Assigning a Num to a Name (line 380):
        
        # Assigning a Num to a Name (line 380):
        int_132706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 380, 18), 'int')
        # Assigning a type to the variable 'nbempty' (line 380)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 380, 8), 'nbempty', int_132706)
        
        # Getting the type of 'names' (line 382)
        names_132707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 20), 'names')
        # Testing the type of a for loop iterable (line 382)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 382, 8), names_132707)
        # Getting the type of the for loop variable (line 382)
        for_loop_var_132708 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 382, 8), names_132707)
        # Assigning a type to the variable 'item' (line 382)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 382, 8), 'item', for_loop_var_132708)
        # SSA begins for a for statement (line 382)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 383):
        
        # Assigning a Call to a Name (line 383):
        
        # Call to strip(...): (line 383)
        # Processing the call keyword arguments (line 383)
        kwargs_132714 = {}
        
        # Call to case_converter(...): (line 383)
        # Processing the call arguments (line 383)
        # Getting the type of 'item' (line 383)
        item_132710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 34), 'item', False)
        # Processing the call keyword arguments (line 383)
        kwargs_132711 = {}
        # Getting the type of 'case_converter' (line 383)
        case_converter_132709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 19), 'case_converter', False)
        # Calling case_converter(args, kwargs) (line 383)
        case_converter_call_result_132712 = invoke(stypy.reporting.localization.Localization(__file__, 383, 19), case_converter_132709, *[item_132710], **kwargs_132711)
        
        # Obtaining the member 'strip' of a type (line 383)
        strip_132713 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 383, 19), case_converter_call_result_132712, 'strip')
        # Calling strip(args, kwargs) (line 383)
        strip_call_result_132715 = invoke(stypy.reporting.localization.Localization(__file__, 383, 19), strip_132713, *[], **kwargs_132714)
        
        # Assigning a type to the variable 'item' (line 383)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 12), 'item', strip_call_result_132715)
        
        # Getting the type of 'replace_space' (line 384)
        replace_space_132716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 15), 'replace_space')
        # Testing the type of an if condition (line 384)
        if_condition_132717 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 384, 12), replace_space_132716)
        # Assigning a type to the variable 'if_condition_132717' (line 384)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 384, 12), 'if_condition_132717', if_condition_132717)
        # SSA begins for if statement (line 384)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 385):
        
        # Assigning a Call to a Name (line 385):
        
        # Call to replace(...): (line 385)
        # Processing the call arguments (line 385)
        str_132720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 385, 36), 'str', ' ')
        # Getting the type of 'replace_space' (line 385)
        replace_space_132721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 41), 'replace_space', False)
        # Processing the call keyword arguments (line 385)
        kwargs_132722 = {}
        # Getting the type of 'item' (line 385)
        item_132718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 23), 'item', False)
        # Obtaining the member 'replace' of a type (line 385)
        replace_132719 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 385, 23), item_132718, 'replace')
        # Calling replace(args, kwargs) (line 385)
        replace_call_result_132723 = invoke(stypy.reporting.localization.Localization(__file__, 385, 23), replace_132719, *[str_132720, replace_space_132721], **kwargs_132722)
        
        # Assigning a type to the variable 'item' (line 385)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 385, 16), 'item', replace_call_result_132723)
        # SSA join for if statement (line 384)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 386):
        
        # Assigning a Call to a Name (line 386):
        
        # Call to join(...): (line 386)
        # Processing the call arguments (line 386)
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'item' (line 386)
        item_132730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 39), 'item', False)
        comprehension_132731 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 386, 28), item_132730)
        # Assigning a type to the variable 'c' (line 386)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 386, 28), 'c', comprehension_132731)
        
        # Getting the type of 'c' (line 386)
        c_132727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 47), 'c', False)
        # Getting the type of 'deletechars' (line 386)
        deletechars_132728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 56), 'deletechars', False)
        # Applying the binary operator 'notin' (line 386)
        result_contains_132729 = python_operator(stypy.reporting.localization.Localization(__file__, 386, 47), 'notin', c_132727, deletechars_132728)
        
        # Getting the type of 'c' (line 386)
        c_132726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 28), 'c', False)
        list_132732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 28), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 386, 28), list_132732, c_132726)
        # Processing the call keyword arguments (line 386)
        kwargs_132733 = {}
        str_132724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 19), 'str', '')
        # Obtaining the member 'join' of a type (line 386)
        join_132725 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 386, 19), str_132724, 'join')
        # Calling join(args, kwargs) (line 386)
        join_call_result_132734 = invoke(stypy.reporting.localization.Localization(__file__, 386, 19), join_132725, *[list_132732], **kwargs_132733)
        
        # Assigning a type to the variable 'item' (line 386)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 386, 12), 'item', join_call_result_132734)
        
        
        # Getting the type of 'item' (line 387)
        item_132735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 15), 'item')
        str_132736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 387, 23), 'str', '')
        # Applying the binary operator '==' (line 387)
        result_eq_132737 = python_operator(stypy.reporting.localization.Localization(__file__, 387, 15), '==', item_132735, str_132736)
        
        # Testing the type of an if condition (line 387)
        if_condition_132738 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 387, 12), result_eq_132737)
        # Assigning a type to the variable 'if_condition_132738' (line 387)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 387, 12), 'if_condition_132738', if_condition_132738)
        # SSA begins for if statement (line 387)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 388):
        
        # Assigning a BinOp to a Name (line 388):
        # Getting the type of 'defaultfmt' (line 388)
        defaultfmt_132739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 23), 'defaultfmt')
        # Getting the type of 'nbempty' (line 388)
        nbempty_132740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 36), 'nbempty')
        # Applying the binary operator '%' (line 388)
        result_mod_132741 = python_operator(stypy.reporting.localization.Localization(__file__, 388, 23), '%', defaultfmt_132739, nbempty_132740)
        
        # Assigning a type to the variable 'item' (line 388)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 388, 16), 'item', result_mod_132741)
        
        
        # Getting the type of 'item' (line 389)
        item_132742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 22), 'item')
        # Getting the type of 'names' (line 389)
        names_132743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 30), 'names')
        # Applying the binary operator 'in' (line 389)
        result_contains_132744 = python_operator(stypy.reporting.localization.Localization(__file__, 389, 22), 'in', item_132742, names_132743)
        
        # Testing the type of an if condition (line 389)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 389, 16), result_contains_132744)
        # SSA begins for while statement (line 389)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Getting the type of 'nbempty' (line 390)
        nbempty_132745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 20), 'nbempty')
        int_132746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 31), 'int')
        # Applying the binary operator '+=' (line 390)
        result_iadd_132747 = python_operator(stypy.reporting.localization.Localization(__file__, 390, 20), '+=', nbempty_132745, int_132746)
        # Assigning a type to the variable 'nbempty' (line 390)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 390, 20), 'nbempty', result_iadd_132747)
        
        
        # Assigning a BinOp to a Name (line 391):
        
        # Assigning a BinOp to a Name (line 391):
        # Getting the type of 'defaultfmt' (line 391)
        defaultfmt_132748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 27), 'defaultfmt')
        # Getting the type of 'nbempty' (line 391)
        nbempty_132749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 40), 'nbempty')
        # Applying the binary operator '%' (line 391)
        result_mod_132750 = python_operator(stypy.reporting.localization.Localization(__file__, 391, 27), '%', defaultfmt_132748, nbempty_132749)
        
        # Assigning a type to the variable 'item' (line 391)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 391, 20), 'item', result_mod_132750)
        # SSA join for while statement (line 389)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'nbempty' (line 392)
        nbempty_132751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 16), 'nbempty')
        int_132752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 392, 27), 'int')
        # Applying the binary operator '+=' (line 392)
        result_iadd_132753 = python_operator(stypy.reporting.localization.Localization(__file__, 392, 16), '+=', nbempty_132751, int_132752)
        # Assigning a type to the variable 'nbempty' (line 392)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 392, 16), 'nbempty', result_iadd_132753)
        
        # SSA branch for the else part of an if statement (line 387)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'item' (line 393)
        item_132754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 17), 'item')
        # Getting the type of 'excludelist' (line 393)
        excludelist_132755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 25), 'excludelist')
        # Applying the binary operator 'in' (line 393)
        result_contains_132756 = python_operator(stypy.reporting.localization.Localization(__file__, 393, 17), 'in', item_132754, excludelist_132755)
        
        # Testing the type of an if condition (line 393)
        if_condition_132757 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 393, 17), result_contains_132756)
        # Assigning a type to the variable 'if_condition_132757' (line 393)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 393, 17), 'if_condition_132757', if_condition_132757)
        # SSA begins for if statement (line 393)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'item' (line 394)
        item_132758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 16), 'item')
        str_132759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 24), 'str', '_')
        # Applying the binary operator '+=' (line 394)
        result_iadd_132760 = python_operator(stypy.reporting.localization.Localization(__file__, 394, 16), '+=', item_132758, str_132759)
        # Assigning a type to the variable 'item' (line 394)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 394, 16), 'item', result_iadd_132760)
        
        # SSA join for if statement (line 393)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 387)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 395):
        
        # Assigning a Call to a Name (line 395):
        
        # Call to get(...): (line 395)
        # Processing the call arguments (line 395)
        # Getting the type of 'item' (line 395)
        item_132763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 27), 'item', False)
        int_132764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 395, 33), 'int')
        # Processing the call keyword arguments (line 395)
        kwargs_132765 = {}
        # Getting the type of 'seen' (line 395)
        seen_132761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 18), 'seen', False)
        # Obtaining the member 'get' of a type (line 395)
        get_132762 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 395, 18), seen_132761, 'get')
        # Calling get(args, kwargs) (line 395)
        get_call_result_132766 = invoke(stypy.reporting.localization.Localization(__file__, 395, 18), get_132762, *[item_132763, int_132764], **kwargs_132765)
        
        # Assigning a type to the variable 'cnt' (line 395)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 395, 12), 'cnt', get_call_result_132766)
        
        
        # Getting the type of 'cnt' (line 396)
        cnt_132767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 15), 'cnt')
        int_132768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 396, 21), 'int')
        # Applying the binary operator '>' (line 396)
        result_gt_132769 = python_operator(stypy.reporting.localization.Localization(__file__, 396, 15), '>', cnt_132767, int_132768)
        
        # Testing the type of an if condition (line 396)
        if_condition_132770 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 396, 12), result_gt_132769)
        # Assigning a type to the variable 'if_condition_132770' (line 396)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 396, 12), 'if_condition_132770', if_condition_132770)
        # SSA begins for if statement (line 396)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 397)
        # Processing the call arguments (line 397)
        # Getting the type of 'item' (line 397)
        item_132773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 38), 'item', False)
        str_132774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 397, 45), 'str', '_%d')
        # Getting the type of 'cnt' (line 397)
        cnt_132775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 53), 'cnt', False)
        # Applying the binary operator '%' (line 397)
        result_mod_132776 = python_operator(stypy.reporting.localization.Localization(__file__, 397, 45), '%', str_132774, cnt_132775)
        
        # Applying the binary operator '+' (line 397)
        result_add_132777 = python_operator(stypy.reporting.localization.Localization(__file__, 397, 38), '+', item_132773, result_mod_132776)
        
        # Processing the call keyword arguments (line 397)
        kwargs_132778 = {}
        # Getting the type of 'validatednames' (line 397)
        validatednames_132771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 16), 'validatednames', False)
        # Obtaining the member 'append' of a type (line 397)
        append_132772 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 397, 16), validatednames_132771, 'append')
        # Calling append(args, kwargs) (line 397)
        append_call_result_132779 = invoke(stypy.reporting.localization.Localization(__file__, 397, 16), append_132772, *[result_add_132777], **kwargs_132778)
        
        # SSA branch for the else part of an if statement (line 396)
        module_type_store.open_ssa_branch('else')
        
        # Call to append(...): (line 399)
        # Processing the call arguments (line 399)
        # Getting the type of 'item' (line 399)
        item_132782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 38), 'item', False)
        # Processing the call keyword arguments (line 399)
        kwargs_132783 = {}
        # Getting the type of 'validatednames' (line 399)
        validatednames_132780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 16), 'validatednames', False)
        # Obtaining the member 'append' of a type (line 399)
        append_132781 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 399, 16), validatednames_132780, 'append')
        # Calling append(args, kwargs) (line 399)
        append_call_result_132784 = invoke(stypy.reporting.localization.Localization(__file__, 399, 16), append_132781, *[item_132782], **kwargs_132783)
        
        # SSA join for if statement (line 396)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Subscript (line 400):
        
        # Assigning a BinOp to a Subscript (line 400):
        # Getting the type of 'cnt' (line 400)
        cnt_132785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 25), 'cnt')
        int_132786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 31), 'int')
        # Applying the binary operator '+' (line 400)
        result_add_132787 = python_operator(stypy.reporting.localization.Localization(__file__, 400, 25), '+', cnt_132785, int_132786)
        
        # Getting the type of 'seen' (line 400)
        seen_132788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 12), 'seen')
        # Getting the type of 'item' (line 400)
        item_132789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 17), 'item')
        # Storing an element on a container (line 400)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 400, 12), seen_132788, (item_132789, result_add_132787))
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to tuple(...): (line 401)
        # Processing the call arguments (line 401)
        # Getting the type of 'validatednames' (line 401)
        validatednames_132791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 21), 'validatednames', False)
        # Processing the call keyword arguments (line 401)
        kwargs_132792 = {}
        # Getting the type of 'tuple' (line 401)
        tuple_132790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 15), 'tuple', False)
        # Calling tuple(args, kwargs) (line 401)
        tuple_call_result_132793 = invoke(stypy.reporting.localization.Localization(__file__, 401, 15), tuple_132790, *[validatednames_132791], **kwargs_132792)
        
        # Assigning a type to the variable 'stypy_return_type' (line 401)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 401, 8), 'stypy_return_type', tuple_call_result_132793)
        
        # ################# End of 'validate(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'validate' in the type store
        # Getting the type of 'stypy_return_type' (line 333)
        stypy_return_type_132794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_132794)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'validate'
        return stypy_return_type_132794


    @norecursion
    def __call__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        str_132795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, 41), 'str', 'f%i')
        # Getting the type of 'None' (line 404)
        None_132796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 57), 'None')
        defaults = [str_132795, None_132796]
        # Create a new context for function '__call__'
        module_type_store = module_type_store.open_function_context('__call__', 404, 4, False)
        # Assigning a type to the variable 'self' (line 405)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 405, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        NameValidator.__call__.__dict__.__setitem__('stypy_localization', localization)
        NameValidator.__call__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        NameValidator.__call__.__dict__.__setitem__('stypy_type_store', module_type_store)
        NameValidator.__call__.__dict__.__setitem__('stypy_function_name', 'NameValidator.__call__')
        NameValidator.__call__.__dict__.__setitem__('stypy_param_names_list', ['names', 'defaultfmt', 'nbfields'])
        NameValidator.__call__.__dict__.__setitem__('stypy_varargs_param_name', None)
        NameValidator.__call__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        NameValidator.__call__.__dict__.__setitem__('stypy_call_defaults', defaults)
        NameValidator.__call__.__dict__.__setitem__('stypy_call_varargs', varargs)
        NameValidator.__call__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        NameValidator.__call__.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NameValidator.__call__', ['names', 'defaultfmt', 'nbfields'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__call__', localization, ['names', 'defaultfmt', 'nbfields'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__call__(...)' code ##################

        
        # Call to validate(...): (line 405)
        # Processing the call arguments (line 405)
        # Getting the type of 'names' (line 405)
        names_132799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 29), 'names', False)
        # Processing the call keyword arguments (line 405)
        # Getting the type of 'defaultfmt' (line 405)
        defaultfmt_132800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 47), 'defaultfmt', False)
        keyword_132801 = defaultfmt_132800
        # Getting the type of 'nbfields' (line 405)
        nbfields_132802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 68), 'nbfields', False)
        keyword_132803 = nbfields_132802
        kwargs_132804 = {'defaultfmt': keyword_132801, 'nbfields': keyword_132803}
        # Getting the type of 'self' (line 405)
        self_132797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 15), 'self', False)
        # Obtaining the member 'validate' of a type (line 405)
        validate_132798 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 405, 15), self_132797, 'validate')
        # Calling validate(args, kwargs) (line 405)
        validate_call_result_132805 = invoke(stypy.reporting.localization.Localization(__file__, 405, 15), validate_132798, *[names_132799], **kwargs_132804)
        
        # Assigning a type to the variable 'stypy_return_type' (line 405)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 405, 8), 'stypy_return_type', validate_call_result_132805)
        
        # ################# End of '__call__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__call__' in the type store
        # Getting the type of 'stypy_return_type' (line 404)
        stypy_return_type_132806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_132806)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__call__'
        return stypy_return_type_132806


# Assigning a type to the variable 'NameValidator' (line 250)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 0), 'NameValidator', NameValidator)

# Assigning a List to a Name (line 302):

# Obtaining an instance of the builtin type 'list' (line 302)
list_132807 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 302)
# Adding element type (line 302)
str_132808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 26), 'str', 'return')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 302, 25), list_132807, str_132808)
# Adding element type (line 302)
str_132809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 36), 'str', 'file')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 302, 25), list_132807, str_132809)
# Adding element type (line 302)
str_132810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 44), 'str', 'print')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 302, 25), list_132807, str_132810)

# Getting the type of 'NameValidator'
NameValidator_132811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NameValidator')
# Setting the type of the member 'defaultexcludelist' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NameValidator_132811, 'defaultexcludelist', list_132807)

# Assigning a Call to a Name (line 303):

# Call to set(...): (line 303)
# Processing the call arguments (line 303)
str_132813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 29), 'str', "~!@#$%^&*()-=+~\\|]}[{';: /?.>,<")
# Processing the call keyword arguments (line 303)
kwargs_132814 = {}
# Getting the type of 'set' (line 303)
set_132812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 25), 'set', False)
# Calling set(args, kwargs) (line 303)
set_call_result_132815 = invoke(stypy.reporting.localization.Localization(__file__, 303, 25), set_132812, *[str_132813], **kwargs_132814)

# Getting the type of 'NameValidator'
NameValidator_132816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NameValidator')
# Setting the type of the member 'defaultdeletechars' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NameValidator_132816, 'defaultdeletechars', set_call_result_132815)

@norecursion
def str2bool(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'str2bool'
    module_type_store = module_type_store.open_function_context('str2bool', 408, 0, False)
    
    # Passed parameters checking function
    str2bool.stypy_localization = localization
    str2bool.stypy_type_of_self = None
    str2bool.stypy_type_store = module_type_store
    str2bool.stypy_function_name = 'str2bool'
    str2bool.stypy_param_names_list = ['value']
    str2bool.stypy_varargs_param_name = None
    str2bool.stypy_kwargs_param_name = None
    str2bool.stypy_call_defaults = defaults
    str2bool.stypy_call_varargs = varargs
    str2bool.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'str2bool', ['value'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'str2bool', localization, ['value'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'str2bool(...)' code ##################

    str_132817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, (-1)), 'str', "\n    Tries to transform a string supposed to represent a boolean to a boolean.\n\n    Parameters\n    ----------\n    value : str\n        The string that is transformed to a boolean.\n\n    Returns\n    -------\n    boolval : bool\n        The boolean representation of `value`.\n\n    Raises\n    ------\n    ValueError\n        If the string is not 'True' or 'False' (case independent)\n\n    Examples\n    --------\n    >>> np.lib._iotools.str2bool('TRUE')\n    True\n    >>> np.lib._iotools.str2bool('false')\n    False\n\n    ")
    
    # Assigning a Call to a Name (line 435):
    
    # Assigning a Call to a Name (line 435):
    
    # Call to upper(...): (line 435)
    # Processing the call keyword arguments (line 435)
    kwargs_132820 = {}
    # Getting the type of 'value' (line 435)
    value_132818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 12), 'value', False)
    # Obtaining the member 'upper' of a type (line 435)
    upper_132819 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 435, 12), value_132818, 'upper')
    # Calling upper(args, kwargs) (line 435)
    upper_call_result_132821 = invoke(stypy.reporting.localization.Localization(__file__, 435, 12), upper_132819, *[], **kwargs_132820)
    
    # Assigning a type to the variable 'value' (line 435)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 435, 4), 'value', upper_call_result_132821)
    
    
    # Getting the type of 'value' (line 436)
    value_132822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 7), 'value')
    
    # Call to asbytes(...): (line 436)
    # Processing the call arguments (line 436)
    str_132824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 436, 24), 'str', 'TRUE')
    # Processing the call keyword arguments (line 436)
    kwargs_132825 = {}
    # Getting the type of 'asbytes' (line 436)
    asbytes_132823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 16), 'asbytes', False)
    # Calling asbytes(args, kwargs) (line 436)
    asbytes_call_result_132826 = invoke(stypy.reporting.localization.Localization(__file__, 436, 16), asbytes_132823, *[str_132824], **kwargs_132825)
    
    # Applying the binary operator '==' (line 436)
    result_eq_132827 = python_operator(stypy.reporting.localization.Localization(__file__, 436, 7), '==', value_132822, asbytes_call_result_132826)
    
    # Testing the type of an if condition (line 436)
    if_condition_132828 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 436, 4), result_eq_132827)
    # Assigning a type to the variable 'if_condition_132828' (line 436)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 436, 4), 'if_condition_132828', if_condition_132828)
    # SSA begins for if statement (line 436)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'True' (line 437)
    True_132829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 15), 'True')
    # Assigning a type to the variable 'stypy_return_type' (line 437)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 437, 8), 'stypy_return_type', True_132829)
    # SSA branch for the else part of an if statement (line 436)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'value' (line 438)
    value_132830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 9), 'value')
    
    # Call to asbytes(...): (line 438)
    # Processing the call arguments (line 438)
    str_132832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 438, 26), 'str', 'FALSE')
    # Processing the call keyword arguments (line 438)
    kwargs_132833 = {}
    # Getting the type of 'asbytes' (line 438)
    asbytes_132831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 18), 'asbytes', False)
    # Calling asbytes(args, kwargs) (line 438)
    asbytes_call_result_132834 = invoke(stypy.reporting.localization.Localization(__file__, 438, 18), asbytes_132831, *[str_132832], **kwargs_132833)
    
    # Applying the binary operator '==' (line 438)
    result_eq_132835 = python_operator(stypy.reporting.localization.Localization(__file__, 438, 9), '==', value_132830, asbytes_call_result_132834)
    
    # Testing the type of an if condition (line 438)
    if_condition_132836 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 438, 9), result_eq_132835)
    # Assigning a type to the variable 'if_condition_132836' (line 438)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 438, 9), 'if_condition_132836', if_condition_132836)
    # SSA begins for if statement (line 438)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'False' (line 439)
    False_132837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 15), 'False')
    # Assigning a type to the variable 'stypy_return_type' (line 439)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 439, 8), 'stypy_return_type', False_132837)
    # SSA branch for the else part of an if statement (line 438)
    module_type_store.open_ssa_branch('else')
    
    # Call to ValueError(...): (line 441)
    # Processing the call arguments (line 441)
    str_132839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 441, 25), 'str', 'Invalid boolean')
    # Processing the call keyword arguments (line 441)
    kwargs_132840 = {}
    # Getting the type of 'ValueError' (line 441)
    ValueError_132838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 441)
    ValueError_call_result_132841 = invoke(stypy.reporting.localization.Localization(__file__, 441, 14), ValueError_132838, *[str_132839], **kwargs_132840)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 441, 8), ValueError_call_result_132841, 'raise parameter', BaseException)
    # SSA join for if statement (line 438)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 436)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'str2bool(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'str2bool' in the type store
    # Getting the type of 'stypy_return_type' (line 408)
    stypy_return_type_132842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_132842)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'str2bool'
    return stypy_return_type_132842

# Assigning a type to the variable 'str2bool' (line 408)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 408, 0), 'str2bool', str2bool)
# Declaration of the 'ConverterError' class
# Getting the type of 'Exception' (line 444)
Exception_132843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 21), 'Exception')

class ConverterError(Exception_132843, ):
    str_132844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 448, (-1)), 'str', '\n    Exception raised when an error occurs in a converter for string values.\n\n    ')
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 444, 0, False)
        # Assigning a type to the variable 'self' (line 445)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 445, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ConverterError.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'ConverterError' (line 444)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 0), 'ConverterError', ConverterError)
# Declaration of the 'ConverterLockError' class
# Getting the type of 'ConverterError' (line 452)
ConverterError_132845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 25), 'ConverterError')

class ConverterLockError(ConverterError_132845, ):
    str_132846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 456, (-1)), 'str', '\n    Exception raised when an attempt is made to upgrade a locked converter.\n\n    ')
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 452, 0, False)
        # Assigning a type to the variable 'self' (line 453)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 453, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ConverterLockError.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'ConverterLockError' (line 452)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 452, 0), 'ConverterLockError', ConverterLockError)
# Declaration of the 'ConversionWarning' class
# Getting the type of 'UserWarning' (line 460)
UserWarning_132847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 24), 'UserWarning')

class ConversionWarning(UserWarning_132847, ):
    str_132848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 469, (-1)), 'str', '\n    Warning issued when a string converter has a problem.\n\n    Notes\n    -----\n    In `genfromtxt` a `ConversionWarning` is issued if raising exceptions\n    is explicitly suppressed with the "invalid_raise" keyword.\n\n    ')
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 460, 0, False)
        # Assigning a type to the variable 'self' (line 461)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 461, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ConversionWarning.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'ConversionWarning' (line 460)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 460, 0), 'ConversionWarning', ConversionWarning)
# Declaration of the 'StringConverter' class

class StringConverter(object, ):
    str_132849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 518, (-1)), 'str', '\n    Factory class for function transforming a string into another object\n    (int, float).\n\n    After initialization, an instance can be called to transform a string\n    into another object. If the string is recognized as representing a\n    missing value, a default value is returned.\n\n    Attributes\n    ----------\n    func : function\n        Function used for the conversion.\n    default : any\n        Default value to return when the input corresponds to a missing\n        value.\n    type : type\n        Type of the output.\n    _status : int\n        Integer representing the order of the conversion.\n    _mapper : sequence of tuples\n        Sequence of tuples (dtype, function, default value) to evaluate in\n        order.\n    _locked : bool\n        Holds `locked` parameter.\n\n    Parameters\n    ----------\n    dtype_or_func : {None, dtype, function}, optional\n        If a `dtype`, specifies the input data type, used to define a basic\n        function and a default value for missing data. For example, when\n        `dtype` is float, the `func` attribute is set to `float` and the\n        default value to `np.nan`.  If a function, this function is used to\n        convert a string to another object. In this case, it is recommended\n        to give an associated default value as input.\n    default : any, optional\n        Value to return by default, that is, when the string to be\n        converted is flagged as missing. If not given, `StringConverter`\n        tries to supply a reasonable default value.\n    missing_values : sequence of str, optional\n        Sequence of strings indicating a missing value.\n    locked : bool, optional\n        Whether the StringConverter should be locked to prevent automatic\n        upgrade or not. Default is False.\n\n    ')
    
    # Assigning a Call to a Tuple (line 533):

    @norecursion
    def _getdtype(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_getdtype'
        module_type_store = module_type_store.open_function_context('_getdtype', 535, 4, False)
        # Assigning a type to the variable 'self' (line 536)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 536, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        StringConverter._getdtype.__dict__.__setitem__('stypy_localization', localization)
        StringConverter._getdtype.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        StringConverter._getdtype.__dict__.__setitem__('stypy_type_store', module_type_store)
        StringConverter._getdtype.__dict__.__setitem__('stypy_function_name', 'StringConverter._getdtype')
        StringConverter._getdtype.__dict__.__setitem__('stypy_param_names_list', ['val'])
        StringConverter._getdtype.__dict__.__setitem__('stypy_varargs_param_name', None)
        StringConverter._getdtype.__dict__.__setitem__('stypy_kwargs_param_name', None)
        StringConverter._getdtype.__dict__.__setitem__('stypy_call_defaults', defaults)
        StringConverter._getdtype.__dict__.__setitem__('stypy_call_varargs', varargs)
        StringConverter._getdtype.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        StringConverter._getdtype.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'StringConverter._getdtype', ['val'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_getdtype', localization, ['val'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_getdtype(...)' code ##################

        str_132850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 537, 8), 'str', 'Returns the dtype of the input variable.')
        
        # Call to array(...): (line 538)
        # Processing the call arguments (line 538)
        # Getting the type of 'val' (line 538)
        val_132853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 24), 'val', False)
        # Processing the call keyword arguments (line 538)
        kwargs_132854 = {}
        # Getting the type of 'np' (line 538)
        np_132851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 15), 'np', False)
        # Obtaining the member 'array' of a type (line 538)
        array_132852 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 538, 15), np_132851, 'array')
        # Calling array(args, kwargs) (line 538)
        array_call_result_132855 = invoke(stypy.reporting.localization.Localization(__file__, 538, 15), array_132852, *[val_132853], **kwargs_132854)
        
        # Obtaining the member 'dtype' of a type (line 538)
        dtype_132856 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 538, 15), array_call_result_132855, 'dtype')
        # Assigning a type to the variable 'stypy_return_type' (line 538)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 538, 8), 'stypy_return_type', dtype_132856)
        
        # ################# End of '_getdtype(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_getdtype' in the type store
        # Getting the type of 'stypy_return_type' (line 535)
        stypy_return_type_132857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_132857)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_getdtype'
        return stypy_return_type_132857


    @norecursion
    def _getsubdtype(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_getsubdtype'
        module_type_store = module_type_store.open_function_context('_getsubdtype', 541, 4, False)
        # Assigning a type to the variable 'self' (line 542)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 542, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        StringConverter._getsubdtype.__dict__.__setitem__('stypy_localization', localization)
        StringConverter._getsubdtype.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        StringConverter._getsubdtype.__dict__.__setitem__('stypy_type_store', module_type_store)
        StringConverter._getsubdtype.__dict__.__setitem__('stypy_function_name', 'StringConverter._getsubdtype')
        StringConverter._getsubdtype.__dict__.__setitem__('stypy_param_names_list', ['val'])
        StringConverter._getsubdtype.__dict__.__setitem__('stypy_varargs_param_name', None)
        StringConverter._getsubdtype.__dict__.__setitem__('stypy_kwargs_param_name', None)
        StringConverter._getsubdtype.__dict__.__setitem__('stypy_call_defaults', defaults)
        StringConverter._getsubdtype.__dict__.__setitem__('stypy_call_varargs', varargs)
        StringConverter._getsubdtype.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        StringConverter._getsubdtype.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'StringConverter._getsubdtype', ['val'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_getsubdtype', localization, ['val'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_getsubdtype(...)' code ##################

        str_132858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 543, 8), 'str', 'Returns the type of the dtype of the input variable.')
        
        # Call to array(...): (line 544)
        # Processing the call arguments (line 544)
        # Getting the type of 'val' (line 544)
        val_132861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 24), 'val', False)
        # Processing the call keyword arguments (line 544)
        kwargs_132862 = {}
        # Getting the type of 'np' (line 544)
        np_132859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 15), 'np', False)
        # Obtaining the member 'array' of a type (line 544)
        array_132860 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 544, 15), np_132859, 'array')
        # Calling array(args, kwargs) (line 544)
        array_call_result_132863 = invoke(stypy.reporting.localization.Localization(__file__, 544, 15), array_132860, *[val_132861], **kwargs_132862)
        
        # Obtaining the member 'dtype' of a type (line 544)
        dtype_132864 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 544, 15), array_call_result_132863, 'dtype')
        # Obtaining the member 'type' of a type (line 544)
        type_132865 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 544, 15), dtype_132864, 'type')
        # Assigning a type to the variable 'stypy_return_type' (line 544)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 544, 8), 'stypy_return_type', type_132865)
        
        # ################# End of '_getsubdtype(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_getsubdtype' in the type store
        # Getting the type of 'stypy_return_type' (line 541)
        stypy_return_type_132866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_132866)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_getsubdtype'
        return stypy_return_type_132866


    @norecursion
    def _dtypeortype(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_dtypeortype'
        module_type_store = module_type_store.open_function_context('_dtypeortype', 551, 4, False)
        # Assigning a type to the variable 'self' (line 552)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 552, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        StringConverter._dtypeortype.__dict__.__setitem__('stypy_localization', localization)
        StringConverter._dtypeortype.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        StringConverter._dtypeortype.__dict__.__setitem__('stypy_type_store', module_type_store)
        StringConverter._dtypeortype.__dict__.__setitem__('stypy_function_name', 'StringConverter._dtypeortype')
        StringConverter._dtypeortype.__dict__.__setitem__('stypy_param_names_list', ['dtype'])
        StringConverter._dtypeortype.__dict__.__setitem__('stypy_varargs_param_name', None)
        StringConverter._dtypeortype.__dict__.__setitem__('stypy_kwargs_param_name', None)
        StringConverter._dtypeortype.__dict__.__setitem__('stypy_call_defaults', defaults)
        StringConverter._dtypeortype.__dict__.__setitem__('stypy_call_varargs', varargs)
        StringConverter._dtypeortype.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        StringConverter._dtypeortype.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'StringConverter._dtypeortype', ['dtype'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_dtypeortype', localization, ['dtype'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_dtypeortype(...)' code ##################

        str_132867 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 553, 8), 'str', 'Returns dtype for datetime64 and type of dtype otherwise.')
        
        
        # Getting the type of 'dtype' (line 554)
        dtype_132868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 11), 'dtype')
        # Obtaining the member 'type' of a type (line 554)
        type_132869 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 554, 11), dtype_132868, 'type')
        # Getting the type of 'np' (line 554)
        np_132870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 25), 'np')
        # Obtaining the member 'datetime64' of a type (line 554)
        datetime64_132871 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 554, 25), np_132870, 'datetime64')
        # Applying the binary operator '==' (line 554)
        result_eq_132872 = python_operator(stypy.reporting.localization.Localization(__file__, 554, 11), '==', type_132869, datetime64_132871)
        
        # Testing the type of an if condition (line 554)
        if_condition_132873 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 554, 8), result_eq_132872)
        # Assigning a type to the variable 'if_condition_132873' (line 554)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 554, 8), 'if_condition_132873', if_condition_132873)
        # SSA begins for if statement (line 554)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'dtype' (line 555)
        dtype_132874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 19), 'dtype')
        # Assigning a type to the variable 'stypy_return_type' (line 555)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 555, 12), 'stypy_return_type', dtype_132874)
        # SSA join for if statement (line 554)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'dtype' (line 556)
        dtype_132875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 15), 'dtype')
        # Obtaining the member 'type' of a type (line 556)
        type_132876 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 556, 15), dtype_132875, 'type')
        # Assigning a type to the variable 'stypy_return_type' (line 556)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 556, 8), 'stypy_return_type', type_132876)
        
        # ################# End of '_dtypeortype(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_dtypeortype' in the type store
        # Getting the type of 'stypy_return_type' (line 551)
        stypy_return_type_132877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_132877)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_dtypeortype'
        return stypy_return_type_132877


    @norecursion
    def upgrade_mapper(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 560)
        None_132878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 42), 'None')
        defaults = [None_132878]
        # Create a new context for function 'upgrade_mapper'
        module_type_store = module_type_store.open_function_context('upgrade_mapper', 559, 4, False)
        # Assigning a type to the variable 'self' (line 560)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 560, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        StringConverter.upgrade_mapper.__dict__.__setitem__('stypy_localization', localization)
        StringConverter.upgrade_mapper.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        StringConverter.upgrade_mapper.__dict__.__setitem__('stypy_type_store', module_type_store)
        StringConverter.upgrade_mapper.__dict__.__setitem__('stypy_function_name', 'StringConverter.upgrade_mapper')
        StringConverter.upgrade_mapper.__dict__.__setitem__('stypy_param_names_list', ['func', 'default'])
        StringConverter.upgrade_mapper.__dict__.__setitem__('stypy_varargs_param_name', None)
        StringConverter.upgrade_mapper.__dict__.__setitem__('stypy_kwargs_param_name', None)
        StringConverter.upgrade_mapper.__dict__.__setitem__('stypy_call_defaults', defaults)
        StringConverter.upgrade_mapper.__dict__.__setitem__('stypy_call_varargs', varargs)
        StringConverter.upgrade_mapper.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        StringConverter.upgrade_mapper.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'StringConverter.upgrade_mapper', ['func', 'default'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'upgrade_mapper', localization, ['func', 'default'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'upgrade_mapper(...)' code ##################

        str_132879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 582, (-1)), 'str', '\n    Upgrade the mapper of a StringConverter by adding a new function and\n    its corresponding default.\n\n    The input function (or sequence of functions) and its associated\n    default value (if any) is inserted in penultimate position of the\n    mapper.  The corresponding type is estimated from the dtype of the\n    default value.\n\n    Parameters\n    ----------\n    func : var\n        Function, or sequence of functions\n\n    Examples\n    --------\n    >>> import dateutil.parser\n    >>> import datetime\n    >>> dateparser = datetustil.parser.parse\n    >>> defaultdate = datetime.date(2000, 1, 1)\n    >>> StringConverter.upgrade_mapper(dateparser, default=defaultdate)\n        ')
        
        # Type idiom detected: calculating its left and rigth part (line 584)
        str_132880 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 584, 25), 'str', '__call__')
        # Getting the type of 'func' (line 584)
        func_132881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 19), 'func')
        
        (may_be_132882, more_types_in_union_132883) = may_provide_member(str_132880, func_132881)

        if may_be_132882:

            if more_types_in_union_132883:
                # Runtime conditional SSA (line 584)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'func' (line 584)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 584, 8), 'func', remove_not_member_provider_from_union(func_132881, '__call__'))
            
            # Call to insert(...): (line 585)
            # Processing the call arguments (line 585)
            int_132887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 585, 31), 'int')
            
            # Obtaining an instance of the builtin type 'tuple' (line 585)
            tuple_132888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 585, 36), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 585)
            # Adding element type (line 585)
            
            # Call to _getsubdtype(...): (line 585)
            # Processing the call arguments (line 585)
            # Getting the type of 'default' (line 585)
            default_132891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 53), 'default', False)
            # Processing the call keyword arguments (line 585)
            kwargs_132892 = {}
            # Getting the type of 'cls' (line 585)
            cls_132889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 36), 'cls', False)
            # Obtaining the member '_getsubdtype' of a type (line 585)
            _getsubdtype_132890 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 585, 36), cls_132889, '_getsubdtype')
            # Calling _getsubdtype(args, kwargs) (line 585)
            _getsubdtype_call_result_132893 = invoke(stypy.reporting.localization.Localization(__file__, 585, 36), _getsubdtype_132890, *[default_132891], **kwargs_132892)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 585, 36), tuple_132888, _getsubdtype_call_result_132893)
            # Adding element type (line 585)
            # Getting the type of 'func' (line 585)
            func_132894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 63), 'func', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 585, 36), tuple_132888, func_132894)
            # Adding element type (line 585)
            # Getting the type of 'default' (line 585)
            default_132895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 69), 'default', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 585, 36), tuple_132888, default_132895)
            
            # Processing the call keyword arguments (line 585)
            kwargs_132896 = {}
            # Getting the type of 'cls' (line 585)
            cls_132884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 12), 'cls', False)
            # Obtaining the member '_mapper' of a type (line 585)
            _mapper_132885 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 585, 12), cls_132884, '_mapper')
            # Obtaining the member 'insert' of a type (line 585)
            insert_132886 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 585, 12), _mapper_132885, 'insert')
            # Calling insert(args, kwargs) (line 585)
            insert_call_result_132897 = invoke(stypy.reporting.localization.Localization(__file__, 585, 12), insert_132886, *[int_132887, tuple_132888], **kwargs_132896)
            
            # Assigning a type to the variable 'stypy_return_type' (line 586)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 586, 12), 'stypy_return_type', types.NoneType)

            if more_types_in_union_132883:
                # Runtime conditional SSA for else branch (line 584)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_132882) or more_types_in_union_132883):
            # Assigning a type to the variable 'func' (line 584)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 584, 8), 'func', remove_member_provider_from_union(func_132881, '__call__'))
            
            # Type idiom detected: calculating its left and rigth part (line 587)
            str_132898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 587, 27), 'str', '__iter__')
            # Getting the type of 'func' (line 587)
            func_132899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 21), 'func')
            
            (may_be_132900, more_types_in_union_132901) = may_provide_member(str_132898, func_132899)

            if may_be_132900:

                if more_types_in_union_132901:
                    # Runtime conditional SSA (line 587)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                # Assigning a type to the variable 'func' (line 587)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 587, 13), 'func', remove_not_member_provider_from_union(func_132899, '__iter__'))
                
                
                # Call to isinstance(...): (line 588)
                # Processing the call arguments (line 588)
                
                # Obtaining the type of the subscript
                int_132903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 588, 31), 'int')
                # Getting the type of 'func' (line 588)
                func_132904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 26), 'func', False)
                # Obtaining the member '__getitem__' of a type (line 588)
                getitem___132905 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 588, 26), func_132904, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 588)
                subscript_call_result_132906 = invoke(stypy.reporting.localization.Localization(__file__, 588, 26), getitem___132905, int_132903)
                
                
                # Obtaining an instance of the builtin type 'tuple' (line 588)
                tuple_132907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 588, 36), 'tuple')
                # Adding type elements to the builtin type 'tuple' instance (line 588)
                # Adding element type (line 588)
                # Getting the type of 'tuple' (line 588)
                tuple_132908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 36), 'tuple', False)
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 588, 36), tuple_132907, tuple_132908)
                # Adding element type (line 588)
                # Getting the type of 'list' (line 588)
                list_132909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 43), 'list', False)
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 588, 36), tuple_132907, list_132909)
                
                # Processing the call keyword arguments (line 588)
                kwargs_132910 = {}
                # Getting the type of 'isinstance' (line 588)
                isinstance_132902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 15), 'isinstance', False)
                # Calling isinstance(args, kwargs) (line 588)
                isinstance_call_result_132911 = invoke(stypy.reporting.localization.Localization(__file__, 588, 15), isinstance_132902, *[subscript_call_result_132906, tuple_132907], **kwargs_132910)
                
                # Testing the type of an if condition (line 588)
                if_condition_132912 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 588, 12), isinstance_call_result_132911)
                # Assigning a type to the variable 'if_condition_132912' (line 588)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 588, 12), 'if_condition_132912', if_condition_132912)
                # SSA begins for if statement (line 588)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Getting the type of 'func' (line 589)
                func_132913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 25), 'func')
                # Testing the type of a for loop iterable (line 589)
                is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 589, 16), func_132913)
                # Getting the type of the for loop variable (line 589)
                for_loop_var_132914 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 589, 16), func_132913)
                # Assigning a type to the variable '_' (line 589)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 589, 16), '_', for_loop_var_132914)
                # SSA begins for a for statement (line 589)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Call to insert(...): (line 590)
                # Processing the call arguments (line 590)
                int_132918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 590, 39), 'int')
                # Getting the type of '_' (line 590)
                __132919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 43), '_', False)
                # Processing the call keyword arguments (line 590)
                kwargs_132920 = {}
                # Getting the type of 'cls' (line 590)
                cls_132915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 20), 'cls', False)
                # Obtaining the member '_mapper' of a type (line 590)
                _mapper_132916 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 590, 20), cls_132915, '_mapper')
                # Obtaining the member 'insert' of a type (line 590)
                insert_132917 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 590, 20), _mapper_132916, 'insert')
                # Calling insert(args, kwargs) (line 590)
                insert_call_result_132921 = invoke(stypy.reporting.localization.Localization(__file__, 590, 20), insert_132917, *[int_132918, __132919], **kwargs_132920)
                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()
                
                # Assigning a type to the variable 'stypy_return_type' (line 591)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 591, 16), 'stypy_return_type', types.NoneType)
                # SSA join for if statement (line 588)
                module_type_store = module_type_store.join_ssa_context()
                
                
                # Type idiom detected: calculating its left and rigth part (line 592)
                # Getting the type of 'default' (line 592)
                default_132922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 15), 'default')
                # Getting the type of 'None' (line 592)
                None_132923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 26), 'None')
                
                (may_be_132924, more_types_in_union_132925) = may_be_none(default_132922, None_132923)

                if may_be_132924:

                    if more_types_in_union_132925:
                        # Runtime conditional SSA (line 592)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                    else:
                        module_type_store = module_type_store

                    
                    # Assigning a BinOp to a Name (line 593):
                    
                    # Assigning a BinOp to a Name (line 593):
                    
                    # Obtaining an instance of the builtin type 'list' (line 593)
                    list_132926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 593, 26), 'list')
                    # Adding type elements to the builtin type 'list' instance (line 593)
                    # Adding element type (line 593)
                    # Getting the type of 'None' (line 593)
                    None_132927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 27), 'None')
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 593, 26), list_132926, None_132927)
                    
                    
                    # Call to len(...): (line 593)
                    # Processing the call arguments (line 593)
                    # Getting the type of 'func' (line 593)
                    func_132929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 39), 'func', False)
                    # Processing the call keyword arguments (line 593)
                    kwargs_132930 = {}
                    # Getting the type of 'len' (line 593)
                    len_132928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 35), 'len', False)
                    # Calling len(args, kwargs) (line 593)
                    len_call_result_132931 = invoke(stypy.reporting.localization.Localization(__file__, 593, 35), len_132928, *[func_132929], **kwargs_132930)
                    
                    # Applying the binary operator '*' (line 593)
                    result_mul_132932 = python_operator(stypy.reporting.localization.Localization(__file__, 593, 26), '*', list_132926, len_call_result_132931)
                    
                    # Assigning a type to the variable 'default' (line 593)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 593, 16), 'default', result_mul_132932)

                    if more_types_in_union_132925:
                        # Runtime conditional SSA for else branch (line 592)
                        module_type_store.open_ssa_branch('idiom else')



                if ((not may_be_132924) or more_types_in_union_132925):
                    
                    # Assigning a Call to a Name (line 595):
                    
                    # Assigning a Call to a Name (line 595):
                    
                    # Call to list(...): (line 595)
                    # Processing the call arguments (line 595)
                    # Getting the type of 'default' (line 595)
                    default_132934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 31), 'default', False)
                    # Processing the call keyword arguments (line 595)
                    kwargs_132935 = {}
                    # Getting the type of 'list' (line 595)
                    list_132933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 26), 'list', False)
                    # Calling list(args, kwargs) (line 595)
                    list_call_result_132936 = invoke(stypy.reporting.localization.Localization(__file__, 595, 26), list_132933, *[default_132934], **kwargs_132935)
                    
                    # Assigning a type to the variable 'default' (line 595)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 595, 16), 'default', list_call_result_132936)
                    
                    # Call to append(...): (line 596)
                    # Processing the call arguments (line 596)
                    
                    # Obtaining an instance of the builtin type 'list' (line 596)
                    list_132939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 596, 31), 'list')
                    # Adding type elements to the builtin type 'list' instance (line 596)
                    # Adding element type (line 596)
                    # Getting the type of 'None' (line 596)
                    None_132940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 32), 'None', False)
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 596, 31), list_132939, None_132940)
                    
                    
                    # Call to len(...): (line 596)
                    # Processing the call arguments (line 596)
                    # Getting the type of 'func' (line 596)
                    func_132942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 45), 'func', False)
                    # Processing the call keyword arguments (line 596)
                    kwargs_132943 = {}
                    # Getting the type of 'len' (line 596)
                    len_132941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 41), 'len', False)
                    # Calling len(args, kwargs) (line 596)
                    len_call_result_132944 = invoke(stypy.reporting.localization.Localization(__file__, 596, 41), len_132941, *[func_132942], **kwargs_132943)
                    
                    
                    # Call to len(...): (line 596)
                    # Processing the call arguments (line 596)
                    # Getting the type of 'default' (line 596)
                    default_132946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 57), 'default', False)
                    # Processing the call keyword arguments (line 596)
                    kwargs_132947 = {}
                    # Getting the type of 'len' (line 596)
                    len_132945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 53), 'len', False)
                    # Calling len(args, kwargs) (line 596)
                    len_call_result_132948 = invoke(stypy.reporting.localization.Localization(__file__, 596, 53), len_132945, *[default_132946], **kwargs_132947)
                    
                    # Applying the binary operator '-' (line 596)
                    result_sub_132949 = python_operator(stypy.reporting.localization.Localization(__file__, 596, 41), '-', len_call_result_132944, len_call_result_132948)
                    
                    # Applying the binary operator '*' (line 596)
                    result_mul_132950 = python_operator(stypy.reporting.localization.Localization(__file__, 596, 31), '*', list_132939, result_sub_132949)
                    
                    # Processing the call keyword arguments (line 596)
                    kwargs_132951 = {}
                    # Getting the type of 'default' (line 596)
                    default_132937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 16), 'default', False)
                    # Obtaining the member 'append' of a type (line 596)
                    append_132938 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 596, 16), default_132937, 'append')
                    # Calling append(args, kwargs) (line 596)
                    append_call_result_132952 = invoke(stypy.reporting.localization.Localization(__file__, 596, 16), append_132938, *[result_mul_132950], **kwargs_132951)
                    

                    if (may_be_132924 and more_types_in_union_132925):
                        # SSA join for if statement (line 592)
                        module_type_store = module_type_store.join_ssa_context()


                
                
                
                # Call to zip(...): (line 597)
                # Processing the call arguments (line 597)
                # Getting the type of 'func' (line 597)
                func_132954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 34), 'func', False)
                # Getting the type of 'default' (line 597)
                default_132955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 40), 'default', False)
                # Processing the call keyword arguments (line 597)
                kwargs_132956 = {}
                # Getting the type of 'zip' (line 597)
                zip_132953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 30), 'zip', False)
                # Calling zip(args, kwargs) (line 597)
                zip_call_result_132957 = invoke(stypy.reporting.localization.Localization(__file__, 597, 30), zip_132953, *[func_132954, default_132955], **kwargs_132956)
                
                # Testing the type of a for loop iterable (line 597)
                is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 597, 12), zip_call_result_132957)
                # Getting the type of the for loop variable (line 597)
                for_loop_var_132958 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 597, 12), zip_call_result_132957)
                # Assigning a type to the variable 'fct' (line 597)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 597, 12), 'fct', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 597, 12), for_loop_var_132958))
                # Assigning a type to the variable 'dft' (line 597)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 597, 12), 'dft', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 597, 12), for_loop_var_132958))
                # SSA begins for a for statement (line 597)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Call to insert(...): (line 598)
                # Processing the call arguments (line 598)
                int_132962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 598, 35), 'int')
                
                # Obtaining an instance of the builtin type 'tuple' (line 598)
                tuple_132963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 598, 40), 'tuple')
                # Adding type elements to the builtin type 'tuple' instance (line 598)
                # Adding element type (line 598)
                
                # Call to _getsubdtype(...): (line 598)
                # Processing the call arguments (line 598)
                # Getting the type of 'dft' (line 598)
                dft_132966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 57), 'dft', False)
                # Processing the call keyword arguments (line 598)
                kwargs_132967 = {}
                # Getting the type of 'cls' (line 598)
                cls_132964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 40), 'cls', False)
                # Obtaining the member '_getsubdtype' of a type (line 598)
                _getsubdtype_132965 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 598, 40), cls_132964, '_getsubdtype')
                # Calling _getsubdtype(args, kwargs) (line 598)
                _getsubdtype_call_result_132968 = invoke(stypy.reporting.localization.Localization(__file__, 598, 40), _getsubdtype_132965, *[dft_132966], **kwargs_132967)
                
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 598, 40), tuple_132963, _getsubdtype_call_result_132968)
                # Adding element type (line 598)
                # Getting the type of 'fct' (line 598)
                fct_132969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 63), 'fct', False)
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 598, 40), tuple_132963, fct_132969)
                # Adding element type (line 598)
                # Getting the type of 'dft' (line 598)
                dft_132970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 68), 'dft', False)
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 598, 40), tuple_132963, dft_132970)
                
                # Processing the call keyword arguments (line 598)
                kwargs_132971 = {}
                # Getting the type of 'cls' (line 598)
                cls_132959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 16), 'cls', False)
                # Obtaining the member '_mapper' of a type (line 598)
                _mapper_132960 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 598, 16), cls_132959, '_mapper')
                # Obtaining the member 'insert' of a type (line 598)
                insert_132961 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 598, 16), _mapper_132960, 'insert')
                # Calling insert(args, kwargs) (line 598)
                insert_call_result_132972 = invoke(stypy.reporting.localization.Localization(__file__, 598, 16), insert_132961, *[int_132962, tuple_132963], **kwargs_132971)
                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()
                

                if more_types_in_union_132901:
                    # SSA join for if statement (line 587)
                    module_type_store = module_type_store.join_ssa_context()


            

            if (may_be_132882 and more_types_in_union_132883):
                # SSA join for if statement (line 584)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # ################# End of 'upgrade_mapper(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'upgrade_mapper' in the type store
        # Getting the type of 'stypy_return_type' (line 559)
        stypy_return_type_132973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_132973)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'upgrade_mapper'
        return stypy_return_type_132973


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 601)
        None_132974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 37), 'None')
        # Getting the type of 'None' (line 601)
        None_132975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 51), 'None')
        # Getting the type of 'None' (line 601)
        None_132976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 72), 'None')
        # Getting the type of 'False' (line 602)
        False_132977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 24), 'False')
        defaults = [None_132974, None_132975, None_132976, False_132977]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 601, 4, False)
        # Assigning a type to the variable 'self' (line 602)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 602, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'StringConverter.__init__', ['dtype_or_func', 'default', 'missing_values', 'locked'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['dtype_or_func', 'default', 'missing_values', 'locked'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Type idiom detected: calculating its left and rigth part (line 604)
        # Getting the type of 'unicode' (line 604)
        unicode_132978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 38), 'unicode')
        # Getting the type of 'missing_values' (line 604)
        missing_values_132979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 22), 'missing_values')
        
        (may_be_132980, more_types_in_union_132981) = may_be_subtype(unicode_132978, missing_values_132979)

        if may_be_132980:

            if more_types_in_union_132981:
                # Runtime conditional SSA (line 604)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'missing_values' (line 604)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 604, 8), 'missing_values', remove_not_subtype_from_union(missing_values_132979, unicode))
            
            # Assigning a Call to a Name (line 605):
            
            # Assigning a Call to a Name (line 605):
            
            # Call to asbytes(...): (line 605)
            # Processing the call arguments (line 605)
            # Getting the type of 'missing_values' (line 605)
            missing_values_132983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 37), 'missing_values', False)
            # Processing the call keyword arguments (line 605)
            kwargs_132984 = {}
            # Getting the type of 'asbytes' (line 605)
            asbytes_132982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 29), 'asbytes', False)
            # Calling asbytes(args, kwargs) (line 605)
            asbytes_call_result_132985 = invoke(stypy.reporting.localization.Localization(__file__, 605, 29), asbytes_132982, *[missing_values_132983], **kwargs_132984)
            
            # Assigning a type to the variable 'missing_values' (line 605)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 605, 12), 'missing_values', asbytes_call_result_132985)

            if more_types_in_union_132981:
                # Runtime conditional SSA for else branch (line 604)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_132980) or more_types_in_union_132981):
            # Assigning a type to the variable 'missing_values' (line 604)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 604, 8), 'missing_values', remove_subtype_from_union(missing_values_132979, unicode))
            
            
            # Call to isinstance(...): (line 606)
            # Processing the call arguments (line 606)
            # Getting the type of 'missing_values' (line 606)
            missing_values_132987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 606, 24), 'missing_values', False)
            
            # Obtaining an instance of the builtin type 'tuple' (line 606)
            tuple_132988 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 606, 41), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 606)
            # Adding element type (line 606)
            # Getting the type of 'list' (line 606)
            list_132989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 606, 41), 'list', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 606, 41), tuple_132988, list_132989)
            # Adding element type (line 606)
            # Getting the type of 'tuple' (line 606)
            tuple_132990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 606, 47), 'tuple', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 606, 41), tuple_132988, tuple_132990)
            
            # Processing the call keyword arguments (line 606)
            kwargs_132991 = {}
            # Getting the type of 'isinstance' (line 606)
            isinstance_132986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 606, 13), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 606)
            isinstance_call_result_132992 = invoke(stypy.reporting.localization.Localization(__file__, 606, 13), isinstance_132986, *[missing_values_132987, tuple_132988], **kwargs_132991)
            
            # Testing the type of an if condition (line 606)
            if_condition_132993 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 606, 13), isinstance_call_result_132992)
            # Assigning a type to the variable 'if_condition_132993' (line 606)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 606, 13), 'if_condition_132993', if_condition_132993)
            # SSA begins for if statement (line 606)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 607):
            
            # Assigning a Call to a Name (line 607):
            
            # Call to asbytes_nested(...): (line 607)
            # Processing the call arguments (line 607)
            # Getting the type of 'missing_values' (line 607)
            missing_values_132995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 607, 44), 'missing_values', False)
            # Processing the call keyword arguments (line 607)
            kwargs_132996 = {}
            # Getting the type of 'asbytes_nested' (line 607)
            asbytes_nested_132994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 607, 29), 'asbytes_nested', False)
            # Calling asbytes_nested(args, kwargs) (line 607)
            asbytes_nested_call_result_132997 = invoke(stypy.reporting.localization.Localization(__file__, 607, 29), asbytes_nested_132994, *[missing_values_132995], **kwargs_132996)
            
            # Assigning a type to the variable 'missing_values' (line 607)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 607, 12), 'missing_values', asbytes_nested_call_result_132997)
            # SSA join for if statement (line 606)
            module_type_store = module_type_store.join_ssa_context()
            

            if (may_be_132980 and more_types_in_union_132981):
                # SSA join for if statement (line 604)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Attribute (line 609):
        
        # Assigning a Call to a Attribute (line 609):
        
        # Call to bool(...): (line 609)
        # Processing the call arguments (line 609)
        # Getting the type of 'locked' (line 609)
        locked_132999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 609, 28), 'locked', False)
        # Processing the call keyword arguments (line 609)
        kwargs_133000 = {}
        # Getting the type of 'bool' (line 609)
        bool_132998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 609, 23), 'bool', False)
        # Calling bool(args, kwargs) (line 609)
        bool_call_result_133001 = invoke(stypy.reporting.localization.Localization(__file__, 609, 23), bool_132998, *[locked_132999], **kwargs_133000)
        
        # Getting the type of 'self' (line 609)
        self_133002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 609, 8), 'self')
        # Setting the type of the member '_locked' of a type (line 609)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 609, 8), self_133002, '_locked', bool_call_result_133001)
        
        # Type idiom detected: calculating its left and rigth part (line 611)
        # Getting the type of 'dtype_or_func' (line 611)
        dtype_or_func_133003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 611, 11), 'dtype_or_func')
        # Getting the type of 'None' (line 611)
        None_133004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 611, 28), 'None')
        
        (may_be_133005, more_types_in_union_133006) = may_be_none(dtype_or_func_133003, None_133004)

        if may_be_133005:

            if more_types_in_union_133006:
                # Runtime conditional SSA (line 611)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Name to a Attribute (line 612):
            
            # Assigning a Name to a Attribute (line 612):
            # Getting the type of 'str2bool' (line 612)
            str2bool_133007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 612, 24), 'str2bool')
            # Getting the type of 'self' (line 612)
            self_133008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 612, 12), 'self')
            # Setting the type of the member 'func' of a type (line 612)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 612, 12), self_133008, 'func', str2bool_133007)
            
            # Assigning a Num to a Attribute (line 613):
            
            # Assigning a Num to a Attribute (line 613):
            int_133009 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 613, 27), 'int')
            # Getting the type of 'self' (line 613)
            self_133010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 12), 'self')
            # Setting the type of the member '_status' of a type (line 613)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 613, 12), self_133010, '_status', int_133009)
            
            # Assigning a BoolOp to a Attribute (line 614):
            
            # Assigning a BoolOp to a Attribute (line 614):
            
            # Evaluating a boolean operation
            # Getting the type of 'default' (line 614)
            default_133011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 614, 27), 'default')
            # Getting the type of 'False' (line 614)
            False_133012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 614, 38), 'False')
            # Applying the binary operator 'or' (line 614)
            result_or_keyword_133013 = python_operator(stypy.reporting.localization.Localization(__file__, 614, 27), 'or', default_133011, False_133012)
            
            # Getting the type of 'self' (line 614)
            self_133014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 614, 12), 'self')
            # Setting the type of the member 'default' of a type (line 614)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 614, 12), self_133014, 'default', result_or_keyword_133013)
            
            # Assigning a Call to a Name (line 615):
            
            # Assigning a Call to a Name (line 615):
            
            # Call to dtype(...): (line 615)
            # Processing the call arguments (line 615)
            str_133017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 615, 29), 'str', 'bool')
            # Processing the call keyword arguments (line 615)
            kwargs_133018 = {}
            # Getting the type of 'np' (line 615)
            np_133015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 20), 'np', False)
            # Obtaining the member 'dtype' of a type (line 615)
            dtype_133016 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 615, 20), np_133015, 'dtype')
            # Calling dtype(args, kwargs) (line 615)
            dtype_call_result_133019 = invoke(stypy.reporting.localization.Localization(__file__, 615, 20), dtype_133016, *[str_133017], **kwargs_133018)
            
            # Assigning a type to the variable 'dtype' (line 615)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 615, 12), 'dtype', dtype_call_result_133019)

            if more_types_in_union_133006:
                # Runtime conditional SSA for else branch (line 611)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_133005) or more_types_in_union_133006):
            
            
            # SSA begins for try-except statement (line 618)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
            
            # Assigning a Name to a Attribute (line 619):
            
            # Assigning a Name to a Attribute (line 619):
            # Getting the type of 'None' (line 619)
            None_133020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 28), 'None')
            # Getting the type of 'self' (line 619)
            self_133021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 16), 'self')
            # Setting the type of the member 'func' of a type (line 619)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 619, 16), self_133021, 'func', None_133020)
            
            # Assigning a Call to a Name (line 620):
            
            # Assigning a Call to a Name (line 620):
            
            # Call to dtype(...): (line 620)
            # Processing the call arguments (line 620)
            # Getting the type of 'dtype_or_func' (line 620)
            dtype_or_func_133024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 33), 'dtype_or_func', False)
            # Processing the call keyword arguments (line 620)
            kwargs_133025 = {}
            # Getting the type of 'np' (line 620)
            np_133022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 24), 'np', False)
            # Obtaining the member 'dtype' of a type (line 620)
            dtype_133023 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 620, 24), np_133022, 'dtype')
            # Calling dtype(args, kwargs) (line 620)
            dtype_call_result_133026 = invoke(stypy.reporting.localization.Localization(__file__, 620, 24), dtype_133023, *[dtype_or_func_133024], **kwargs_133025)
            
            # Assigning a type to the variable 'dtype' (line 620)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 620, 16), 'dtype', dtype_call_result_133026)
            # SSA branch for the except part of a try statement (line 618)
            # SSA branch for the except 'TypeError' branch of a try statement (line 618)
            module_type_store.open_ssa_branch('except')
            
            # Type idiom detected: calculating its left and rigth part (line 623)
            str_133027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 623, 46), 'str', '__call__')
            # Getting the type of 'dtype_or_func' (line 623)
            dtype_or_func_133028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 31), 'dtype_or_func')
            
            (may_be_133029, more_types_in_union_133030) = may_not_provide_member(str_133027, dtype_or_func_133028)

            if may_be_133029:

                if more_types_in_union_133030:
                    # Runtime conditional SSA (line 623)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                # Assigning a type to the variable 'dtype_or_func' (line 623)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 623, 16), 'dtype_or_func', remove_member_provider_from_union(dtype_or_func_133028, '__call__'))
                
                # Assigning a Str to a Name (line 624):
                
                # Assigning a Str to a Name (line 624):
                str_133031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 624, 30), 'str', "The input argument `dtype` is neither a function nor a dtype (got '%s' instead)")
                # Assigning a type to the variable 'errmsg' (line 624)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 624, 20), 'errmsg', str_133031)
                
                # Call to TypeError(...): (line 626)
                # Processing the call arguments (line 626)
                # Getting the type of 'errmsg' (line 626)
                errmsg_133033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 626, 36), 'errmsg', False)
                
                # Call to type(...): (line 626)
                # Processing the call arguments (line 626)
                # Getting the type of 'dtype_or_func' (line 626)
                dtype_or_func_133035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 626, 50), 'dtype_or_func', False)
                # Processing the call keyword arguments (line 626)
                kwargs_133036 = {}
                # Getting the type of 'type' (line 626)
                type_133034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 626, 45), 'type', False)
                # Calling type(args, kwargs) (line 626)
                type_call_result_133037 = invoke(stypy.reporting.localization.Localization(__file__, 626, 45), type_133034, *[dtype_or_func_133035], **kwargs_133036)
                
                # Applying the binary operator '%' (line 626)
                result_mod_133038 = python_operator(stypy.reporting.localization.Localization(__file__, 626, 36), '%', errmsg_133033, type_call_result_133037)
                
                # Processing the call keyword arguments (line 626)
                kwargs_133039 = {}
                # Getting the type of 'TypeError' (line 626)
                TypeError_133032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 626, 26), 'TypeError', False)
                # Calling TypeError(args, kwargs) (line 626)
                TypeError_call_result_133040 = invoke(stypy.reporting.localization.Localization(__file__, 626, 26), TypeError_133032, *[result_mod_133038], **kwargs_133039)
                
                ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 626, 20), TypeError_call_result_133040, 'raise parameter', BaseException)

                if more_types_in_union_133030:
                    # SSA join for if statement (line 623)
                    module_type_store = module_type_store.join_ssa_context()


            
            
            # Assigning a Name to a Attribute (line 628):
            
            # Assigning a Name to a Attribute (line 628):
            # Getting the type of 'dtype_or_func' (line 628)
            dtype_or_func_133041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 628, 28), 'dtype_or_func')
            # Getting the type of 'self' (line 628)
            self_133042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 628, 16), 'self')
            # Setting the type of the member 'func' of a type (line 628)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 628, 16), self_133042, 'func', dtype_or_func_133041)
            
            # Type idiom detected: calculating its left and rigth part (line 631)
            # Getting the type of 'default' (line 631)
            default_133043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 631, 19), 'default')
            # Getting the type of 'None' (line 631)
            None_133044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 631, 30), 'None')
            
            (may_be_133045, more_types_in_union_133046) = may_be_none(default_133043, None_133044)

            if may_be_133045:

                if more_types_in_union_133046:
                    # Runtime conditional SSA (line 631)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                
                
                # SSA begins for try-except statement (line 632)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
                
                # Assigning a Call to a Name (line 633):
                
                # Assigning a Call to a Name (line 633):
                
                # Call to func(...): (line 633)
                # Processing the call arguments (line 633)
                
                # Call to asbytes(...): (line 633)
                # Processing the call arguments (line 633)
                str_133050 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 633, 52), 'str', '0')
                # Processing the call keyword arguments (line 633)
                kwargs_133051 = {}
                # Getting the type of 'asbytes' (line 633)
                asbytes_133049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 633, 44), 'asbytes', False)
                # Calling asbytes(args, kwargs) (line 633)
                asbytes_call_result_133052 = invoke(stypy.reporting.localization.Localization(__file__, 633, 44), asbytes_133049, *[str_133050], **kwargs_133051)
                
                # Processing the call keyword arguments (line 633)
                kwargs_133053 = {}
                # Getting the type of 'self' (line 633)
                self_133047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 633, 34), 'self', False)
                # Obtaining the member 'func' of a type (line 633)
                func_133048 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 633, 34), self_133047, 'func')
                # Calling func(args, kwargs) (line 633)
                func_call_result_133054 = invoke(stypy.reporting.localization.Localization(__file__, 633, 34), func_133048, *[asbytes_call_result_133052], **kwargs_133053)
                
                # Assigning a type to the variable 'default' (line 633)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 633, 24), 'default', func_call_result_133054)
                # SSA branch for the except part of a try statement (line 632)
                # SSA branch for the except 'ValueError' branch of a try statement (line 632)
                module_type_store.open_ssa_branch('except')
                
                # Assigning a Name to a Name (line 635):
                
                # Assigning a Name to a Name (line 635):
                # Getting the type of 'None' (line 635)
                None_133055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 34), 'None')
                # Assigning a type to the variable 'default' (line 635)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 635, 24), 'default', None_133055)
                # SSA join for try-except statement (line 632)
                module_type_store = module_type_store.join_ssa_context()
                

                if more_types_in_union_133046:
                    # SSA join for if statement (line 631)
                    module_type_store = module_type_store.join_ssa_context()


            
            
            # Assigning a Call to a Name (line 636):
            
            # Assigning a Call to a Name (line 636):
            
            # Call to _getdtype(...): (line 636)
            # Processing the call arguments (line 636)
            # Getting the type of 'default' (line 636)
            default_133058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 39), 'default', False)
            # Processing the call keyword arguments (line 636)
            kwargs_133059 = {}
            # Getting the type of 'self' (line 636)
            self_133056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 24), 'self', False)
            # Obtaining the member '_getdtype' of a type (line 636)
            _getdtype_133057 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 636, 24), self_133056, '_getdtype')
            # Calling _getdtype(args, kwargs) (line 636)
            _getdtype_call_result_133060 = invoke(stypy.reporting.localization.Localization(__file__, 636, 24), _getdtype_133057, *[default_133058], **kwargs_133059)
            
            # Assigning a type to the variable 'dtype' (line 636)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 636, 16), 'dtype', _getdtype_call_result_133060)
            # SSA join for try-except statement (line 618)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Assigning a Num to a Name (line 638):
            
            # Assigning a Num to a Name (line 638):
            int_133061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 638, 22), 'int')
            # Assigning a type to the variable '_status' (line 638)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 638, 12), '_status', int_133061)
            
            
            # Call to enumerate(...): (line 639)
            # Processing the call arguments (line 639)
            # Getting the type of 'self' (line 639)
            self_133063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 639, 63), 'self', False)
            # Obtaining the member '_mapper' of a type (line 639)
            _mapper_133064 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 639, 63), self_133063, '_mapper')
            # Processing the call keyword arguments (line 639)
            kwargs_133065 = {}
            # Getting the type of 'enumerate' (line 639)
            enumerate_133062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 639, 53), 'enumerate', False)
            # Calling enumerate(args, kwargs) (line 639)
            enumerate_call_result_133066 = invoke(stypy.reporting.localization.Localization(__file__, 639, 53), enumerate_133062, *[_mapper_133064], **kwargs_133065)
            
            # Testing the type of a for loop iterable (line 639)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 639, 12), enumerate_call_result_133066)
            # Getting the type of the for loop variable (line 639)
            for_loop_var_133067 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 639, 12), enumerate_call_result_133066)
            # Assigning a type to the variable 'i' (line 639)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 639, 12), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 639, 12), for_loop_var_133067))
            # Assigning a type to the variable 'deftype' (line 639)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 639, 12), 'deftype', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 639, 12), for_loop_var_133067))
            # Assigning a type to the variable 'func' (line 639)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 639, 12), 'func', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 639, 12), for_loop_var_133067))
            # Assigning a type to the variable 'default_def' (line 639)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 639, 12), 'default_def', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 639, 12), for_loop_var_133067))
            # SSA begins for a for statement (line 639)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            
            # Call to issubdtype(...): (line 640)
            # Processing the call arguments (line 640)
            # Getting the type of 'dtype' (line 640)
            dtype_133070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 640, 33), 'dtype', False)
            # Obtaining the member 'type' of a type (line 640)
            type_133071 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 640, 33), dtype_133070, 'type')
            # Getting the type of 'deftype' (line 640)
            deftype_133072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 640, 45), 'deftype', False)
            # Processing the call keyword arguments (line 640)
            kwargs_133073 = {}
            # Getting the type of 'np' (line 640)
            np_133068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 640, 19), 'np', False)
            # Obtaining the member 'issubdtype' of a type (line 640)
            issubdtype_133069 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 640, 19), np_133068, 'issubdtype')
            # Calling issubdtype(args, kwargs) (line 640)
            issubdtype_call_result_133074 = invoke(stypy.reporting.localization.Localization(__file__, 640, 19), issubdtype_133069, *[type_133071, deftype_133072], **kwargs_133073)
            
            # Testing the type of an if condition (line 640)
            if_condition_133075 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 640, 16), issubdtype_call_result_133074)
            # Assigning a type to the variable 'if_condition_133075' (line 640)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 640, 16), 'if_condition_133075', if_condition_133075)
            # SSA begins for if statement (line 640)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Name to a Name (line 641):
            
            # Assigning a Name to a Name (line 641):
            # Getting the type of 'i' (line 641)
            i_133076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 641, 30), 'i')
            # Assigning a type to the variable '_status' (line 641)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 641, 20), '_status', i_133076)
            
            # Type idiom detected: calculating its left and rigth part (line 642)
            # Getting the type of 'default' (line 642)
            default_133077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 642, 23), 'default')
            # Getting the type of 'None' (line 642)
            None_133078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 642, 34), 'None')
            
            (may_be_133079, more_types_in_union_133080) = may_be_none(default_133077, None_133078)

            if may_be_133079:

                if more_types_in_union_133080:
                    # Runtime conditional SSA (line 642)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                
                # Assigning a Name to a Attribute (line 643):
                
                # Assigning a Name to a Attribute (line 643):
                # Getting the type of 'default_def' (line 643)
                default_def_133081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 643, 39), 'default_def')
                # Getting the type of 'self' (line 643)
                self_133082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 643, 24), 'self')
                # Setting the type of the member 'default' of a type (line 643)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 643, 24), self_133082, 'default', default_def_133081)

                if more_types_in_union_133080:
                    # Runtime conditional SSA for else branch (line 642)
                    module_type_store.open_ssa_branch('idiom else')



            if ((not may_be_133079) or more_types_in_union_133080):
                
                # Assigning a Name to a Attribute (line 645):
                
                # Assigning a Name to a Attribute (line 645):
                # Getting the type of 'default' (line 645)
                default_133083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 645, 39), 'default')
                # Getting the type of 'self' (line 645)
                self_133084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 645, 24), 'self')
                # Setting the type of the member 'default' of a type (line 645)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 645, 24), self_133084, 'default', default_133083)

                if (may_be_133079 and more_types_in_union_133080):
                    # SSA join for if statement (line 642)
                    module_type_store = module_type_store.join_ssa_context()


            
            # SSA join for if statement (line 640)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Assigning a Name to a Name (line 648):
            
            # Assigning a Name to a Name (line 648):
            # Getting the type of 'func' (line 648)
            func_133085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 648, 24), 'func')
            # Assigning a type to the variable 'last_func' (line 648)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 648, 12), 'last_func', func_133085)
            
            
            # Call to enumerate(...): (line 649)
            # Processing the call arguments (line 649)
            # Getting the type of 'self' (line 649)
            self_133087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 63), 'self', False)
            # Obtaining the member '_mapper' of a type (line 649)
            _mapper_133088 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 649, 63), self_133087, '_mapper')
            # Processing the call keyword arguments (line 649)
            kwargs_133089 = {}
            # Getting the type of 'enumerate' (line 649)
            enumerate_133086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 53), 'enumerate', False)
            # Calling enumerate(args, kwargs) (line 649)
            enumerate_call_result_133090 = invoke(stypy.reporting.localization.Localization(__file__, 649, 53), enumerate_133086, *[_mapper_133088], **kwargs_133089)
            
            # Testing the type of a for loop iterable (line 649)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 649, 12), enumerate_call_result_133090)
            # Getting the type of the for loop variable (line 649)
            for_loop_var_133091 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 649, 12), enumerate_call_result_133090)
            # Assigning a type to the variable 'i' (line 649)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 649, 12), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 649, 12), for_loop_var_133091))
            # Assigning a type to the variable 'deftype' (line 649)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 649, 12), 'deftype', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 649, 12), for_loop_var_133091))
            # Assigning a type to the variable 'func' (line 649)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 649, 12), 'func', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 649, 12), for_loop_var_133091))
            # Assigning a type to the variable 'default_def' (line 649)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 649, 12), 'default_def', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 649, 12), for_loop_var_133091))
            # SSA begins for a for statement (line 649)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            
            # Getting the type of 'dtype' (line 650)
            dtype_133092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 19), 'dtype')
            # Obtaining the member 'type' of a type (line 650)
            type_133093 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 650, 19), dtype_133092, 'type')
            # Getting the type of 'deftype' (line 650)
            deftype_133094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 33), 'deftype')
            # Applying the binary operator '==' (line 650)
            result_eq_133095 = python_operator(stypy.reporting.localization.Localization(__file__, 650, 19), '==', type_133093, deftype_133094)
            
            # Testing the type of an if condition (line 650)
            if_condition_133096 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 650, 16), result_eq_133095)
            # Assigning a type to the variable 'if_condition_133096' (line 650)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 650, 16), 'if_condition_133096', if_condition_133096)
            # SSA begins for if statement (line 650)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Name to a Name (line 651):
            
            # Assigning a Name to a Name (line 651):
            # Getting the type of 'i' (line 651)
            i_133097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 30), 'i')
            # Assigning a type to the variable '_status' (line 651)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 651, 20), '_status', i_133097)
            
            # Assigning a Name to a Name (line 652):
            
            # Assigning a Name to a Name (line 652):
            # Getting the type of 'func' (line 652)
            func_133098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 652, 32), 'func')
            # Assigning a type to the variable 'last_func' (line 652)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 652, 20), 'last_func', func_133098)
            
            # Type idiom detected: calculating its left and rigth part (line 653)
            # Getting the type of 'default' (line 653)
            default_133099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 23), 'default')
            # Getting the type of 'None' (line 653)
            None_133100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 34), 'None')
            
            (may_be_133101, more_types_in_union_133102) = may_be_none(default_133099, None_133100)

            if may_be_133101:

                if more_types_in_union_133102:
                    # Runtime conditional SSA (line 653)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                
                # Assigning a Name to a Attribute (line 654):
                
                # Assigning a Name to a Attribute (line 654):
                # Getting the type of 'default_def' (line 654)
                default_def_133103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 654, 39), 'default_def')
                # Getting the type of 'self' (line 654)
                self_133104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 654, 24), 'self')
                # Setting the type of the member 'default' of a type (line 654)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 654, 24), self_133104, 'default', default_def_133103)

                if more_types_in_union_133102:
                    # Runtime conditional SSA for else branch (line 653)
                    module_type_store.open_ssa_branch('idiom else')



            if ((not may_be_133101) or more_types_in_union_133102):
                
                # Assigning a Name to a Attribute (line 656):
                
                # Assigning a Name to a Attribute (line 656):
                # Getting the type of 'default' (line 656)
                default_133105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 39), 'default')
                # Getting the type of 'self' (line 656)
                self_133106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 24), 'self')
                # Setting the type of the member 'default' of a type (line 656)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 656, 24), self_133106, 'default', default_133105)

                if (may_be_133101 and more_types_in_union_133102):
                    # SSA join for if statement (line 653)
                    module_type_store = module_type_store.join_ssa_context()


            
            # SSA join for if statement (line 650)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Assigning a Name to a Name (line 658):
            
            # Assigning a Name to a Name (line 658):
            # Getting the type of 'last_func' (line 658)
            last_func_133107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 658, 19), 'last_func')
            # Assigning a type to the variable 'func' (line 658)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 658, 12), 'func', last_func_133107)
            
            
            # Getting the type of '_status' (line 659)
            _status_133108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 15), '_status')
            int_133109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 659, 26), 'int')
            # Applying the binary operator '==' (line 659)
            result_eq_133110 = python_operator(stypy.reporting.localization.Localization(__file__, 659, 15), '==', _status_133108, int_133109)
            
            # Testing the type of an if condition (line 659)
            if_condition_133111 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 659, 12), result_eq_133110)
            # Assigning a type to the variable 'if_condition_133111' (line 659)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 659, 12), 'if_condition_133111', if_condition_133111)
            # SSA begins for if statement (line 659)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Num to a Name (line 661):
            
            # Assigning a Num to a Name (line 661):
            int_133112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 661, 26), 'int')
            # Assigning a type to the variable '_status' (line 661)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 661, 16), '_status', int_133112)
            
            # Assigning a Name to a Attribute (line 662):
            
            # Assigning a Name to a Attribute (line 662):
            # Getting the type of 'default' (line 662)
            default_133113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 31), 'default')
            # Getting the type of 'self' (line 662)
            self_133114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 16), 'self')
            # Setting the type of the member 'default' of a type (line 662)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 662, 16), self_133114, 'default', default_133113)
            # SSA join for if statement (line 659)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Assigning a Name to a Attribute (line 663):
            
            # Assigning a Name to a Attribute (line 663):
            # Getting the type of '_status' (line 663)
            _status_133115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 27), '_status')
            # Getting the type of 'self' (line 663)
            self_133116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 12), 'self')
            # Setting the type of the member '_status' of a type (line 663)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 663, 12), self_133116, '_status', _status_133115)
            
            # Type idiom detected: calculating its left and rigth part (line 665)
            # Getting the type of 'self' (line 665)
            self_133117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 665, 15), 'self')
            # Obtaining the member 'func' of a type (line 665)
            func_133118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 665, 15), self_133117, 'func')
            # Getting the type of 'None' (line 665)
            None_133119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 665, 28), 'None')
            
            (may_be_133120, more_types_in_union_133121) = may_be_none(func_133118, None_133119)

            if may_be_133120:

                if more_types_in_union_133121:
                    # Runtime conditional SSA (line 665)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                
                # Assigning a Name to a Attribute (line 666):
                
                # Assigning a Name to a Attribute (line 666):
                # Getting the type of 'func' (line 666)
                func_133122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 28), 'func')
                # Getting the type of 'self' (line 666)
                self_133123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 16), 'self')
                # Setting the type of the member 'func' of a type (line 666)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 666, 16), self_133123, 'func', func_133122)

                if more_types_in_union_133121:
                    # SSA join for if statement (line 665)
                    module_type_store = module_type_store.join_ssa_context()


            
            
            
            # Getting the type of 'self' (line 669)
            self_133124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 669, 15), 'self')
            # Obtaining the member 'func' of a type (line 669)
            func_133125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 669, 15), self_133124, 'func')
            
            # Obtaining the type of the subscript
            int_133126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 669, 44), 'int')
            
            # Obtaining the type of the subscript
            int_133127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 669, 41), 'int')
            # Getting the type of 'self' (line 669)
            self_133128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 669, 28), 'self')
            # Obtaining the member '_mapper' of a type (line 669)
            _mapper_133129 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 669, 28), self_133128, '_mapper')
            # Obtaining the member '__getitem__' of a type (line 669)
            getitem___133130 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 669, 28), _mapper_133129, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 669)
            subscript_call_result_133131 = invoke(stypy.reporting.localization.Localization(__file__, 669, 28), getitem___133130, int_133127)
            
            # Obtaining the member '__getitem__' of a type (line 669)
            getitem___133132 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 669, 28), subscript_call_result_133131, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 669)
            subscript_call_result_133133 = invoke(stypy.reporting.localization.Localization(__file__, 669, 28), getitem___133132, int_133126)
            
            # Applying the binary operator '==' (line 669)
            result_eq_133134 = python_operator(stypy.reporting.localization.Localization(__file__, 669, 15), '==', func_133125, subscript_call_result_133133)
            
            # Testing the type of an if condition (line 669)
            if_condition_133135 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 669, 12), result_eq_133134)
            # Assigning a type to the variable 'if_condition_133135' (line 669)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 669, 12), 'if_condition_133135', if_condition_133135)
            # SSA begins for if statement (line 669)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            
            # Call to issubclass(...): (line 670)
            # Processing the call arguments (line 670)
            # Getting the type of 'dtype' (line 670)
            dtype_133137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 670, 30), 'dtype', False)
            # Obtaining the member 'type' of a type (line 670)
            type_133138 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 670, 30), dtype_133137, 'type')
            # Getting the type of 'np' (line 670)
            np_133139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 670, 42), 'np', False)
            # Obtaining the member 'uint64' of a type (line 670)
            uint64_133140 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 670, 42), np_133139, 'uint64')
            # Processing the call keyword arguments (line 670)
            kwargs_133141 = {}
            # Getting the type of 'issubclass' (line 670)
            issubclass_133136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 670, 19), 'issubclass', False)
            # Calling issubclass(args, kwargs) (line 670)
            issubclass_call_result_133142 = invoke(stypy.reporting.localization.Localization(__file__, 670, 19), issubclass_133136, *[type_133138, uint64_133140], **kwargs_133141)
            
            # Testing the type of an if condition (line 670)
            if_condition_133143 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 670, 16), issubclass_call_result_133142)
            # Assigning a type to the variable 'if_condition_133143' (line 670)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 670, 16), 'if_condition_133143', if_condition_133143)
            # SSA begins for if statement (line 670)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Attribute to a Attribute (line 671):
            
            # Assigning a Attribute to a Attribute (line 671):
            # Getting the type of 'np' (line 671)
            np_133144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 671, 32), 'np')
            # Obtaining the member 'uint64' of a type (line 671)
            uint64_133145 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 671, 32), np_133144, 'uint64')
            # Getting the type of 'self' (line 671)
            self_133146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 671, 20), 'self')
            # Setting the type of the member 'func' of a type (line 671)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 671, 20), self_133146, 'func', uint64_133145)
            # SSA branch for the else part of an if statement (line 670)
            module_type_store.open_ssa_branch('else')
            
            
            # Call to issubclass(...): (line 672)
            # Processing the call arguments (line 672)
            # Getting the type of 'dtype' (line 672)
            dtype_133148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 672, 32), 'dtype', False)
            # Obtaining the member 'type' of a type (line 672)
            type_133149 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 672, 32), dtype_133148, 'type')
            # Getting the type of 'np' (line 672)
            np_133150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 672, 44), 'np', False)
            # Obtaining the member 'int64' of a type (line 672)
            int64_133151 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 672, 44), np_133150, 'int64')
            # Processing the call keyword arguments (line 672)
            kwargs_133152 = {}
            # Getting the type of 'issubclass' (line 672)
            issubclass_133147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 672, 21), 'issubclass', False)
            # Calling issubclass(args, kwargs) (line 672)
            issubclass_call_result_133153 = invoke(stypy.reporting.localization.Localization(__file__, 672, 21), issubclass_133147, *[type_133149, int64_133151], **kwargs_133152)
            
            # Testing the type of an if condition (line 672)
            if_condition_133154 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 672, 21), issubclass_call_result_133153)
            # Assigning a type to the variable 'if_condition_133154' (line 672)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 672, 21), 'if_condition_133154', if_condition_133154)
            # SSA begins for if statement (line 672)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Attribute to a Attribute (line 673):
            
            # Assigning a Attribute to a Attribute (line 673):
            # Getting the type of 'np' (line 673)
            np_133155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 673, 32), 'np')
            # Obtaining the member 'int64' of a type (line 673)
            int64_133156 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 673, 32), np_133155, 'int64')
            # Getting the type of 'self' (line 673)
            self_133157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 673, 20), 'self')
            # Setting the type of the member 'func' of a type (line 673)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 673, 20), self_133157, 'func', int64_133156)
            # SSA branch for the else part of an if statement (line 672)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Lambda to a Attribute (line 675):
            
            # Assigning a Lambda to a Attribute (line 675):

            @norecursion
            def _stypy_temp_lambda_36(localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function '_stypy_temp_lambda_36'
                module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_36', 675, 32, True)
                # Passed parameters checking function
                _stypy_temp_lambda_36.stypy_localization = localization
                _stypy_temp_lambda_36.stypy_type_of_self = None
                _stypy_temp_lambda_36.stypy_type_store = module_type_store
                _stypy_temp_lambda_36.stypy_function_name = '_stypy_temp_lambda_36'
                _stypy_temp_lambda_36.stypy_param_names_list = ['x']
                _stypy_temp_lambda_36.stypy_varargs_param_name = None
                _stypy_temp_lambda_36.stypy_kwargs_param_name = None
                _stypy_temp_lambda_36.stypy_call_defaults = defaults
                _stypy_temp_lambda_36.stypy_call_varargs = varargs
                _stypy_temp_lambda_36.stypy_call_kwargs = kwargs
                arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_36', ['x'], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Stacktrace push for error reporting
                localization.set_stack_trace('_stypy_temp_lambda_36', ['x'], arguments)
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of the lambda function code ##################

                
                # Call to int(...): (line 675)
                # Processing the call arguments (line 675)
                
                # Call to float(...): (line 675)
                # Processing the call arguments (line 675)
                # Getting the type of 'x' (line 675)
                x_133160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 52), 'x', False)
                # Processing the call keyword arguments (line 675)
                kwargs_133161 = {}
                # Getting the type of 'float' (line 675)
                float_133159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 46), 'float', False)
                # Calling float(args, kwargs) (line 675)
                float_call_result_133162 = invoke(stypy.reporting.localization.Localization(__file__, 675, 46), float_133159, *[x_133160], **kwargs_133161)
                
                # Processing the call keyword arguments (line 675)
                kwargs_133163 = {}
                # Getting the type of 'int' (line 675)
                int_133158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 42), 'int', False)
                # Calling int(args, kwargs) (line 675)
                int_call_result_133164 = invoke(stypy.reporting.localization.Localization(__file__, 675, 42), int_133158, *[float_call_result_133162], **kwargs_133163)
                
                # Assigning the return type of the lambda function
                # Assigning a type to the variable 'stypy_return_type' (line 675)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 675, 32), 'stypy_return_type', int_call_result_133164)
                
                # ################# End of the lambda function code ##################

                # Stacktrace pop (error reporting)
                localization.unset_stack_trace()
                
                # Storing the return type of function '_stypy_temp_lambda_36' in the type store
                # Getting the type of 'stypy_return_type' (line 675)
                stypy_return_type_133165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 32), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_133165)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function '_stypy_temp_lambda_36'
                return stypy_return_type_133165

            # Assigning a type to the variable '_stypy_temp_lambda_36' (line 675)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 675, 32), '_stypy_temp_lambda_36', _stypy_temp_lambda_36)
            # Getting the type of '_stypy_temp_lambda_36' (line 675)
            _stypy_temp_lambda_36_133166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 32), '_stypy_temp_lambda_36')
            # Getting the type of 'self' (line 675)
            self_133167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 20), 'self')
            # Setting the type of the member 'func' of a type (line 675)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 675, 20), self_133167, 'func', _stypy_temp_lambda_36_133166)
            # SSA join for if statement (line 672)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 670)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 669)
            module_type_store = module_type_store.join_ssa_context()
            

            if (may_be_133005 and more_types_in_union_133006):
                # SSA join for if statement (line 611)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 677)
        # Getting the type of 'missing_values' (line 677)
        missing_values_133168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 677, 11), 'missing_values')
        # Getting the type of 'None' (line 677)
        None_133169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 677, 29), 'None')
        
        (may_be_133170, more_types_in_union_133171) = may_be_none(missing_values_133168, None_133169)

        if may_be_133170:

            if more_types_in_union_133171:
                # Runtime conditional SSA (line 677)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Attribute (line 678):
            
            # Assigning a Call to a Attribute (line 678):
            
            # Call to set(...): (line 678)
            # Processing the call arguments (line 678)
            
            # Obtaining an instance of the builtin type 'list' (line 678)
            list_133173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 678, 38), 'list')
            # Adding type elements to the builtin type 'list' instance (line 678)
            # Adding element type (line 678)
            
            # Call to asbytes(...): (line 678)
            # Processing the call arguments (line 678)
            str_133175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 678, 47), 'str', '')
            # Processing the call keyword arguments (line 678)
            kwargs_133176 = {}
            # Getting the type of 'asbytes' (line 678)
            asbytes_133174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 678, 39), 'asbytes', False)
            # Calling asbytes(args, kwargs) (line 678)
            asbytes_call_result_133177 = invoke(stypy.reporting.localization.Localization(__file__, 678, 39), asbytes_133174, *[str_133175], **kwargs_133176)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 678, 38), list_133173, asbytes_call_result_133177)
            
            # Processing the call keyword arguments (line 678)
            kwargs_133178 = {}
            # Getting the type of 'set' (line 678)
            set_133172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 678, 34), 'set', False)
            # Calling set(args, kwargs) (line 678)
            set_call_result_133179 = invoke(stypy.reporting.localization.Localization(__file__, 678, 34), set_133172, *[list_133173], **kwargs_133178)
            
            # Getting the type of 'self' (line 678)
            self_133180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 678, 12), 'self')
            # Setting the type of the member 'missing_values' of a type (line 678)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 678, 12), self_133180, 'missing_values', set_call_result_133179)

            if more_types_in_union_133171:
                # Runtime conditional SSA for else branch (line 677)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_133170) or more_types_in_union_133171):
            
            # Type idiom detected: calculating its left and rigth part (line 680)
            # Getting the type of 'bytes' (line 680)
            bytes_133181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 680, 42), 'bytes')
            # Getting the type of 'missing_values' (line 680)
            missing_values_133182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 680, 26), 'missing_values')
            
            (may_be_133183, more_types_in_union_133184) = may_be_subtype(bytes_133181, missing_values_133182)

            if may_be_133183:

                if more_types_in_union_133184:
                    # Runtime conditional SSA (line 680)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                # Assigning a type to the variable 'missing_values' (line 680)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 680, 12), 'missing_values', remove_not_subtype_from_union(missing_values_133182, bytes))
                
                # Assigning a Call to a Name (line 681):
                
                # Assigning a Call to a Name (line 681):
                
                # Call to split(...): (line 681)
                # Processing the call arguments (line 681)
                
                # Call to asbytes(...): (line 681)
                # Processing the call arguments (line 681)
                str_133188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 681, 62), 'str', ',')
                # Processing the call keyword arguments (line 681)
                kwargs_133189 = {}
                # Getting the type of 'asbytes' (line 681)
                asbytes_133187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 54), 'asbytes', False)
                # Calling asbytes(args, kwargs) (line 681)
                asbytes_call_result_133190 = invoke(stypy.reporting.localization.Localization(__file__, 681, 54), asbytes_133187, *[str_133188], **kwargs_133189)
                
                # Processing the call keyword arguments (line 681)
                kwargs_133191 = {}
                # Getting the type of 'missing_values' (line 681)
                missing_values_133185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 33), 'missing_values', False)
                # Obtaining the member 'split' of a type (line 681)
                split_133186 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 681, 33), missing_values_133185, 'split')
                # Calling split(args, kwargs) (line 681)
                split_call_result_133192 = invoke(stypy.reporting.localization.Localization(__file__, 681, 33), split_133186, *[asbytes_call_result_133190], **kwargs_133191)
                
                # Assigning a type to the variable 'missing_values' (line 681)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 681, 16), 'missing_values', split_call_result_133192)

                if more_types_in_union_133184:
                    # SSA join for if statement (line 680)
                    module_type_store = module_type_store.join_ssa_context()


            
            
            # Assigning a Call to a Attribute (line 682):
            
            # Assigning a Call to a Attribute (line 682):
            
            # Call to set(...): (line 682)
            # Processing the call arguments (line 682)
            
            # Call to list(...): (line 682)
            # Processing the call arguments (line 682)
            # Getting the type of 'missing_values' (line 682)
            missing_values_133195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 682, 43), 'missing_values', False)
            # Processing the call keyword arguments (line 682)
            kwargs_133196 = {}
            # Getting the type of 'list' (line 682)
            list_133194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 682, 38), 'list', False)
            # Calling list(args, kwargs) (line 682)
            list_call_result_133197 = invoke(stypy.reporting.localization.Localization(__file__, 682, 38), list_133194, *[missing_values_133195], **kwargs_133196)
            
            
            # Obtaining an instance of the builtin type 'list' (line 682)
            list_133198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 682, 61), 'list')
            # Adding type elements to the builtin type 'list' instance (line 682)
            # Adding element type (line 682)
            
            # Call to asbytes(...): (line 682)
            # Processing the call arguments (line 682)
            str_133200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 682, 70), 'str', '')
            # Processing the call keyword arguments (line 682)
            kwargs_133201 = {}
            # Getting the type of 'asbytes' (line 682)
            asbytes_133199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 682, 62), 'asbytes', False)
            # Calling asbytes(args, kwargs) (line 682)
            asbytes_call_result_133202 = invoke(stypy.reporting.localization.Localization(__file__, 682, 62), asbytes_133199, *[str_133200], **kwargs_133201)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 682, 61), list_133198, asbytes_call_result_133202)
            
            # Applying the binary operator '+' (line 682)
            result_add_133203 = python_operator(stypy.reporting.localization.Localization(__file__, 682, 38), '+', list_call_result_133197, list_133198)
            
            # Processing the call keyword arguments (line 682)
            kwargs_133204 = {}
            # Getting the type of 'set' (line 682)
            set_133193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 682, 34), 'set', False)
            # Calling set(args, kwargs) (line 682)
            set_call_result_133205 = invoke(stypy.reporting.localization.Localization(__file__, 682, 34), set_133193, *[result_add_133203], **kwargs_133204)
            
            # Getting the type of 'self' (line 682)
            self_133206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 682, 12), 'self')
            # Setting the type of the member 'missing_values' of a type (line 682)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 682, 12), self_133206, 'missing_values', set_call_result_133205)

            if (may_be_133170 and more_types_in_union_133171):
                # SSA join for if statement (line 677)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Attribute to a Attribute (line 684):
        
        # Assigning a Attribute to a Attribute (line 684):
        # Getting the type of 'self' (line 684)
        self_133207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 684, 32), 'self')
        # Obtaining the member '_strict_call' of a type (line 684)
        _strict_call_133208 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 684, 32), self_133207, '_strict_call')
        # Getting the type of 'self' (line 684)
        self_133209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 684, 8), 'self')
        # Setting the type of the member '_callingfunction' of a type (line 684)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 684, 8), self_133209, '_callingfunction', _strict_call_133208)
        
        # Assigning a Call to a Attribute (line 685):
        
        # Assigning a Call to a Attribute (line 685):
        
        # Call to _dtypeortype(...): (line 685)
        # Processing the call arguments (line 685)
        # Getting the type of 'dtype' (line 685)
        dtype_133212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 685, 38), 'dtype', False)
        # Processing the call keyword arguments (line 685)
        kwargs_133213 = {}
        # Getting the type of 'self' (line 685)
        self_133210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 685, 20), 'self', False)
        # Obtaining the member '_dtypeortype' of a type (line 685)
        _dtypeortype_133211 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 685, 20), self_133210, '_dtypeortype')
        # Calling _dtypeortype(args, kwargs) (line 685)
        _dtypeortype_call_result_133214 = invoke(stypy.reporting.localization.Localization(__file__, 685, 20), _dtypeortype_133211, *[dtype_133212], **kwargs_133213)
        
        # Getting the type of 'self' (line 685)
        self_133215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 685, 8), 'self')
        # Setting the type of the member 'type' of a type (line 685)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 685, 8), self_133215, 'type', _dtypeortype_call_result_133214)
        
        # Assigning a Name to a Attribute (line 686):
        
        # Assigning a Name to a Attribute (line 686):
        # Getting the type of 'False' (line 686)
        False_133216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 686, 24), 'False')
        # Getting the type of 'self' (line 686)
        self_133217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 686, 8), 'self')
        # Setting the type of the member '_checked' of a type (line 686)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 686, 8), self_133217, '_checked', False_133216)
        
        # Assigning a Name to a Attribute (line 687):
        
        # Assigning a Name to a Attribute (line 687):
        # Getting the type of 'default' (line 687)
        default_133218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 687, 32), 'default')
        # Getting the type of 'self' (line 687)
        self_133219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 687, 8), 'self')
        # Setting the type of the member '_initial_default' of a type (line 687)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 687, 8), self_133219, '_initial_default', default_133218)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def _loose_call(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_loose_call'
        module_type_store = module_type_store.open_function_context('_loose_call', 690, 4, False)
        # Assigning a type to the variable 'self' (line 691)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 691, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        StringConverter._loose_call.__dict__.__setitem__('stypy_localization', localization)
        StringConverter._loose_call.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        StringConverter._loose_call.__dict__.__setitem__('stypy_type_store', module_type_store)
        StringConverter._loose_call.__dict__.__setitem__('stypy_function_name', 'StringConverter._loose_call')
        StringConverter._loose_call.__dict__.__setitem__('stypy_param_names_list', ['value'])
        StringConverter._loose_call.__dict__.__setitem__('stypy_varargs_param_name', None)
        StringConverter._loose_call.__dict__.__setitem__('stypy_kwargs_param_name', None)
        StringConverter._loose_call.__dict__.__setitem__('stypy_call_defaults', defaults)
        StringConverter._loose_call.__dict__.__setitem__('stypy_call_varargs', varargs)
        StringConverter._loose_call.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        StringConverter._loose_call.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'StringConverter._loose_call', ['value'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_loose_call', localization, ['value'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_loose_call(...)' code ##################

        
        
        # SSA begins for try-except statement (line 691)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to func(...): (line 692)
        # Processing the call arguments (line 692)
        # Getting the type of 'value' (line 692)
        value_133222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 29), 'value', False)
        # Processing the call keyword arguments (line 692)
        kwargs_133223 = {}
        # Getting the type of 'self' (line 692)
        self_133220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 19), 'self', False)
        # Obtaining the member 'func' of a type (line 692)
        func_133221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 692, 19), self_133220, 'func')
        # Calling func(args, kwargs) (line 692)
        func_call_result_133224 = invoke(stypy.reporting.localization.Localization(__file__, 692, 19), func_133221, *[value_133222], **kwargs_133223)
        
        # Assigning a type to the variable 'stypy_return_type' (line 692)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 692, 12), 'stypy_return_type', func_call_result_133224)
        # SSA branch for the except part of a try statement (line 691)
        # SSA branch for the except 'ValueError' branch of a try statement (line 691)
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'self' (line 694)
        self_133225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 19), 'self')
        # Obtaining the member 'default' of a type (line 694)
        default_133226 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 694, 19), self_133225, 'default')
        # Assigning a type to the variable 'stypy_return_type' (line 694)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 694, 12), 'stypy_return_type', default_133226)
        # SSA join for try-except statement (line 691)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_loose_call(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_loose_call' in the type store
        # Getting the type of 'stypy_return_type' (line 690)
        stypy_return_type_133227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_133227)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_loose_call'
        return stypy_return_type_133227


    @norecursion
    def _strict_call(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_strict_call'
        module_type_store = module_type_store.open_function_context('_strict_call', 697, 4, False)
        # Assigning a type to the variable 'self' (line 698)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 698, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        StringConverter._strict_call.__dict__.__setitem__('stypy_localization', localization)
        StringConverter._strict_call.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        StringConverter._strict_call.__dict__.__setitem__('stypy_type_store', module_type_store)
        StringConverter._strict_call.__dict__.__setitem__('stypy_function_name', 'StringConverter._strict_call')
        StringConverter._strict_call.__dict__.__setitem__('stypy_param_names_list', ['value'])
        StringConverter._strict_call.__dict__.__setitem__('stypy_varargs_param_name', None)
        StringConverter._strict_call.__dict__.__setitem__('stypy_kwargs_param_name', None)
        StringConverter._strict_call.__dict__.__setitem__('stypy_call_defaults', defaults)
        StringConverter._strict_call.__dict__.__setitem__('stypy_call_varargs', varargs)
        StringConverter._strict_call.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        StringConverter._strict_call.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'StringConverter._strict_call', ['value'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_strict_call', localization, ['value'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_strict_call(...)' code ##################

        
        
        # SSA begins for try-except statement (line 698)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Name (line 701):
        
        # Assigning a Call to a Name (line 701):
        
        # Call to func(...): (line 701)
        # Processing the call arguments (line 701)
        # Getting the type of 'value' (line 701)
        value_133230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 701, 34), 'value', False)
        # Processing the call keyword arguments (line 701)
        kwargs_133231 = {}
        # Getting the type of 'self' (line 701)
        self_133228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 701, 24), 'self', False)
        # Obtaining the member 'func' of a type (line 701)
        func_133229 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 701, 24), self_133228, 'func')
        # Calling func(args, kwargs) (line 701)
        func_call_result_133232 = invoke(stypy.reporting.localization.Localization(__file__, 701, 24), func_133229, *[value_133230], **kwargs_133231)
        
        # Assigning a type to the variable 'new_value' (line 701)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 701, 12), 'new_value', func_call_result_133232)
        
        
        # Getting the type of 'self' (line 706)
        self_133233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 706, 15), 'self')
        # Obtaining the member 'func' of a type (line 706)
        func_133234 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 706, 15), self_133233, 'func')
        # Getting the type of 'int' (line 706)
        int_133235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 706, 28), 'int')
        # Applying the binary operator 'is' (line 706)
        result_is__133236 = python_operator(stypy.reporting.localization.Localization(__file__, 706, 15), 'is', func_133234, int_133235)
        
        # Testing the type of an if condition (line 706)
        if_condition_133237 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 706, 12), result_is__133236)
        # Assigning a type to the variable 'if_condition_133237' (line 706)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 706, 12), 'if_condition_133237', if_condition_133237)
        # SSA begins for if statement (line 706)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # SSA begins for try-except statement (line 707)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to array(...): (line 708)
        # Processing the call arguments (line 708)
        # Getting the type of 'value' (line 708)
        value_133240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 708, 29), 'value', False)
        # Processing the call keyword arguments (line 708)
        # Getting the type of 'self' (line 708)
        self_133241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 708, 42), 'self', False)
        # Obtaining the member 'type' of a type (line 708)
        type_133242 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 708, 42), self_133241, 'type')
        keyword_133243 = type_133242
        kwargs_133244 = {'dtype': keyword_133243}
        # Getting the type of 'np' (line 708)
        np_133238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 708, 20), 'np', False)
        # Obtaining the member 'array' of a type (line 708)
        array_133239 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 708, 20), np_133238, 'array')
        # Calling array(args, kwargs) (line 708)
        array_call_result_133245 = invoke(stypy.reporting.localization.Localization(__file__, 708, 20), array_133239, *[value_133240], **kwargs_133244)
        
        # SSA branch for the except part of a try statement (line 707)
        # SSA branch for the except 'OverflowError' branch of a try statement (line 707)
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'ValueError' (line 710)
        ValueError_133246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 710, 26), 'ValueError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 710, 20), ValueError_133246, 'raise parameter', BaseException)
        # SSA join for try-except statement (line 707)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 706)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'new_value' (line 713)
        new_value_133247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 713, 19), 'new_value')
        # Assigning a type to the variable 'stypy_return_type' (line 713)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 713, 12), 'stypy_return_type', new_value_133247)
        # SSA branch for the except part of a try statement (line 698)
        # SSA branch for the except 'ValueError' branch of a try statement (line 698)
        module_type_store.open_ssa_branch('except')
        
        
        
        # Call to strip(...): (line 716)
        # Processing the call keyword arguments (line 716)
        kwargs_133250 = {}
        # Getting the type of 'value' (line 716)
        value_133248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 716, 15), 'value', False)
        # Obtaining the member 'strip' of a type (line 716)
        strip_133249 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 716, 15), value_133248, 'strip')
        # Calling strip(args, kwargs) (line 716)
        strip_call_result_133251 = invoke(stypy.reporting.localization.Localization(__file__, 716, 15), strip_133249, *[], **kwargs_133250)
        
        # Getting the type of 'self' (line 716)
        self_133252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 716, 32), 'self')
        # Obtaining the member 'missing_values' of a type (line 716)
        missing_values_133253 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 716, 32), self_133252, 'missing_values')
        # Applying the binary operator 'in' (line 716)
        result_contains_133254 = python_operator(stypy.reporting.localization.Localization(__file__, 716, 15), 'in', strip_call_result_133251, missing_values_133253)
        
        # Testing the type of an if condition (line 716)
        if_condition_133255 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 716, 12), result_contains_133254)
        # Assigning a type to the variable 'if_condition_133255' (line 716)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 716, 12), 'if_condition_133255', if_condition_133255)
        # SSA begins for if statement (line 716)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Getting the type of 'self' (line 717)
        self_133256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 717, 23), 'self')
        # Obtaining the member '_status' of a type (line 717)
        _status_133257 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 717, 23), self_133256, '_status')
        # Applying the 'not' unary operator (line 717)
        result_not__133258 = python_operator(stypy.reporting.localization.Localization(__file__, 717, 19), 'not', _status_133257)
        
        # Testing the type of an if condition (line 717)
        if_condition_133259 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 717, 16), result_not__133258)
        # Assigning a type to the variable 'if_condition_133259' (line 717)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 717, 16), 'if_condition_133259', if_condition_133259)
        # SSA begins for if statement (line 717)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Attribute (line 718):
        
        # Assigning a Name to a Attribute (line 718):
        # Getting the type of 'False' (line 718)
        False_133260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 36), 'False')
        # Getting the type of 'self' (line 718)
        self_133261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 20), 'self')
        # Setting the type of the member '_checked' of a type (line 718)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 718, 20), self_133261, '_checked', False_133260)
        # SSA join for if statement (line 717)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'self' (line 719)
        self_133262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 719, 23), 'self')
        # Obtaining the member 'default' of a type (line 719)
        default_133263 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 719, 23), self_133262, 'default')
        # Assigning a type to the variable 'stypy_return_type' (line 719)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 719, 16), 'stypy_return_type', default_133263)
        # SSA join for if statement (line 716)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to ValueError(...): (line 720)
        # Processing the call arguments (line 720)
        str_133265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 720, 29), 'str', "Cannot convert string '%s'")
        # Getting the type of 'value' (line 720)
        value_133266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 720, 60), 'value', False)
        # Applying the binary operator '%' (line 720)
        result_mod_133267 = python_operator(stypy.reporting.localization.Localization(__file__, 720, 29), '%', str_133265, value_133266)
        
        # Processing the call keyword arguments (line 720)
        kwargs_133268 = {}
        # Getting the type of 'ValueError' (line 720)
        ValueError_133264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 720, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 720)
        ValueError_call_result_133269 = invoke(stypy.reporting.localization.Localization(__file__, 720, 18), ValueError_133264, *[result_mod_133267], **kwargs_133268)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 720, 12), ValueError_call_result_133269, 'raise parameter', BaseException)
        # SSA join for try-except statement (line 698)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_strict_call(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_strict_call' in the type store
        # Getting the type of 'stypy_return_type' (line 697)
        stypy_return_type_133270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_133270)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_strict_call'
        return stypy_return_type_133270


    @norecursion
    def __call__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__call__'
        module_type_store = module_type_store.open_function_context('__call__', 723, 4, False)
        # Assigning a type to the variable 'self' (line 724)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 724, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        StringConverter.__call__.__dict__.__setitem__('stypy_localization', localization)
        StringConverter.__call__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        StringConverter.__call__.__dict__.__setitem__('stypy_type_store', module_type_store)
        StringConverter.__call__.__dict__.__setitem__('stypy_function_name', 'StringConverter.__call__')
        StringConverter.__call__.__dict__.__setitem__('stypy_param_names_list', ['value'])
        StringConverter.__call__.__dict__.__setitem__('stypy_varargs_param_name', None)
        StringConverter.__call__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        StringConverter.__call__.__dict__.__setitem__('stypy_call_defaults', defaults)
        StringConverter.__call__.__dict__.__setitem__('stypy_call_varargs', varargs)
        StringConverter.__call__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        StringConverter.__call__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'StringConverter.__call__', ['value'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__call__', localization, ['value'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__call__(...)' code ##################

        
        # Call to _callingfunction(...): (line 724)
        # Processing the call arguments (line 724)
        # Getting the type of 'value' (line 724)
        value_133273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 724, 37), 'value', False)
        # Processing the call keyword arguments (line 724)
        kwargs_133274 = {}
        # Getting the type of 'self' (line 724)
        self_133271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 724, 15), 'self', False)
        # Obtaining the member '_callingfunction' of a type (line 724)
        _callingfunction_133272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 724, 15), self_133271, '_callingfunction')
        # Calling _callingfunction(args, kwargs) (line 724)
        _callingfunction_call_result_133275 = invoke(stypy.reporting.localization.Localization(__file__, 724, 15), _callingfunction_133272, *[value_133273], **kwargs_133274)
        
        # Assigning a type to the variable 'stypy_return_type' (line 724)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 724, 8), 'stypy_return_type', _callingfunction_call_result_133275)
        
        # ################# End of '__call__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__call__' in the type store
        # Getting the type of 'stypy_return_type' (line 723)
        stypy_return_type_133276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 723, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_133276)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__call__'
        return stypy_return_type_133276


    @norecursion
    def upgrade(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'upgrade'
        module_type_store = module_type_store.open_function_context('upgrade', 727, 4, False)
        # Assigning a type to the variable 'self' (line 728)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 728, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        StringConverter.upgrade.__dict__.__setitem__('stypy_localization', localization)
        StringConverter.upgrade.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        StringConverter.upgrade.__dict__.__setitem__('stypy_type_store', module_type_store)
        StringConverter.upgrade.__dict__.__setitem__('stypy_function_name', 'StringConverter.upgrade')
        StringConverter.upgrade.__dict__.__setitem__('stypy_param_names_list', ['value'])
        StringConverter.upgrade.__dict__.__setitem__('stypy_varargs_param_name', None)
        StringConverter.upgrade.__dict__.__setitem__('stypy_kwargs_param_name', None)
        StringConverter.upgrade.__dict__.__setitem__('stypy_call_defaults', defaults)
        StringConverter.upgrade.__dict__.__setitem__('stypy_call_varargs', varargs)
        StringConverter.upgrade.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        StringConverter.upgrade.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'StringConverter.upgrade', ['value'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'upgrade', localization, ['value'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'upgrade(...)' code ##################

        str_133277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 747, (-1)), 'str', '\n        Find the best converter for a given string, and return the result.\n\n        The supplied string `value` is converted by testing different\n        converters in order. First the `func` method of the\n        `StringConverter` instance is tried, if this fails other available\n        converters are tried.  The order in which these other converters\n        are tried is determined by the `_status` attribute of the instance.\n\n        Parameters\n        ----------\n        value : str\n            The string to convert.\n\n        Returns\n        -------\n        out : any\n            The result of converting `value` with the appropriate converter.\n\n        ')
        
        # Assigning a Name to a Attribute (line 748):
        
        # Assigning a Name to a Attribute (line 748):
        # Getting the type of 'True' (line 748)
        True_133278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 748, 24), 'True')
        # Getting the type of 'self' (line 748)
        self_133279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 748, 8), 'self')
        # Setting the type of the member '_checked' of a type (line 748)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 748, 8), self_133279, '_checked', True_133278)
        
        
        # SSA begins for try-except statement (line 749)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to _strict_call(...): (line 750)
        # Processing the call arguments (line 750)
        # Getting the type of 'value' (line 750)
        value_133282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 750, 37), 'value', False)
        # Processing the call keyword arguments (line 750)
        kwargs_133283 = {}
        # Getting the type of 'self' (line 750)
        self_133280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 750, 19), 'self', False)
        # Obtaining the member '_strict_call' of a type (line 750)
        _strict_call_133281 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 750, 19), self_133280, '_strict_call')
        # Calling _strict_call(args, kwargs) (line 750)
        _strict_call_call_result_133284 = invoke(stypy.reporting.localization.Localization(__file__, 750, 19), _strict_call_133281, *[value_133282], **kwargs_133283)
        
        # Assigning a type to the variable 'stypy_return_type' (line 750)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 750, 12), 'stypy_return_type', _strict_call_call_result_133284)
        # SSA branch for the except part of a try statement (line 749)
        # SSA branch for the except 'ValueError' branch of a try statement (line 749)
        module_type_store.open_ssa_branch('except')
        
        # Getting the type of 'self' (line 753)
        self_133285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 753, 15), 'self')
        # Obtaining the member '_locked' of a type (line 753)
        _locked_133286 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 753, 15), self_133285, '_locked')
        # Testing the type of an if condition (line 753)
        if_condition_133287 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 753, 12), _locked_133286)
        # Assigning a type to the variable 'if_condition_133287' (line 753)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 753, 12), 'if_condition_133287', if_condition_133287)
        # SSA begins for if statement (line 753)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Str to a Name (line 754):
        
        # Assigning a Str to a Name (line 754):
        str_133288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 754, 25), 'str', 'Converter is locked and cannot be upgraded')
        # Assigning a type to the variable 'errmsg' (line 754)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 754, 16), 'errmsg', str_133288)
        
        # Call to ConverterLockError(...): (line 755)
        # Processing the call arguments (line 755)
        # Getting the type of 'errmsg' (line 755)
        errmsg_133290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 755, 41), 'errmsg', False)
        # Processing the call keyword arguments (line 755)
        kwargs_133291 = {}
        # Getting the type of 'ConverterLockError' (line 755)
        ConverterLockError_133289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 755, 22), 'ConverterLockError', False)
        # Calling ConverterLockError(args, kwargs) (line 755)
        ConverterLockError_call_result_133292 = invoke(stypy.reporting.localization.Localization(__file__, 755, 22), ConverterLockError_133289, *[errmsg_133290], **kwargs_133291)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 755, 16), ConverterLockError_call_result_133292, 'raise parameter', BaseException)
        # SSA join for if statement (line 753)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 756):
        
        # Assigning a Call to a Name (line 756):
        
        # Call to len(...): (line 756)
        # Processing the call arguments (line 756)
        # Getting the type of 'self' (line 756)
        self_133294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 756, 29), 'self', False)
        # Obtaining the member '_mapper' of a type (line 756)
        _mapper_133295 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 756, 29), self_133294, '_mapper')
        # Processing the call keyword arguments (line 756)
        kwargs_133296 = {}
        # Getting the type of 'len' (line 756)
        len_133293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 756, 25), 'len', False)
        # Calling len(args, kwargs) (line 756)
        len_call_result_133297 = invoke(stypy.reporting.localization.Localization(__file__, 756, 25), len_133293, *[_mapper_133295], **kwargs_133296)
        
        # Assigning a type to the variable '_statusmax' (line 756)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 756, 12), '_statusmax', len_call_result_133297)
        
        # Assigning a Attribute to a Name (line 758):
        
        # Assigning a Attribute to a Name (line 758):
        # Getting the type of 'self' (line 758)
        self_133298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 758, 22), 'self')
        # Obtaining the member '_status' of a type (line 758)
        _status_133299 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 758, 22), self_133298, '_status')
        # Assigning a type to the variable '_status' (line 758)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 758, 12), '_status', _status_133299)
        
        
        # Getting the type of '_status' (line 759)
        _status_133300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 759, 15), '_status')
        # Getting the type of '_statusmax' (line 759)
        _statusmax_133301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 759, 26), '_statusmax')
        # Applying the binary operator '==' (line 759)
        result_eq_133302 = python_operator(stypy.reporting.localization.Localization(__file__, 759, 15), '==', _status_133300, _statusmax_133301)
        
        # Testing the type of an if condition (line 759)
        if_condition_133303 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 759, 12), result_eq_133302)
        # Assigning a type to the variable 'if_condition_133303' (line 759)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 759, 12), 'if_condition_133303', if_condition_133303)
        # SSA begins for if statement (line 759)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Str to a Name (line 760):
        
        # Assigning a Str to a Name (line 760):
        str_133304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 760, 25), 'str', 'Could not find a valid conversion function')
        # Assigning a type to the variable 'errmsg' (line 760)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 760, 16), 'errmsg', str_133304)
        
        # Call to ConverterError(...): (line 761)
        # Processing the call arguments (line 761)
        # Getting the type of 'errmsg' (line 761)
        errmsg_133306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 761, 37), 'errmsg', False)
        # Processing the call keyword arguments (line 761)
        kwargs_133307 = {}
        # Getting the type of 'ConverterError' (line 761)
        ConverterError_133305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 761, 22), 'ConverterError', False)
        # Calling ConverterError(args, kwargs) (line 761)
        ConverterError_call_result_133308 = invoke(stypy.reporting.localization.Localization(__file__, 761, 22), ConverterError_133305, *[errmsg_133306], **kwargs_133307)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 761, 16), ConverterError_call_result_133308, 'raise parameter', BaseException)
        # SSA branch for the else part of an if statement (line 759)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of '_status' (line 762)
        _status_133309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 762, 17), '_status')
        # Getting the type of '_statusmax' (line 762)
        _statusmax_133310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 762, 27), '_statusmax')
        int_133311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 762, 40), 'int')
        # Applying the binary operator '-' (line 762)
        result_sub_133312 = python_operator(stypy.reporting.localization.Localization(__file__, 762, 27), '-', _statusmax_133310, int_133311)
        
        # Applying the binary operator '<' (line 762)
        result_lt_133313 = python_operator(stypy.reporting.localization.Localization(__file__, 762, 17), '<', _status_133309, result_sub_133312)
        
        # Testing the type of an if condition (line 762)
        if_condition_133314 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 762, 17), result_lt_133313)
        # Assigning a type to the variable 'if_condition_133314' (line 762)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 762, 17), 'if_condition_133314', if_condition_133314)
        # SSA begins for if statement (line 762)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of '_status' (line 763)
        _status_133315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 763, 16), '_status')
        int_133316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 763, 27), 'int')
        # Applying the binary operator '+=' (line 763)
        result_iadd_133317 = python_operator(stypy.reporting.localization.Localization(__file__, 763, 16), '+=', _status_133315, int_133316)
        # Assigning a type to the variable '_status' (line 763)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 763, 16), '_status', result_iadd_133317)
        
        # SSA join for if statement (line 762)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 759)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Subscript to a Tuple (line 764):
        
        # Assigning a Subscript to a Name (line 764):
        
        # Obtaining the type of the subscript
        int_133318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 764, 12), 'int')
        
        # Obtaining the type of the subscript
        # Getting the type of '_status' (line 764)
        _status_133319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 764, 59), '_status')
        # Getting the type of 'self' (line 764)
        self_133320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 764, 46), 'self')
        # Obtaining the member '_mapper' of a type (line 764)
        _mapper_133321 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 764, 46), self_133320, '_mapper')
        # Obtaining the member '__getitem__' of a type (line 764)
        getitem___133322 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 764, 46), _mapper_133321, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 764)
        subscript_call_result_133323 = invoke(stypy.reporting.localization.Localization(__file__, 764, 46), getitem___133322, _status_133319)
        
        # Obtaining the member '__getitem__' of a type (line 764)
        getitem___133324 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 764, 12), subscript_call_result_133323, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 764)
        subscript_call_result_133325 = invoke(stypy.reporting.localization.Localization(__file__, 764, 12), getitem___133324, int_133318)
        
        # Assigning a type to the variable 'tuple_var_assignment_132138' (line 764)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 764, 12), 'tuple_var_assignment_132138', subscript_call_result_133325)
        
        # Assigning a Subscript to a Name (line 764):
        
        # Obtaining the type of the subscript
        int_133326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 764, 12), 'int')
        
        # Obtaining the type of the subscript
        # Getting the type of '_status' (line 764)
        _status_133327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 764, 59), '_status')
        # Getting the type of 'self' (line 764)
        self_133328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 764, 46), 'self')
        # Obtaining the member '_mapper' of a type (line 764)
        _mapper_133329 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 764, 46), self_133328, '_mapper')
        # Obtaining the member '__getitem__' of a type (line 764)
        getitem___133330 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 764, 46), _mapper_133329, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 764)
        subscript_call_result_133331 = invoke(stypy.reporting.localization.Localization(__file__, 764, 46), getitem___133330, _status_133327)
        
        # Obtaining the member '__getitem__' of a type (line 764)
        getitem___133332 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 764, 12), subscript_call_result_133331, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 764)
        subscript_call_result_133333 = invoke(stypy.reporting.localization.Localization(__file__, 764, 12), getitem___133332, int_133326)
        
        # Assigning a type to the variable 'tuple_var_assignment_132139' (line 764)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 764, 12), 'tuple_var_assignment_132139', subscript_call_result_133333)
        
        # Assigning a Subscript to a Name (line 764):
        
        # Obtaining the type of the subscript
        int_133334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 764, 12), 'int')
        
        # Obtaining the type of the subscript
        # Getting the type of '_status' (line 764)
        _status_133335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 764, 59), '_status')
        # Getting the type of 'self' (line 764)
        self_133336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 764, 46), 'self')
        # Obtaining the member '_mapper' of a type (line 764)
        _mapper_133337 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 764, 46), self_133336, '_mapper')
        # Obtaining the member '__getitem__' of a type (line 764)
        getitem___133338 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 764, 46), _mapper_133337, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 764)
        subscript_call_result_133339 = invoke(stypy.reporting.localization.Localization(__file__, 764, 46), getitem___133338, _status_133335)
        
        # Obtaining the member '__getitem__' of a type (line 764)
        getitem___133340 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 764, 12), subscript_call_result_133339, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 764)
        subscript_call_result_133341 = invoke(stypy.reporting.localization.Localization(__file__, 764, 12), getitem___133340, int_133334)
        
        # Assigning a type to the variable 'tuple_var_assignment_132140' (line 764)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 764, 12), 'tuple_var_assignment_132140', subscript_call_result_133341)
        
        # Assigning a Name to a Attribute (line 764):
        # Getting the type of 'tuple_var_assignment_132138' (line 764)
        tuple_var_assignment_132138_133342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 764, 12), 'tuple_var_assignment_132138')
        # Getting the type of 'self' (line 764)
        self_133343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 764, 13), 'self')
        # Setting the type of the member 'type' of a type (line 764)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 764, 13), self_133343, 'type', tuple_var_assignment_132138_133342)
        
        # Assigning a Name to a Attribute (line 764):
        # Getting the type of 'tuple_var_assignment_132139' (line 764)
        tuple_var_assignment_132139_133344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 764, 12), 'tuple_var_assignment_132139')
        # Getting the type of 'self' (line 764)
        self_133345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 764, 24), 'self')
        # Setting the type of the member 'func' of a type (line 764)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 764, 24), self_133345, 'func', tuple_var_assignment_132139_133344)
        
        # Assigning a Name to a Name (line 764):
        # Getting the type of 'tuple_var_assignment_132140' (line 764)
        tuple_var_assignment_132140_133346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 764, 12), 'tuple_var_assignment_132140')
        # Assigning a type to the variable 'default' (line 764)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 764, 35), 'default', tuple_var_assignment_132140_133346)
        
        # Assigning a Name to a Attribute (line 765):
        
        # Assigning a Name to a Attribute (line 765):
        # Getting the type of '_status' (line 765)
        _status_133347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 765, 27), '_status')
        # Getting the type of 'self' (line 765)
        self_133348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 765, 12), 'self')
        # Setting the type of the member '_status' of a type (line 765)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 765, 12), self_133348, '_status', _status_133347)
        
        
        # Getting the type of 'self' (line 766)
        self_133349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 766, 15), 'self')
        # Obtaining the member '_initial_default' of a type (line 766)
        _initial_default_133350 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 766, 15), self_133349, '_initial_default')
        # Getting the type of 'None' (line 766)
        None_133351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 766, 44), 'None')
        # Applying the binary operator 'isnot' (line 766)
        result_is_not_133352 = python_operator(stypy.reporting.localization.Localization(__file__, 766, 15), 'isnot', _initial_default_133350, None_133351)
        
        # Testing the type of an if condition (line 766)
        if_condition_133353 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 766, 12), result_is_not_133352)
        # Assigning a type to the variable 'if_condition_133353' (line 766)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 766, 12), 'if_condition_133353', if_condition_133353)
        # SSA begins for if statement (line 766)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Attribute (line 767):
        
        # Assigning a Attribute to a Attribute (line 767):
        # Getting the type of 'self' (line 767)
        self_133354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 767, 31), 'self')
        # Obtaining the member '_initial_default' of a type (line 767)
        _initial_default_133355 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 767, 31), self_133354, '_initial_default')
        # Getting the type of 'self' (line 767)
        self_133356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 767, 16), 'self')
        # Setting the type of the member 'default' of a type (line 767)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 767, 16), self_133356, 'default', _initial_default_133355)
        # SSA branch for the else part of an if statement (line 766)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Name to a Attribute (line 769):
        
        # Assigning a Name to a Attribute (line 769):
        # Getting the type of 'default' (line 769)
        default_133357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 769, 31), 'default')
        # Getting the type of 'self' (line 769)
        self_133358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 769, 16), 'self')
        # Setting the type of the member 'default' of a type (line 769)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 769, 16), self_133358, 'default', default_133357)
        # SSA join for if statement (line 766)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to upgrade(...): (line 770)
        # Processing the call arguments (line 770)
        # Getting the type of 'value' (line 770)
        value_133361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 770, 32), 'value', False)
        # Processing the call keyword arguments (line 770)
        kwargs_133362 = {}
        # Getting the type of 'self' (line 770)
        self_133359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 770, 19), 'self', False)
        # Obtaining the member 'upgrade' of a type (line 770)
        upgrade_133360 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 770, 19), self_133359, 'upgrade')
        # Calling upgrade(args, kwargs) (line 770)
        upgrade_call_result_133363 = invoke(stypy.reporting.localization.Localization(__file__, 770, 19), upgrade_133360, *[value_133361], **kwargs_133362)
        
        # Assigning a type to the variable 'stypy_return_type' (line 770)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 770, 12), 'stypy_return_type', upgrade_call_result_133363)
        # SSA join for try-except statement (line 749)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'upgrade(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'upgrade' in the type store
        # Getting the type of 'stypy_return_type' (line 727)
        stypy_return_type_133364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 727, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_133364)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'upgrade'
        return stypy_return_type_133364


    @norecursion
    def iterupgrade(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'iterupgrade'
        module_type_store = module_type_store.open_function_context('iterupgrade', 772, 4, False)
        # Assigning a type to the variable 'self' (line 773)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 773, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        StringConverter.iterupgrade.__dict__.__setitem__('stypy_localization', localization)
        StringConverter.iterupgrade.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        StringConverter.iterupgrade.__dict__.__setitem__('stypy_type_store', module_type_store)
        StringConverter.iterupgrade.__dict__.__setitem__('stypy_function_name', 'StringConverter.iterupgrade')
        StringConverter.iterupgrade.__dict__.__setitem__('stypy_param_names_list', ['value'])
        StringConverter.iterupgrade.__dict__.__setitem__('stypy_varargs_param_name', None)
        StringConverter.iterupgrade.__dict__.__setitem__('stypy_kwargs_param_name', None)
        StringConverter.iterupgrade.__dict__.__setitem__('stypy_call_defaults', defaults)
        StringConverter.iterupgrade.__dict__.__setitem__('stypy_call_varargs', varargs)
        StringConverter.iterupgrade.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        StringConverter.iterupgrade.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'StringConverter.iterupgrade', ['value'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'iterupgrade', localization, ['value'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'iterupgrade(...)' code ##################

        
        # Assigning a Name to a Attribute (line 773):
        
        # Assigning a Name to a Attribute (line 773):
        # Getting the type of 'True' (line 773)
        True_133365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 773, 24), 'True')
        # Getting the type of 'self' (line 773)
        self_133366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 773, 8), 'self')
        # Setting the type of the member '_checked' of a type (line 773)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 773, 8), self_133366, '_checked', True_133365)
        
        # Type idiom detected: calculating its left and rigth part (line 774)
        str_133367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 774, 30), 'str', '__iter__')
        # Getting the type of 'value' (line 774)
        value_133368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 774, 23), 'value')
        
        (may_be_133369, more_types_in_union_133370) = may_not_provide_member(str_133367, value_133368)

        if may_be_133369:

            if more_types_in_union_133370:
                # Runtime conditional SSA (line 774)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'value' (line 774)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 774, 8), 'value', remove_member_provider_from_union(value_133368, '__iter__'))
            
            # Assigning a Tuple to a Name (line 775):
            
            # Assigning a Tuple to a Name (line 775):
            
            # Obtaining an instance of the builtin type 'tuple' (line 775)
            tuple_133371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 775, 21), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 775)
            # Adding element type (line 775)
            # Getting the type of 'value' (line 775)
            value_133372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 775, 21), 'value')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 775, 21), tuple_133371, value_133372)
            
            # Assigning a type to the variable 'value' (line 775)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 775, 12), 'value', tuple_133371)

            if more_types_in_union_133370:
                # SSA join for if statement (line 774)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Attribute to a Name (line 776):
        
        # Assigning a Attribute to a Name (line 776):
        # Getting the type of 'self' (line 776)
        self_133373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 776, 23), 'self')
        # Obtaining the member '_strict_call' of a type (line 776)
        _strict_call_133374 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 776, 23), self_133373, '_strict_call')
        # Assigning a type to the variable '_strict_call' (line 776)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 776, 8), '_strict_call', _strict_call_133374)
        
        
        # SSA begins for try-except statement (line 777)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Getting the type of 'value' (line 778)
        value_133375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 778, 22), 'value')
        # Testing the type of a for loop iterable (line 778)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 778, 12), value_133375)
        # Getting the type of the for loop variable (line 778)
        for_loop_var_133376 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 778, 12), value_133375)
        # Assigning a type to the variable '_m' (line 778)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 778, 12), '_m', for_loop_var_133376)
        # SSA begins for a for statement (line 778)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to _strict_call(...): (line 779)
        # Processing the call arguments (line 779)
        # Getting the type of '_m' (line 779)
        _m_133378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 779, 29), '_m', False)
        # Processing the call keyword arguments (line 779)
        kwargs_133379 = {}
        # Getting the type of '_strict_call' (line 779)
        _strict_call_133377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 779, 16), '_strict_call', False)
        # Calling _strict_call(args, kwargs) (line 779)
        _strict_call_call_result_133380 = invoke(stypy.reporting.localization.Localization(__file__, 779, 16), _strict_call_133377, *[_m_133378], **kwargs_133379)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the except part of a try statement (line 777)
        # SSA branch for the except 'ValueError' branch of a try statement (line 777)
        module_type_store.open_ssa_branch('except')
        
        # Getting the type of 'self' (line 782)
        self_133381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 782, 15), 'self')
        # Obtaining the member '_locked' of a type (line 782)
        _locked_133382 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 782, 15), self_133381, '_locked')
        # Testing the type of an if condition (line 782)
        if_condition_133383 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 782, 12), _locked_133382)
        # Assigning a type to the variable 'if_condition_133383' (line 782)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 782, 12), 'if_condition_133383', if_condition_133383)
        # SSA begins for if statement (line 782)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Str to a Name (line 783):
        
        # Assigning a Str to a Name (line 783):
        str_133384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 783, 25), 'str', 'Converter is locked and cannot be upgraded')
        # Assigning a type to the variable 'errmsg' (line 783)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 783, 16), 'errmsg', str_133384)
        
        # Call to ConverterLockError(...): (line 784)
        # Processing the call arguments (line 784)
        # Getting the type of 'errmsg' (line 784)
        errmsg_133386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 784, 41), 'errmsg', False)
        # Processing the call keyword arguments (line 784)
        kwargs_133387 = {}
        # Getting the type of 'ConverterLockError' (line 784)
        ConverterLockError_133385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 784, 22), 'ConverterLockError', False)
        # Calling ConverterLockError(args, kwargs) (line 784)
        ConverterLockError_call_result_133388 = invoke(stypy.reporting.localization.Localization(__file__, 784, 22), ConverterLockError_133385, *[errmsg_133386], **kwargs_133387)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 784, 16), ConverterLockError_call_result_133388, 'raise parameter', BaseException)
        # SSA join for if statement (line 782)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 785):
        
        # Assigning a Call to a Name (line 785):
        
        # Call to len(...): (line 785)
        # Processing the call arguments (line 785)
        # Getting the type of 'self' (line 785)
        self_133390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 785, 29), 'self', False)
        # Obtaining the member '_mapper' of a type (line 785)
        _mapper_133391 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 785, 29), self_133390, '_mapper')
        # Processing the call keyword arguments (line 785)
        kwargs_133392 = {}
        # Getting the type of 'len' (line 785)
        len_133389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 785, 25), 'len', False)
        # Calling len(args, kwargs) (line 785)
        len_call_result_133393 = invoke(stypy.reporting.localization.Localization(__file__, 785, 25), len_133389, *[_mapper_133391], **kwargs_133392)
        
        # Assigning a type to the variable '_statusmax' (line 785)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 785, 12), '_statusmax', len_call_result_133393)
        
        # Assigning a Attribute to a Name (line 787):
        
        # Assigning a Attribute to a Name (line 787):
        # Getting the type of 'self' (line 787)
        self_133394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 787, 22), 'self')
        # Obtaining the member '_status' of a type (line 787)
        _status_133395 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 787, 22), self_133394, '_status')
        # Assigning a type to the variable '_status' (line 787)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 787, 12), '_status', _status_133395)
        
        
        # Getting the type of '_status' (line 788)
        _status_133396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 788, 15), '_status')
        # Getting the type of '_statusmax' (line 788)
        _statusmax_133397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 788, 26), '_statusmax')
        # Applying the binary operator '==' (line 788)
        result_eq_133398 = python_operator(stypy.reporting.localization.Localization(__file__, 788, 15), '==', _status_133396, _statusmax_133397)
        
        # Testing the type of an if condition (line 788)
        if_condition_133399 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 788, 12), result_eq_133398)
        # Assigning a type to the variable 'if_condition_133399' (line 788)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 788, 12), 'if_condition_133399', if_condition_133399)
        # SSA begins for if statement (line 788)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ConverterError(...): (line 789)
        # Processing the call arguments (line 789)
        str_133401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 790, 20), 'str', 'Could not find a valid conversion function')
        # Processing the call keyword arguments (line 789)
        kwargs_133402 = {}
        # Getting the type of 'ConverterError' (line 789)
        ConverterError_133400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 789, 22), 'ConverterError', False)
        # Calling ConverterError(args, kwargs) (line 789)
        ConverterError_call_result_133403 = invoke(stypy.reporting.localization.Localization(__file__, 789, 22), ConverterError_133400, *[str_133401], **kwargs_133402)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 789, 16), ConverterError_call_result_133403, 'raise parameter', BaseException)
        # SSA branch for the else part of an if statement (line 788)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of '_status' (line 792)
        _status_133404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 792, 17), '_status')
        # Getting the type of '_statusmax' (line 792)
        _statusmax_133405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 792, 27), '_statusmax')
        int_133406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 792, 40), 'int')
        # Applying the binary operator '-' (line 792)
        result_sub_133407 = python_operator(stypy.reporting.localization.Localization(__file__, 792, 27), '-', _statusmax_133405, int_133406)
        
        # Applying the binary operator '<' (line 792)
        result_lt_133408 = python_operator(stypy.reporting.localization.Localization(__file__, 792, 17), '<', _status_133404, result_sub_133407)
        
        # Testing the type of an if condition (line 792)
        if_condition_133409 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 792, 17), result_lt_133408)
        # Assigning a type to the variable 'if_condition_133409' (line 792)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 792, 17), 'if_condition_133409', if_condition_133409)
        # SSA begins for if statement (line 792)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of '_status' (line 793)
        _status_133410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 793, 16), '_status')
        int_133411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 793, 27), 'int')
        # Applying the binary operator '+=' (line 793)
        result_iadd_133412 = python_operator(stypy.reporting.localization.Localization(__file__, 793, 16), '+=', _status_133410, int_133411)
        # Assigning a type to the variable '_status' (line 793)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 793, 16), '_status', result_iadd_133412)
        
        # SSA join for if statement (line 792)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 788)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Subscript to a Tuple (line 794):
        
        # Assigning a Subscript to a Name (line 794):
        
        # Obtaining the type of the subscript
        int_133413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 794, 12), 'int')
        
        # Obtaining the type of the subscript
        # Getting the type of '_status' (line 794)
        _status_133414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 794, 59), '_status')
        # Getting the type of 'self' (line 794)
        self_133415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 794, 46), 'self')
        # Obtaining the member '_mapper' of a type (line 794)
        _mapper_133416 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 794, 46), self_133415, '_mapper')
        # Obtaining the member '__getitem__' of a type (line 794)
        getitem___133417 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 794, 46), _mapper_133416, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 794)
        subscript_call_result_133418 = invoke(stypy.reporting.localization.Localization(__file__, 794, 46), getitem___133417, _status_133414)
        
        # Obtaining the member '__getitem__' of a type (line 794)
        getitem___133419 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 794, 12), subscript_call_result_133418, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 794)
        subscript_call_result_133420 = invoke(stypy.reporting.localization.Localization(__file__, 794, 12), getitem___133419, int_133413)
        
        # Assigning a type to the variable 'tuple_var_assignment_132141' (line 794)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 794, 12), 'tuple_var_assignment_132141', subscript_call_result_133420)
        
        # Assigning a Subscript to a Name (line 794):
        
        # Obtaining the type of the subscript
        int_133421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 794, 12), 'int')
        
        # Obtaining the type of the subscript
        # Getting the type of '_status' (line 794)
        _status_133422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 794, 59), '_status')
        # Getting the type of 'self' (line 794)
        self_133423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 794, 46), 'self')
        # Obtaining the member '_mapper' of a type (line 794)
        _mapper_133424 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 794, 46), self_133423, '_mapper')
        # Obtaining the member '__getitem__' of a type (line 794)
        getitem___133425 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 794, 46), _mapper_133424, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 794)
        subscript_call_result_133426 = invoke(stypy.reporting.localization.Localization(__file__, 794, 46), getitem___133425, _status_133422)
        
        # Obtaining the member '__getitem__' of a type (line 794)
        getitem___133427 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 794, 12), subscript_call_result_133426, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 794)
        subscript_call_result_133428 = invoke(stypy.reporting.localization.Localization(__file__, 794, 12), getitem___133427, int_133421)
        
        # Assigning a type to the variable 'tuple_var_assignment_132142' (line 794)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 794, 12), 'tuple_var_assignment_132142', subscript_call_result_133428)
        
        # Assigning a Subscript to a Name (line 794):
        
        # Obtaining the type of the subscript
        int_133429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 794, 12), 'int')
        
        # Obtaining the type of the subscript
        # Getting the type of '_status' (line 794)
        _status_133430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 794, 59), '_status')
        # Getting the type of 'self' (line 794)
        self_133431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 794, 46), 'self')
        # Obtaining the member '_mapper' of a type (line 794)
        _mapper_133432 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 794, 46), self_133431, '_mapper')
        # Obtaining the member '__getitem__' of a type (line 794)
        getitem___133433 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 794, 46), _mapper_133432, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 794)
        subscript_call_result_133434 = invoke(stypy.reporting.localization.Localization(__file__, 794, 46), getitem___133433, _status_133430)
        
        # Obtaining the member '__getitem__' of a type (line 794)
        getitem___133435 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 794, 12), subscript_call_result_133434, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 794)
        subscript_call_result_133436 = invoke(stypy.reporting.localization.Localization(__file__, 794, 12), getitem___133435, int_133429)
        
        # Assigning a type to the variable 'tuple_var_assignment_132143' (line 794)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 794, 12), 'tuple_var_assignment_132143', subscript_call_result_133436)
        
        # Assigning a Name to a Attribute (line 794):
        # Getting the type of 'tuple_var_assignment_132141' (line 794)
        tuple_var_assignment_132141_133437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 794, 12), 'tuple_var_assignment_132141')
        # Getting the type of 'self' (line 794)
        self_133438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 794, 13), 'self')
        # Setting the type of the member 'type' of a type (line 794)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 794, 13), self_133438, 'type', tuple_var_assignment_132141_133437)
        
        # Assigning a Name to a Attribute (line 794):
        # Getting the type of 'tuple_var_assignment_132142' (line 794)
        tuple_var_assignment_132142_133439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 794, 12), 'tuple_var_assignment_132142')
        # Getting the type of 'self' (line 794)
        self_133440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 794, 24), 'self')
        # Setting the type of the member 'func' of a type (line 794)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 794, 24), self_133440, 'func', tuple_var_assignment_132142_133439)
        
        # Assigning a Name to a Name (line 794):
        # Getting the type of 'tuple_var_assignment_132143' (line 794)
        tuple_var_assignment_132143_133441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 794, 12), 'tuple_var_assignment_132143')
        # Assigning a type to the variable 'default' (line 794)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 794, 35), 'default', tuple_var_assignment_132143_133441)
        
        
        # Getting the type of 'self' (line 795)
        self_133442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 795, 15), 'self')
        # Obtaining the member '_initial_default' of a type (line 795)
        _initial_default_133443 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 795, 15), self_133442, '_initial_default')
        # Getting the type of 'None' (line 795)
        None_133444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 795, 44), 'None')
        # Applying the binary operator 'isnot' (line 795)
        result_is_not_133445 = python_operator(stypy.reporting.localization.Localization(__file__, 795, 15), 'isnot', _initial_default_133443, None_133444)
        
        # Testing the type of an if condition (line 795)
        if_condition_133446 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 795, 12), result_is_not_133445)
        # Assigning a type to the variable 'if_condition_133446' (line 795)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 795, 12), 'if_condition_133446', if_condition_133446)
        # SSA begins for if statement (line 795)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Attribute (line 796):
        
        # Assigning a Attribute to a Attribute (line 796):
        # Getting the type of 'self' (line 796)
        self_133447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 796, 31), 'self')
        # Obtaining the member '_initial_default' of a type (line 796)
        _initial_default_133448 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 796, 31), self_133447, '_initial_default')
        # Getting the type of 'self' (line 796)
        self_133449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 796, 16), 'self')
        # Setting the type of the member 'default' of a type (line 796)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 796, 16), self_133449, 'default', _initial_default_133448)
        # SSA branch for the else part of an if statement (line 795)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Name to a Attribute (line 798):
        
        # Assigning a Name to a Attribute (line 798):
        # Getting the type of 'default' (line 798)
        default_133450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 798, 31), 'default')
        # Getting the type of 'self' (line 798)
        self_133451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 798, 16), 'self')
        # Setting the type of the member 'default' of a type (line 798)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 798, 16), self_133451, 'default', default_133450)
        # SSA join for if statement (line 795)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 799):
        
        # Assigning a Name to a Attribute (line 799):
        # Getting the type of '_status' (line 799)
        _status_133452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 799, 27), '_status')
        # Getting the type of 'self' (line 799)
        self_133453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 799, 12), 'self')
        # Setting the type of the member '_status' of a type (line 799)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 799, 12), self_133453, '_status', _status_133452)
        
        # Call to iterupgrade(...): (line 800)
        # Processing the call arguments (line 800)
        # Getting the type of 'value' (line 800)
        value_133456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 800, 29), 'value', False)
        # Processing the call keyword arguments (line 800)
        kwargs_133457 = {}
        # Getting the type of 'self' (line 800)
        self_133454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 800, 12), 'self', False)
        # Obtaining the member 'iterupgrade' of a type (line 800)
        iterupgrade_133455 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 800, 12), self_133454, 'iterupgrade')
        # Calling iterupgrade(args, kwargs) (line 800)
        iterupgrade_call_result_133458 = invoke(stypy.reporting.localization.Localization(__file__, 800, 12), iterupgrade_133455, *[value_133456], **kwargs_133457)
        
        # SSA join for try-except statement (line 777)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'iterupgrade(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'iterupgrade' in the type store
        # Getting the type of 'stypy_return_type' (line 772)
        stypy_return_type_133459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 772, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_133459)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'iterupgrade'
        return stypy_return_type_133459


    @norecursion
    def update(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 802)
        None_133460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 802, 35), 'None')
        # Getting the type of 'None' (line 802)
        None_133461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 802, 55), 'None')
        
        # Call to asbytes(...): (line 803)
        # Processing the call arguments (line 803)
        str_133463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 803, 38), 'str', '')
        # Processing the call keyword arguments (line 803)
        kwargs_133464 = {}
        # Getting the type of 'asbytes' (line 803)
        asbytes_133462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 803, 30), 'asbytes', False)
        # Calling asbytes(args, kwargs) (line 803)
        asbytes_call_result_133465 = invoke(stypy.reporting.localization.Localization(__file__, 803, 30), asbytes_133462, *[str_133463], **kwargs_133464)
        
        # Getting the type of 'False' (line 803)
        False_133466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 803, 50), 'False')
        defaults = [None_133460, None_133461, asbytes_call_result_133465, False_133466]
        # Create a new context for function 'update'
        module_type_store = module_type_store.open_function_context('update', 802, 4, False)
        # Assigning a type to the variable 'self' (line 803)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 803, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        StringConverter.update.__dict__.__setitem__('stypy_localization', localization)
        StringConverter.update.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        StringConverter.update.__dict__.__setitem__('stypy_type_store', module_type_store)
        StringConverter.update.__dict__.__setitem__('stypy_function_name', 'StringConverter.update')
        StringConverter.update.__dict__.__setitem__('stypy_param_names_list', ['func', 'default', 'testing_value', 'missing_values', 'locked'])
        StringConverter.update.__dict__.__setitem__('stypy_varargs_param_name', None)
        StringConverter.update.__dict__.__setitem__('stypy_kwargs_param_name', None)
        StringConverter.update.__dict__.__setitem__('stypy_call_defaults', defaults)
        StringConverter.update.__dict__.__setitem__('stypy_call_varargs', varargs)
        StringConverter.update.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        StringConverter.update.__dict__.__setitem__('stypy_declared_arg_number', 6)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'StringConverter.update', ['func', 'default', 'testing_value', 'missing_values', 'locked'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'update', localization, ['func', 'default', 'testing_value', 'missing_values', 'locked'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'update(...)' code ##################

        str_133467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 831, (-1)), 'str', '\n        Set StringConverter attributes directly.\n\n        Parameters\n        ----------\n        func : function\n            Conversion function.\n        default : any, optional\n            Value to return by default, that is, when the string to be\n            converted is flagged as missing. If not given,\n            `StringConverter` tries to supply a reasonable default value.\n        testing_value : str, optional\n            A string representing a standard input value of the converter.\n            This string is used to help defining a reasonable default\n            value.\n        missing_values : sequence of str, optional\n            Sequence of strings indicating a missing value.\n        locked : bool, optional\n            Whether the StringConverter should be locked to prevent\n            automatic upgrade or not. Default is False.\n\n        Notes\n        -----\n        `update` takes the same parameters as the constructor of\n        `StringConverter`, except that `func` does not accept a `dtype`\n        whereas `dtype_or_func` in the constructor does.\n\n        ')
        
        # Assigning a Name to a Attribute (line 832):
        
        # Assigning a Name to a Attribute (line 832):
        # Getting the type of 'func' (line 832)
        func_133468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 832, 20), 'func')
        # Getting the type of 'self' (line 832)
        self_133469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 832, 8), 'self')
        # Setting the type of the member 'func' of a type (line 832)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 832, 8), self_133469, 'func', func_133468)
        
        # Assigning a Name to a Attribute (line 833):
        
        # Assigning a Name to a Attribute (line 833):
        # Getting the type of 'locked' (line 833)
        locked_133470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 833, 23), 'locked')
        # Getting the type of 'self' (line 833)
        self_133471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 833, 8), 'self')
        # Setting the type of the member '_locked' of a type (line 833)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 833, 8), self_133471, '_locked', locked_133470)
        
        # Type idiom detected: calculating its left and rigth part (line 835)
        # Getting the type of 'default' (line 835)
        default_133472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 835, 8), 'default')
        # Getting the type of 'None' (line 835)
        None_133473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 835, 26), 'None')
        
        (may_be_133474, more_types_in_union_133475) = may_not_be_none(default_133472, None_133473)

        if may_be_133474:

            if more_types_in_union_133475:
                # Runtime conditional SSA (line 835)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Name to a Attribute (line 836):
            
            # Assigning a Name to a Attribute (line 836):
            # Getting the type of 'default' (line 836)
            default_133476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 836, 27), 'default')
            # Getting the type of 'self' (line 836)
            self_133477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 836, 12), 'self')
            # Setting the type of the member 'default' of a type (line 836)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 836, 12), self_133477, 'default', default_133476)
            
            # Assigning a Call to a Attribute (line 837):
            
            # Assigning a Call to a Attribute (line 837):
            
            # Call to _dtypeortype(...): (line 837)
            # Processing the call arguments (line 837)
            
            # Call to _getdtype(...): (line 837)
            # Processing the call arguments (line 837)
            # Getting the type of 'default' (line 837)
            default_133482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 837, 57), 'default', False)
            # Processing the call keyword arguments (line 837)
            kwargs_133483 = {}
            # Getting the type of 'self' (line 837)
            self_133480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 837, 42), 'self', False)
            # Obtaining the member '_getdtype' of a type (line 837)
            _getdtype_133481 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 837, 42), self_133480, '_getdtype')
            # Calling _getdtype(args, kwargs) (line 837)
            _getdtype_call_result_133484 = invoke(stypy.reporting.localization.Localization(__file__, 837, 42), _getdtype_133481, *[default_133482], **kwargs_133483)
            
            # Processing the call keyword arguments (line 837)
            kwargs_133485 = {}
            # Getting the type of 'self' (line 837)
            self_133478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 837, 24), 'self', False)
            # Obtaining the member '_dtypeortype' of a type (line 837)
            _dtypeortype_133479 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 837, 24), self_133478, '_dtypeortype')
            # Calling _dtypeortype(args, kwargs) (line 837)
            _dtypeortype_call_result_133486 = invoke(stypy.reporting.localization.Localization(__file__, 837, 24), _dtypeortype_133479, *[_getdtype_call_result_133484], **kwargs_133485)
            
            # Getting the type of 'self' (line 837)
            self_133487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 837, 12), 'self')
            # Setting the type of the member 'type' of a type (line 837)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 837, 12), self_133487, 'type', _dtypeortype_call_result_133486)

            if more_types_in_union_133475:
                # Runtime conditional SSA for else branch (line 835)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_133474) or more_types_in_union_133475):
            
            
            # SSA begins for try-except statement (line 839)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
            
            # Assigning a Call to a Name (line 840):
            
            # Assigning a Call to a Name (line 840):
            
            # Call to func(...): (line 840)
            # Processing the call arguments (line 840)
            
            # Evaluating a boolean operation
            # Getting the type of 'testing_value' (line 840)
            testing_value_133489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 840, 30), 'testing_value', False)
            
            # Call to asbytes(...): (line 840)
            # Processing the call arguments (line 840)
            str_133491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 840, 55), 'str', '1')
            # Processing the call keyword arguments (line 840)
            kwargs_133492 = {}
            # Getting the type of 'asbytes' (line 840)
            asbytes_133490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 840, 47), 'asbytes', False)
            # Calling asbytes(args, kwargs) (line 840)
            asbytes_call_result_133493 = invoke(stypy.reporting.localization.Localization(__file__, 840, 47), asbytes_133490, *[str_133491], **kwargs_133492)
            
            # Applying the binary operator 'or' (line 840)
            result_or_keyword_133494 = python_operator(stypy.reporting.localization.Localization(__file__, 840, 30), 'or', testing_value_133489, asbytes_call_result_133493)
            
            # Processing the call keyword arguments (line 840)
            kwargs_133495 = {}
            # Getting the type of 'func' (line 840)
            func_133488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 840, 25), 'func', False)
            # Calling func(args, kwargs) (line 840)
            func_call_result_133496 = invoke(stypy.reporting.localization.Localization(__file__, 840, 25), func_133488, *[result_or_keyword_133494], **kwargs_133495)
            
            # Assigning a type to the variable 'tester' (line 840)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 840, 16), 'tester', func_call_result_133496)
            # SSA branch for the except part of a try statement (line 839)
            # SSA branch for the except 'Tuple' branch of a try statement (line 839)
            module_type_store.open_ssa_branch('except')
            
            # Assigning a Name to a Name (line 842):
            
            # Assigning a Name to a Name (line 842):
            # Getting the type of 'None' (line 842)
            None_133497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 842, 25), 'None')
            # Assigning a type to the variable 'tester' (line 842)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 842, 16), 'tester', None_133497)
            # SSA join for try-except statement (line 839)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Assigning a Call to a Attribute (line 843):
            
            # Assigning a Call to a Attribute (line 843):
            
            # Call to _dtypeortype(...): (line 843)
            # Processing the call arguments (line 843)
            
            # Call to _getdtype(...): (line 843)
            # Processing the call arguments (line 843)
            # Getting the type of 'tester' (line 843)
            tester_133502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 843, 57), 'tester', False)
            # Processing the call keyword arguments (line 843)
            kwargs_133503 = {}
            # Getting the type of 'self' (line 843)
            self_133500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 843, 42), 'self', False)
            # Obtaining the member '_getdtype' of a type (line 843)
            _getdtype_133501 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 843, 42), self_133500, '_getdtype')
            # Calling _getdtype(args, kwargs) (line 843)
            _getdtype_call_result_133504 = invoke(stypy.reporting.localization.Localization(__file__, 843, 42), _getdtype_133501, *[tester_133502], **kwargs_133503)
            
            # Processing the call keyword arguments (line 843)
            kwargs_133505 = {}
            # Getting the type of 'self' (line 843)
            self_133498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 843, 24), 'self', False)
            # Obtaining the member '_dtypeortype' of a type (line 843)
            _dtypeortype_133499 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 843, 24), self_133498, '_dtypeortype')
            # Calling _dtypeortype(args, kwargs) (line 843)
            _dtypeortype_call_result_133506 = invoke(stypy.reporting.localization.Localization(__file__, 843, 24), _dtypeortype_133499, *[_getdtype_call_result_133504], **kwargs_133505)
            
            # Getting the type of 'self' (line 843)
            self_133507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 843, 12), 'self')
            # Setting the type of the member 'type' of a type (line 843)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 843, 12), self_133507, 'type', _dtypeortype_call_result_133506)

            if (may_be_133474 and more_types_in_union_133475):
                # SSA join for if statement (line 835)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 845)
        # Getting the type of 'missing_values' (line 845)
        missing_values_133508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 845, 8), 'missing_values')
        # Getting the type of 'None' (line 845)
        None_133509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 845, 33), 'None')
        
        (may_be_133510, more_types_in_union_133511) = may_not_be_none(missing_values_133508, None_133509)

        if may_be_133510:

            if more_types_in_union_133511:
                # Runtime conditional SSA (line 845)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            
            # Call to _is_bytes_like(...): (line 846)
            # Processing the call arguments (line 846)
            # Getting the type of 'missing_values' (line 846)
            missing_values_133513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 846, 30), 'missing_values', False)
            # Processing the call keyword arguments (line 846)
            kwargs_133514 = {}
            # Getting the type of '_is_bytes_like' (line 846)
            _is_bytes_like_133512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 846, 15), '_is_bytes_like', False)
            # Calling _is_bytes_like(args, kwargs) (line 846)
            _is_bytes_like_call_result_133515 = invoke(stypy.reporting.localization.Localization(__file__, 846, 15), _is_bytes_like_133512, *[missing_values_133513], **kwargs_133514)
            
            # Testing the type of an if condition (line 846)
            if_condition_133516 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 846, 12), _is_bytes_like_call_result_133515)
            # Assigning a type to the variable 'if_condition_133516' (line 846)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 846, 12), 'if_condition_133516', if_condition_133516)
            # SSA begins for if statement (line 846)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to add(...): (line 847)
            # Processing the call arguments (line 847)
            # Getting the type of 'missing_values' (line 847)
            missing_values_133520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 847, 40), 'missing_values', False)
            # Processing the call keyword arguments (line 847)
            kwargs_133521 = {}
            # Getting the type of 'self' (line 847)
            self_133517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 847, 16), 'self', False)
            # Obtaining the member 'missing_values' of a type (line 847)
            missing_values_133518 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 847, 16), self_133517, 'missing_values')
            # Obtaining the member 'add' of a type (line 847)
            add_133519 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 847, 16), missing_values_133518, 'add')
            # Calling add(args, kwargs) (line 847)
            add_call_result_133522 = invoke(stypy.reporting.localization.Localization(__file__, 847, 16), add_133519, *[missing_values_133520], **kwargs_133521)
            
            # SSA branch for the else part of an if statement (line 846)
            module_type_store.open_ssa_branch('else')
            
            # Type idiom detected: calculating its left and rigth part (line 848)
            str_133523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 848, 41), 'str', '__iter__')
            # Getting the type of 'missing_values' (line 848)
            missing_values_133524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 848, 25), 'missing_values')
            
            (may_be_133525, more_types_in_union_133526) = may_provide_member(str_133523, missing_values_133524)

            if may_be_133525:

                if more_types_in_union_133526:
                    # Runtime conditional SSA (line 848)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                # Assigning a type to the variable 'missing_values' (line 848)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 848, 17), 'missing_values', remove_not_member_provider_from_union(missing_values_133524, '__iter__'))
                
                # Getting the type of 'missing_values' (line 849)
                missing_values_133527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 849, 27), 'missing_values')
                # Testing the type of a for loop iterable (line 849)
                is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 849, 16), missing_values_133527)
                # Getting the type of the for loop variable (line 849)
                for_loop_var_133528 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 849, 16), missing_values_133527)
                # Assigning a type to the variable 'val' (line 849)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 849, 16), 'val', for_loop_var_133528)
                # SSA begins for a for statement (line 849)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Call to add(...): (line 850)
                # Processing the call arguments (line 850)
                # Getting the type of 'val' (line 850)
                val_133532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 850, 44), 'val', False)
                # Processing the call keyword arguments (line 850)
                kwargs_133533 = {}
                # Getting the type of 'self' (line 850)
                self_133529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 850, 20), 'self', False)
                # Obtaining the member 'missing_values' of a type (line 850)
                missing_values_133530 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 850, 20), self_133529, 'missing_values')
                # Obtaining the member 'add' of a type (line 850)
                add_133531 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 850, 20), missing_values_133530, 'add')
                # Calling add(args, kwargs) (line 850)
                add_call_result_133534 = invoke(stypy.reporting.localization.Localization(__file__, 850, 20), add_133531, *[val_133532], **kwargs_133533)
                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()
                

                if more_types_in_union_133526:
                    # SSA join for if statement (line 848)
                    module_type_store = module_type_store.join_ssa_context()


            
            # SSA join for if statement (line 846)
            module_type_store = module_type_store.join_ssa_context()
            

            if more_types_in_union_133511:
                # Runtime conditional SSA for else branch (line 845)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_133510) or more_types_in_union_133511):
            
            # Assigning a List to a Attribute (line 852):
            
            # Assigning a List to a Attribute (line 852):
            
            # Obtaining an instance of the builtin type 'list' (line 852)
            list_133535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 852, 34), 'list')
            # Adding type elements to the builtin type 'list' instance (line 852)
            
            # Getting the type of 'self' (line 852)
            self_133536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 852, 12), 'self')
            # Setting the type of the member 'missing_values' of a type (line 852)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 852, 12), self_133536, 'missing_values', list_133535)

            if (may_be_133510 and more_types_in_union_133511):
                # SSA join for if statement (line 845)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # ################# End of 'update(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'update' in the type store
        # Getting the type of 'stypy_return_type' (line 802)
        stypy_return_type_133537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 802, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_133537)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'update'
        return stypy_return_type_133537


# Assigning a type to the variable 'StringConverter' (line 473)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 473, 0), 'StringConverter', StringConverter)

# Assigning a List to a Name (line 520):

# Obtaining an instance of the builtin type 'list' (line 520)
list_133538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 520, 14), 'list')
# Adding type elements to the builtin type 'list' instance (line 520)
# Adding element type (line 520)

# Obtaining an instance of the builtin type 'tuple' (line 520)
tuple_133539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 520, 16), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 520)
# Adding element type (line 520)
# Getting the type of 'nx' (line 520)
nx_133540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 16), 'nx')
# Obtaining the member 'bool_' of a type (line 520)
bool__133541 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 520, 16), nx_133540, 'bool_')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 520, 16), tuple_133539, bool__133541)
# Adding element type (line 520)
# Getting the type of 'str2bool' (line 520)
str2bool_133542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 26), 'str2bool')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 520, 16), tuple_133539, str2bool_133542)
# Adding element type (line 520)
# Getting the type of 'False' (line 520)
False_133543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 36), 'False')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 520, 16), tuple_133539, False_133543)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 520, 14), list_133538, tuple_133539)
# Adding element type (line 520)

# Obtaining an instance of the builtin type 'tuple' (line 521)
tuple_133544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 521, 16), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 521)
# Adding element type (line 521)
# Getting the type of 'nx' (line 521)
nx_133545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 16), 'nx')
# Obtaining the member 'integer' of a type (line 521)
integer_133546 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 521, 16), nx_133545, 'integer')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 521, 16), tuple_133544, integer_133546)
# Adding element type (line 521)
# Getting the type of 'int' (line 521)
int_133547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 28), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 521, 16), tuple_133544, int_133547)
# Adding element type (line 521)
int_133548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 521, 33), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 521, 16), tuple_133544, int_133548)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 520, 14), list_133538, tuple_133544)

# Getting the type of 'StringConverter'
StringConverter_133549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'StringConverter')
# Setting the type of the member '_mapper' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), StringConverter_133549, '_mapper', list_133538)

# Assigning a List to a Name (line 520):



# Call to dtype(...): (line 525)
# Processing the call arguments (line 525)
# Getting the type of 'nx' (line 525)
nx_133552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 16), 'nx', False)
# Obtaining the member 'integer' of a type (line 525)
integer_133553 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 525, 16), nx_133552, 'integer')
# Processing the call keyword arguments (line 525)
kwargs_133554 = {}
# Getting the type of 'nx' (line 525)
nx_133550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 7), 'nx', False)
# Obtaining the member 'dtype' of a type (line 525)
dtype_133551 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 525, 7), nx_133550, 'dtype')
# Calling dtype(args, kwargs) (line 525)
dtype_call_result_133555 = invoke(stypy.reporting.localization.Localization(__file__, 525, 7), dtype_133551, *[integer_133553], **kwargs_133554)

# Obtaining the member 'itemsize' of a type (line 525)
itemsize_133556 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 525, 7), dtype_call_result_133555, 'itemsize')

# Call to dtype(...): (line 525)
# Processing the call arguments (line 525)
# Getting the type of 'nx' (line 525)
nx_133559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 48), 'nx', False)
# Obtaining the member 'int64' of a type (line 525)
int64_133560 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 525, 48), nx_133559, 'int64')
# Processing the call keyword arguments (line 525)
kwargs_133561 = {}
# Getting the type of 'nx' (line 525)
nx_133557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 39), 'nx', False)
# Obtaining the member 'dtype' of a type (line 525)
dtype_133558 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 525, 39), nx_133557, 'dtype')
# Calling dtype(args, kwargs) (line 525)
dtype_call_result_133562 = invoke(stypy.reporting.localization.Localization(__file__, 525, 39), dtype_133558, *[int64_133560], **kwargs_133561)

# Obtaining the member 'itemsize' of a type (line 525)
itemsize_133563 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 525, 39), dtype_call_result_133562, 'itemsize')
# Applying the binary operator '<' (line 525)
result_lt_133564 = python_operator(stypy.reporting.localization.Localization(__file__, 525, 7), '<', itemsize_133556, itemsize_133563)

# Testing the type of an if condition (line 525)
if_condition_133565 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 525, 4), result_lt_133564)
# Assigning a type to the variable 'if_condition_133565' (line 525)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 525, 4), 'if_condition_133565', if_condition_133565)
# SSA begins for if statement (line 525)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Call to append(...): (line 526)
# Processing the call arguments (line 526)

# Obtaining an instance of the builtin type 'tuple' (line 526)
tuple_133569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 526, 24), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 526)
# Adding element type (line 526)
# Getting the type of 'nx' (line 526)
nx_133570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 24), 'nx', False)
# Obtaining the member 'int64' of a type (line 526)
int64_133571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 526, 24), nx_133570, 'int64')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 526, 24), tuple_133569, int64_133571)
# Adding element type (line 526)
# Getting the type of 'int' (line 526)
int_133572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 34), 'int', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 526, 24), tuple_133569, int_133572)
# Adding element type (line 526)
int_133573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 526, 39), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 526, 24), tuple_133569, int_133573)

# Processing the call keyword arguments (line 526)
kwargs_133574 = {}
# Getting the type of 'StringConverter'
StringConverter_133566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'StringConverter', False)
# Obtaining the member '_mapper' of a type
_mapper_133567 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), StringConverter_133566, '_mapper')
# Obtaining the member 'append' of a type (line 526)
append_133568 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 526, 8), _mapper_133567, 'append')
# Calling append(args, kwargs) (line 526)
append_call_result_133575 = invoke(stypy.reporting.localization.Localization(__file__, 526, 8), append_133568, *[tuple_133569], **kwargs_133574)

# SSA join for if statement (line 525)
module_type_store = module_type_store.join_ssa_context()


# Call to extend(...): (line 528)
# Processing the call arguments (line 528)

# Obtaining an instance of the builtin type 'list' (line 528)
list_133579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 528, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 528)
# Adding element type (line 528)

# Obtaining an instance of the builtin type 'tuple' (line 528)
tuple_133580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 528, 21), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 528)
# Adding element type (line 528)
# Getting the type of 'nx' (line 528)
nx_133581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 21), 'nx', False)
# Obtaining the member 'floating' of a type (line 528)
floating_133582 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 528, 21), nx_133581, 'floating')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 528, 21), tuple_133580, floating_133582)
# Adding element type (line 528)
# Getting the type of 'float' (line 528)
float_133583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 34), 'float', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 528, 21), tuple_133580, float_133583)
# Adding element type (line 528)
# Getting the type of 'nx' (line 528)
nx_133584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 41), 'nx', False)
# Obtaining the member 'nan' of a type (line 528)
nan_133585 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 528, 41), nx_133584, 'nan')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 528, 21), tuple_133580, nan_133585)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 528, 19), list_133579, tuple_133580)
# Adding element type (line 528)

# Obtaining an instance of the builtin type 'tuple' (line 529)
tuple_133586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 529, 21), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 529)
# Adding element type (line 529)
# Getting the type of 'complex' (line 529)
complex_133587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 21), 'complex', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 529, 21), tuple_133586, complex_133587)
# Adding element type (line 529)
# Getting the type of '_bytes_to_complex' (line 529)
_bytes_to_complex_133588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 30), '_bytes_to_complex', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 529, 21), tuple_133586, _bytes_to_complex_133588)
# Adding element type (line 529)
# Getting the type of 'nx' (line 529)
nx_133589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 49), 'nx', False)
# Obtaining the member 'nan' of a type (line 529)
nan_133590 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 529, 49), nx_133589, 'nan')
complex_133591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 529, 58), 'complex')
# Applying the binary operator '+' (line 529)
result_add_133592 = python_operator(stypy.reporting.localization.Localization(__file__, 529, 49), '+', nan_133590, complex_133591)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 529, 21), tuple_133586, result_add_133592)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 528, 19), list_133579, tuple_133586)
# Adding element type (line 528)

# Obtaining an instance of the builtin type 'tuple' (line 530)
tuple_133593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 530, 21), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 530)
# Adding element type (line 530)
# Getting the type of 'nx' (line 530)
nx_133594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 21), 'nx', False)
# Obtaining the member 'longdouble' of a type (line 530)
longdouble_133595 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 530, 21), nx_133594, 'longdouble')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 530, 21), tuple_133593, longdouble_133595)
# Adding element type (line 530)
# Getting the type of 'nx' (line 530)
nx_133596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 36), 'nx', False)
# Obtaining the member 'longdouble' of a type (line 530)
longdouble_133597 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 530, 36), nx_133596, 'longdouble')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 530, 21), tuple_133593, longdouble_133597)
# Adding element type (line 530)
# Getting the type of 'nx' (line 530)
nx_133598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 51), 'nx', False)
# Obtaining the member 'nan' of a type (line 530)
nan_133599 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 530, 51), nx_133598, 'nan')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 530, 21), tuple_133593, nan_133599)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 528, 19), list_133579, tuple_133593)
# Adding element type (line 528)

# Obtaining an instance of the builtin type 'tuple' (line 531)
tuple_133600 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 531, 21), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 531)
# Adding element type (line 531)
# Getting the type of 'nx' (line 531)
nx_133601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 21), 'nx', False)
# Obtaining the member 'string_' of a type (line 531)
string__133602 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 531, 21), nx_133601, 'string_')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 531, 21), tuple_133600, string__133602)
# Adding element type (line 531)
# Getting the type of 'bytes' (line 531)
bytes_133603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 33), 'bytes', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 531, 21), tuple_133600, bytes_133603)
# Adding element type (line 531)

# Call to asbytes(...): (line 531)
# Processing the call arguments (line 531)
str_133605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 531, 48), 'str', '???')
# Processing the call keyword arguments (line 531)
kwargs_133606 = {}
# Getting the type of 'asbytes' (line 531)
asbytes_133604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 40), 'asbytes', False)
# Calling asbytes(args, kwargs) (line 531)
asbytes_call_result_133607 = invoke(stypy.reporting.localization.Localization(__file__, 531, 40), asbytes_133604, *[str_133605], **kwargs_133606)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 531, 21), tuple_133600, asbytes_call_result_133607)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 528, 19), list_133579, tuple_133600)

# Processing the call keyword arguments (line 528)
kwargs_133608 = {}
# Getting the type of 'StringConverter'
StringConverter_133576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'StringConverter', False)
# Obtaining the member '_mapper' of a type
_mapper_133577 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), StringConverter_133576, '_mapper')
# Obtaining the member 'extend' of a type (line 528)
extend_133578 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 528, 4), _mapper_133577, 'extend')
# Calling extend(args, kwargs) (line 528)
extend_call_result_133609 = invoke(stypy.reporting.localization.Localization(__file__, 528, 4), extend_133578, *[list_133579], **kwargs_133608)


# Assigning a Call to a Name:

# Call to zip(...): (line 533)
# Getting the type of 'StringConverter'
StringConverter_133611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'StringConverter', False)
# Obtaining the member '_mapper' of a type
_mapper_133612 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), StringConverter_133611, '_mapper')
# Processing the call keyword arguments (line 533)
kwargs_133613 = {}
# Getting the type of 'zip' (line 533)
zip_133610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 49), 'zip', False)
# Calling zip(args, kwargs) (line 533)
zip_call_result_133614 = invoke(stypy.reporting.localization.Localization(__file__, 533, 49), zip_133610, *[_mapper_133612], **kwargs_133613)

# Getting the type of 'StringConverter'
StringConverter_133615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'StringConverter')
# Setting the type of the member 'call_assignment_132134' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), StringConverter_133615, 'call_assignment_132134', zip_call_result_133614)

# Assigning a Call to a Name (line 533):

# Call to __getitem__(...):
# Processing the call arguments
int_133619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 533, 4), 'int')
# Processing the call keyword arguments
kwargs_133620 = {}
# Getting the type of 'StringConverter'
StringConverter_133616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'StringConverter', False)
# Obtaining the member 'call_assignment_132134' of a type
call_assignment_132134_133617 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), StringConverter_133616, 'call_assignment_132134')
# Obtaining the member '__getitem__' of a type
getitem___133618 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), call_assignment_132134_133617, '__getitem__')
# Calling __getitem__(args, kwargs)
getitem___call_result_133621 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___133618, *[int_133619], **kwargs_133620)

# Getting the type of 'StringConverter'
StringConverter_133622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'StringConverter')
# Setting the type of the member 'call_assignment_132135' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), StringConverter_133622, 'call_assignment_132135', getitem___call_result_133621)

# Assigning a Name to a Name (line 533):
# Getting the type of 'StringConverter'
StringConverter_133623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'StringConverter')
# Obtaining the member 'call_assignment_132135' of a type
call_assignment_132135_133624 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), StringConverter_133623, 'call_assignment_132135')
# Getting the type of 'StringConverter'
StringConverter_133625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'StringConverter')
# Setting the type of the member '_defaulttype' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), StringConverter_133625, '_defaulttype', call_assignment_132135_133624)

# Assigning a Call to a Name (line 533):

# Call to __getitem__(...):
# Processing the call arguments
int_133629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 533, 4), 'int')
# Processing the call keyword arguments
kwargs_133630 = {}
# Getting the type of 'StringConverter'
StringConverter_133626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'StringConverter', False)
# Obtaining the member 'call_assignment_132134' of a type
call_assignment_132134_133627 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), StringConverter_133626, 'call_assignment_132134')
# Obtaining the member '__getitem__' of a type
getitem___133628 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), call_assignment_132134_133627, '__getitem__')
# Calling __getitem__(args, kwargs)
getitem___call_result_133631 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___133628, *[int_133629], **kwargs_133630)

# Getting the type of 'StringConverter'
StringConverter_133632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'StringConverter')
# Setting the type of the member 'call_assignment_132136' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), StringConverter_133632, 'call_assignment_132136', getitem___call_result_133631)

# Assigning a Name to a Name (line 533):
# Getting the type of 'StringConverter'
StringConverter_133633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'StringConverter')
# Obtaining the member 'call_assignment_132136' of a type
call_assignment_132136_133634 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), StringConverter_133633, 'call_assignment_132136')
# Getting the type of 'StringConverter'
StringConverter_133635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'StringConverter')
# Setting the type of the member '_defaultfunc' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), StringConverter_133635, '_defaultfunc', call_assignment_132136_133634)

# Assigning a Call to a Name (line 533):

# Call to __getitem__(...):
# Processing the call arguments
int_133639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 533, 4), 'int')
# Processing the call keyword arguments
kwargs_133640 = {}
# Getting the type of 'StringConverter'
StringConverter_133636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'StringConverter', False)
# Obtaining the member 'call_assignment_132134' of a type
call_assignment_132134_133637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), StringConverter_133636, 'call_assignment_132134')
# Obtaining the member '__getitem__' of a type
getitem___133638 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), call_assignment_132134_133637, '__getitem__')
# Calling __getitem__(args, kwargs)
getitem___call_result_133641 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___133638, *[int_133639], **kwargs_133640)

# Getting the type of 'StringConverter'
StringConverter_133642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'StringConverter')
# Setting the type of the member 'call_assignment_132137' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), StringConverter_133642, 'call_assignment_132137', getitem___call_result_133641)

# Assigning a Name to a Name (line 533):
# Getting the type of 'StringConverter'
StringConverter_133643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'StringConverter')
# Obtaining the member 'call_assignment_132137' of a type
call_assignment_132137_133644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), StringConverter_133643, 'call_assignment_132137')
# Getting the type of 'StringConverter'
StringConverter_133645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'StringConverter')
# Setting the type of the member '_defaultfill' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), StringConverter_133645, '_defaultfill', call_assignment_132137_133644)

@norecursion
def easy_dtype(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 855)
    None_133646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 855, 29), 'None')
    str_133647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 855, 46), 'str', 'f%i')
    defaults = [None_133646, str_133647]
    # Create a new context for function 'easy_dtype'
    module_type_store = module_type_store.open_function_context('easy_dtype', 855, 0, False)
    
    # Passed parameters checking function
    easy_dtype.stypy_localization = localization
    easy_dtype.stypy_type_of_self = None
    easy_dtype.stypy_type_store = module_type_store
    easy_dtype.stypy_function_name = 'easy_dtype'
    easy_dtype.stypy_param_names_list = ['ndtype', 'names', 'defaultfmt']
    easy_dtype.stypy_varargs_param_name = None
    easy_dtype.stypy_kwargs_param_name = 'validationargs'
    easy_dtype.stypy_call_defaults = defaults
    easy_dtype.stypy_call_varargs = varargs
    easy_dtype.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'easy_dtype', ['ndtype', 'names', 'defaultfmt'], None, 'validationargs', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'easy_dtype', localization, ['ndtype', 'names', 'defaultfmt'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'easy_dtype(...)' code ##################

    str_133648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 892, (-1)), 'str', '\n    Convenience function to create a `np.dtype` object.\n\n    The function processes the input `dtype` and matches it with the given\n    names.\n\n    Parameters\n    ----------\n    ndtype : var\n        Definition of the dtype. Can be any string or dictionary recognized\n        by the `np.dtype` function, or a sequence of types.\n    names : str or sequence, optional\n        Sequence of strings to use as field names for a structured dtype.\n        For convenience, `names` can be a string of a comma-separated list\n        of names.\n    defaultfmt : str, optional\n        Format string used to define missing names, such as ``"f%i"``\n        (default) or ``"fields_%02i"``.\n    validationargs : optional\n        A series of optional arguments used to initialize a\n        `NameValidator`.\n\n    Examples\n    --------\n    >>> np.lib._iotools.easy_dtype(float)\n    dtype(\'float64\')\n    >>> np.lib._iotools.easy_dtype("i4, f8")\n    dtype([(\'f0\', \'<i4\'), (\'f1\', \'<f8\')])\n    >>> np.lib._iotools.easy_dtype("i4, f8", defaultfmt="field_%03i")\n    dtype([(\'field_000\', \'<i4\'), (\'field_001\', \'<f8\')])\n\n    >>> np.lib._iotools.easy_dtype((int, float, float), names="a,b,c")\n    dtype([(\'a\', \'<i8\'), (\'b\', \'<f8\'), (\'c\', \'<f8\')])\n    >>> np.lib._iotools.easy_dtype(float, names="a,b,c")\n    dtype([(\'a\', \'<f8\'), (\'b\', \'<f8\'), (\'c\', \'<f8\')])\n\n    ')
    
    
    # SSA begins for try-except statement (line 893)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Name (line 894):
    
    # Assigning a Call to a Name (line 894):
    
    # Call to dtype(...): (line 894)
    # Processing the call arguments (line 894)
    # Getting the type of 'ndtype' (line 894)
    ndtype_133651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 894, 26), 'ndtype', False)
    # Processing the call keyword arguments (line 894)
    kwargs_133652 = {}
    # Getting the type of 'np' (line 894)
    np_133649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 894, 17), 'np', False)
    # Obtaining the member 'dtype' of a type (line 894)
    dtype_133650 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 894, 17), np_133649, 'dtype')
    # Calling dtype(args, kwargs) (line 894)
    dtype_call_result_133653 = invoke(stypy.reporting.localization.Localization(__file__, 894, 17), dtype_133650, *[ndtype_133651], **kwargs_133652)
    
    # Assigning a type to the variable 'ndtype' (line 894)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 894, 8), 'ndtype', dtype_call_result_133653)
    # SSA branch for the except part of a try statement (line 893)
    # SSA branch for the except 'TypeError' branch of a try statement (line 893)
    module_type_store.open_ssa_branch('except')
    
    # Assigning a Call to a Name (line 896):
    
    # Assigning a Call to a Name (line 896):
    
    # Call to NameValidator(...): (line 896)
    # Processing the call keyword arguments (line 896)
    # Getting the type of 'validationargs' (line 896)
    validationargs_133655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 896, 35), 'validationargs', False)
    kwargs_133656 = {'validationargs_133655': validationargs_133655}
    # Getting the type of 'NameValidator' (line 896)
    NameValidator_133654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 896, 19), 'NameValidator', False)
    # Calling NameValidator(args, kwargs) (line 896)
    NameValidator_call_result_133657 = invoke(stypy.reporting.localization.Localization(__file__, 896, 19), NameValidator_133654, *[], **kwargs_133656)
    
    # Assigning a type to the variable 'validate' (line 896)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 896, 8), 'validate', NameValidator_call_result_133657)
    
    # Assigning a Call to a Name (line 897):
    
    # Assigning a Call to a Name (line 897):
    
    # Call to len(...): (line 897)
    # Processing the call arguments (line 897)
    # Getting the type of 'ndtype' (line 897)
    ndtype_133659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 897, 23), 'ndtype', False)
    # Processing the call keyword arguments (line 897)
    kwargs_133660 = {}
    # Getting the type of 'len' (line 897)
    len_133658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 897, 19), 'len', False)
    # Calling len(args, kwargs) (line 897)
    len_call_result_133661 = invoke(stypy.reporting.localization.Localization(__file__, 897, 19), len_133658, *[ndtype_133659], **kwargs_133660)
    
    # Assigning a type to the variable 'nbfields' (line 897)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 897, 8), 'nbfields', len_call_result_133661)
    
    # Type idiom detected: calculating its left and rigth part (line 898)
    # Getting the type of 'names' (line 898)
    names_133662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 898, 11), 'names')
    # Getting the type of 'None' (line 898)
    None_133663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 898, 20), 'None')
    
    (may_be_133664, more_types_in_union_133665) = may_be_none(names_133662, None_133663)

    if may_be_133664:

        if more_types_in_union_133665:
            # Runtime conditional SSA (line 898)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a BinOp to a Name (line 899):
        
        # Assigning a BinOp to a Name (line 899):
        
        # Obtaining an instance of the builtin type 'list' (line 899)
        list_133666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 899, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 899)
        # Adding element type (line 899)
        str_133667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 899, 21), 'str', '')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 899, 20), list_133666, str_133667)
        
        
        # Call to len(...): (line 899)
        # Processing the call arguments (line 899)
        # Getting the type of 'ndtype' (line 899)
        ndtype_133669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 899, 31), 'ndtype', False)
        # Processing the call keyword arguments (line 899)
        kwargs_133670 = {}
        # Getting the type of 'len' (line 899)
        len_133668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 899, 27), 'len', False)
        # Calling len(args, kwargs) (line 899)
        len_call_result_133671 = invoke(stypy.reporting.localization.Localization(__file__, 899, 27), len_133668, *[ndtype_133669], **kwargs_133670)
        
        # Applying the binary operator '*' (line 899)
        result_mul_133672 = python_operator(stypy.reporting.localization.Localization(__file__, 899, 20), '*', list_133666, len_call_result_133671)
        
        # Assigning a type to the variable 'names' (line 899)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 899, 12), 'names', result_mul_133672)

        if more_types_in_union_133665:
            # Runtime conditional SSA for else branch (line 898)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_133664) or more_types_in_union_133665):
        
        # Type idiom detected: calculating its left and rigth part (line 900)
        # Getting the type of 'basestring' (line 900)
        basestring_133673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 900, 31), 'basestring')
        # Getting the type of 'names' (line 900)
        names_133674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 900, 24), 'names')
        
        (may_be_133675, more_types_in_union_133676) = may_be_subtype(basestring_133673, names_133674)

        if may_be_133675:

            if more_types_in_union_133676:
                # Runtime conditional SSA (line 900)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'names' (line 900)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 900, 13), 'names', remove_not_subtype_from_union(names_133674, basestring))
            
            # Assigning a Call to a Name (line 901):
            
            # Assigning a Call to a Name (line 901):
            
            # Call to split(...): (line 901)
            # Processing the call arguments (line 901)
            str_133679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 901, 32), 'str', ',')
            # Processing the call keyword arguments (line 901)
            kwargs_133680 = {}
            # Getting the type of 'names' (line 901)
            names_133677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 901, 20), 'names', False)
            # Obtaining the member 'split' of a type (line 901)
            split_133678 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 901, 20), names_133677, 'split')
            # Calling split(args, kwargs) (line 901)
            split_call_result_133681 = invoke(stypy.reporting.localization.Localization(__file__, 901, 20), split_133678, *[str_133679], **kwargs_133680)
            
            # Assigning a type to the variable 'names' (line 901)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 901, 12), 'names', split_call_result_133681)

            if more_types_in_union_133676:
                # SSA join for if statement (line 900)
                module_type_store = module_type_store.join_ssa_context()


        

        if (may_be_133664 and more_types_in_union_133665):
            # SSA join for if statement (line 898)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 902):
    
    # Assigning a Call to a Name (line 902):
    
    # Call to validate(...): (line 902)
    # Processing the call arguments (line 902)
    # Getting the type of 'names' (line 902)
    names_133683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 902, 25), 'names', False)
    # Processing the call keyword arguments (line 902)
    # Getting the type of 'nbfields' (line 902)
    nbfields_133684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 902, 41), 'nbfields', False)
    keyword_133685 = nbfields_133684
    # Getting the type of 'defaultfmt' (line 902)
    defaultfmt_133686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 902, 62), 'defaultfmt', False)
    keyword_133687 = defaultfmt_133686
    kwargs_133688 = {'defaultfmt': keyword_133687, 'nbfields': keyword_133685}
    # Getting the type of 'validate' (line 902)
    validate_133682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 902, 16), 'validate', False)
    # Calling validate(args, kwargs) (line 902)
    validate_call_result_133689 = invoke(stypy.reporting.localization.Localization(__file__, 902, 16), validate_133682, *[names_133683], **kwargs_133688)
    
    # Assigning a type to the variable 'names' (line 902)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 902, 8), 'names', validate_call_result_133689)
    
    # Assigning a Call to a Name (line 903):
    
    # Assigning a Call to a Name (line 903):
    
    # Call to dtype(...): (line 903)
    # Processing the call arguments (line 903)
    
    # Call to dict(...): (line 903)
    # Processing the call keyword arguments (line 903)
    # Getting the type of 'ndtype' (line 903)
    ndtype_133693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 903, 39), 'ndtype', False)
    keyword_133694 = ndtype_133693
    # Getting the type of 'names' (line 903)
    names_133695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 903, 53), 'names', False)
    keyword_133696 = names_133695
    kwargs_133697 = {'names': keyword_133696, 'formats': keyword_133694}
    # Getting the type of 'dict' (line 903)
    dict_133692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 903, 26), 'dict', False)
    # Calling dict(args, kwargs) (line 903)
    dict_call_result_133698 = invoke(stypy.reporting.localization.Localization(__file__, 903, 26), dict_133692, *[], **kwargs_133697)
    
    # Processing the call keyword arguments (line 903)
    kwargs_133699 = {}
    # Getting the type of 'np' (line 903)
    np_133690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 903, 17), 'np', False)
    # Obtaining the member 'dtype' of a type (line 903)
    dtype_133691 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 903, 17), np_133690, 'dtype')
    # Calling dtype(args, kwargs) (line 903)
    dtype_call_result_133700 = invoke(stypy.reporting.localization.Localization(__file__, 903, 17), dtype_133691, *[dict_call_result_133698], **kwargs_133699)
    
    # Assigning a type to the variable 'ndtype' (line 903)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 903, 8), 'ndtype', dtype_call_result_133700)
    # SSA branch for the else branch of a try statement (line 893)
    module_type_store.open_ssa_branch('except else')
    
    # Assigning a Call to a Name (line 905):
    
    # Assigning a Call to a Name (line 905):
    
    # Call to len(...): (line 905)
    # Processing the call arguments (line 905)
    # Getting the type of 'ndtype' (line 905)
    ndtype_133702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 905, 22), 'ndtype', False)
    # Processing the call keyword arguments (line 905)
    kwargs_133703 = {}
    # Getting the type of 'len' (line 905)
    len_133701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 905, 18), 'len', False)
    # Calling len(args, kwargs) (line 905)
    len_call_result_133704 = invoke(stypy.reporting.localization.Localization(__file__, 905, 18), len_133701, *[ndtype_133702], **kwargs_133703)
    
    # Assigning a type to the variable 'nbtypes' (line 905)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 905, 8), 'nbtypes', len_call_result_133704)
    
    # Type idiom detected: calculating its left and rigth part (line 907)
    # Getting the type of 'names' (line 907)
    names_133705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 907, 8), 'names')
    # Getting the type of 'None' (line 907)
    None_133706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 907, 24), 'None')
    
    (may_be_133707, more_types_in_union_133708) = may_not_be_none(names_133705, None_133706)

    if may_be_133707:

        if more_types_in_union_133708:
            # Runtime conditional SSA (line 907)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 908):
        
        # Assigning a Call to a Name (line 908):
        
        # Call to NameValidator(...): (line 908)
        # Processing the call keyword arguments (line 908)
        # Getting the type of 'validationargs' (line 908)
        validationargs_133710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 908, 39), 'validationargs', False)
        kwargs_133711 = {'validationargs_133710': validationargs_133710}
        # Getting the type of 'NameValidator' (line 908)
        NameValidator_133709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 908, 23), 'NameValidator', False)
        # Calling NameValidator(args, kwargs) (line 908)
        NameValidator_call_result_133712 = invoke(stypy.reporting.localization.Localization(__file__, 908, 23), NameValidator_133709, *[], **kwargs_133711)
        
        # Assigning a type to the variable 'validate' (line 908)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 908, 12), 'validate', NameValidator_call_result_133712)
        
        # Type idiom detected: calculating its left and rigth part (line 909)
        # Getting the type of 'basestring' (line 909)
        basestring_133713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 909, 33), 'basestring')
        # Getting the type of 'names' (line 909)
        names_133714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 909, 26), 'names')
        
        (may_be_133715, more_types_in_union_133716) = may_be_subtype(basestring_133713, names_133714)

        if may_be_133715:

            if more_types_in_union_133716:
                # Runtime conditional SSA (line 909)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'names' (line 909)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 909, 12), 'names', remove_not_subtype_from_union(names_133714, basestring))
            
            # Assigning a Call to a Name (line 910):
            
            # Assigning a Call to a Name (line 910):
            
            # Call to split(...): (line 910)
            # Processing the call arguments (line 910)
            str_133719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 910, 36), 'str', ',')
            # Processing the call keyword arguments (line 910)
            kwargs_133720 = {}
            # Getting the type of 'names' (line 910)
            names_133717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 910, 24), 'names', False)
            # Obtaining the member 'split' of a type (line 910)
            split_133718 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 910, 24), names_133717, 'split')
            # Calling split(args, kwargs) (line 910)
            split_call_result_133721 = invoke(stypy.reporting.localization.Localization(__file__, 910, 24), split_133718, *[str_133719], **kwargs_133720)
            
            # Assigning a type to the variable 'names' (line 910)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 910, 16), 'names', split_call_result_133721)

            if more_types_in_union_133716:
                # SSA join for if statement (line 909)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        # Getting the type of 'nbtypes' (line 912)
        nbtypes_133722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 912, 15), 'nbtypes')
        int_133723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 912, 26), 'int')
        # Applying the binary operator '==' (line 912)
        result_eq_133724 = python_operator(stypy.reporting.localization.Localization(__file__, 912, 15), '==', nbtypes_133722, int_133723)
        
        # Testing the type of an if condition (line 912)
        if_condition_133725 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 912, 12), result_eq_133724)
        # Assigning a type to the variable 'if_condition_133725' (line 912)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 912, 12), 'if_condition_133725', if_condition_133725)
        # SSA begins for if statement (line 912)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 913):
        
        # Assigning a Call to a Name (line 913):
        
        # Call to tuple(...): (line 913)
        # Processing the call arguments (line 913)
        
        # Obtaining an instance of the builtin type 'list' (line 913)
        list_133727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 913, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 913)
        # Adding element type (line 913)
        # Getting the type of 'ndtype' (line 913)
        ndtype_133728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 913, 33), 'ndtype', False)
        # Obtaining the member 'type' of a type (line 913)
        type_133729 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 913, 33), ndtype_133728, 'type')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 913, 32), list_133727, type_133729)
        
        
        # Call to len(...): (line 913)
        # Processing the call arguments (line 913)
        # Getting the type of 'names' (line 913)
        names_133731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 913, 52), 'names', False)
        # Processing the call keyword arguments (line 913)
        kwargs_133732 = {}
        # Getting the type of 'len' (line 913)
        len_133730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 913, 48), 'len', False)
        # Calling len(args, kwargs) (line 913)
        len_call_result_133733 = invoke(stypy.reporting.localization.Localization(__file__, 913, 48), len_133730, *[names_133731], **kwargs_133732)
        
        # Applying the binary operator '*' (line 913)
        result_mul_133734 = python_operator(stypy.reporting.localization.Localization(__file__, 913, 32), '*', list_133727, len_call_result_133733)
        
        # Processing the call keyword arguments (line 913)
        kwargs_133735 = {}
        # Getting the type of 'tuple' (line 913)
        tuple_133726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 913, 26), 'tuple', False)
        # Calling tuple(args, kwargs) (line 913)
        tuple_call_result_133736 = invoke(stypy.reporting.localization.Localization(__file__, 913, 26), tuple_133726, *[result_mul_133734], **kwargs_133735)
        
        # Assigning a type to the variable 'formats' (line 913)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 913, 16), 'formats', tuple_call_result_133736)
        
        # Assigning a Call to a Name (line 914):
        
        # Assigning a Call to a Name (line 914):
        
        # Call to validate(...): (line 914)
        # Processing the call arguments (line 914)
        # Getting the type of 'names' (line 914)
        names_133738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 914, 33), 'names', False)
        # Processing the call keyword arguments (line 914)
        # Getting the type of 'defaultfmt' (line 914)
        defaultfmt_133739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 914, 51), 'defaultfmt', False)
        keyword_133740 = defaultfmt_133739
        kwargs_133741 = {'defaultfmt': keyword_133740}
        # Getting the type of 'validate' (line 914)
        validate_133737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 914, 24), 'validate', False)
        # Calling validate(args, kwargs) (line 914)
        validate_call_result_133742 = invoke(stypy.reporting.localization.Localization(__file__, 914, 24), validate_133737, *[names_133738], **kwargs_133741)
        
        # Assigning a type to the variable 'names' (line 914)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 914, 16), 'names', validate_call_result_133742)
        
        # Assigning a Call to a Name (line 915):
        
        # Assigning a Call to a Name (line 915):
        
        # Call to dtype(...): (line 915)
        # Processing the call arguments (line 915)
        
        # Call to list(...): (line 915)
        # Processing the call arguments (line 915)
        
        # Call to zip(...): (line 915)
        # Processing the call arguments (line 915)
        # Getting the type of 'names' (line 915)
        names_133747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 915, 43), 'names', False)
        # Getting the type of 'formats' (line 915)
        formats_133748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 915, 50), 'formats', False)
        # Processing the call keyword arguments (line 915)
        kwargs_133749 = {}
        # Getting the type of 'zip' (line 915)
        zip_133746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 915, 39), 'zip', False)
        # Calling zip(args, kwargs) (line 915)
        zip_call_result_133750 = invoke(stypy.reporting.localization.Localization(__file__, 915, 39), zip_133746, *[names_133747, formats_133748], **kwargs_133749)
        
        # Processing the call keyword arguments (line 915)
        kwargs_133751 = {}
        # Getting the type of 'list' (line 915)
        list_133745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 915, 34), 'list', False)
        # Calling list(args, kwargs) (line 915)
        list_call_result_133752 = invoke(stypy.reporting.localization.Localization(__file__, 915, 34), list_133745, *[zip_call_result_133750], **kwargs_133751)
        
        # Processing the call keyword arguments (line 915)
        kwargs_133753 = {}
        # Getting the type of 'np' (line 915)
        np_133743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 915, 25), 'np', False)
        # Obtaining the member 'dtype' of a type (line 915)
        dtype_133744 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 915, 25), np_133743, 'dtype')
        # Calling dtype(args, kwargs) (line 915)
        dtype_call_result_133754 = invoke(stypy.reporting.localization.Localization(__file__, 915, 25), dtype_133744, *[list_call_result_133752], **kwargs_133753)
        
        # Assigning a type to the variable 'ndtype' (line 915)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 915, 16), 'ndtype', dtype_call_result_133754)
        # SSA branch for the else part of an if statement (line 912)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Attribute (line 918):
        
        # Assigning a Call to a Attribute (line 918):
        
        # Call to validate(...): (line 918)
        # Processing the call arguments (line 918)
        # Getting the type of 'names' (line 918)
        names_133756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 918, 40), 'names', False)
        # Processing the call keyword arguments (line 918)
        # Getting the type of 'nbtypes' (line 918)
        nbtypes_133757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 918, 56), 'nbtypes', False)
        keyword_133758 = nbtypes_133757
        # Getting the type of 'defaultfmt' (line 919)
        defaultfmt_133759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 919, 51), 'defaultfmt', False)
        keyword_133760 = defaultfmt_133759
        kwargs_133761 = {'defaultfmt': keyword_133760, 'nbfields': keyword_133758}
        # Getting the type of 'validate' (line 918)
        validate_133755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 918, 31), 'validate', False)
        # Calling validate(args, kwargs) (line 918)
        validate_call_result_133762 = invoke(stypy.reporting.localization.Localization(__file__, 918, 31), validate_133755, *[names_133756], **kwargs_133761)
        
        # Getting the type of 'ndtype' (line 918)
        ndtype_133763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 918, 16), 'ndtype')
        # Setting the type of the member 'names' of a type (line 918)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 918, 16), ndtype_133763, 'names', validate_call_result_133762)
        # SSA join for if statement (line 912)
        module_type_store = module_type_store.join_ssa_context()
        

        if more_types_in_union_133708:
            # Runtime conditional SSA for else branch (line 907)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_133707) or more_types_in_union_133708):
        
        
        # Getting the type of 'nbtypes' (line 921)
        nbtypes_133764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 921, 14), 'nbtypes')
        int_133765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 921, 24), 'int')
        # Applying the binary operator '>' (line 921)
        result_gt_133766 = python_operator(stypy.reporting.localization.Localization(__file__, 921, 14), '>', nbtypes_133764, int_133765)
        
        # Testing the type of an if condition (line 921)
        if_condition_133767 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 921, 13), result_gt_133766)
        # Assigning a type to the variable 'if_condition_133767' (line 921)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 921, 13), 'if_condition_133767', if_condition_133767)
        # SSA begins for if statement (line 921)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 922):
        
        # Assigning a Call to a Name (line 922):
        
        # Call to NameValidator(...): (line 922)
        # Processing the call keyword arguments (line 922)
        # Getting the type of 'validationargs' (line 922)
        validationargs_133769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 922, 39), 'validationargs', False)
        kwargs_133770 = {'validationargs_133769': validationargs_133769}
        # Getting the type of 'NameValidator' (line 922)
        NameValidator_133768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 922, 23), 'NameValidator', False)
        # Calling NameValidator(args, kwargs) (line 922)
        NameValidator_call_result_133771 = invoke(stypy.reporting.localization.Localization(__file__, 922, 23), NameValidator_133768, *[], **kwargs_133770)
        
        # Assigning a type to the variable 'validate' (line 922)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 922, 12), 'validate', NameValidator_call_result_133771)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'ndtype' (line 924)
        ndtype_133772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 924, 17), 'ndtype')
        # Obtaining the member 'names' of a type (line 924)
        names_133773 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 924, 17), ndtype_133772, 'names')
        
        # Call to tuple(...): (line 924)
        # Processing the call arguments (line 924)
        # Calculating generator expression
        module_type_store = module_type_store.open_function_context('list comprehension expression', 924, 39, True)
        # Calculating comprehension expression
        
        # Call to range(...): (line 924)
        # Processing the call arguments (line 924)
        # Getting the type of 'nbtypes' (line 924)
        nbtypes_133779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 924, 64), 'nbtypes', False)
        # Processing the call keyword arguments (line 924)
        kwargs_133780 = {}
        # Getting the type of 'range' (line 924)
        range_133778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 924, 58), 'range', False)
        # Calling range(args, kwargs) (line 924)
        range_call_result_133781 = invoke(stypy.reporting.localization.Localization(__file__, 924, 58), range_133778, *[nbtypes_133779], **kwargs_133780)
        
        comprehension_133782 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 924, 39), range_call_result_133781)
        # Assigning a type to the variable 'i' (line 924)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 924, 39), 'i', comprehension_133782)
        str_133775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 924, 39), 'str', 'f%i')
        # Getting the type of 'i' (line 924)
        i_133776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 924, 47), 'i', False)
        # Applying the binary operator '%' (line 924)
        result_mod_133777 = python_operator(stypy.reporting.localization.Localization(__file__, 924, 39), '%', str_133775, i_133776)
        
        list_133783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 924, 39), 'list')
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 924, 39), list_133783, result_mod_133777)
        # Processing the call keyword arguments (line 924)
        kwargs_133784 = {}
        # Getting the type of 'tuple' (line 924)
        tuple_133774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 924, 33), 'tuple', False)
        # Calling tuple(args, kwargs) (line 924)
        tuple_call_result_133785 = invoke(stypy.reporting.localization.Localization(__file__, 924, 33), tuple_133774, *[list_133783], **kwargs_133784)
        
        # Applying the binary operator '==' (line 924)
        result_eq_133786 = python_operator(stypy.reporting.localization.Localization(__file__, 924, 17), '==', names_133773, tuple_call_result_133785)
        
        
        # Getting the type of 'defaultfmt' (line 925)
        defaultfmt_133787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 925, 21), 'defaultfmt')
        str_133788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 925, 35), 'str', 'f%i')
        # Applying the binary operator '!=' (line 925)
        result_ne_133789 = python_operator(stypy.reporting.localization.Localization(__file__, 925, 21), '!=', defaultfmt_133787, str_133788)
        
        # Applying the binary operator 'and' (line 924)
        result_and_keyword_133790 = python_operator(stypy.reporting.localization.Localization(__file__, 924, 16), 'and', result_eq_133786, result_ne_133789)
        
        # Testing the type of an if condition (line 924)
        if_condition_133791 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 924, 12), result_and_keyword_133790)
        # Assigning a type to the variable 'if_condition_133791' (line 924)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 924, 12), 'if_condition_133791', if_condition_133791)
        # SSA begins for if statement (line 924)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Attribute (line 926):
        
        # Assigning a Call to a Attribute (line 926):
        
        # Call to validate(...): (line 926)
        # Processing the call arguments (line 926)
        
        # Obtaining an instance of the builtin type 'list' (line 926)
        list_133793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 926, 40), 'list')
        # Adding type elements to the builtin type 'list' instance (line 926)
        # Adding element type (line 926)
        str_133794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 926, 41), 'str', '')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 926, 40), list_133793, str_133794)
        
        # Getting the type of 'nbtypes' (line 926)
        nbtypes_133795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 926, 47), 'nbtypes', False)
        # Applying the binary operator '*' (line 926)
        result_mul_133796 = python_operator(stypy.reporting.localization.Localization(__file__, 926, 40), '*', list_133793, nbtypes_133795)
        
        # Processing the call keyword arguments (line 926)
        # Getting the type of 'defaultfmt' (line 926)
        defaultfmt_133797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 926, 67), 'defaultfmt', False)
        keyword_133798 = defaultfmt_133797
        kwargs_133799 = {'defaultfmt': keyword_133798}
        # Getting the type of 'validate' (line 926)
        validate_133792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 926, 31), 'validate', False)
        # Calling validate(args, kwargs) (line 926)
        validate_call_result_133800 = invoke(stypy.reporting.localization.Localization(__file__, 926, 31), validate_133792, *[result_mul_133796], **kwargs_133799)
        
        # Getting the type of 'ndtype' (line 926)
        ndtype_133801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 926, 16), 'ndtype')
        # Setting the type of the member 'names' of a type (line 926)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 926, 16), ndtype_133801, 'names', validate_call_result_133800)
        # SSA branch for the else part of an if statement (line 924)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Attribute (line 929):
        
        # Assigning a Call to a Attribute (line 929):
        
        # Call to validate(...): (line 929)
        # Processing the call arguments (line 929)
        # Getting the type of 'ndtype' (line 929)
        ndtype_133803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 929, 40), 'ndtype', False)
        # Obtaining the member 'names' of a type (line 929)
        names_133804 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 929, 40), ndtype_133803, 'names')
        # Processing the call keyword arguments (line 929)
        # Getting the type of 'defaultfmt' (line 929)
        defaultfmt_133805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 929, 65), 'defaultfmt', False)
        keyword_133806 = defaultfmt_133805
        kwargs_133807 = {'defaultfmt': keyword_133806}
        # Getting the type of 'validate' (line 929)
        validate_133802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 929, 31), 'validate', False)
        # Calling validate(args, kwargs) (line 929)
        validate_call_result_133808 = invoke(stypy.reporting.localization.Localization(__file__, 929, 31), validate_133802, *[names_133804], **kwargs_133807)
        
        # Getting the type of 'ndtype' (line 929)
        ndtype_133809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 929, 16), 'ndtype')
        # Setting the type of the member 'names' of a type (line 929)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 929, 16), ndtype_133809, 'names', validate_call_result_133808)
        # SSA join for if statement (line 924)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 921)
        module_type_store = module_type_store.join_ssa_context()
        

        if (may_be_133707 and more_types_in_union_133708):
            # SSA join for if statement (line 907)
            module_type_store = module_type_store.join_ssa_context()


    
    # SSA join for try-except statement (line 893)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'ndtype' (line 930)
    ndtype_133810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 930, 11), 'ndtype')
    # Assigning a type to the variable 'stypy_return_type' (line 930)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 930, 4), 'stypy_return_type', ndtype_133810)
    
    # ################# End of 'easy_dtype(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'easy_dtype' in the type store
    # Getting the type of 'stypy_return_type' (line 855)
    stypy_return_type_133811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 855, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_133811)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'easy_dtype'
    return stypy_return_type_133811

# Assigning a type to the variable 'easy_dtype' (line 855)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 855, 0), 'easy_dtype', easy_dtype)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
