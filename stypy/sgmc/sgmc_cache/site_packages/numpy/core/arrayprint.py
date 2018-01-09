
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''Array printing function
2: 
3: $Id: arrayprint.py,v 1.9 2005/09/13 13:58:44 teoliphant Exp $
4: 
5: '''
6: from __future__ import division, absolute_import, print_function
7: 
8: __all__ = ["array2string", "set_printoptions", "get_printoptions"]
9: __docformat__ = 'restructuredtext'
10: 
11: #
12: # Written by Konrad Hinsen <hinsenk@ere.umontreal.ca>
13: # last revision: 1996-3-13
14: # modified by Jim Hugunin 1997-3-3 for repr's and str's (and other details)
15: # and by Perry Greenfield 2000-4-1 for numarray
16: # and by Travis Oliphant  2005-8-22 for numpy
17: 
18: import sys
19: from functools import reduce
20: from . import numerictypes as _nt
21: from .umath import maximum, minimum, absolute, not_equal, isnan, isinf
22: from .multiarray import (array, format_longfloat, datetime_as_string,
23:                          datetime_data)
24: from .fromnumeric import ravel
25: from .numeric import asarray
26: 
27: if sys.version_info[0] >= 3:
28:     _MAXINT = sys.maxsize
29:     _MININT = -sys.maxsize - 1
30: else:
31:     _MAXINT = sys.maxint
32:     _MININT = -sys.maxint - 1
33: 
34: def product(x, y):
35:     return x*y
36: 
37: _summaryEdgeItems = 3     # repr N leading and trailing items of each dimension
38: _summaryThreshold = 1000  # total items > triggers array summarization
39: 
40: _float_output_precision = 8
41: _float_output_suppress_small = False
42: _line_width = 75
43: _nan_str = 'nan'
44: _inf_str = 'inf'
45: _formatter = None  # formatting function for array elements
46: 
47: 
48: def set_printoptions(precision=None, threshold=None, edgeitems=None,
49:                      linewidth=None, suppress=None,
50:                      nanstr=None, infstr=None,
51:                      formatter=None):
52:     '''
53:     Set printing options.
54: 
55:     These options determine the way floating point numbers, arrays and
56:     other NumPy objects are displayed.
57: 
58:     Parameters
59:     ----------
60:     precision : int, optional
61:         Number of digits of precision for floating point output (default 8).
62:     threshold : int, optional
63:         Total number of array elements which trigger summarization
64:         rather than full repr (default 1000).
65:     edgeitems : int, optional
66:         Number of array items in summary at beginning and end of
67:         each dimension (default 3).
68:     linewidth : int, optional
69:         The number of characters per line for the purpose of inserting
70:         line breaks (default 75).
71:     suppress : bool, optional
72:         Whether or not suppress printing of small floating point values
73:         using scientific notation (default False).
74:     nanstr : str, optional
75:         String representation of floating point not-a-number (default nan).
76:     infstr : str, optional
77:         String representation of floating point infinity (default inf).
78:     formatter : dict of callables, optional
79:         If not None, the keys should indicate the type(s) that the respective
80:         formatting function applies to.  Callables should return a string.
81:         Types that are not specified (by their corresponding keys) are handled
82:         by the default formatters.  Individual types for which a formatter
83:         can be set are::
84: 
85:             - 'bool'
86:             - 'int'
87:             - 'timedelta' : a `numpy.timedelta64`
88:             - 'datetime' : a `numpy.datetime64`
89:             - 'float'
90:             - 'longfloat' : 128-bit floats
91:             - 'complexfloat'
92:             - 'longcomplexfloat' : composed of two 128-bit floats
93:             - 'numpy_str' : types `numpy.string_` and `numpy.unicode_`
94:             - 'str' : all other strings
95: 
96:         Other keys that can be used to set a group of types at once are::
97: 
98:             - 'all' : sets all types
99:             - 'int_kind' : sets 'int'
100:             - 'float_kind' : sets 'float' and 'longfloat'
101:             - 'complex_kind' : sets 'complexfloat' and 'longcomplexfloat'
102:             - 'str_kind' : sets 'str' and 'numpystr'
103: 
104:     See Also
105:     --------
106:     get_printoptions, set_string_function, array2string
107: 
108:     Notes
109:     -----
110:     `formatter` is always reset with a call to `set_printoptions`.
111: 
112:     Examples
113:     --------
114:     Floating point precision can be set:
115: 
116:     >>> np.set_printoptions(precision=4)
117:     >>> print(np.array([1.123456789]))
118:     [ 1.1235]
119: 
120:     Long arrays can be summarised:
121: 
122:     >>> np.set_printoptions(threshold=5)
123:     >>> print(np.arange(10))
124:     [0 1 2 ..., 7 8 9]
125: 
126:     Small results can be suppressed:
127: 
128:     >>> eps = np.finfo(float).eps
129:     >>> x = np.arange(4.)
130:     >>> x**2 - (x + eps)**2
131:     array([ -4.9304e-32,  -4.4409e-16,   0.0000e+00,   0.0000e+00])
132:     >>> np.set_printoptions(suppress=True)
133:     >>> x**2 - (x + eps)**2
134:     array([-0., -0.,  0.,  0.])
135: 
136:     A custom formatter can be used to display array elements as desired:
137: 
138:     >>> np.set_printoptions(formatter={'all':lambda x: 'int: '+str(-x)})
139:     >>> x = np.arange(3)
140:     >>> x
141:     array([int: 0, int: -1, int: -2])
142:     >>> np.set_printoptions()  # formatter gets reset
143:     >>> x
144:     array([0, 1, 2])
145: 
146:     To put back the default options, you can use:
147: 
148:     >>> np.set_printoptions(edgeitems=3,infstr='inf',
149:     ... linewidth=75, nanstr='nan', precision=8,
150:     ... suppress=False, threshold=1000, formatter=None)
151:     '''
152: 
153:     global _summaryThreshold, _summaryEdgeItems, _float_output_precision
154:     global _line_width, _float_output_suppress_small, _nan_str, _inf_str
155:     global _formatter
156: 
157:     if linewidth is not None:
158:         _line_width = linewidth
159:     if threshold is not None:
160:         _summaryThreshold = threshold
161:     if edgeitems is not None:
162:         _summaryEdgeItems = edgeitems
163:     if precision is not None:
164:         _float_output_precision = precision
165:     if suppress is not None:
166:         _float_output_suppress_small = not not suppress
167:     if nanstr is not None:
168:         _nan_str = nanstr
169:     if infstr is not None:
170:         _inf_str = infstr
171:     _formatter = formatter
172: 
173: def get_printoptions():
174:     '''
175:     Return the current print options.
176: 
177:     Returns
178:     -------
179:     print_opts : dict
180:         Dictionary of current print options with keys
181: 
182:           - precision : int
183:           - threshold : int
184:           - edgeitems : int
185:           - linewidth : int
186:           - suppress : bool
187:           - nanstr : str
188:           - infstr : str
189:           - formatter : dict of callables
190: 
191:         For a full description of these options, see `set_printoptions`.
192: 
193:     See Also
194:     --------
195:     set_printoptions, set_string_function
196: 
197:     '''
198:     d = dict(precision=_float_output_precision,
199:              threshold=_summaryThreshold,
200:              edgeitems=_summaryEdgeItems,
201:              linewidth=_line_width,
202:              suppress=_float_output_suppress_small,
203:              nanstr=_nan_str,
204:              infstr=_inf_str,
205:              formatter=_formatter)
206:     return d
207: 
208: def _leading_trailing(a):
209:     from . import numeric as _nc
210:     if a.ndim == 1:
211:         if len(a) > 2*_summaryEdgeItems:
212:             b = _nc.concatenate((a[:_summaryEdgeItems],
213:                                      a[-_summaryEdgeItems:]))
214:         else:
215:             b = a
216:     else:
217:         if len(a) > 2*_summaryEdgeItems:
218:             l = [_leading_trailing(a[i]) for i in range(
219:                 min(len(a), _summaryEdgeItems))]
220:             l.extend([_leading_trailing(a[-i]) for i in range(
221:                 min(len(a), _summaryEdgeItems), 0, -1)])
222:         else:
223:             l = [_leading_trailing(a[i]) for i in range(0, len(a))]
224:         b = _nc.concatenate(tuple(l))
225:     return b
226: 
227: def _boolFormatter(x):
228:     if x:
229:         return ' True'
230:     else:
231:         return 'False'
232: 
233: 
234: def repr_format(x):
235:     return repr(x)
236: 
237: def _array2string(a, max_line_width, precision, suppress_small, separator=' ',
238:                   prefix="", formatter=None):
239: 
240:     if max_line_width is None:
241:         max_line_width = _line_width
242: 
243:     if precision is None:
244:         precision = _float_output_precision
245: 
246:     if suppress_small is None:
247:         suppress_small = _float_output_suppress_small
248: 
249:     if formatter is None:
250:         formatter = _formatter
251: 
252:     if a.size > _summaryThreshold:
253:         summary_insert = "..., "
254:         data = _leading_trailing(a)
255:     else:
256:         summary_insert = ""
257:         data = ravel(asarray(a))
258: 
259:     formatdict = {'bool': _boolFormatter,
260:                   'int': IntegerFormat(data),
261:                   'float': FloatFormat(data, precision, suppress_small),
262:                   'longfloat': LongFloatFormat(precision),
263:                   'complexfloat': ComplexFormat(data, precision,
264:                                                  suppress_small),
265:                   'longcomplexfloat': LongComplexFormat(precision),
266:                   'datetime': DatetimeFormat(data),
267:                   'timedelta': TimedeltaFormat(data),
268:                   'numpystr': repr_format,
269:                   'str': str}
270: 
271:     if formatter is not None:
272:         fkeys = [k for k in formatter.keys() if formatter[k] is not None]
273:         if 'all' in fkeys:
274:             for key in formatdict.keys():
275:                 formatdict[key] = formatter['all']
276:         if 'int_kind' in fkeys:
277:             for key in ['int']:
278:                 formatdict[key] = formatter['int_kind']
279:         if 'float_kind' in fkeys:
280:             for key in ['float', 'longfloat']:
281:                 formatdict[key] = formatter['float_kind']
282:         if 'complex_kind' in fkeys:
283:             for key in ['complexfloat', 'longcomplexfloat']:
284:                 formatdict[key] = formatter['complex_kind']
285:         if 'str_kind' in fkeys:
286:             for key in ['numpystr', 'str']:
287:                 formatdict[key] = formatter['str_kind']
288:         for key in formatdict.keys():
289:             if key in fkeys:
290:                 formatdict[key] = formatter[key]
291: 
292:     # find the right formatting function for the array
293:     dtypeobj = a.dtype.type
294:     if issubclass(dtypeobj, _nt.bool_):
295:         format_function = formatdict['bool']
296:     elif issubclass(dtypeobj, _nt.integer):
297:         if issubclass(dtypeobj, _nt.timedelta64):
298:             format_function = formatdict['timedelta']
299:         else:
300:             format_function = formatdict['int']
301:     elif issubclass(dtypeobj, _nt.floating):
302:         if issubclass(dtypeobj, _nt.longfloat):
303:             format_function = formatdict['longfloat']
304:         else:
305:             format_function = formatdict['float']
306:     elif issubclass(dtypeobj, _nt.complexfloating):
307:         if issubclass(dtypeobj, _nt.clongfloat):
308:             format_function = formatdict['longcomplexfloat']
309:         else:
310:             format_function = formatdict['complexfloat']
311:     elif issubclass(dtypeobj, (_nt.unicode_, _nt.string_)):
312:         format_function = formatdict['numpystr']
313:     elif issubclass(dtypeobj, _nt.datetime64):
314:         format_function = formatdict['datetime']
315:     else:
316:         format_function = formatdict['numpystr']
317: 
318:     # skip over "["
319:     next_line_prefix = " "
320:     # skip over array(
321:     next_line_prefix += " "*len(prefix)
322: 
323:     lst = _formatArray(a, format_function, len(a.shape), max_line_width,
324:                        next_line_prefix, separator,
325:                        _summaryEdgeItems, summary_insert)[:-1]
326:     return lst
327: 
328: def _convert_arrays(obj):
329:     from . import numeric as _nc
330:     newtup = []
331:     for k in obj:
332:         if isinstance(k, _nc.ndarray):
333:             k = k.tolist()
334:         elif isinstance(k, tuple):
335:             k = _convert_arrays(k)
336:         newtup.append(k)
337:     return tuple(newtup)
338: 
339: 
340: def array2string(a, max_line_width=None, precision=None,
341:                  suppress_small=None, separator=' ', prefix="",
342:                  style=repr, formatter=None):
343:     '''
344:     Return a string representation of an array.
345: 
346:     Parameters
347:     ----------
348:     a : ndarray
349:         Input array.
350:     max_line_width : int, optional
351:         The maximum number of columns the string should span. Newline
352:         characters splits the string appropriately after array elements.
353:     precision : int, optional
354:         Floating point precision. Default is the current printing
355:         precision (usually 8), which can be altered using `set_printoptions`.
356:     suppress_small : bool, optional
357:         Represent very small numbers as zero. A number is "very small" if it
358:         is smaller than the current printing precision.
359:     separator : str, optional
360:         Inserted between elements.
361:     prefix : str, optional
362:         An array is typically printed as::
363: 
364:           'prefix(' + array2string(a) + ')'
365: 
366:         The length of the prefix string is used to align the
367:         output correctly.
368:     style : function, optional
369:         A function that accepts an ndarray and returns a string.  Used only
370:         when the shape of `a` is equal to ``()``, i.e. for 0-D arrays.
371:     formatter : dict of callables, optional
372:         If not None, the keys should indicate the type(s) that the respective
373:         formatting function applies to.  Callables should return a string.
374:         Types that are not specified (by their corresponding keys) are handled
375:         by the default formatters.  Individual types for which a formatter
376:         can be set are::
377: 
378:             - 'bool'
379:             - 'int'
380:             - 'timedelta' : a `numpy.timedelta64`
381:             - 'datetime' : a `numpy.datetime64`
382:             - 'float'
383:             - 'longfloat' : 128-bit floats
384:             - 'complexfloat'
385:             - 'longcomplexfloat' : composed of two 128-bit floats
386:             - 'numpy_str' : types `numpy.string_` and `numpy.unicode_`
387:             - 'str' : all other strings
388: 
389:         Other keys that can be used to set a group of types at once are::
390: 
391:             - 'all' : sets all types
392:             - 'int_kind' : sets 'int'
393:             - 'float_kind' : sets 'float' and 'longfloat'
394:             - 'complex_kind' : sets 'complexfloat' and 'longcomplexfloat'
395:             - 'str_kind' : sets 'str' and 'numpystr'
396: 
397:     Returns
398:     -------
399:     array_str : str
400:         String representation of the array.
401: 
402:     Raises
403:     ------
404:     TypeError
405:         if a callable in `formatter` does not return a string.
406: 
407:     See Also
408:     --------
409:     array_str, array_repr, set_printoptions, get_printoptions
410: 
411:     Notes
412:     -----
413:     If a formatter is specified for a certain type, the `precision` keyword is
414:     ignored for that type.
415: 
416:     This is a very flexible function; `array_repr` and `array_str` are using
417:     `array2string` internally so keywords with the same name should work
418:     identically in all three functions.
419: 
420:     Examples
421:     --------
422:     >>> x = np.array([1e-16,1,2,3])
423:     >>> print(np.array2string(x, precision=2, separator=',',
424:     ...                       suppress_small=True))
425:     [ 0., 1., 2., 3.]
426: 
427:     >>> x  = np.arange(3.)
428:     >>> np.array2string(x, formatter={'float_kind':lambda x: "%.2f" % x})
429:     '[0.00 1.00 2.00]'
430: 
431:     >>> x  = np.arange(3)
432:     >>> np.array2string(x, formatter={'int':lambda x: hex(x)})
433:     '[0x0L 0x1L 0x2L]'
434: 
435:     '''
436: 
437:     if a.shape == ():
438:         x = a.item()
439:         if isinstance(x, tuple):
440:             x = _convert_arrays(x)
441:         lst = style(x)
442:     elif reduce(product, a.shape) == 0:
443:         # treat as a null array if any of shape elements == 0
444:         lst = "[]"
445:     else:
446:         lst = _array2string(a, max_line_width, precision, suppress_small,
447:                             separator, prefix, formatter=formatter)
448:     return lst
449: 
450: def _extendLine(s, line, word, max_line_len, next_line_prefix):
451:     if len(line.rstrip()) + len(word.rstrip()) >= max_line_len:
452:         s += line.rstrip() + "\n"
453:         line = next_line_prefix
454:     line += word
455:     return s, line
456: 
457: 
458: def _formatArray(a, format_function, rank, max_line_len,
459:                  next_line_prefix, separator, edge_items, summary_insert):
460:     '''formatArray is designed for two modes of operation:
461: 
462:     1. Full output
463: 
464:     2. Summarized output
465: 
466:     '''
467:     if rank == 0:
468:         obj = a.item()
469:         if isinstance(obj, tuple):
470:             obj = _convert_arrays(obj)
471:         return str(obj)
472: 
473:     if summary_insert and 2*edge_items < len(a):
474:         leading_items = edge_items
475:         trailing_items = edge_items
476:         summary_insert1 = summary_insert
477:     else:
478:         leading_items = 0
479:         trailing_items = len(a)
480:         summary_insert1 = ""
481: 
482:     if rank == 1:
483:         s = ""
484:         line = next_line_prefix
485:         for i in range(leading_items):
486:             word = format_function(a[i]) + separator
487:             s, line = _extendLine(s, line, word, max_line_len, next_line_prefix)
488: 
489:         if summary_insert1:
490:             s, line = _extendLine(s, line, summary_insert1, max_line_len, next_line_prefix)
491: 
492:         for i in range(trailing_items, 1, -1):
493:             word = format_function(a[-i]) + separator
494:             s, line = _extendLine(s, line, word, max_line_len, next_line_prefix)
495: 
496:         word = format_function(a[-1])
497:         s, line = _extendLine(s, line, word, max_line_len, next_line_prefix)
498:         s += line + "]\n"
499:         s = '[' + s[len(next_line_prefix):]
500:     else:
501:         s = '['
502:         sep = separator.rstrip()
503:         for i in range(leading_items):
504:             if i > 0:
505:                 s += next_line_prefix
506:             s += _formatArray(a[i], format_function, rank-1, max_line_len,
507:                               " " + next_line_prefix, separator, edge_items,
508:                               summary_insert)
509:             s = s.rstrip() + sep.rstrip() + '\n'*max(rank-1, 1)
510: 
511:         if summary_insert1:
512:             s += next_line_prefix + summary_insert1 + "\n"
513: 
514:         for i in range(trailing_items, 1, -1):
515:             if leading_items or i != trailing_items:
516:                 s += next_line_prefix
517:             s += _formatArray(a[-i], format_function, rank-1, max_line_len,
518:                               " " + next_line_prefix, separator, edge_items,
519:                               summary_insert)
520:             s = s.rstrip() + sep.rstrip() + '\n'*max(rank-1, 1)
521:         if leading_items or trailing_items > 1:
522:             s += next_line_prefix
523:         s += _formatArray(a[-1], format_function, rank-1, max_line_len,
524:                           " " + next_line_prefix, separator, edge_items,
525:                           summary_insert).rstrip()+']\n'
526:     return s
527: 
528: class FloatFormat(object):
529:     def __init__(self, data, precision, suppress_small, sign=False):
530:         self.precision = precision
531:         self.suppress_small = suppress_small
532:         self.sign = sign
533:         self.exp_format = False
534:         self.large_exponent = False
535:         self.max_str_len = 0
536:         try:
537:             self.fillFormat(data)
538:         except (TypeError, NotImplementedError):
539:             # if reduce(data) fails, this instance will not be called, just
540:             # instantiated in formatdict.
541:             pass
542: 
543:     def fillFormat(self, data):
544:         from . import numeric as _nc
545: 
546:         with _nc.errstate(all='ignore'):
547:             special = isnan(data) | isinf(data)
548:             valid = not_equal(data, 0) & ~special
549:             non_zero = absolute(data.compress(valid))
550:             if len(non_zero) == 0:
551:                 max_val = 0.
552:                 min_val = 0.
553:             else:
554:                 max_val = maximum.reduce(non_zero)
555:                 min_val = minimum.reduce(non_zero)
556:                 if max_val >= 1.e8:
557:                     self.exp_format = True
558:                 if not self.suppress_small and (min_val < 0.0001
559:                                            or max_val/min_val > 1000.):
560:                     self.exp_format = True
561: 
562:         if self.exp_format:
563:             self.large_exponent = 0 < min_val < 1e-99 or max_val >= 1e100
564:             self.max_str_len = 8 + self.precision
565:             if self.large_exponent:
566:                 self.max_str_len += 1
567:             if self.sign:
568:                 format = '%+'
569:             else:
570:                 format = '%'
571:             format = format + '%d.%de' % (self.max_str_len, self.precision)
572:         else:
573:             format = '%%.%df' % (self.precision,)
574:             if len(non_zero):
575:                 precision = max([_digits(x, self.precision, format)
576:                                  for x in non_zero])
577:             else:
578:                 precision = 0
579:             precision = min(self.precision, precision)
580:             self.max_str_len = len(str(int(max_val))) + precision + 2
581:             if _nc.any(special):
582:                 self.max_str_len = max(self.max_str_len,
583:                                        len(_nan_str),
584:                                        len(_inf_str)+1)
585:             if self.sign:
586:                 format = '%#+'
587:             else:
588:                 format = '%#'
589:             format = format + '%d.%df' % (self.max_str_len, precision)
590: 
591:         self.special_fmt = '%%%ds' % (self.max_str_len,)
592:         self.format = format
593: 
594:     def __call__(self, x, strip_zeros=True):
595:         from . import numeric as _nc
596: 
597:         with _nc.errstate(invalid='ignore'):
598:             if isnan(x):
599:                 if self.sign:
600:                     return self.special_fmt % ('+' + _nan_str,)
601:                 else:
602:                     return self.special_fmt % (_nan_str,)
603:             elif isinf(x):
604:                 if x > 0:
605:                     if self.sign:
606:                         return self.special_fmt % ('+' + _inf_str,)
607:                     else:
608:                         return self.special_fmt % (_inf_str,)
609:                 else:
610:                     return self.special_fmt % ('-' + _inf_str,)
611: 
612:         s = self.format % x
613:         if self.large_exponent:
614:             # 3-digit exponent
615:             expsign = s[-3]
616:             if expsign == '+' or expsign == '-':
617:                 s = s[1:-2] + '0' + s[-2:]
618:         elif self.exp_format:
619:             # 2-digit exponent
620:             if s[-3] == '0':
621:                 s = ' ' + s[:-3] + s[-2:]
622:         elif strip_zeros:
623:             z = s.rstrip('0')
624:             s = z + ' '*(len(s)-len(z))
625:         return s
626: 
627: 
628: def _digits(x, precision, format):
629:     s = format % x
630:     z = s.rstrip('0')
631:     return precision - len(s) + len(z)
632: 
633: 
634: class IntegerFormat(object):
635:     def __init__(self, data):
636:         try:
637:             max_str_len = max(len(str(maximum.reduce(data))),
638:                               len(str(minimum.reduce(data))))
639:             self.format = '%' + str(max_str_len) + 'd'
640:         except (TypeError, NotImplementedError):
641:             # if reduce(data) fails, this instance will not be called, just
642:             # instantiated in formatdict.
643:             pass
644:         except ValueError:
645:             # this occurs when everything is NA
646:             pass
647: 
648:     def __call__(self, x):
649:         if _MININT < x < _MAXINT:
650:             return self.format % x
651:         else:
652:             return "%s" % x
653: 
654: class LongFloatFormat(object):
655:     # XXX Have to add something to determine the width to use a la FloatFormat
656:     # Right now, things won't line up properly
657:     def __init__(self, precision, sign=False):
658:         self.precision = precision
659:         self.sign = sign
660: 
661:     def __call__(self, x):
662:         if isnan(x):
663:             if self.sign:
664:                 return '+' + _nan_str
665:             else:
666:                 return ' ' + _nan_str
667:         elif isinf(x):
668:             if x > 0:
669:                 if self.sign:
670:                     return '+' + _inf_str
671:                 else:
672:                     return ' ' + _inf_str
673:             else:
674:                 return '-' + _inf_str
675:         elif x >= 0:
676:             if self.sign:
677:                 return '+' + format_longfloat(x, self.precision)
678:             else:
679:                 return ' ' + format_longfloat(x, self.precision)
680:         else:
681:             return format_longfloat(x, self.precision)
682: 
683: 
684: class LongComplexFormat(object):
685:     def __init__(self, precision):
686:         self.real_format = LongFloatFormat(precision)
687:         self.imag_format = LongFloatFormat(precision, sign=True)
688: 
689:     def __call__(self, x):
690:         r = self.real_format(x.real)
691:         i = self.imag_format(x.imag)
692:         return r + i + 'j'
693: 
694: 
695: class ComplexFormat(object):
696:     def __init__(self, x, precision, suppress_small):
697:         self.real_format = FloatFormat(x.real, precision, suppress_small)
698:         self.imag_format = FloatFormat(x.imag, precision, suppress_small,
699:                                        sign=True)
700: 
701:     def __call__(self, x):
702:         r = self.real_format(x.real, strip_zeros=False)
703:         i = self.imag_format(x.imag, strip_zeros=False)
704:         if not self.imag_format.exp_format:
705:             z = i.rstrip('0')
706:             i = z + 'j' + ' '*(len(i)-len(z))
707:         else:
708:             i = i + 'j'
709:         return r + i
710: 
711: 
712: class DatetimeFormat(object):
713:     def __init__(self, x, unit=None, timezone=None, casting='same_kind'):
714:         # Get the unit from the dtype
715:         if unit is None:
716:             if x.dtype.kind == 'M':
717:                 unit = datetime_data(x.dtype)[0]
718:             else:
719:                 unit = 's'
720: 
721:         if timezone is None:
722:             timezone = 'naive'
723:         self.timezone = timezone
724:         self.unit = unit
725:         self.casting = casting
726: 
727:     def __call__(self, x):
728:         return "'%s'" % datetime_as_string(x,
729:                                     unit=self.unit,
730:                                     timezone=self.timezone,
731:                                     casting=self.casting)
732: 
733: class TimedeltaFormat(object):
734:     def __init__(self, data):
735:         if data.dtype.kind == 'm':
736:             nat_value = array(['NaT'], dtype=data.dtype)[0]
737:             v = data[not_equal(data, nat_value)].view('i8')
738:             if len(v) > 0:
739:                 # Max str length of non-NaT elements
740:                 max_str_len = max(len(str(maximum.reduce(v))),
741:                                   len(str(minimum.reduce(v))))
742:             else:
743:                 max_str_len = 0
744:             if len(v) < len(data):
745:                 # data contains a NaT
746:                 max_str_len = max(max_str_len, 5)
747:             self.format = '%' + str(max_str_len) + 'd'
748:             self._nat = "'NaT'".rjust(max_str_len)
749: 
750:     def __call__(self, x):
751:         if x + 1 == x:
752:             return self._nat
753:         else:
754:             return self.format % x.astype('i8')
755: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, (-1)), 'str', 'Array printing function\n\n$Id: arrayprint.py,v 1.9 2005/09/13 13:58:44 teoliphant Exp $\n\n')

# Assigning a List to a Name (line 8):

# Assigning a List to a Name (line 8):
__all__ = ['array2string', 'set_printoptions', 'get_printoptions']
module_type_store.set_exportable_members(['array2string', 'set_printoptions', 'get_printoptions'])

# Obtaining an instance of the builtin type 'list' (line 8)
list_189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 8)
# Adding element type (line 8)
str_190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 11), 'str', 'array2string')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 10), list_189, str_190)
# Adding element type (line 8)
str_191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 27), 'str', 'set_printoptions')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 10), list_189, str_191)
# Adding element type (line 8)
str_192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 47), 'str', 'get_printoptions')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 10), list_189, str_192)

# Assigning a type to the variable '__all__' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), '__all__', list_189)

# Assigning a Str to a Name (line 9):

# Assigning a Str to a Name (line 9):
str_193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 16), 'str', 'restructuredtext')
# Assigning a type to the variable '__docformat__' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), '__docformat__', str_193)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 18, 0))

# 'import sys' statement (line 18)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 19, 0))

# 'from functools import reduce' statement (line 19)
from functools import reduce

import_from_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'functools', None, module_type_store, ['reduce'], [reduce])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 20, 0))

# 'from numpy.core import _nt' statement (line 20)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/core/')
import_194 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'numpy.core')

if (type(import_194) is not StypyTypeError):

    if (import_194 != 'pyd_module'):
        __import__(import_194)
        sys_modules_195 = sys.modules[import_194]
        import_from_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'numpy.core', sys_modules_195.module_type_store, module_type_store, ['numerictypes'])
        nest_module(stypy.reporting.localization.Localization(__file__, 20, 0), __file__, sys_modules_195, sys_modules_195.module_type_store, module_type_store)
    else:
        from numpy.core import numerictypes as _nt

        import_from_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'numpy.core', None, module_type_store, ['numerictypes'], [_nt])

else:
    # Assigning a type to the variable 'numpy.core' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'numpy.core', import_194)

# Adding an alias
module_type_store.add_alias('_nt', 'numerictypes')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/core/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 21, 0))

# 'from numpy.core.umath import maximum, minimum, absolute, not_equal, isnan, isinf' statement (line 21)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/core/')
import_196 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'numpy.core.umath')

if (type(import_196) is not StypyTypeError):

    if (import_196 != 'pyd_module'):
        __import__(import_196)
        sys_modules_197 = sys.modules[import_196]
        import_from_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'numpy.core.umath', sys_modules_197.module_type_store, module_type_store, ['maximum', 'minimum', 'absolute', 'not_equal', 'isnan', 'isinf'])
        nest_module(stypy.reporting.localization.Localization(__file__, 21, 0), __file__, sys_modules_197, sys_modules_197.module_type_store, module_type_store)
    else:
        from numpy.core.umath import maximum, minimum, absolute, not_equal, isnan, isinf

        import_from_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'numpy.core.umath', None, module_type_store, ['maximum', 'minimum', 'absolute', 'not_equal', 'isnan', 'isinf'], [maximum, minimum, absolute, not_equal, isnan, isinf])

else:
    # Assigning a type to the variable 'numpy.core.umath' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'numpy.core.umath', import_196)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/core/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 22, 0))

# 'from numpy.core.multiarray import array, format_longfloat, datetime_as_string, datetime_data' statement (line 22)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/core/')
import_198 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'numpy.core.multiarray')

if (type(import_198) is not StypyTypeError):

    if (import_198 != 'pyd_module'):
        __import__(import_198)
        sys_modules_199 = sys.modules[import_198]
        import_from_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'numpy.core.multiarray', sys_modules_199.module_type_store, module_type_store, ['array', 'format_longfloat', 'datetime_as_string', 'datetime_data'])
        nest_module(stypy.reporting.localization.Localization(__file__, 22, 0), __file__, sys_modules_199, sys_modules_199.module_type_store, module_type_store)
    else:
        from numpy.core.multiarray import array, format_longfloat, datetime_as_string, datetime_data

        import_from_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'numpy.core.multiarray', None, module_type_store, ['array', 'format_longfloat', 'datetime_as_string', 'datetime_data'], [array, format_longfloat, datetime_as_string, datetime_data])

else:
    # Assigning a type to the variable 'numpy.core.multiarray' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'numpy.core.multiarray', import_198)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/core/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 24, 0))

# 'from numpy.core.fromnumeric import ravel' statement (line 24)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/core/')
import_200 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'numpy.core.fromnumeric')

if (type(import_200) is not StypyTypeError):

    if (import_200 != 'pyd_module'):
        __import__(import_200)
        sys_modules_201 = sys.modules[import_200]
        import_from_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'numpy.core.fromnumeric', sys_modules_201.module_type_store, module_type_store, ['ravel'])
        nest_module(stypy.reporting.localization.Localization(__file__, 24, 0), __file__, sys_modules_201, sys_modules_201.module_type_store, module_type_store)
    else:
        from numpy.core.fromnumeric import ravel

        import_from_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'numpy.core.fromnumeric', None, module_type_store, ['ravel'], [ravel])

else:
    # Assigning a type to the variable 'numpy.core.fromnumeric' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'numpy.core.fromnumeric', import_200)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/core/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 25, 0))

# 'from numpy.core.numeric import asarray' statement (line 25)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/core/')
import_202 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 25, 0), 'numpy.core.numeric')

if (type(import_202) is not StypyTypeError):

    if (import_202 != 'pyd_module'):
        __import__(import_202)
        sys_modules_203 = sys.modules[import_202]
        import_from_module(stypy.reporting.localization.Localization(__file__, 25, 0), 'numpy.core.numeric', sys_modules_203.module_type_store, module_type_store, ['asarray'])
        nest_module(stypy.reporting.localization.Localization(__file__, 25, 0), __file__, sys_modules_203, sys_modules_203.module_type_store, module_type_store)
    else:
        from numpy.core.numeric import asarray

        import_from_module(stypy.reporting.localization.Localization(__file__, 25, 0), 'numpy.core.numeric', None, module_type_store, ['asarray'], [asarray])

else:
    # Assigning a type to the variable 'numpy.core.numeric' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'numpy.core.numeric', import_202)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/core/')




# Obtaining the type of the subscript
int_204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 20), 'int')
# Getting the type of 'sys' (line 27)
sys_205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 3), 'sys')
# Obtaining the member 'version_info' of a type (line 27)
version_info_206 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 3), sys_205, 'version_info')
# Obtaining the member '__getitem__' of a type (line 27)
getitem___207 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 3), version_info_206, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 27)
subscript_call_result_208 = invoke(stypy.reporting.localization.Localization(__file__, 27, 3), getitem___207, int_204)

int_209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 26), 'int')
# Applying the binary operator '>=' (line 27)
result_ge_210 = python_operator(stypy.reporting.localization.Localization(__file__, 27, 3), '>=', subscript_call_result_208, int_209)

# Testing the type of an if condition (line 27)
if_condition_211 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 27, 0), result_ge_210)
# Assigning a type to the variable 'if_condition_211' (line 27)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 0), 'if_condition_211', if_condition_211)
# SSA begins for if statement (line 27)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Assigning a Attribute to a Name (line 28):

# Assigning a Attribute to a Name (line 28):
# Getting the type of 'sys' (line 28)
sys_212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 14), 'sys')
# Obtaining the member 'maxsize' of a type (line 28)
maxsize_213 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 14), sys_212, 'maxsize')
# Assigning a type to the variable '_MAXINT' (line 28)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), '_MAXINT', maxsize_213)

# Assigning a BinOp to a Name (line 29):

# Assigning a BinOp to a Name (line 29):

# Getting the type of 'sys' (line 29)
sys_214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 15), 'sys')
# Obtaining the member 'maxsize' of a type (line 29)
maxsize_215 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 15), sys_214, 'maxsize')
# Applying the 'usub' unary operator (line 29)
result___neg___216 = python_operator(stypy.reporting.localization.Localization(__file__, 29, 14), 'usub', maxsize_215)

int_217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 29), 'int')
# Applying the binary operator '-' (line 29)
result_sub_218 = python_operator(stypy.reporting.localization.Localization(__file__, 29, 14), '-', result___neg___216, int_217)

# Assigning a type to the variable '_MININT' (line 29)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), '_MININT', result_sub_218)
# SSA branch for the else part of an if statement (line 27)
module_type_store.open_ssa_branch('else')

# Assigning a Attribute to a Name (line 31):

# Assigning a Attribute to a Name (line 31):
# Getting the type of 'sys' (line 31)
sys_219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 14), 'sys')
# Obtaining the member 'maxint' of a type (line 31)
maxint_220 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 14), sys_219, 'maxint')
# Assigning a type to the variable '_MAXINT' (line 31)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), '_MAXINT', maxint_220)

# Assigning a BinOp to a Name (line 32):

# Assigning a BinOp to a Name (line 32):

# Getting the type of 'sys' (line 32)
sys_221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 15), 'sys')
# Obtaining the member 'maxint' of a type (line 32)
maxint_222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 15), sys_221, 'maxint')
# Applying the 'usub' unary operator (line 32)
result___neg___223 = python_operator(stypy.reporting.localization.Localization(__file__, 32, 14), 'usub', maxint_222)

int_224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 28), 'int')
# Applying the binary operator '-' (line 32)
result_sub_225 = python_operator(stypy.reporting.localization.Localization(__file__, 32, 14), '-', result___neg___223, int_224)

# Assigning a type to the variable '_MININT' (line 32)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), '_MININT', result_sub_225)
# SSA join for if statement (line 27)
module_type_store = module_type_store.join_ssa_context()


@norecursion
def product(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'product'
    module_type_store = module_type_store.open_function_context('product', 34, 0, False)
    
    # Passed parameters checking function
    product.stypy_localization = localization
    product.stypy_type_of_self = None
    product.stypy_type_store = module_type_store
    product.stypy_function_name = 'product'
    product.stypy_param_names_list = ['x', 'y']
    product.stypy_varargs_param_name = None
    product.stypy_kwargs_param_name = None
    product.stypy_call_defaults = defaults
    product.stypy_call_varargs = varargs
    product.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'product', ['x', 'y'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'product', localization, ['x', 'y'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'product(...)' code ##################

    # Getting the type of 'x' (line 35)
    x_226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 11), 'x')
    # Getting the type of 'y' (line 35)
    y_227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 13), 'y')
    # Applying the binary operator '*' (line 35)
    result_mul_228 = python_operator(stypy.reporting.localization.Localization(__file__, 35, 11), '*', x_226, y_227)
    
    # Assigning a type to the variable 'stypy_return_type' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'stypy_return_type', result_mul_228)
    
    # ################# End of 'product(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'product' in the type store
    # Getting the type of 'stypy_return_type' (line 34)
    stypy_return_type_229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_229)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'product'
    return stypy_return_type_229

# Assigning a type to the variable 'product' (line 34)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 0), 'product', product)

# Assigning a Num to a Name (line 37):

# Assigning a Num to a Name (line 37):
int_230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 20), 'int')
# Assigning a type to the variable '_summaryEdgeItems' (line 37)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 0), '_summaryEdgeItems', int_230)

# Assigning a Num to a Name (line 38):

# Assigning a Num to a Name (line 38):
int_231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 20), 'int')
# Assigning a type to the variable '_summaryThreshold' (line 38)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 0), '_summaryThreshold', int_231)

# Assigning a Num to a Name (line 40):

# Assigning a Num to a Name (line 40):
int_232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 26), 'int')
# Assigning a type to the variable '_float_output_precision' (line 40)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 0), '_float_output_precision', int_232)

# Assigning a Name to a Name (line 41):

# Assigning a Name to a Name (line 41):
# Getting the type of 'False' (line 41)
False_233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 31), 'False')
# Assigning a type to the variable '_float_output_suppress_small' (line 41)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 0), '_float_output_suppress_small', False_233)

# Assigning a Num to a Name (line 42):

# Assigning a Num to a Name (line 42):
int_234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 14), 'int')
# Assigning a type to the variable '_line_width' (line 42)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 0), '_line_width', int_234)

# Assigning a Str to a Name (line 43):

# Assigning a Str to a Name (line 43):
str_235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 11), 'str', 'nan')
# Assigning a type to the variable '_nan_str' (line 43)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 0), '_nan_str', str_235)

# Assigning a Str to a Name (line 44):

# Assigning a Str to a Name (line 44):
str_236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 11), 'str', 'inf')
# Assigning a type to the variable '_inf_str' (line 44)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 0), '_inf_str', str_236)

# Assigning a Name to a Name (line 45):

# Assigning a Name to a Name (line 45):
# Getting the type of 'None' (line 45)
None_237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 13), 'None')
# Assigning a type to the variable '_formatter' (line 45)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 0), '_formatter', None_237)

@norecursion
def set_printoptions(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 48)
    None_238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 31), 'None')
    # Getting the type of 'None' (line 48)
    None_239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 47), 'None')
    # Getting the type of 'None' (line 48)
    None_240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 63), 'None')
    # Getting the type of 'None' (line 49)
    None_241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 31), 'None')
    # Getting the type of 'None' (line 49)
    None_242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 46), 'None')
    # Getting the type of 'None' (line 50)
    None_243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 28), 'None')
    # Getting the type of 'None' (line 50)
    None_244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 41), 'None')
    # Getting the type of 'None' (line 51)
    None_245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 31), 'None')
    defaults = [None_238, None_239, None_240, None_241, None_242, None_243, None_244, None_245]
    # Create a new context for function 'set_printoptions'
    module_type_store = module_type_store.open_function_context('set_printoptions', 48, 0, False)
    
    # Passed parameters checking function
    set_printoptions.stypy_localization = localization
    set_printoptions.stypy_type_of_self = None
    set_printoptions.stypy_type_store = module_type_store
    set_printoptions.stypy_function_name = 'set_printoptions'
    set_printoptions.stypy_param_names_list = ['precision', 'threshold', 'edgeitems', 'linewidth', 'suppress', 'nanstr', 'infstr', 'formatter']
    set_printoptions.stypy_varargs_param_name = None
    set_printoptions.stypy_kwargs_param_name = None
    set_printoptions.stypy_call_defaults = defaults
    set_printoptions.stypy_call_varargs = varargs
    set_printoptions.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'set_printoptions', ['precision', 'threshold', 'edgeitems', 'linewidth', 'suppress', 'nanstr', 'infstr', 'formatter'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'set_printoptions', localization, ['precision', 'threshold', 'edgeitems', 'linewidth', 'suppress', 'nanstr', 'infstr', 'formatter'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'set_printoptions(...)' code ##################

    str_246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, (-1)), 'str', "\n    Set printing options.\n\n    These options determine the way floating point numbers, arrays and\n    other NumPy objects are displayed.\n\n    Parameters\n    ----------\n    precision : int, optional\n        Number of digits of precision for floating point output (default 8).\n    threshold : int, optional\n        Total number of array elements which trigger summarization\n        rather than full repr (default 1000).\n    edgeitems : int, optional\n        Number of array items in summary at beginning and end of\n        each dimension (default 3).\n    linewidth : int, optional\n        The number of characters per line for the purpose of inserting\n        line breaks (default 75).\n    suppress : bool, optional\n        Whether or not suppress printing of small floating point values\n        using scientific notation (default False).\n    nanstr : str, optional\n        String representation of floating point not-a-number (default nan).\n    infstr : str, optional\n        String representation of floating point infinity (default inf).\n    formatter : dict of callables, optional\n        If not None, the keys should indicate the type(s) that the respective\n        formatting function applies to.  Callables should return a string.\n        Types that are not specified (by their corresponding keys) are handled\n        by the default formatters.  Individual types for which a formatter\n        can be set are::\n\n            - 'bool'\n            - 'int'\n            - 'timedelta' : a `numpy.timedelta64`\n            - 'datetime' : a `numpy.datetime64`\n            - 'float'\n            - 'longfloat' : 128-bit floats\n            - 'complexfloat'\n            - 'longcomplexfloat' : composed of two 128-bit floats\n            - 'numpy_str' : types `numpy.string_` and `numpy.unicode_`\n            - 'str' : all other strings\n\n        Other keys that can be used to set a group of types at once are::\n\n            - 'all' : sets all types\n            - 'int_kind' : sets 'int'\n            - 'float_kind' : sets 'float' and 'longfloat'\n            - 'complex_kind' : sets 'complexfloat' and 'longcomplexfloat'\n            - 'str_kind' : sets 'str' and 'numpystr'\n\n    See Also\n    --------\n    get_printoptions, set_string_function, array2string\n\n    Notes\n    -----\n    `formatter` is always reset with a call to `set_printoptions`.\n\n    Examples\n    --------\n    Floating point precision can be set:\n\n    >>> np.set_printoptions(precision=4)\n    >>> print(np.array([1.123456789]))\n    [ 1.1235]\n\n    Long arrays can be summarised:\n\n    >>> np.set_printoptions(threshold=5)\n    >>> print(np.arange(10))\n    [0 1 2 ..., 7 8 9]\n\n    Small results can be suppressed:\n\n    >>> eps = np.finfo(float).eps\n    >>> x = np.arange(4.)\n    >>> x**2 - (x + eps)**2\n    array([ -4.9304e-32,  -4.4409e-16,   0.0000e+00,   0.0000e+00])\n    >>> np.set_printoptions(suppress=True)\n    >>> x**2 - (x + eps)**2\n    array([-0., -0.,  0.,  0.])\n\n    A custom formatter can be used to display array elements as desired:\n\n    >>> np.set_printoptions(formatter={'all':lambda x: 'int: '+str(-x)})\n    >>> x = np.arange(3)\n    >>> x\n    array([int: 0, int: -1, int: -2])\n    >>> np.set_printoptions()  # formatter gets reset\n    >>> x\n    array([0, 1, 2])\n\n    To put back the default options, you can use:\n\n    >>> np.set_printoptions(edgeitems=3,infstr='inf',\n    ... linewidth=75, nanstr='nan', precision=8,\n    ... suppress=False, threshold=1000, formatter=None)\n    ")
    # Marking variables as global (line 153)
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 153, 4), '_summaryThreshold')
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 153, 4), '_summaryEdgeItems')
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 153, 4), '_float_output_precision')
    # Marking variables as global (line 154)
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 154, 4), '_line_width')
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 154, 4), '_float_output_suppress_small')
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 154, 4), '_nan_str')
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 154, 4), '_inf_str')
    # Marking variables as global (line 155)
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 155, 4), '_formatter')
    
    # Type idiom detected: calculating its left and rigth part (line 157)
    # Getting the type of 'linewidth' (line 157)
    linewidth_247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 4), 'linewidth')
    # Getting the type of 'None' (line 157)
    None_248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 24), 'None')
    
    (may_be_249, more_types_in_union_250) = may_not_be_none(linewidth_247, None_248)

    if may_be_249:

        if more_types_in_union_250:
            # Runtime conditional SSA (line 157)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Name to a Name (line 158):
        
        # Assigning a Name to a Name (line 158):
        # Getting the type of 'linewidth' (line 158)
        linewidth_251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 22), 'linewidth')
        # Assigning a type to the variable '_line_width' (line 158)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), '_line_width', linewidth_251)

        if more_types_in_union_250:
            # SSA join for if statement (line 157)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 159)
    # Getting the type of 'threshold' (line 159)
    threshold_252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 4), 'threshold')
    # Getting the type of 'None' (line 159)
    None_253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 24), 'None')
    
    (may_be_254, more_types_in_union_255) = may_not_be_none(threshold_252, None_253)

    if may_be_254:

        if more_types_in_union_255:
            # Runtime conditional SSA (line 159)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Name to a Name (line 160):
        
        # Assigning a Name to a Name (line 160):
        # Getting the type of 'threshold' (line 160)
        threshold_256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 28), 'threshold')
        # Assigning a type to the variable '_summaryThreshold' (line 160)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), '_summaryThreshold', threshold_256)

        if more_types_in_union_255:
            # SSA join for if statement (line 159)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 161)
    # Getting the type of 'edgeitems' (line 161)
    edgeitems_257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 4), 'edgeitems')
    # Getting the type of 'None' (line 161)
    None_258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 24), 'None')
    
    (may_be_259, more_types_in_union_260) = may_not_be_none(edgeitems_257, None_258)

    if may_be_259:

        if more_types_in_union_260:
            # Runtime conditional SSA (line 161)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Name to a Name (line 162):
        
        # Assigning a Name to a Name (line 162):
        # Getting the type of 'edgeitems' (line 162)
        edgeitems_261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 28), 'edgeitems')
        # Assigning a type to the variable '_summaryEdgeItems' (line 162)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 8), '_summaryEdgeItems', edgeitems_261)

        if more_types_in_union_260:
            # SSA join for if statement (line 161)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 163)
    # Getting the type of 'precision' (line 163)
    precision_262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 4), 'precision')
    # Getting the type of 'None' (line 163)
    None_263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 24), 'None')
    
    (may_be_264, more_types_in_union_265) = may_not_be_none(precision_262, None_263)

    if may_be_264:

        if more_types_in_union_265:
            # Runtime conditional SSA (line 163)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Name to a Name (line 164):
        
        # Assigning a Name to a Name (line 164):
        # Getting the type of 'precision' (line 164)
        precision_266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 34), 'precision')
        # Assigning a type to the variable '_float_output_precision' (line 164)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 8), '_float_output_precision', precision_266)

        if more_types_in_union_265:
            # SSA join for if statement (line 163)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 165)
    # Getting the type of 'suppress' (line 165)
    suppress_267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 4), 'suppress')
    # Getting the type of 'None' (line 165)
    None_268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 23), 'None')
    
    (may_be_269, more_types_in_union_270) = may_not_be_none(suppress_267, None_268)

    if may_be_269:

        if more_types_in_union_270:
            # Runtime conditional SSA (line 165)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a UnaryOp to a Name (line 166):
        
        # Assigning a UnaryOp to a Name (line 166):
        
        
        # Getting the type of 'suppress' (line 166)
        suppress_271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 47), 'suppress')
        # Applying the 'not' unary operator (line 166)
        result_not__272 = python_operator(stypy.reporting.localization.Localization(__file__, 166, 43), 'not', suppress_271)
        
        # Applying the 'not' unary operator (line 166)
        result_not__273 = python_operator(stypy.reporting.localization.Localization(__file__, 166, 39), 'not', result_not__272)
        
        # Assigning a type to the variable '_float_output_suppress_small' (line 166)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), '_float_output_suppress_small', result_not__273)

        if more_types_in_union_270:
            # SSA join for if statement (line 165)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 167)
    # Getting the type of 'nanstr' (line 167)
    nanstr_274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 4), 'nanstr')
    # Getting the type of 'None' (line 167)
    None_275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 21), 'None')
    
    (may_be_276, more_types_in_union_277) = may_not_be_none(nanstr_274, None_275)

    if may_be_276:

        if more_types_in_union_277:
            # Runtime conditional SSA (line 167)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Name to a Name (line 168):
        
        # Assigning a Name to a Name (line 168):
        # Getting the type of 'nanstr' (line 168)
        nanstr_278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 19), 'nanstr')
        # Assigning a type to the variable '_nan_str' (line 168)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 8), '_nan_str', nanstr_278)

        if more_types_in_union_277:
            # SSA join for if statement (line 167)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 169)
    # Getting the type of 'infstr' (line 169)
    infstr_279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 4), 'infstr')
    # Getting the type of 'None' (line 169)
    None_280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 21), 'None')
    
    (may_be_281, more_types_in_union_282) = may_not_be_none(infstr_279, None_280)

    if may_be_281:

        if more_types_in_union_282:
            # Runtime conditional SSA (line 169)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Name to a Name (line 170):
        
        # Assigning a Name to a Name (line 170):
        # Getting the type of 'infstr' (line 170)
        infstr_283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 19), 'infstr')
        # Assigning a type to the variable '_inf_str' (line 170)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 8), '_inf_str', infstr_283)

        if more_types_in_union_282:
            # SSA join for if statement (line 169)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Name to a Name (line 171):
    
    # Assigning a Name to a Name (line 171):
    # Getting the type of 'formatter' (line 171)
    formatter_284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 17), 'formatter')
    # Assigning a type to the variable '_formatter' (line 171)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 4), '_formatter', formatter_284)
    
    # ################# End of 'set_printoptions(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'set_printoptions' in the type store
    # Getting the type of 'stypy_return_type' (line 48)
    stypy_return_type_285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_285)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'set_printoptions'
    return stypy_return_type_285

# Assigning a type to the variable 'set_printoptions' (line 48)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 0), 'set_printoptions', set_printoptions)

@norecursion
def get_printoptions(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'get_printoptions'
    module_type_store = module_type_store.open_function_context('get_printoptions', 173, 0, False)
    
    # Passed parameters checking function
    get_printoptions.stypy_localization = localization
    get_printoptions.stypy_type_of_self = None
    get_printoptions.stypy_type_store = module_type_store
    get_printoptions.stypy_function_name = 'get_printoptions'
    get_printoptions.stypy_param_names_list = []
    get_printoptions.stypy_varargs_param_name = None
    get_printoptions.stypy_kwargs_param_name = None
    get_printoptions.stypy_call_defaults = defaults
    get_printoptions.stypy_call_varargs = varargs
    get_printoptions.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'get_printoptions', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'get_printoptions', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'get_printoptions(...)' code ##################

    str_286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, (-1)), 'str', '\n    Return the current print options.\n\n    Returns\n    -------\n    print_opts : dict\n        Dictionary of current print options with keys\n\n          - precision : int\n          - threshold : int\n          - edgeitems : int\n          - linewidth : int\n          - suppress : bool\n          - nanstr : str\n          - infstr : str\n          - formatter : dict of callables\n\n        For a full description of these options, see `set_printoptions`.\n\n    See Also\n    --------\n    set_printoptions, set_string_function\n\n    ')
    
    # Assigning a Call to a Name (line 198):
    
    # Assigning a Call to a Name (line 198):
    
    # Call to dict(...): (line 198)
    # Processing the call keyword arguments (line 198)
    # Getting the type of '_float_output_precision' (line 198)
    _float_output_precision_288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 23), '_float_output_precision', False)
    keyword_289 = _float_output_precision_288
    # Getting the type of '_summaryThreshold' (line 199)
    _summaryThreshold_290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 23), '_summaryThreshold', False)
    keyword_291 = _summaryThreshold_290
    # Getting the type of '_summaryEdgeItems' (line 200)
    _summaryEdgeItems_292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 23), '_summaryEdgeItems', False)
    keyword_293 = _summaryEdgeItems_292
    # Getting the type of '_line_width' (line 201)
    _line_width_294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 23), '_line_width', False)
    keyword_295 = _line_width_294
    # Getting the type of '_float_output_suppress_small' (line 202)
    _float_output_suppress_small_296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 22), '_float_output_suppress_small', False)
    keyword_297 = _float_output_suppress_small_296
    # Getting the type of '_nan_str' (line 203)
    _nan_str_298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 20), '_nan_str', False)
    keyword_299 = _nan_str_298
    # Getting the type of '_inf_str' (line 204)
    _inf_str_300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 20), '_inf_str', False)
    keyword_301 = _inf_str_300
    # Getting the type of '_formatter' (line 205)
    _formatter_302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 23), '_formatter', False)
    keyword_303 = _formatter_302
    kwargs_304 = {'infstr': keyword_301, 'formatter': keyword_303, 'suppress': keyword_297, 'edgeitems': keyword_293, 'precision': keyword_289, 'threshold': keyword_291, 'linewidth': keyword_295, 'nanstr': keyword_299}
    # Getting the type of 'dict' (line 198)
    dict_287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 8), 'dict', False)
    # Calling dict(args, kwargs) (line 198)
    dict_call_result_305 = invoke(stypy.reporting.localization.Localization(__file__, 198, 8), dict_287, *[], **kwargs_304)
    
    # Assigning a type to the variable 'd' (line 198)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 4), 'd', dict_call_result_305)
    # Getting the type of 'd' (line 206)
    d_306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 11), 'd')
    # Assigning a type to the variable 'stypy_return_type' (line 206)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 4), 'stypy_return_type', d_306)
    
    # ################# End of 'get_printoptions(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_printoptions' in the type store
    # Getting the type of 'stypy_return_type' (line 173)
    stypy_return_type_307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_307)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_printoptions'
    return stypy_return_type_307

# Assigning a type to the variable 'get_printoptions' (line 173)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 0), 'get_printoptions', get_printoptions)

@norecursion
def _leading_trailing(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_leading_trailing'
    module_type_store = module_type_store.open_function_context('_leading_trailing', 208, 0, False)
    
    # Passed parameters checking function
    _leading_trailing.stypy_localization = localization
    _leading_trailing.stypy_type_of_self = None
    _leading_trailing.stypy_type_store = module_type_store
    _leading_trailing.stypy_function_name = '_leading_trailing'
    _leading_trailing.stypy_param_names_list = ['a']
    _leading_trailing.stypy_varargs_param_name = None
    _leading_trailing.stypy_kwargs_param_name = None
    _leading_trailing.stypy_call_defaults = defaults
    _leading_trailing.stypy_call_varargs = varargs
    _leading_trailing.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_leading_trailing', ['a'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_leading_trailing', localization, ['a'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_leading_trailing(...)' code ##################

    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 209, 4))
    
    # 'from numpy.core import _nc' statement (line 209)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/core/')
    import_308 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 209, 4), 'numpy.core')

    if (type(import_308) is not StypyTypeError):

        if (import_308 != 'pyd_module'):
            __import__(import_308)
            sys_modules_309 = sys.modules[import_308]
            import_from_module(stypy.reporting.localization.Localization(__file__, 209, 4), 'numpy.core', sys_modules_309.module_type_store, module_type_store, ['numeric'])
            nest_module(stypy.reporting.localization.Localization(__file__, 209, 4), __file__, sys_modules_309, sys_modules_309.module_type_store, module_type_store)
        else:
            from numpy.core import numeric as _nc

            import_from_module(stypy.reporting.localization.Localization(__file__, 209, 4), 'numpy.core', None, module_type_store, ['numeric'], [_nc])

    else:
        # Assigning a type to the variable 'numpy.core' (line 209)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 4), 'numpy.core', import_308)

    # Adding an alias
    module_type_store.add_alias('_nc', 'numeric')
    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/core/')
    
    
    
    # Getting the type of 'a' (line 210)
    a_310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 7), 'a')
    # Obtaining the member 'ndim' of a type (line 210)
    ndim_311 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 7), a_310, 'ndim')
    int_312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 17), 'int')
    # Applying the binary operator '==' (line 210)
    result_eq_313 = python_operator(stypy.reporting.localization.Localization(__file__, 210, 7), '==', ndim_311, int_312)
    
    # Testing the type of an if condition (line 210)
    if_condition_314 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 210, 4), result_eq_313)
    # Assigning a type to the variable 'if_condition_314' (line 210)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 4), 'if_condition_314', if_condition_314)
    # SSA begins for if statement (line 210)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    
    # Call to len(...): (line 211)
    # Processing the call arguments (line 211)
    # Getting the type of 'a' (line 211)
    a_316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 15), 'a', False)
    # Processing the call keyword arguments (line 211)
    kwargs_317 = {}
    # Getting the type of 'len' (line 211)
    len_315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 11), 'len', False)
    # Calling len(args, kwargs) (line 211)
    len_call_result_318 = invoke(stypy.reporting.localization.Localization(__file__, 211, 11), len_315, *[a_316], **kwargs_317)
    
    int_319 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 20), 'int')
    # Getting the type of '_summaryEdgeItems' (line 211)
    _summaryEdgeItems_320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 22), '_summaryEdgeItems')
    # Applying the binary operator '*' (line 211)
    result_mul_321 = python_operator(stypy.reporting.localization.Localization(__file__, 211, 20), '*', int_319, _summaryEdgeItems_320)
    
    # Applying the binary operator '>' (line 211)
    result_gt_322 = python_operator(stypy.reporting.localization.Localization(__file__, 211, 11), '>', len_call_result_318, result_mul_321)
    
    # Testing the type of an if condition (line 211)
    if_condition_323 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 211, 8), result_gt_322)
    # Assigning a type to the variable 'if_condition_323' (line 211)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 8), 'if_condition_323', if_condition_323)
    # SSA begins for if statement (line 211)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 212):
    
    # Assigning a Call to a Name (line 212):
    
    # Call to concatenate(...): (line 212)
    # Processing the call arguments (line 212)
    
    # Obtaining an instance of the builtin type 'tuple' (line 212)
    tuple_326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 33), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 212)
    # Adding element type (line 212)
    
    # Obtaining the type of the subscript
    # Getting the type of '_summaryEdgeItems' (line 212)
    _summaryEdgeItems_327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 36), '_summaryEdgeItems', False)
    slice_328 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 212, 33), None, _summaryEdgeItems_327, None)
    # Getting the type of 'a' (line 212)
    a_329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 33), 'a', False)
    # Obtaining the member '__getitem__' of a type (line 212)
    getitem___330 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 33), a_329, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 212)
    subscript_call_result_331 = invoke(stypy.reporting.localization.Localization(__file__, 212, 33), getitem___330, slice_328)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 212, 33), tuple_326, subscript_call_result_331)
    # Adding element type (line 212)
    
    # Obtaining the type of the subscript
    
    # Getting the type of '_summaryEdgeItems' (line 213)
    _summaryEdgeItems_332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 40), '_summaryEdgeItems', False)
    # Applying the 'usub' unary operator (line 213)
    result___neg___333 = python_operator(stypy.reporting.localization.Localization(__file__, 213, 39), 'usub', _summaryEdgeItems_332)
    
    slice_334 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 213, 37), result___neg___333, None, None)
    # Getting the type of 'a' (line 213)
    a_335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 37), 'a', False)
    # Obtaining the member '__getitem__' of a type (line 213)
    getitem___336 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 37), a_335, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 213)
    subscript_call_result_337 = invoke(stypy.reporting.localization.Localization(__file__, 213, 37), getitem___336, slice_334)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 212, 33), tuple_326, subscript_call_result_337)
    
    # Processing the call keyword arguments (line 212)
    kwargs_338 = {}
    # Getting the type of '_nc' (line 212)
    _nc_324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 16), '_nc', False)
    # Obtaining the member 'concatenate' of a type (line 212)
    concatenate_325 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 16), _nc_324, 'concatenate')
    # Calling concatenate(args, kwargs) (line 212)
    concatenate_call_result_339 = invoke(stypy.reporting.localization.Localization(__file__, 212, 16), concatenate_325, *[tuple_326], **kwargs_338)
    
    # Assigning a type to the variable 'b' (line 212)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 12), 'b', concatenate_call_result_339)
    # SSA branch for the else part of an if statement (line 211)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Name to a Name (line 215):
    
    # Assigning a Name to a Name (line 215):
    # Getting the type of 'a' (line 215)
    a_340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 16), 'a')
    # Assigning a type to the variable 'b' (line 215)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 12), 'b', a_340)
    # SSA join for if statement (line 211)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 210)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Call to len(...): (line 217)
    # Processing the call arguments (line 217)
    # Getting the type of 'a' (line 217)
    a_342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 15), 'a', False)
    # Processing the call keyword arguments (line 217)
    kwargs_343 = {}
    # Getting the type of 'len' (line 217)
    len_341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 11), 'len', False)
    # Calling len(args, kwargs) (line 217)
    len_call_result_344 = invoke(stypy.reporting.localization.Localization(__file__, 217, 11), len_341, *[a_342], **kwargs_343)
    
    int_345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 20), 'int')
    # Getting the type of '_summaryEdgeItems' (line 217)
    _summaryEdgeItems_346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 22), '_summaryEdgeItems')
    # Applying the binary operator '*' (line 217)
    result_mul_347 = python_operator(stypy.reporting.localization.Localization(__file__, 217, 20), '*', int_345, _summaryEdgeItems_346)
    
    # Applying the binary operator '>' (line 217)
    result_gt_348 = python_operator(stypy.reporting.localization.Localization(__file__, 217, 11), '>', len_call_result_344, result_mul_347)
    
    # Testing the type of an if condition (line 217)
    if_condition_349 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 217, 8), result_gt_348)
    # Assigning a type to the variable 'if_condition_349' (line 217)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 8), 'if_condition_349', if_condition_349)
    # SSA begins for if statement (line 217)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a ListComp to a Name (line 218):
    
    # Assigning a ListComp to a Name (line 218):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to range(...): (line 218)
    # Processing the call arguments (line 218)
    
    # Call to min(...): (line 219)
    # Processing the call arguments (line 219)
    
    # Call to len(...): (line 219)
    # Processing the call arguments (line 219)
    # Getting the type of 'a' (line 219)
    a_360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 24), 'a', False)
    # Processing the call keyword arguments (line 219)
    kwargs_361 = {}
    # Getting the type of 'len' (line 219)
    len_359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 20), 'len', False)
    # Calling len(args, kwargs) (line 219)
    len_call_result_362 = invoke(stypy.reporting.localization.Localization(__file__, 219, 20), len_359, *[a_360], **kwargs_361)
    
    # Getting the type of '_summaryEdgeItems' (line 219)
    _summaryEdgeItems_363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 28), '_summaryEdgeItems', False)
    # Processing the call keyword arguments (line 219)
    kwargs_364 = {}
    # Getting the type of 'min' (line 219)
    min_358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 16), 'min', False)
    # Calling min(args, kwargs) (line 219)
    min_call_result_365 = invoke(stypy.reporting.localization.Localization(__file__, 219, 16), min_358, *[len_call_result_362, _summaryEdgeItems_363], **kwargs_364)
    
    # Processing the call keyword arguments (line 218)
    kwargs_366 = {}
    # Getting the type of 'range' (line 218)
    range_357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 50), 'range', False)
    # Calling range(args, kwargs) (line 218)
    range_call_result_367 = invoke(stypy.reporting.localization.Localization(__file__, 218, 50), range_357, *[min_call_result_365], **kwargs_366)
    
    comprehension_368 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 17), range_call_result_367)
    # Assigning a type to the variable 'i' (line 218)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 17), 'i', comprehension_368)
    
    # Call to _leading_trailing(...): (line 218)
    # Processing the call arguments (line 218)
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 218)
    i_351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 37), 'i', False)
    # Getting the type of 'a' (line 218)
    a_352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 35), 'a', False)
    # Obtaining the member '__getitem__' of a type (line 218)
    getitem___353 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 35), a_352, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 218)
    subscript_call_result_354 = invoke(stypy.reporting.localization.Localization(__file__, 218, 35), getitem___353, i_351)
    
    # Processing the call keyword arguments (line 218)
    kwargs_355 = {}
    # Getting the type of '_leading_trailing' (line 218)
    _leading_trailing_350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 17), '_leading_trailing', False)
    # Calling _leading_trailing(args, kwargs) (line 218)
    _leading_trailing_call_result_356 = invoke(stypy.reporting.localization.Localization(__file__, 218, 17), _leading_trailing_350, *[subscript_call_result_354], **kwargs_355)
    
    list_369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 17), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 17), list_369, _leading_trailing_call_result_356)
    # Assigning a type to the variable 'l' (line 218)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 12), 'l', list_369)
    
    # Call to extend(...): (line 220)
    # Processing the call arguments (line 220)
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to range(...): (line 220)
    # Processing the call arguments (line 220)
    
    # Call to min(...): (line 221)
    # Processing the call arguments (line 221)
    
    # Call to len(...): (line 221)
    # Processing the call arguments (line 221)
    # Getting the type of 'a' (line 221)
    a_383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 24), 'a', False)
    # Processing the call keyword arguments (line 221)
    kwargs_384 = {}
    # Getting the type of 'len' (line 221)
    len_382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 20), 'len', False)
    # Calling len(args, kwargs) (line 221)
    len_call_result_385 = invoke(stypy.reporting.localization.Localization(__file__, 221, 20), len_382, *[a_383], **kwargs_384)
    
    # Getting the type of '_summaryEdgeItems' (line 221)
    _summaryEdgeItems_386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 28), '_summaryEdgeItems', False)
    # Processing the call keyword arguments (line 221)
    kwargs_387 = {}
    # Getting the type of 'min' (line 221)
    min_381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 16), 'min', False)
    # Calling min(args, kwargs) (line 221)
    min_call_result_388 = invoke(stypy.reporting.localization.Localization(__file__, 221, 16), min_381, *[len_call_result_385, _summaryEdgeItems_386], **kwargs_387)
    
    int_389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 48), 'int')
    int_390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 51), 'int')
    # Processing the call keyword arguments (line 220)
    kwargs_391 = {}
    # Getting the type of 'range' (line 220)
    range_380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 56), 'range', False)
    # Calling range(args, kwargs) (line 220)
    range_call_result_392 = invoke(stypy.reporting.localization.Localization(__file__, 220, 56), range_380, *[min_call_result_388, int_389, int_390], **kwargs_391)
    
    comprehension_393 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 220, 22), range_call_result_392)
    # Assigning a type to the variable 'i' (line 220)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 22), 'i', comprehension_393)
    
    # Call to _leading_trailing(...): (line 220)
    # Processing the call arguments (line 220)
    
    # Obtaining the type of the subscript
    
    # Getting the type of 'i' (line 220)
    i_373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 43), 'i', False)
    # Applying the 'usub' unary operator (line 220)
    result___neg___374 = python_operator(stypy.reporting.localization.Localization(__file__, 220, 42), 'usub', i_373)
    
    # Getting the type of 'a' (line 220)
    a_375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 40), 'a', False)
    # Obtaining the member '__getitem__' of a type (line 220)
    getitem___376 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 40), a_375, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 220)
    subscript_call_result_377 = invoke(stypy.reporting.localization.Localization(__file__, 220, 40), getitem___376, result___neg___374)
    
    # Processing the call keyword arguments (line 220)
    kwargs_378 = {}
    # Getting the type of '_leading_trailing' (line 220)
    _leading_trailing_372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 22), '_leading_trailing', False)
    # Calling _leading_trailing(args, kwargs) (line 220)
    _leading_trailing_call_result_379 = invoke(stypy.reporting.localization.Localization(__file__, 220, 22), _leading_trailing_372, *[subscript_call_result_377], **kwargs_378)
    
    list_394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 22), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 220, 22), list_394, _leading_trailing_call_result_379)
    # Processing the call keyword arguments (line 220)
    kwargs_395 = {}
    # Getting the type of 'l' (line 220)
    l_370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 12), 'l', False)
    # Obtaining the member 'extend' of a type (line 220)
    extend_371 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 12), l_370, 'extend')
    # Calling extend(args, kwargs) (line 220)
    extend_call_result_396 = invoke(stypy.reporting.localization.Localization(__file__, 220, 12), extend_371, *[list_394], **kwargs_395)
    
    # SSA branch for the else part of an if statement (line 217)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a ListComp to a Name (line 223):
    
    # Assigning a ListComp to a Name (line 223):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to range(...): (line 223)
    # Processing the call arguments (line 223)
    int_405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 56), 'int')
    
    # Call to len(...): (line 223)
    # Processing the call arguments (line 223)
    # Getting the type of 'a' (line 223)
    a_407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 63), 'a', False)
    # Processing the call keyword arguments (line 223)
    kwargs_408 = {}
    # Getting the type of 'len' (line 223)
    len_406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 59), 'len', False)
    # Calling len(args, kwargs) (line 223)
    len_call_result_409 = invoke(stypy.reporting.localization.Localization(__file__, 223, 59), len_406, *[a_407], **kwargs_408)
    
    # Processing the call keyword arguments (line 223)
    kwargs_410 = {}
    # Getting the type of 'range' (line 223)
    range_404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 50), 'range', False)
    # Calling range(args, kwargs) (line 223)
    range_call_result_411 = invoke(stypy.reporting.localization.Localization(__file__, 223, 50), range_404, *[int_405, len_call_result_409], **kwargs_410)
    
    comprehension_412 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 223, 17), range_call_result_411)
    # Assigning a type to the variable 'i' (line 223)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 17), 'i', comprehension_412)
    
    # Call to _leading_trailing(...): (line 223)
    # Processing the call arguments (line 223)
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 223)
    i_398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 37), 'i', False)
    # Getting the type of 'a' (line 223)
    a_399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 35), 'a', False)
    # Obtaining the member '__getitem__' of a type (line 223)
    getitem___400 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 35), a_399, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 223)
    subscript_call_result_401 = invoke(stypy.reporting.localization.Localization(__file__, 223, 35), getitem___400, i_398)
    
    # Processing the call keyword arguments (line 223)
    kwargs_402 = {}
    # Getting the type of '_leading_trailing' (line 223)
    _leading_trailing_397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 17), '_leading_trailing', False)
    # Calling _leading_trailing(args, kwargs) (line 223)
    _leading_trailing_call_result_403 = invoke(stypy.reporting.localization.Localization(__file__, 223, 17), _leading_trailing_397, *[subscript_call_result_401], **kwargs_402)
    
    list_413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 17), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 223, 17), list_413, _leading_trailing_call_result_403)
    # Assigning a type to the variable 'l' (line 223)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 12), 'l', list_413)
    # SSA join for if statement (line 217)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 224):
    
    # Assigning a Call to a Name (line 224):
    
    # Call to concatenate(...): (line 224)
    # Processing the call arguments (line 224)
    
    # Call to tuple(...): (line 224)
    # Processing the call arguments (line 224)
    # Getting the type of 'l' (line 224)
    l_417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 34), 'l', False)
    # Processing the call keyword arguments (line 224)
    kwargs_418 = {}
    # Getting the type of 'tuple' (line 224)
    tuple_416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 28), 'tuple', False)
    # Calling tuple(args, kwargs) (line 224)
    tuple_call_result_419 = invoke(stypy.reporting.localization.Localization(__file__, 224, 28), tuple_416, *[l_417], **kwargs_418)
    
    # Processing the call keyword arguments (line 224)
    kwargs_420 = {}
    # Getting the type of '_nc' (line 224)
    _nc_414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 12), '_nc', False)
    # Obtaining the member 'concatenate' of a type (line 224)
    concatenate_415 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 12), _nc_414, 'concatenate')
    # Calling concatenate(args, kwargs) (line 224)
    concatenate_call_result_421 = invoke(stypy.reporting.localization.Localization(__file__, 224, 12), concatenate_415, *[tuple_call_result_419], **kwargs_420)
    
    # Assigning a type to the variable 'b' (line 224)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 8), 'b', concatenate_call_result_421)
    # SSA join for if statement (line 210)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'b' (line 225)
    b_422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 11), 'b')
    # Assigning a type to the variable 'stypy_return_type' (line 225)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 4), 'stypy_return_type', b_422)
    
    # ################# End of '_leading_trailing(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_leading_trailing' in the type store
    # Getting the type of 'stypy_return_type' (line 208)
    stypy_return_type_423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_423)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_leading_trailing'
    return stypy_return_type_423

# Assigning a type to the variable '_leading_trailing' (line 208)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 0), '_leading_trailing', _leading_trailing)

@norecursion
def _boolFormatter(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_boolFormatter'
    module_type_store = module_type_store.open_function_context('_boolFormatter', 227, 0, False)
    
    # Passed parameters checking function
    _boolFormatter.stypy_localization = localization
    _boolFormatter.stypy_type_of_self = None
    _boolFormatter.stypy_type_store = module_type_store
    _boolFormatter.stypy_function_name = '_boolFormatter'
    _boolFormatter.stypy_param_names_list = ['x']
    _boolFormatter.stypy_varargs_param_name = None
    _boolFormatter.stypy_kwargs_param_name = None
    _boolFormatter.stypy_call_defaults = defaults
    _boolFormatter.stypy_call_varargs = varargs
    _boolFormatter.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_boolFormatter', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_boolFormatter', localization, ['x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_boolFormatter(...)' code ##################

    
    # Getting the type of 'x' (line 228)
    x_424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 7), 'x')
    # Testing the type of an if condition (line 228)
    if_condition_425 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 228, 4), x_424)
    # Assigning a type to the variable 'if_condition_425' (line 228)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 4), 'if_condition_425', if_condition_425)
    # SSA begins for if statement (line 228)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    str_426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 15), 'str', ' True')
    # Assigning a type to the variable 'stypy_return_type' (line 229)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 8), 'stypy_return_type', str_426)
    # SSA branch for the else part of an if statement (line 228)
    module_type_store.open_ssa_branch('else')
    str_427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 15), 'str', 'False')
    # Assigning a type to the variable 'stypy_return_type' (line 231)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 8), 'stypy_return_type', str_427)
    # SSA join for if statement (line 228)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of '_boolFormatter(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_boolFormatter' in the type store
    # Getting the type of 'stypy_return_type' (line 227)
    stypy_return_type_428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_428)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_boolFormatter'
    return stypy_return_type_428

# Assigning a type to the variable '_boolFormatter' (line 227)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 0), '_boolFormatter', _boolFormatter)

@norecursion
def repr_format(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'repr_format'
    module_type_store = module_type_store.open_function_context('repr_format', 234, 0, False)
    
    # Passed parameters checking function
    repr_format.stypy_localization = localization
    repr_format.stypy_type_of_self = None
    repr_format.stypy_type_store = module_type_store
    repr_format.stypy_function_name = 'repr_format'
    repr_format.stypy_param_names_list = ['x']
    repr_format.stypy_varargs_param_name = None
    repr_format.stypy_kwargs_param_name = None
    repr_format.stypy_call_defaults = defaults
    repr_format.stypy_call_varargs = varargs
    repr_format.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'repr_format', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'repr_format', localization, ['x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'repr_format(...)' code ##################

    
    # Call to repr(...): (line 235)
    # Processing the call arguments (line 235)
    # Getting the type of 'x' (line 235)
    x_430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 16), 'x', False)
    # Processing the call keyword arguments (line 235)
    kwargs_431 = {}
    # Getting the type of 'repr' (line 235)
    repr_429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 11), 'repr', False)
    # Calling repr(args, kwargs) (line 235)
    repr_call_result_432 = invoke(stypy.reporting.localization.Localization(__file__, 235, 11), repr_429, *[x_430], **kwargs_431)
    
    # Assigning a type to the variable 'stypy_return_type' (line 235)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 4), 'stypy_return_type', repr_call_result_432)
    
    # ################# End of 'repr_format(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'repr_format' in the type store
    # Getting the type of 'stypy_return_type' (line 234)
    stypy_return_type_433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_433)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'repr_format'
    return stypy_return_type_433

# Assigning a type to the variable 'repr_format' (line 234)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 0), 'repr_format', repr_format)

@norecursion
def _array2string(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 74), 'str', ' ')
    str_435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 25), 'str', '')
    # Getting the type of 'None' (line 238)
    None_436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 39), 'None')
    defaults = [str_434, str_435, None_436]
    # Create a new context for function '_array2string'
    module_type_store = module_type_store.open_function_context('_array2string', 237, 0, False)
    
    # Passed parameters checking function
    _array2string.stypy_localization = localization
    _array2string.stypy_type_of_self = None
    _array2string.stypy_type_store = module_type_store
    _array2string.stypy_function_name = '_array2string'
    _array2string.stypy_param_names_list = ['a', 'max_line_width', 'precision', 'suppress_small', 'separator', 'prefix', 'formatter']
    _array2string.stypy_varargs_param_name = None
    _array2string.stypy_kwargs_param_name = None
    _array2string.stypy_call_defaults = defaults
    _array2string.stypy_call_varargs = varargs
    _array2string.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_array2string', ['a', 'max_line_width', 'precision', 'suppress_small', 'separator', 'prefix', 'formatter'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_array2string', localization, ['a', 'max_line_width', 'precision', 'suppress_small', 'separator', 'prefix', 'formatter'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_array2string(...)' code ##################

    
    # Type idiom detected: calculating its left and rigth part (line 240)
    # Getting the type of 'max_line_width' (line 240)
    max_line_width_437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 7), 'max_line_width')
    # Getting the type of 'None' (line 240)
    None_438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 25), 'None')
    
    (may_be_439, more_types_in_union_440) = may_be_none(max_line_width_437, None_438)

    if may_be_439:

        if more_types_in_union_440:
            # Runtime conditional SSA (line 240)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Name to a Name (line 241):
        
        # Assigning a Name to a Name (line 241):
        # Getting the type of '_line_width' (line 241)
        _line_width_441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 25), '_line_width')
        # Assigning a type to the variable 'max_line_width' (line 241)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 8), 'max_line_width', _line_width_441)

        if more_types_in_union_440:
            # SSA join for if statement (line 240)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 243)
    # Getting the type of 'precision' (line 243)
    precision_442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 7), 'precision')
    # Getting the type of 'None' (line 243)
    None_443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 20), 'None')
    
    (may_be_444, more_types_in_union_445) = may_be_none(precision_442, None_443)

    if may_be_444:

        if more_types_in_union_445:
            # Runtime conditional SSA (line 243)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Name to a Name (line 244):
        
        # Assigning a Name to a Name (line 244):
        # Getting the type of '_float_output_precision' (line 244)
        _float_output_precision_446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 20), '_float_output_precision')
        # Assigning a type to the variable 'precision' (line 244)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 8), 'precision', _float_output_precision_446)

        if more_types_in_union_445:
            # SSA join for if statement (line 243)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 246)
    # Getting the type of 'suppress_small' (line 246)
    suppress_small_447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 7), 'suppress_small')
    # Getting the type of 'None' (line 246)
    None_448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 25), 'None')
    
    (may_be_449, more_types_in_union_450) = may_be_none(suppress_small_447, None_448)

    if may_be_449:

        if more_types_in_union_450:
            # Runtime conditional SSA (line 246)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Name to a Name (line 247):
        
        # Assigning a Name to a Name (line 247):
        # Getting the type of '_float_output_suppress_small' (line 247)
        _float_output_suppress_small_451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 25), '_float_output_suppress_small')
        # Assigning a type to the variable 'suppress_small' (line 247)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 8), 'suppress_small', _float_output_suppress_small_451)

        if more_types_in_union_450:
            # SSA join for if statement (line 246)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 249)
    # Getting the type of 'formatter' (line 249)
    formatter_452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 7), 'formatter')
    # Getting the type of 'None' (line 249)
    None_453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 20), 'None')
    
    (may_be_454, more_types_in_union_455) = may_be_none(formatter_452, None_453)

    if may_be_454:

        if more_types_in_union_455:
            # Runtime conditional SSA (line 249)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Name to a Name (line 250):
        
        # Assigning a Name to a Name (line 250):
        # Getting the type of '_formatter' (line 250)
        _formatter_456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 20), '_formatter')
        # Assigning a type to the variable 'formatter' (line 250)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 8), 'formatter', _formatter_456)

        if more_types_in_union_455:
            # SSA join for if statement (line 249)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Getting the type of 'a' (line 252)
    a_457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 7), 'a')
    # Obtaining the member 'size' of a type (line 252)
    size_458 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 7), a_457, 'size')
    # Getting the type of '_summaryThreshold' (line 252)
    _summaryThreshold_459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 16), '_summaryThreshold')
    # Applying the binary operator '>' (line 252)
    result_gt_460 = python_operator(stypy.reporting.localization.Localization(__file__, 252, 7), '>', size_458, _summaryThreshold_459)
    
    # Testing the type of an if condition (line 252)
    if_condition_461 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 252, 4), result_gt_460)
    # Assigning a type to the variable 'if_condition_461' (line 252)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 4), 'if_condition_461', if_condition_461)
    # SSA begins for if statement (line 252)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 253):
    
    # Assigning a Str to a Name (line 253):
    str_462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 25), 'str', '..., ')
    # Assigning a type to the variable 'summary_insert' (line 253)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 8), 'summary_insert', str_462)
    
    # Assigning a Call to a Name (line 254):
    
    # Assigning a Call to a Name (line 254):
    
    # Call to _leading_trailing(...): (line 254)
    # Processing the call arguments (line 254)
    # Getting the type of 'a' (line 254)
    a_464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 33), 'a', False)
    # Processing the call keyword arguments (line 254)
    kwargs_465 = {}
    # Getting the type of '_leading_trailing' (line 254)
    _leading_trailing_463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 15), '_leading_trailing', False)
    # Calling _leading_trailing(args, kwargs) (line 254)
    _leading_trailing_call_result_466 = invoke(stypy.reporting.localization.Localization(__file__, 254, 15), _leading_trailing_463, *[a_464], **kwargs_465)
    
    # Assigning a type to the variable 'data' (line 254)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 8), 'data', _leading_trailing_call_result_466)
    # SSA branch for the else part of an if statement (line 252)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Str to a Name (line 256):
    
    # Assigning a Str to a Name (line 256):
    str_467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 25), 'str', '')
    # Assigning a type to the variable 'summary_insert' (line 256)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 8), 'summary_insert', str_467)
    
    # Assigning a Call to a Name (line 257):
    
    # Assigning a Call to a Name (line 257):
    
    # Call to ravel(...): (line 257)
    # Processing the call arguments (line 257)
    
    # Call to asarray(...): (line 257)
    # Processing the call arguments (line 257)
    # Getting the type of 'a' (line 257)
    a_470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 29), 'a', False)
    # Processing the call keyword arguments (line 257)
    kwargs_471 = {}
    # Getting the type of 'asarray' (line 257)
    asarray_469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 21), 'asarray', False)
    # Calling asarray(args, kwargs) (line 257)
    asarray_call_result_472 = invoke(stypy.reporting.localization.Localization(__file__, 257, 21), asarray_469, *[a_470], **kwargs_471)
    
    # Processing the call keyword arguments (line 257)
    kwargs_473 = {}
    # Getting the type of 'ravel' (line 257)
    ravel_468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 15), 'ravel', False)
    # Calling ravel(args, kwargs) (line 257)
    ravel_call_result_474 = invoke(stypy.reporting.localization.Localization(__file__, 257, 15), ravel_468, *[asarray_call_result_472], **kwargs_473)
    
    # Assigning a type to the variable 'data' (line 257)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 8), 'data', ravel_call_result_474)
    # SSA join for if statement (line 252)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Dict to a Name (line 259):
    
    # Assigning a Dict to a Name (line 259):
    
    # Obtaining an instance of the builtin type 'dict' (line 259)
    dict_475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 17), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 259)
    # Adding element type (key, value) (line 259)
    str_476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 18), 'str', 'bool')
    # Getting the type of '_boolFormatter' (line 259)
    _boolFormatter_477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 26), '_boolFormatter')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 17), dict_475, (str_476, _boolFormatter_477))
    # Adding element type (key, value) (line 259)
    str_478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 18), 'str', 'int')
    
    # Call to IntegerFormat(...): (line 260)
    # Processing the call arguments (line 260)
    # Getting the type of 'data' (line 260)
    data_480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 39), 'data', False)
    # Processing the call keyword arguments (line 260)
    kwargs_481 = {}
    # Getting the type of 'IntegerFormat' (line 260)
    IntegerFormat_479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 25), 'IntegerFormat', False)
    # Calling IntegerFormat(args, kwargs) (line 260)
    IntegerFormat_call_result_482 = invoke(stypy.reporting.localization.Localization(__file__, 260, 25), IntegerFormat_479, *[data_480], **kwargs_481)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 17), dict_475, (str_478, IntegerFormat_call_result_482))
    # Adding element type (key, value) (line 259)
    str_483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 18), 'str', 'float')
    
    # Call to FloatFormat(...): (line 261)
    # Processing the call arguments (line 261)
    # Getting the type of 'data' (line 261)
    data_485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 39), 'data', False)
    # Getting the type of 'precision' (line 261)
    precision_486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 45), 'precision', False)
    # Getting the type of 'suppress_small' (line 261)
    suppress_small_487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 56), 'suppress_small', False)
    # Processing the call keyword arguments (line 261)
    kwargs_488 = {}
    # Getting the type of 'FloatFormat' (line 261)
    FloatFormat_484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 27), 'FloatFormat', False)
    # Calling FloatFormat(args, kwargs) (line 261)
    FloatFormat_call_result_489 = invoke(stypy.reporting.localization.Localization(__file__, 261, 27), FloatFormat_484, *[data_485, precision_486, suppress_small_487], **kwargs_488)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 17), dict_475, (str_483, FloatFormat_call_result_489))
    # Adding element type (key, value) (line 259)
    str_490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 18), 'str', 'longfloat')
    
    # Call to LongFloatFormat(...): (line 262)
    # Processing the call arguments (line 262)
    # Getting the type of 'precision' (line 262)
    precision_492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 47), 'precision', False)
    # Processing the call keyword arguments (line 262)
    kwargs_493 = {}
    # Getting the type of 'LongFloatFormat' (line 262)
    LongFloatFormat_491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 31), 'LongFloatFormat', False)
    # Calling LongFloatFormat(args, kwargs) (line 262)
    LongFloatFormat_call_result_494 = invoke(stypy.reporting.localization.Localization(__file__, 262, 31), LongFloatFormat_491, *[precision_492], **kwargs_493)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 17), dict_475, (str_490, LongFloatFormat_call_result_494))
    # Adding element type (key, value) (line 259)
    str_495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 18), 'str', 'complexfloat')
    
    # Call to ComplexFormat(...): (line 263)
    # Processing the call arguments (line 263)
    # Getting the type of 'data' (line 263)
    data_497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 48), 'data', False)
    # Getting the type of 'precision' (line 263)
    precision_498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 54), 'precision', False)
    # Getting the type of 'suppress_small' (line 264)
    suppress_small_499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 49), 'suppress_small', False)
    # Processing the call keyword arguments (line 263)
    kwargs_500 = {}
    # Getting the type of 'ComplexFormat' (line 263)
    ComplexFormat_496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 34), 'ComplexFormat', False)
    # Calling ComplexFormat(args, kwargs) (line 263)
    ComplexFormat_call_result_501 = invoke(stypy.reporting.localization.Localization(__file__, 263, 34), ComplexFormat_496, *[data_497, precision_498, suppress_small_499], **kwargs_500)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 17), dict_475, (str_495, ComplexFormat_call_result_501))
    # Adding element type (key, value) (line 259)
    str_502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 18), 'str', 'longcomplexfloat')
    
    # Call to LongComplexFormat(...): (line 265)
    # Processing the call arguments (line 265)
    # Getting the type of 'precision' (line 265)
    precision_504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 56), 'precision', False)
    # Processing the call keyword arguments (line 265)
    kwargs_505 = {}
    # Getting the type of 'LongComplexFormat' (line 265)
    LongComplexFormat_503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 38), 'LongComplexFormat', False)
    # Calling LongComplexFormat(args, kwargs) (line 265)
    LongComplexFormat_call_result_506 = invoke(stypy.reporting.localization.Localization(__file__, 265, 38), LongComplexFormat_503, *[precision_504], **kwargs_505)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 17), dict_475, (str_502, LongComplexFormat_call_result_506))
    # Adding element type (key, value) (line 259)
    str_507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 18), 'str', 'datetime')
    
    # Call to DatetimeFormat(...): (line 266)
    # Processing the call arguments (line 266)
    # Getting the type of 'data' (line 266)
    data_509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 45), 'data', False)
    # Processing the call keyword arguments (line 266)
    kwargs_510 = {}
    # Getting the type of 'DatetimeFormat' (line 266)
    DatetimeFormat_508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 30), 'DatetimeFormat', False)
    # Calling DatetimeFormat(args, kwargs) (line 266)
    DatetimeFormat_call_result_511 = invoke(stypy.reporting.localization.Localization(__file__, 266, 30), DatetimeFormat_508, *[data_509], **kwargs_510)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 17), dict_475, (str_507, DatetimeFormat_call_result_511))
    # Adding element type (key, value) (line 259)
    str_512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 18), 'str', 'timedelta')
    
    # Call to TimedeltaFormat(...): (line 267)
    # Processing the call arguments (line 267)
    # Getting the type of 'data' (line 267)
    data_514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 47), 'data', False)
    # Processing the call keyword arguments (line 267)
    kwargs_515 = {}
    # Getting the type of 'TimedeltaFormat' (line 267)
    TimedeltaFormat_513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 31), 'TimedeltaFormat', False)
    # Calling TimedeltaFormat(args, kwargs) (line 267)
    TimedeltaFormat_call_result_516 = invoke(stypy.reporting.localization.Localization(__file__, 267, 31), TimedeltaFormat_513, *[data_514], **kwargs_515)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 17), dict_475, (str_512, TimedeltaFormat_call_result_516))
    # Adding element type (key, value) (line 259)
    str_517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 18), 'str', 'numpystr')
    # Getting the type of 'repr_format' (line 268)
    repr_format_518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 30), 'repr_format')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 17), dict_475, (str_517, repr_format_518))
    # Adding element type (key, value) (line 259)
    str_519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 18), 'str', 'str')
    # Getting the type of 'str' (line 269)
    str_520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 25), 'str')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 17), dict_475, (str_519, str_520))
    
    # Assigning a type to the variable 'formatdict' (line 259)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 4), 'formatdict', dict_475)
    
    # Type idiom detected: calculating its left and rigth part (line 271)
    # Getting the type of 'formatter' (line 271)
    formatter_521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 4), 'formatter')
    # Getting the type of 'None' (line 271)
    None_522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 24), 'None')
    
    (may_be_523, more_types_in_union_524) = may_not_be_none(formatter_521, None_522)

    if may_be_523:

        if more_types_in_union_524:
            # Runtime conditional SSA (line 271)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a ListComp to a Name (line 272):
        
        # Assigning a ListComp to a Name (line 272):
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to keys(...): (line 272)
        # Processing the call keyword arguments (line 272)
        kwargs_534 = {}
        # Getting the type of 'formatter' (line 272)
        formatter_532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 28), 'formatter', False)
        # Obtaining the member 'keys' of a type (line 272)
        keys_533 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 28), formatter_532, 'keys')
        # Calling keys(args, kwargs) (line 272)
        keys_call_result_535 = invoke(stypy.reporting.localization.Localization(__file__, 272, 28), keys_533, *[], **kwargs_534)
        
        comprehension_536 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 272, 17), keys_call_result_535)
        # Assigning a type to the variable 'k' (line 272)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 17), 'k', comprehension_536)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'k' (line 272)
        k_526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 58), 'k')
        # Getting the type of 'formatter' (line 272)
        formatter_527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 48), 'formatter')
        # Obtaining the member '__getitem__' of a type (line 272)
        getitem___528 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 48), formatter_527, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 272)
        subscript_call_result_529 = invoke(stypy.reporting.localization.Localization(__file__, 272, 48), getitem___528, k_526)
        
        # Getting the type of 'None' (line 272)
        None_530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 68), 'None')
        # Applying the binary operator 'isnot' (line 272)
        result_is_not_531 = python_operator(stypy.reporting.localization.Localization(__file__, 272, 48), 'isnot', subscript_call_result_529, None_530)
        
        # Getting the type of 'k' (line 272)
        k_525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 17), 'k')
        list_537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 17), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 272, 17), list_537, k_525)
        # Assigning a type to the variable 'fkeys' (line 272)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 8), 'fkeys', list_537)
        
        
        str_538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 11), 'str', 'all')
        # Getting the type of 'fkeys' (line 273)
        fkeys_539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 20), 'fkeys')
        # Applying the binary operator 'in' (line 273)
        result_contains_540 = python_operator(stypy.reporting.localization.Localization(__file__, 273, 11), 'in', str_538, fkeys_539)
        
        # Testing the type of an if condition (line 273)
        if_condition_541 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 273, 8), result_contains_540)
        # Assigning a type to the variable 'if_condition_541' (line 273)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 8), 'if_condition_541', if_condition_541)
        # SSA begins for if statement (line 273)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Call to keys(...): (line 274)
        # Processing the call keyword arguments (line 274)
        kwargs_544 = {}
        # Getting the type of 'formatdict' (line 274)
        formatdict_542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 23), 'formatdict', False)
        # Obtaining the member 'keys' of a type (line 274)
        keys_543 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 23), formatdict_542, 'keys')
        # Calling keys(args, kwargs) (line 274)
        keys_call_result_545 = invoke(stypy.reporting.localization.Localization(__file__, 274, 23), keys_543, *[], **kwargs_544)
        
        # Testing the type of a for loop iterable (line 274)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 274, 12), keys_call_result_545)
        # Getting the type of the for loop variable (line 274)
        for_loop_var_546 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 274, 12), keys_call_result_545)
        # Assigning a type to the variable 'key' (line 274)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 12), 'key', for_loop_var_546)
        # SSA begins for a for statement (line 274)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Subscript to a Subscript (line 275):
        
        # Assigning a Subscript to a Subscript (line 275):
        
        # Obtaining the type of the subscript
        str_547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 44), 'str', 'all')
        # Getting the type of 'formatter' (line 275)
        formatter_548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 34), 'formatter')
        # Obtaining the member '__getitem__' of a type (line 275)
        getitem___549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 34), formatter_548, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 275)
        subscript_call_result_550 = invoke(stypy.reporting.localization.Localization(__file__, 275, 34), getitem___549, str_547)
        
        # Getting the type of 'formatdict' (line 275)
        formatdict_551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 16), 'formatdict')
        # Getting the type of 'key' (line 275)
        key_552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 27), 'key')
        # Storing an element on a container (line 275)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 16), formatdict_551, (key_552, subscript_call_result_550))
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 273)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        str_553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 11), 'str', 'int_kind')
        # Getting the type of 'fkeys' (line 276)
        fkeys_554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 25), 'fkeys')
        # Applying the binary operator 'in' (line 276)
        result_contains_555 = python_operator(stypy.reporting.localization.Localization(__file__, 276, 11), 'in', str_553, fkeys_554)
        
        # Testing the type of an if condition (line 276)
        if_condition_556 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 276, 8), result_contains_555)
        # Assigning a type to the variable 'if_condition_556' (line 276)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 8), 'if_condition_556', if_condition_556)
        # SSA begins for if statement (line 276)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Obtaining an instance of the builtin type 'list' (line 277)
        list_557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 277)
        # Adding element type (line 277)
        str_558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 24), 'str', 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 277, 23), list_557, str_558)
        
        # Testing the type of a for loop iterable (line 277)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 277, 12), list_557)
        # Getting the type of the for loop variable (line 277)
        for_loop_var_559 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 277, 12), list_557)
        # Assigning a type to the variable 'key' (line 277)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 12), 'key', for_loop_var_559)
        # SSA begins for a for statement (line 277)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Subscript to a Subscript (line 278):
        
        # Assigning a Subscript to a Subscript (line 278):
        
        # Obtaining the type of the subscript
        str_560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 44), 'str', 'int_kind')
        # Getting the type of 'formatter' (line 278)
        formatter_561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 34), 'formatter')
        # Obtaining the member '__getitem__' of a type (line 278)
        getitem___562 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 34), formatter_561, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 278)
        subscript_call_result_563 = invoke(stypy.reporting.localization.Localization(__file__, 278, 34), getitem___562, str_560)
        
        # Getting the type of 'formatdict' (line 278)
        formatdict_564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 16), 'formatdict')
        # Getting the type of 'key' (line 278)
        key_565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 27), 'key')
        # Storing an element on a container (line 278)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 278, 16), formatdict_564, (key_565, subscript_call_result_563))
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 276)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        str_566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 11), 'str', 'float_kind')
        # Getting the type of 'fkeys' (line 279)
        fkeys_567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 27), 'fkeys')
        # Applying the binary operator 'in' (line 279)
        result_contains_568 = python_operator(stypy.reporting.localization.Localization(__file__, 279, 11), 'in', str_566, fkeys_567)
        
        # Testing the type of an if condition (line 279)
        if_condition_569 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 279, 8), result_contains_568)
        # Assigning a type to the variable 'if_condition_569' (line 279)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 8), 'if_condition_569', if_condition_569)
        # SSA begins for if statement (line 279)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Obtaining an instance of the builtin type 'list' (line 280)
        list_570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 280)
        # Adding element type (line 280)
        str_571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 24), 'str', 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 280, 23), list_570, str_571)
        # Adding element type (line 280)
        str_572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 33), 'str', 'longfloat')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 280, 23), list_570, str_572)
        
        # Testing the type of a for loop iterable (line 280)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 280, 12), list_570)
        # Getting the type of the for loop variable (line 280)
        for_loop_var_573 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 280, 12), list_570)
        # Assigning a type to the variable 'key' (line 280)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 12), 'key', for_loop_var_573)
        # SSA begins for a for statement (line 280)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Subscript to a Subscript (line 281):
        
        # Assigning a Subscript to a Subscript (line 281):
        
        # Obtaining the type of the subscript
        str_574 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 44), 'str', 'float_kind')
        # Getting the type of 'formatter' (line 281)
        formatter_575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 34), 'formatter')
        # Obtaining the member '__getitem__' of a type (line 281)
        getitem___576 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 34), formatter_575, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 281)
        subscript_call_result_577 = invoke(stypy.reporting.localization.Localization(__file__, 281, 34), getitem___576, str_574)
        
        # Getting the type of 'formatdict' (line 281)
        formatdict_578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 16), 'formatdict')
        # Getting the type of 'key' (line 281)
        key_579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 27), 'key')
        # Storing an element on a container (line 281)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 281, 16), formatdict_578, (key_579, subscript_call_result_577))
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 279)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        str_580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 11), 'str', 'complex_kind')
        # Getting the type of 'fkeys' (line 282)
        fkeys_581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 29), 'fkeys')
        # Applying the binary operator 'in' (line 282)
        result_contains_582 = python_operator(stypy.reporting.localization.Localization(__file__, 282, 11), 'in', str_580, fkeys_581)
        
        # Testing the type of an if condition (line 282)
        if_condition_583 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 282, 8), result_contains_582)
        # Assigning a type to the variable 'if_condition_583' (line 282)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 8), 'if_condition_583', if_condition_583)
        # SSA begins for if statement (line 282)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Obtaining an instance of the builtin type 'list' (line 283)
        list_584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 283)
        # Adding element type (line 283)
        str_585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 24), 'str', 'complexfloat')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 283, 23), list_584, str_585)
        # Adding element type (line 283)
        str_586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 40), 'str', 'longcomplexfloat')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 283, 23), list_584, str_586)
        
        # Testing the type of a for loop iterable (line 283)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 283, 12), list_584)
        # Getting the type of the for loop variable (line 283)
        for_loop_var_587 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 283, 12), list_584)
        # Assigning a type to the variable 'key' (line 283)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 12), 'key', for_loop_var_587)
        # SSA begins for a for statement (line 283)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Subscript to a Subscript (line 284):
        
        # Assigning a Subscript to a Subscript (line 284):
        
        # Obtaining the type of the subscript
        str_588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 44), 'str', 'complex_kind')
        # Getting the type of 'formatter' (line 284)
        formatter_589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 34), 'formatter')
        # Obtaining the member '__getitem__' of a type (line 284)
        getitem___590 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 34), formatter_589, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 284)
        subscript_call_result_591 = invoke(stypy.reporting.localization.Localization(__file__, 284, 34), getitem___590, str_588)
        
        # Getting the type of 'formatdict' (line 284)
        formatdict_592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 16), 'formatdict')
        # Getting the type of 'key' (line 284)
        key_593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 27), 'key')
        # Storing an element on a container (line 284)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 284, 16), formatdict_592, (key_593, subscript_call_result_591))
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 282)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        str_594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 11), 'str', 'str_kind')
        # Getting the type of 'fkeys' (line 285)
        fkeys_595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 25), 'fkeys')
        # Applying the binary operator 'in' (line 285)
        result_contains_596 = python_operator(stypy.reporting.localization.Localization(__file__, 285, 11), 'in', str_594, fkeys_595)
        
        # Testing the type of an if condition (line 285)
        if_condition_597 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 285, 8), result_contains_596)
        # Assigning a type to the variable 'if_condition_597' (line 285)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 8), 'if_condition_597', if_condition_597)
        # SSA begins for if statement (line 285)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Obtaining an instance of the builtin type 'list' (line 286)
        list_598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 286)
        # Adding element type (line 286)
        str_599 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 24), 'str', 'numpystr')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 286, 23), list_598, str_599)
        # Adding element type (line 286)
        str_600 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 36), 'str', 'str')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 286, 23), list_598, str_600)
        
        # Testing the type of a for loop iterable (line 286)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 286, 12), list_598)
        # Getting the type of the for loop variable (line 286)
        for_loop_var_601 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 286, 12), list_598)
        # Assigning a type to the variable 'key' (line 286)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 12), 'key', for_loop_var_601)
        # SSA begins for a for statement (line 286)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Subscript to a Subscript (line 287):
        
        # Assigning a Subscript to a Subscript (line 287):
        
        # Obtaining the type of the subscript
        str_602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 44), 'str', 'str_kind')
        # Getting the type of 'formatter' (line 287)
        formatter_603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 34), 'formatter')
        # Obtaining the member '__getitem__' of a type (line 287)
        getitem___604 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 34), formatter_603, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 287)
        subscript_call_result_605 = invoke(stypy.reporting.localization.Localization(__file__, 287, 34), getitem___604, str_602)
        
        # Getting the type of 'formatdict' (line 287)
        formatdict_606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 16), 'formatdict')
        # Getting the type of 'key' (line 287)
        key_607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 27), 'key')
        # Storing an element on a container (line 287)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 287, 16), formatdict_606, (key_607, subscript_call_result_605))
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 285)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to keys(...): (line 288)
        # Processing the call keyword arguments (line 288)
        kwargs_610 = {}
        # Getting the type of 'formatdict' (line 288)
        formatdict_608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 19), 'formatdict', False)
        # Obtaining the member 'keys' of a type (line 288)
        keys_609 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 19), formatdict_608, 'keys')
        # Calling keys(args, kwargs) (line 288)
        keys_call_result_611 = invoke(stypy.reporting.localization.Localization(__file__, 288, 19), keys_609, *[], **kwargs_610)
        
        # Testing the type of a for loop iterable (line 288)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 288, 8), keys_call_result_611)
        # Getting the type of the for loop variable (line 288)
        for_loop_var_612 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 288, 8), keys_call_result_611)
        # Assigning a type to the variable 'key' (line 288)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 8), 'key', for_loop_var_612)
        # SSA begins for a for statement (line 288)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Getting the type of 'key' (line 289)
        key_613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 15), 'key')
        # Getting the type of 'fkeys' (line 289)
        fkeys_614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 22), 'fkeys')
        # Applying the binary operator 'in' (line 289)
        result_contains_615 = python_operator(stypy.reporting.localization.Localization(__file__, 289, 15), 'in', key_613, fkeys_614)
        
        # Testing the type of an if condition (line 289)
        if_condition_616 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 289, 12), result_contains_615)
        # Assigning a type to the variable 'if_condition_616' (line 289)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 12), 'if_condition_616', if_condition_616)
        # SSA begins for if statement (line 289)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Subscript (line 290):
        
        # Assigning a Subscript to a Subscript (line 290):
        
        # Obtaining the type of the subscript
        # Getting the type of 'key' (line 290)
        key_617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 44), 'key')
        # Getting the type of 'formatter' (line 290)
        formatter_618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 34), 'formatter')
        # Obtaining the member '__getitem__' of a type (line 290)
        getitem___619 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 34), formatter_618, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 290)
        subscript_call_result_620 = invoke(stypy.reporting.localization.Localization(__file__, 290, 34), getitem___619, key_617)
        
        # Getting the type of 'formatdict' (line 290)
        formatdict_621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 16), 'formatdict')
        # Getting the type of 'key' (line 290)
        key_622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 27), 'key')
        # Storing an element on a container (line 290)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 290, 16), formatdict_621, (key_622, subscript_call_result_620))
        # SSA join for if statement (line 289)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        

        if more_types_in_union_524:
            # SSA join for if statement (line 271)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Attribute to a Name (line 293):
    
    # Assigning a Attribute to a Name (line 293):
    # Getting the type of 'a' (line 293)
    a_623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 15), 'a')
    # Obtaining the member 'dtype' of a type (line 293)
    dtype_624 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 15), a_623, 'dtype')
    # Obtaining the member 'type' of a type (line 293)
    type_625 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 15), dtype_624, 'type')
    # Assigning a type to the variable 'dtypeobj' (line 293)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 4), 'dtypeobj', type_625)
    
    
    # Call to issubclass(...): (line 294)
    # Processing the call arguments (line 294)
    # Getting the type of 'dtypeobj' (line 294)
    dtypeobj_627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 18), 'dtypeobj', False)
    # Getting the type of '_nt' (line 294)
    _nt_628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 28), '_nt', False)
    # Obtaining the member 'bool_' of a type (line 294)
    bool__629 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 28), _nt_628, 'bool_')
    # Processing the call keyword arguments (line 294)
    kwargs_630 = {}
    # Getting the type of 'issubclass' (line 294)
    issubclass_626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 7), 'issubclass', False)
    # Calling issubclass(args, kwargs) (line 294)
    issubclass_call_result_631 = invoke(stypy.reporting.localization.Localization(__file__, 294, 7), issubclass_626, *[dtypeobj_627, bool__629], **kwargs_630)
    
    # Testing the type of an if condition (line 294)
    if_condition_632 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 294, 4), issubclass_call_result_631)
    # Assigning a type to the variable 'if_condition_632' (line 294)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 4), 'if_condition_632', if_condition_632)
    # SSA begins for if statement (line 294)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 295):
    
    # Assigning a Subscript to a Name (line 295):
    
    # Obtaining the type of the subscript
    str_633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 37), 'str', 'bool')
    # Getting the type of 'formatdict' (line 295)
    formatdict_634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 26), 'formatdict')
    # Obtaining the member '__getitem__' of a type (line 295)
    getitem___635 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 26), formatdict_634, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 295)
    subscript_call_result_636 = invoke(stypy.reporting.localization.Localization(__file__, 295, 26), getitem___635, str_633)
    
    # Assigning a type to the variable 'format_function' (line 295)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 8), 'format_function', subscript_call_result_636)
    # SSA branch for the else part of an if statement (line 294)
    module_type_store.open_ssa_branch('else')
    
    
    # Call to issubclass(...): (line 296)
    # Processing the call arguments (line 296)
    # Getting the type of 'dtypeobj' (line 296)
    dtypeobj_638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 20), 'dtypeobj', False)
    # Getting the type of '_nt' (line 296)
    _nt_639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 30), '_nt', False)
    # Obtaining the member 'integer' of a type (line 296)
    integer_640 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 30), _nt_639, 'integer')
    # Processing the call keyword arguments (line 296)
    kwargs_641 = {}
    # Getting the type of 'issubclass' (line 296)
    issubclass_637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 9), 'issubclass', False)
    # Calling issubclass(args, kwargs) (line 296)
    issubclass_call_result_642 = invoke(stypy.reporting.localization.Localization(__file__, 296, 9), issubclass_637, *[dtypeobj_638, integer_640], **kwargs_641)
    
    # Testing the type of an if condition (line 296)
    if_condition_643 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 296, 9), issubclass_call_result_642)
    # Assigning a type to the variable 'if_condition_643' (line 296)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 296, 9), 'if_condition_643', if_condition_643)
    # SSA begins for if statement (line 296)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Call to issubclass(...): (line 297)
    # Processing the call arguments (line 297)
    # Getting the type of 'dtypeobj' (line 297)
    dtypeobj_645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 22), 'dtypeobj', False)
    # Getting the type of '_nt' (line 297)
    _nt_646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 32), '_nt', False)
    # Obtaining the member 'timedelta64' of a type (line 297)
    timedelta64_647 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 297, 32), _nt_646, 'timedelta64')
    # Processing the call keyword arguments (line 297)
    kwargs_648 = {}
    # Getting the type of 'issubclass' (line 297)
    issubclass_644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 11), 'issubclass', False)
    # Calling issubclass(args, kwargs) (line 297)
    issubclass_call_result_649 = invoke(stypy.reporting.localization.Localization(__file__, 297, 11), issubclass_644, *[dtypeobj_645, timedelta64_647], **kwargs_648)
    
    # Testing the type of an if condition (line 297)
    if_condition_650 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 297, 8), issubclass_call_result_649)
    # Assigning a type to the variable 'if_condition_650' (line 297)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 8), 'if_condition_650', if_condition_650)
    # SSA begins for if statement (line 297)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 298):
    
    # Assigning a Subscript to a Name (line 298):
    
    # Obtaining the type of the subscript
    str_651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 41), 'str', 'timedelta')
    # Getting the type of 'formatdict' (line 298)
    formatdict_652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 30), 'formatdict')
    # Obtaining the member '__getitem__' of a type (line 298)
    getitem___653 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 30), formatdict_652, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 298)
    subscript_call_result_654 = invoke(stypy.reporting.localization.Localization(__file__, 298, 30), getitem___653, str_651)
    
    # Assigning a type to the variable 'format_function' (line 298)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 12), 'format_function', subscript_call_result_654)
    # SSA branch for the else part of an if statement (line 297)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Subscript to a Name (line 300):
    
    # Assigning a Subscript to a Name (line 300):
    
    # Obtaining the type of the subscript
    str_655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, 41), 'str', 'int')
    # Getting the type of 'formatdict' (line 300)
    formatdict_656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 30), 'formatdict')
    # Obtaining the member '__getitem__' of a type (line 300)
    getitem___657 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 30), formatdict_656, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 300)
    subscript_call_result_658 = invoke(stypy.reporting.localization.Localization(__file__, 300, 30), getitem___657, str_655)
    
    # Assigning a type to the variable 'format_function' (line 300)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 12), 'format_function', subscript_call_result_658)
    # SSA join for if statement (line 297)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 296)
    module_type_store.open_ssa_branch('else')
    
    
    # Call to issubclass(...): (line 301)
    # Processing the call arguments (line 301)
    # Getting the type of 'dtypeobj' (line 301)
    dtypeobj_660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 20), 'dtypeobj', False)
    # Getting the type of '_nt' (line 301)
    _nt_661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 30), '_nt', False)
    # Obtaining the member 'floating' of a type (line 301)
    floating_662 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 30), _nt_661, 'floating')
    # Processing the call keyword arguments (line 301)
    kwargs_663 = {}
    # Getting the type of 'issubclass' (line 301)
    issubclass_659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 9), 'issubclass', False)
    # Calling issubclass(args, kwargs) (line 301)
    issubclass_call_result_664 = invoke(stypy.reporting.localization.Localization(__file__, 301, 9), issubclass_659, *[dtypeobj_660, floating_662], **kwargs_663)
    
    # Testing the type of an if condition (line 301)
    if_condition_665 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 301, 9), issubclass_call_result_664)
    # Assigning a type to the variable 'if_condition_665' (line 301)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 9), 'if_condition_665', if_condition_665)
    # SSA begins for if statement (line 301)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Call to issubclass(...): (line 302)
    # Processing the call arguments (line 302)
    # Getting the type of 'dtypeobj' (line 302)
    dtypeobj_667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 22), 'dtypeobj', False)
    # Getting the type of '_nt' (line 302)
    _nt_668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 32), '_nt', False)
    # Obtaining the member 'longfloat' of a type (line 302)
    longfloat_669 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 32), _nt_668, 'longfloat')
    # Processing the call keyword arguments (line 302)
    kwargs_670 = {}
    # Getting the type of 'issubclass' (line 302)
    issubclass_666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 11), 'issubclass', False)
    # Calling issubclass(args, kwargs) (line 302)
    issubclass_call_result_671 = invoke(stypy.reporting.localization.Localization(__file__, 302, 11), issubclass_666, *[dtypeobj_667, longfloat_669], **kwargs_670)
    
    # Testing the type of an if condition (line 302)
    if_condition_672 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 302, 8), issubclass_call_result_671)
    # Assigning a type to the variable 'if_condition_672' (line 302)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 302, 8), 'if_condition_672', if_condition_672)
    # SSA begins for if statement (line 302)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 303):
    
    # Assigning a Subscript to a Name (line 303):
    
    # Obtaining the type of the subscript
    str_673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 41), 'str', 'longfloat')
    # Getting the type of 'formatdict' (line 303)
    formatdict_674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 30), 'formatdict')
    # Obtaining the member '__getitem__' of a type (line 303)
    getitem___675 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 30), formatdict_674, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 303)
    subscript_call_result_676 = invoke(stypy.reporting.localization.Localization(__file__, 303, 30), getitem___675, str_673)
    
    # Assigning a type to the variable 'format_function' (line 303)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 12), 'format_function', subscript_call_result_676)
    # SSA branch for the else part of an if statement (line 302)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Subscript to a Name (line 305):
    
    # Assigning a Subscript to a Name (line 305):
    
    # Obtaining the type of the subscript
    str_677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 41), 'str', 'float')
    # Getting the type of 'formatdict' (line 305)
    formatdict_678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 30), 'formatdict')
    # Obtaining the member '__getitem__' of a type (line 305)
    getitem___679 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 305, 30), formatdict_678, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 305)
    subscript_call_result_680 = invoke(stypy.reporting.localization.Localization(__file__, 305, 30), getitem___679, str_677)
    
    # Assigning a type to the variable 'format_function' (line 305)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 305, 12), 'format_function', subscript_call_result_680)
    # SSA join for if statement (line 302)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 301)
    module_type_store.open_ssa_branch('else')
    
    
    # Call to issubclass(...): (line 306)
    # Processing the call arguments (line 306)
    # Getting the type of 'dtypeobj' (line 306)
    dtypeobj_682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 20), 'dtypeobj', False)
    # Getting the type of '_nt' (line 306)
    _nt_683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 30), '_nt', False)
    # Obtaining the member 'complexfloating' of a type (line 306)
    complexfloating_684 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 30), _nt_683, 'complexfloating')
    # Processing the call keyword arguments (line 306)
    kwargs_685 = {}
    # Getting the type of 'issubclass' (line 306)
    issubclass_681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 9), 'issubclass', False)
    # Calling issubclass(args, kwargs) (line 306)
    issubclass_call_result_686 = invoke(stypy.reporting.localization.Localization(__file__, 306, 9), issubclass_681, *[dtypeobj_682, complexfloating_684], **kwargs_685)
    
    # Testing the type of an if condition (line 306)
    if_condition_687 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 306, 9), issubclass_call_result_686)
    # Assigning a type to the variable 'if_condition_687' (line 306)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 9), 'if_condition_687', if_condition_687)
    # SSA begins for if statement (line 306)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Call to issubclass(...): (line 307)
    # Processing the call arguments (line 307)
    # Getting the type of 'dtypeobj' (line 307)
    dtypeobj_689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 22), 'dtypeobj', False)
    # Getting the type of '_nt' (line 307)
    _nt_690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 32), '_nt', False)
    # Obtaining the member 'clongfloat' of a type (line 307)
    clongfloat_691 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 32), _nt_690, 'clongfloat')
    # Processing the call keyword arguments (line 307)
    kwargs_692 = {}
    # Getting the type of 'issubclass' (line 307)
    issubclass_688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 11), 'issubclass', False)
    # Calling issubclass(args, kwargs) (line 307)
    issubclass_call_result_693 = invoke(stypy.reporting.localization.Localization(__file__, 307, 11), issubclass_688, *[dtypeobj_689, clongfloat_691], **kwargs_692)
    
    # Testing the type of an if condition (line 307)
    if_condition_694 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 307, 8), issubclass_call_result_693)
    # Assigning a type to the variable 'if_condition_694' (line 307)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 307, 8), 'if_condition_694', if_condition_694)
    # SSA begins for if statement (line 307)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 308):
    
    # Assigning a Subscript to a Name (line 308):
    
    # Obtaining the type of the subscript
    str_695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, 41), 'str', 'longcomplexfloat')
    # Getting the type of 'formatdict' (line 308)
    formatdict_696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 30), 'formatdict')
    # Obtaining the member '__getitem__' of a type (line 308)
    getitem___697 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 30), formatdict_696, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 308)
    subscript_call_result_698 = invoke(stypy.reporting.localization.Localization(__file__, 308, 30), getitem___697, str_695)
    
    # Assigning a type to the variable 'format_function' (line 308)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 308, 12), 'format_function', subscript_call_result_698)
    # SSA branch for the else part of an if statement (line 307)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Subscript to a Name (line 310):
    
    # Assigning a Subscript to a Name (line 310):
    
    # Obtaining the type of the subscript
    str_699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 41), 'str', 'complexfloat')
    # Getting the type of 'formatdict' (line 310)
    formatdict_700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 30), 'formatdict')
    # Obtaining the member '__getitem__' of a type (line 310)
    getitem___701 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 30), formatdict_700, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 310)
    subscript_call_result_702 = invoke(stypy.reporting.localization.Localization(__file__, 310, 30), getitem___701, str_699)
    
    # Assigning a type to the variable 'format_function' (line 310)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 12), 'format_function', subscript_call_result_702)
    # SSA join for if statement (line 307)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 306)
    module_type_store.open_ssa_branch('else')
    
    
    # Call to issubclass(...): (line 311)
    # Processing the call arguments (line 311)
    # Getting the type of 'dtypeobj' (line 311)
    dtypeobj_704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 20), 'dtypeobj', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 311)
    tuple_705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 31), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 311)
    # Adding element type (line 311)
    # Getting the type of '_nt' (line 311)
    _nt_706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 31), '_nt', False)
    # Obtaining the member 'unicode_' of a type (line 311)
    unicode__707 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 31), _nt_706, 'unicode_')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 311, 31), tuple_705, unicode__707)
    # Adding element type (line 311)
    # Getting the type of '_nt' (line 311)
    _nt_708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 45), '_nt', False)
    # Obtaining the member 'string_' of a type (line 311)
    string__709 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 45), _nt_708, 'string_')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 311, 31), tuple_705, string__709)
    
    # Processing the call keyword arguments (line 311)
    kwargs_710 = {}
    # Getting the type of 'issubclass' (line 311)
    issubclass_703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 9), 'issubclass', False)
    # Calling issubclass(args, kwargs) (line 311)
    issubclass_call_result_711 = invoke(stypy.reporting.localization.Localization(__file__, 311, 9), issubclass_703, *[dtypeobj_704, tuple_705], **kwargs_710)
    
    # Testing the type of an if condition (line 311)
    if_condition_712 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 311, 9), issubclass_call_result_711)
    # Assigning a type to the variable 'if_condition_712' (line 311)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 9), 'if_condition_712', if_condition_712)
    # SSA begins for if statement (line 311)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 312):
    
    # Assigning a Subscript to a Name (line 312):
    
    # Obtaining the type of the subscript
    str_713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 37), 'str', 'numpystr')
    # Getting the type of 'formatdict' (line 312)
    formatdict_714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 26), 'formatdict')
    # Obtaining the member '__getitem__' of a type (line 312)
    getitem___715 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 26), formatdict_714, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 312)
    subscript_call_result_716 = invoke(stypy.reporting.localization.Localization(__file__, 312, 26), getitem___715, str_713)
    
    # Assigning a type to the variable 'format_function' (line 312)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 8), 'format_function', subscript_call_result_716)
    # SSA branch for the else part of an if statement (line 311)
    module_type_store.open_ssa_branch('else')
    
    
    # Call to issubclass(...): (line 313)
    # Processing the call arguments (line 313)
    # Getting the type of 'dtypeobj' (line 313)
    dtypeobj_718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 20), 'dtypeobj', False)
    # Getting the type of '_nt' (line 313)
    _nt_719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 30), '_nt', False)
    # Obtaining the member 'datetime64' of a type (line 313)
    datetime64_720 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 30), _nt_719, 'datetime64')
    # Processing the call keyword arguments (line 313)
    kwargs_721 = {}
    # Getting the type of 'issubclass' (line 313)
    issubclass_717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 9), 'issubclass', False)
    # Calling issubclass(args, kwargs) (line 313)
    issubclass_call_result_722 = invoke(stypy.reporting.localization.Localization(__file__, 313, 9), issubclass_717, *[dtypeobj_718, datetime64_720], **kwargs_721)
    
    # Testing the type of an if condition (line 313)
    if_condition_723 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 313, 9), issubclass_call_result_722)
    # Assigning a type to the variable 'if_condition_723' (line 313)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 313, 9), 'if_condition_723', if_condition_723)
    # SSA begins for if statement (line 313)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 314):
    
    # Assigning a Subscript to a Name (line 314):
    
    # Obtaining the type of the subscript
    str_724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 37), 'str', 'datetime')
    # Getting the type of 'formatdict' (line 314)
    formatdict_725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 26), 'formatdict')
    # Obtaining the member '__getitem__' of a type (line 314)
    getitem___726 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 26), formatdict_725, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 314)
    subscript_call_result_727 = invoke(stypy.reporting.localization.Localization(__file__, 314, 26), getitem___726, str_724)
    
    # Assigning a type to the variable 'format_function' (line 314)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 314, 8), 'format_function', subscript_call_result_727)
    # SSA branch for the else part of an if statement (line 313)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Subscript to a Name (line 316):
    
    # Assigning a Subscript to a Name (line 316):
    
    # Obtaining the type of the subscript
    str_728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 316, 37), 'str', 'numpystr')
    # Getting the type of 'formatdict' (line 316)
    formatdict_729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 26), 'formatdict')
    # Obtaining the member '__getitem__' of a type (line 316)
    getitem___730 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 26), formatdict_729, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 316)
    subscript_call_result_731 = invoke(stypy.reporting.localization.Localization(__file__, 316, 26), getitem___730, str_728)
    
    # Assigning a type to the variable 'format_function' (line 316)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 316, 8), 'format_function', subscript_call_result_731)
    # SSA join for if statement (line 313)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 311)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 306)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 301)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 296)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 294)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Str to a Name (line 319):
    
    # Assigning a Str to a Name (line 319):
    str_732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 23), 'str', ' ')
    # Assigning a type to the variable 'next_line_prefix' (line 319)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 319, 4), 'next_line_prefix', str_732)
    
    # Getting the type of 'next_line_prefix' (line 321)
    next_line_prefix_733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 4), 'next_line_prefix')
    str_734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 321, 24), 'str', ' ')
    
    # Call to len(...): (line 321)
    # Processing the call arguments (line 321)
    # Getting the type of 'prefix' (line 321)
    prefix_736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 32), 'prefix', False)
    # Processing the call keyword arguments (line 321)
    kwargs_737 = {}
    # Getting the type of 'len' (line 321)
    len_735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 28), 'len', False)
    # Calling len(args, kwargs) (line 321)
    len_call_result_738 = invoke(stypy.reporting.localization.Localization(__file__, 321, 28), len_735, *[prefix_736], **kwargs_737)
    
    # Applying the binary operator '*' (line 321)
    result_mul_739 = python_operator(stypy.reporting.localization.Localization(__file__, 321, 24), '*', str_734, len_call_result_738)
    
    # Applying the binary operator '+=' (line 321)
    result_iadd_740 = python_operator(stypy.reporting.localization.Localization(__file__, 321, 4), '+=', next_line_prefix_733, result_mul_739)
    # Assigning a type to the variable 'next_line_prefix' (line 321)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 321, 4), 'next_line_prefix', result_iadd_740)
    
    
    # Assigning a Subscript to a Name (line 323):
    
    # Assigning a Subscript to a Name (line 323):
    
    # Obtaining the type of the subscript
    int_741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 59), 'int')
    slice_742 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 323, 10), None, int_741, None)
    
    # Call to _formatArray(...): (line 323)
    # Processing the call arguments (line 323)
    # Getting the type of 'a' (line 323)
    a_744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 23), 'a', False)
    # Getting the type of 'format_function' (line 323)
    format_function_745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 26), 'format_function', False)
    
    # Call to len(...): (line 323)
    # Processing the call arguments (line 323)
    # Getting the type of 'a' (line 323)
    a_747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 47), 'a', False)
    # Obtaining the member 'shape' of a type (line 323)
    shape_748 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 323, 47), a_747, 'shape')
    # Processing the call keyword arguments (line 323)
    kwargs_749 = {}
    # Getting the type of 'len' (line 323)
    len_746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 43), 'len', False)
    # Calling len(args, kwargs) (line 323)
    len_call_result_750 = invoke(stypy.reporting.localization.Localization(__file__, 323, 43), len_746, *[shape_748], **kwargs_749)
    
    # Getting the type of 'max_line_width' (line 323)
    max_line_width_751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 57), 'max_line_width', False)
    # Getting the type of 'next_line_prefix' (line 324)
    next_line_prefix_752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 23), 'next_line_prefix', False)
    # Getting the type of 'separator' (line 324)
    separator_753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 41), 'separator', False)
    # Getting the type of '_summaryEdgeItems' (line 325)
    _summaryEdgeItems_754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 23), '_summaryEdgeItems', False)
    # Getting the type of 'summary_insert' (line 325)
    summary_insert_755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 42), 'summary_insert', False)
    # Processing the call keyword arguments (line 323)
    kwargs_756 = {}
    # Getting the type of '_formatArray' (line 323)
    _formatArray_743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 10), '_formatArray', False)
    # Calling _formatArray(args, kwargs) (line 323)
    _formatArray_call_result_757 = invoke(stypy.reporting.localization.Localization(__file__, 323, 10), _formatArray_743, *[a_744, format_function_745, len_call_result_750, max_line_width_751, next_line_prefix_752, separator_753, _summaryEdgeItems_754, summary_insert_755], **kwargs_756)
    
    # Obtaining the member '__getitem__' of a type (line 323)
    getitem___758 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 323, 10), _formatArray_call_result_757, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 323)
    subscript_call_result_759 = invoke(stypy.reporting.localization.Localization(__file__, 323, 10), getitem___758, slice_742)
    
    # Assigning a type to the variable 'lst' (line 323)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 4), 'lst', subscript_call_result_759)
    # Getting the type of 'lst' (line 326)
    lst_760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 11), 'lst')
    # Assigning a type to the variable 'stypy_return_type' (line 326)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 4), 'stypy_return_type', lst_760)
    
    # ################# End of '_array2string(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_array2string' in the type store
    # Getting the type of 'stypy_return_type' (line 237)
    stypy_return_type_761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_761)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_array2string'
    return stypy_return_type_761

# Assigning a type to the variable '_array2string' (line 237)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 0), '_array2string', _array2string)

@norecursion
def _convert_arrays(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_convert_arrays'
    module_type_store = module_type_store.open_function_context('_convert_arrays', 328, 0, False)
    
    # Passed parameters checking function
    _convert_arrays.stypy_localization = localization
    _convert_arrays.stypy_type_of_self = None
    _convert_arrays.stypy_type_store = module_type_store
    _convert_arrays.stypy_function_name = '_convert_arrays'
    _convert_arrays.stypy_param_names_list = ['obj']
    _convert_arrays.stypy_varargs_param_name = None
    _convert_arrays.stypy_kwargs_param_name = None
    _convert_arrays.stypy_call_defaults = defaults
    _convert_arrays.stypy_call_varargs = varargs
    _convert_arrays.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_convert_arrays', ['obj'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_convert_arrays', localization, ['obj'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_convert_arrays(...)' code ##################

    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 329, 4))
    
    # 'from numpy.core import _nc' statement (line 329)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/core/')
    import_762 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 329, 4), 'numpy.core')

    if (type(import_762) is not StypyTypeError):

        if (import_762 != 'pyd_module'):
            __import__(import_762)
            sys_modules_763 = sys.modules[import_762]
            import_from_module(stypy.reporting.localization.Localization(__file__, 329, 4), 'numpy.core', sys_modules_763.module_type_store, module_type_store, ['numeric'])
            nest_module(stypy.reporting.localization.Localization(__file__, 329, 4), __file__, sys_modules_763, sys_modules_763.module_type_store, module_type_store)
        else:
            from numpy.core import numeric as _nc

            import_from_module(stypy.reporting.localization.Localization(__file__, 329, 4), 'numpy.core', None, module_type_store, ['numeric'], [_nc])

    else:
        # Assigning a type to the variable 'numpy.core' (line 329)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 4), 'numpy.core', import_762)

    # Adding an alias
    module_type_store.add_alias('_nc', 'numeric')
    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/core/')
    
    
    # Assigning a List to a Name (line 330):
    
    # Assigning a List to a Name (line 330):
    
    # Obtaining an instance of the builtin type 'list' (line 330)
    list_764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 13), 'list')
    # Adding type elements to the builtin type 'list' instance (line 330)
    
    # Assigning a type to the variable 'newtup' (line 330)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 4), 'newtup', list_764)
    
    # Getting the type of 'obj' (line 331)
    obj_765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 13), 'obj')
    # Testing the type of a for loop iterable (line 331)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 331, 4), obj_765)
    # Getting the type of the for loop variable (line 331)
    for_loop_var_766 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 331, 4), obj_765)
    # Assigning a type to the variable 'k' (line 331)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 4), 'k', for_loop_var_766)
    # SSA begins for a for statement (line 331)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Call to isinstance(...): (line 332)
    # Processing the call arguments (line 332)
    # Getting the type of 'k' (line 332)
    k_768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 22), 'k', False)
    # Getting the type of '_nc' (line 332)
    _nc_769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 25), '_nc', False)
    # Obtaining the member 'ndarray' of a type (line 332)
    ndarray_770 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 25), _nc_769, 'ndarray')
    # Processing the call keyword arguments (line 332)
    kwargs_771 = {}
    # Getting the type of 'isinstance' (line 332)
    isinstance_767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 11), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 332)
    isinstance_call_result_772 = invoke(stypy.reporting.localization.Localization(__file__, 332, 11), isinstance_767, *[k_768, ndarray_770], **kwargs_771)
    
    # Testing the type of an if condition (line 332)
    if_condition_773 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 332, 8), isinstance_call_result_772)
    # Assigning a type to the variable 'if_condition_773' (line 332)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 332, 8), 'if_condition_773', if_condition_773)
    # SSA begins for if statement (line 332)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 333):
    
    # Assigning a Call to a Name (line 333):
    
    # Call to tolist(...): (line 333)
    # Processing the call keyword arguments (line 333)
    kwargs_776 = {}
    # Getting the type of 'k' (line 333)
    k_774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 16), 'k', False)
    # Obtaining the member 'tolist' of a type (line 333)
    tolist_775 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 16), k_774, 'tolist')
    # Calling tolist(args, kwargs) (line 333)
    tolist_call_result_777 = invoke(stypy.reporting.localization.Localization(__file__, 333, 16), tolist_775, *[], **kwargs_776)
    
    # Assigning a type to the variable 'k' (line 333)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 333, 12), 'k', tolist_call_result_777)
    # SSA branch for the else part of an if statement (line 332)
    module_type_store.open_ssa_branch('else')
    
    # Type idiom detected: calculating its left and rigth part (line 334)
    # Getting the type of 'tuple' (line 334)
    tuple_778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 27), 'tuple')
    # Getting the type of 'k' (line 334)
    k_779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 24), 'k')
    
    (may_be_780, more_types_in_union_781) = may_be_subtype(tuple_778, k_779)

    if may_be_780:

        if more_types_in_union_781:
            # Runtime conditional SSA (line 334)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'k' (line 334)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 13), 'k', remove_not_subtype_from_union(k_779, tuple))
        
        # Assigning a Call to a Name (line 335):
        
        # Assigning a Call to a Name (line 335):
        
        # Call to _convert_arrays(...): (line 335)
        # Processing the call arguments (line 335)
        # Getting the type of 'k' (line 335)
        k_783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 32), 'k', False)
        # Processing the call keyword arguments (line 335)
        kwargs_784 = {}
        # Getting the type of '_convert_arrays' (line 335)
        _convert_arrays_782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 16), '_convert_arrays', False)
        # Calling _convert_arrays(args, kwargs) (line 335)
        _convert_arrays_call_result_785 = invoke(stypy.reporting.localization.Localization(__file__, 335, 16), _convert_arrays_782, *[k_783], **kwargs_784)
        
        # Assigning a type to the variable 'k' (line 335)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 12), 'k', _convert_arrays_call_result_785)

        if more_types_in_union_781:
            # SSA join for if statement (line 334)
            module_type_store = module_type_store.join_ssa_context()


    
    # SSA join for if statement (line 332)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to append(...): (line 336)
    # Processing the call arguments (line 336)
    # Getting the type of 'k' (line 336)
    k_788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 22), 'k', False)
    # Processing the call keyword arguments (line 336)
    kwargs_789 = {}
    # Getting the type of 'newtup' (line 336)
    newtup_786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 8), 'newtup', False)
    # Obtaining the member 'append' of a type (line 336)
    append_787 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 8), newtup_786, 'append')
    # Calling append(args, kwargs) (line 336)
    append_call_result_790 = invoke(stypy.reporting.localization.Localization(__file__, 336, 8), append_787, *[k_788], **kwargs_789)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to tuple(...): (line 337)
    # Processing the call arguments (line 337)
    # Getting the type of 'newtup' (line 337)
    newtup_792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 17), 'newtup', False)
    # Processing the call keyword arguments (line 337)
    kwargs_793 = {}
    # Getting the type of 'tuple' (line 337)
    tuple_791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 11), 'tuple', False)
    # Calling tuple(args, kwargs) (line 337)
    tuple_call_result_794 = invoke(stypy.reporting.localization.Localization(__file__, 337, 11), tuple_791, *[newtup_792], **kwargs_793)
    
    # Assigning a type to the variable 'stypy_return_type' (line 337)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 337, 4), 'stypy_return_type', tuple_call_result_794)
    
    # ################# End of '_convert_arrays(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_convert_arrays' in the type store
    # Getting the type of 'stypy_return_type' (line 328)
    stypy_return_type_795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_795)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_convert_arrays'
    return stypy_return_type_795

# Assigning a type to the variable '_convert_arrays' (line 328)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 328, 0), '_convert_arrays', _convert_arrays)

@norecursion
def array2string(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 340)
    None_796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 35), 'None')
    # Getting the type of 'None' (line 340)
    None_797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 51), 'None')
    # Getting the type of 'None' (line 341)
    None_798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 32), 'None')
    str_799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 48), 'str', ' ')
    str_800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 60), 'str', '')
    # Getting the type of 'repr' (line 342)
    repr_801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 23), 'repr')
    # Getting the type of 'None' (line 342)
    None_802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 39), 'None')
    defaults = [None_796, None_797, None_798, str_799, str_800, repr_801, None_802]
    # Create a new context for function 'array2string'
    module_type_store = module_type_store.open_function_context('array2string', 340, 0, False)
    
    # Passed parameters checking function
    array2string.stypy_localization = localization
    array2string.stypy_type_of_self = None
    array2string.stypy_type_store = module_type_store
    array2string.stypy_function_name = 'array2string'
    array2string.stypy_param_names_list = ['a', 'max_line_width', 'precision', 'suppress_small', 'separator', 'prefix', 'style', 'formatter']
    array2string.stypy_varargs_param_name = None
    array2string.stypy_kwargs_param_name = None
    array2string.stypy_call_defaults = defaults
    array2string.stypy_call_varargs = varargs
    array2string.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'array2string', ['a', 'max_line_width', 'precision', 'suppress_small', 'separator', 'prefix', 'style', 'formatter'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'array2string', localization, ['a', 'max_line_width', 'precision', 'suppress_small', 'separator', 'prefix', 'style', 'formatter'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'array2string(...)' code ##################

    str_803 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 435, (-1)), 'str', '\n    Return a string representation of an array.\n\n    Parameters\n    ----------\n    a : ndarray\n        Input array.\n    max_line_width : int, optional\n        The maximum number of columns the string should span. Newline\n        characters splits the string appropriately after array elements.\n    precision : int, optional\n        Floating point precision. Default is the current printing\n        precision (usually 8), which can be altered using `set_printoptions`.\n    suppress_small : bool, optional\n        Represent very small numbers as zero. A number is "very small" if it\n        is smaller than the current printing precision.\n    separator : str, optional\n        Inserted between elements.\n    prefix : str, optional\n        An array is typically printed as::\n\n          \'prefix(\' + array2string(a) + \')\'\n\n        The length of the prefix string is used to align the\n        output correctly.\n    style : function, optional\n        A function that accepts an ndarray and returns a string.  Used only\n        when the shape of `a` is equal to ``()``, i.e. for 0-D arrays.\n    formatter : dict of callables, optional\n        If not None, the keys should indicate the type(s) that the respective\n        formatting function applies to.  Callables should return a string.\n        Types that are not specified (by their corresponding keys) are handled\n        by the default formatters.  Individual types for which a formatter\n        can be set are::\n\n            - \'bool\'\n            - \'int\'\n            - \'timedelta\' : a `numpy.timedelta64`\n            - \'datetime\' : a `numpy.datetime64`\n            - \'float\'\n            - \'longfloat\' : 128-bit floats\n            - \'complexfloat\'\n            - \'longcomplexfloat\' : composed of two 128-bit floats\n            - \'numpy_str\' : types `numpy.string_` and `numpy.unicode_`\n            - \'str\' : all other strings\n\n        Other keys that can be used to set a group of types at once are::\n\n            - \'all\' : sets all types\n            - \'int_kind\' : sets \'int\'\n            - \'float_kind\' : sets \'float\' and \'longfloat\'\n            - \'complex_kind\' : sets \'complexfloat\' and \'longcomplexfloat\'\n            - \'str_kind\' : sets \'str\' and \'numpystr\'\n\n    Returns\n    -------\n    array_str : str\n        String representation of the array.\n\n    Raises\n    ------\n    TypeError\n        if a callable in `formatter` does not return a string.\n\n    See Also\n    --------\n    array_str, array_repr, set_printoptions, get_printoptions\n\n    Notes\n    -----\n    If a formatter is specified for a certain type, the `precision` keyword is\n    ignored for that type.\n\n    This is a very flexible function; `array_repr` and `array_str` are using\n    `array2string` internally so keywords with the same name should work\n    identically in all three functions.\n\n    Examples\n    --------\n    >>> x = np.array([1e-16,1,2,3])\n    >>> print(np.array2string(x, precision=2, separator=\',\',\n    ...                       suppress_small=True))\n    [ 0., 1., 2., 3.]\n\n    >>> x  = np.arange(3.)\n    >>> np.array2string(x, formatter={\'float_kind\':lambda x: "%.2f" % x})\n    \'[0.00 1.00 2.00]\'\n\n    >>> x  = np.arange(3)\n    >>> np.array2string(x, formatter={\'int\':lambda x: hex(x)})\n    \'[0x0L 0x1L 0x2L]\'\n\n    ')
    
    
    # Getting the type of 'a' (line 437)
    a_804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 7), 'a')
    # Obtaining the member 'shape' of a type (line 437)
    shape_805 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 437, 7), a_804, 'shape')
    
    # Obtaining an instance of the builtin type 'tuple' (line 437)
    tuple_806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 437, 18), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 437)
    
    # Applying the binary operator '==' (line 437)
    result_eq_807 = python_operator(stypy.reporting.localization.Localization(__file__, 437, 7), '==', shape_805, tuple_806)
    
    # Testing the type of an if condition (line 437)
    if_condition_808 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 437, 4), result_eq_807)
    # Assigning a type to the variable 'if_condition_808' (line 437)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 437, 4), 'if_condition_808', if_condition_808)
    # SSA begins for if statement (line 437)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 438):
    
    # Assigning a Call to a Name (line 438):
    
    # Call to item(...): (line 438)
    # Processing the call keyword arguments (line 438)
    kwargs_811 = {}
    # Getting the type of 'a' (line 438)
    a_809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 12), 'a', False)
    # Obtaining the member 'item' of a type (line 438)
    item_810 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 438, 12), a_809, 'item')
    # Calling item(args, kwargs) (line 438)
    item_call_result_812 = invoke(stypy.reporting.localization.Localization(__file__, 438, 12), item_810, *[], **kwargs_811)
    
    # Assigning a type to the variable 'x' (line 438)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 438, 8), 'x', item_call_result_812)
    
    # Type idiom detected: calculating its left and rigth part (line 439)
    # Getting the type of 'tuple' (line 439)
    tuple_813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 25), 'tuple')
    # Getting the type of 'x' (line 439)
    x_814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 22), 'x')
    
    (may_be_815, more_types_in_union_816) = may_be_subtype(tuple_813, x_814)

    if may_be_815:

        if more_types_in_union_816:
            # Runtime conditional SSA (line 439)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'x' (line 439)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 439, 8), 'x', remove_not_subtype_from_union(x_814, tuple))
        
        # Assigning a Call to a Name (line 440):
        
        # Assigning a Call to a Name (line 440):
        
        # Call to _convert_arrays(...): (line 440)
        # Processing the call arguments (line 440)
        # Getting the type of 'x' (line 440)
        x_818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 32), 'x', False)
        # Processing the call keyword arguments (line 440)
        kwargs_819 = {}
        # Getting the type of '_convert_arrays' (line 440)
        _convert_arrays_817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 16), '_convert_arrays', False)
        # Calling _convert_arrays(args, kwargs) (line 440)
        _convert_arrays_call_result_820 = invoke(stypy.reporting.localization.Localization(__file__, 440, 16), _convert_arrays_817, *[x_818], **kwargs_819)
        
        # Assigning a type to the variable 'x' (line 440)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 440, 12), 'x', _convert_arrays_call_result_820)

        if more_types_in_union_816:
            # SSA join for if statement (line 439)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 441):
    
    # Assigning a Call to a Name (line 441):
    
    # Call to style(...): (line 441)
    # Processing the call arguments (line 441)
    # Getting the type of 'x' (line 441)
    x_822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 20), 'x', False)
    # Processing the call keyword arguments (line 441)
    kwargs_823 = {}
    # Getting the type of 'style' (line 441)
    style_821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 14), 'style', False)
    # Calling style(args, kwargs) (line 441)
    style_call_result_824 = invoke(stypy.reporting.localization.Localization(__file__, 441, 14), style_821, *[x_822], **kwargs_823)
    
    # Assigning a type to the variable 'lst' (line 441)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 441, 8), 'lst', style_call_result_824)
    # SSA branch for the else part of an if statement (line 437)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Call to reduce(...): (line 442)
    # Processing the call arguments (line 442)
    # Getting the type of 'product' (line 442)
    product_826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 16), 'product', False)
    # Getting the type of 'a' (line 442)
    a_827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 25), 'a', False)
    # Obtaining the member 'shape' of a type (line 442)
    shape_828 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 442, 25), a_827, 'shape')
    # Processing the call keyword arguments (line 442)
    kwargs_829 = {}
    # Getting the type of 'reduce' (line 442)
    reduce_825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 9), 'reduce', False)
    # Calling reduce(args, kwargs) (line 442)
    reduce_call_result_830 = invoke(stypy.reporting.localization.Localization(__file__, 442, 9), reduce_825, *[product_826, shape_828], **kwargs_829)
    
    int_831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 442, 37), 'int')
    # Applying the binary operator '==' (line 442)
    result_eq_832 = python_operator(stypy.reporting.localization.Localization(__file__, 442, 9), '==', reduce_call_result_830, int_831)
    
    # Testing the type of an if condition (line 442)
    if_condition_833 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 442, 9), result_eq_832)
    # Assigning a type to the variable 'if_condition_833' (line 442)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 442, 9), 'if_condition_833', if_condition_833)
    # SSA begins for if statement (line 442)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 444):
    
    # Assigning a Str to a Name (line 444):
    str_834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 444, 14), 'str', '[]')
    # Assigning a type to the variable 'lst' (line 444)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 8), 'lst', str_834)
    # SSA branch for the else part of an if statement (line 442)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 446):
    
    # Assigning a Call to a Name (line 446):
    
    # Call to _array2string(...): (line 446)
    # Processing the call arguments (line 446)
    # Getting the type of 'a' (line 446)
    a_836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 28), 'a', False)
    # Getting the type of 'max_line_width' (line 446)
    max_line_width_837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 31), 'max_line_width', False)
    # Getting the type of 'precision' (line 446)
    precision_838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 47), 'precision', False)
    # Getting the type of 'suppress_small' (line 446)
    suppress_small_839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 58), 'suppress_small', False)
    # Getting the type of 'separator' (line 447)
    separator_840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 28), 'separator', False)
    # Getting the type of 'prefix' (line 447)
    prefix_841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 39), 'prefix', False)
    # Processing the call keyword arguments (line 446)
    # Getting the type of 'formatter' (line 447)
    formatter_842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 57), 'formatter', False)
    keyword_843 = formatter_842
    kwargs_844 = {'formatter': keyword_843}
    # Getting the type of '_array2string' (line 446)
    _array2string_835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 14), '_array2string', False)
    # Calling _array2string(args, kwargs) (line 446)
    _array2string_call_result_845 = invoke(stypy.reporting.localization.Localization(__file__, 446, 14), _array2string_835, *[a_836, max_line_width_837, precision_838, suppress_small_839, separator_840, prefix_841], **kwargs_844)
    
    # Assigning a type to the variable 'lst' (line 446)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 446, 8), 'lst', _array2string_call_result_845)
    # SSA join for if statement (line 442)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 437)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'lst' (line 448)
    lst_846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 11), 'lst')
    # Assigning a type to the variable 'stypy_return_type' (line 448)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 448, 4), 'stypy_return_type', lst_846)
    
    # ################# End of 'array2string(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'array2string' in the type store
    # Getting the type of 'stypy_return_type' (line 340)
    stypy_return_type_847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_847)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'array2string'
    return stypy_return_type_847

# Assigning a type to the variable 'array2string' (line 340)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 0), 'array2string', array2string)

@norecursion
def _extendLine(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_extendLine'
    module_type_store = module_type_store.open_function_context('_extendLine', 450, 0, False)
    
    # Passed parameters checking function
    _extendLine.stypy_localization = localization
    _extendLine.stypy_type_of_self = None
    _extendLine.stypy_type_store = module_type_store
    _extendLine.stypy_function_name = '_extendLine'
    _extendLine.stypy_param_names_list = ['s', 'line', 'word', 'max_line_len', 'next_line_prefix']
    _extendLine.stypy_varargs_param_name = None
    _extendLine.stypy_kwargs_param_name = None
    _extendLine.stypy_call_defaults = defaults
    _extendLine.stypy_call_varargs = varargs
    _extendLine.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_extendLine', ['s', 'line', 'word', 'max_line_len', 'next_line_prefix'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_extendLine', localization, ['s', 'line', 'word', 'max_line_len', 'next_line_prefix'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_extendLine(...)' code ##################

    
    
    
    # Call to len(...): (line 451)
    # Processing the call arguments (line 451)
    
    # Call to rstrip(...): (line 451)
    # Processing the call keyword arguments (line 451)
    kwargs_851 = {}
    # Getting the type of 'line' (line 451)
    line_849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 11), 'line', False)
    # Obtaining the member 'rstrip' of a type (line 451)
    rstrip_850 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 451, 11), line_849, 'rstrip')
    # Calling rstrip(args, kwargs) (line 451)
    rstrip_call_result_852 = invoke(stypy.reporting.localization.Localization(__file__, 451, 11), rstrip_850, *[], **kwargs_851)
    
    # Processing the call keyword arguments (line 451)
    kwargs_853 = {}
    # Getting the type of 'len' (line 451)
    len_848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 7), 'len', False)
    # Calling len(args, kwargs) (line 451)
    len_call_result_854 = invoke(stypy.reporting.localization.Localization(__file__, 451, 7), len_848, *[rstrip_call_result_852], **kwargs_853)
    
    
    # Call to len(...): (line 451)
    # Processing the call arguments (line 451)
    
    # Call to rstrip(...): (line 451)
    # Processing the call keyword arguments (line 451)
    kwargs_858 = {}
    # Getting the type of 'word' (line 451)
    word_856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 32), 'word', False)
    # Obtaining the member 'rstrip' of a type (line 451)
    rstrip_857 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 451, 32), word_856, 'rstrip')
    # Calling rstrip(args, kwargs) (line 451)
    rstrip_call_result_859 = invoke(stypy.reporting.localization.Localization(__file__, 451, 32), rstrip_857, *[], **kwargs_858)
    
    # Processing the call keyword arguments (line 451)
    kwargs_860 = {}
    # Getting the type of 'len' (line 451)
    len_855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 28), 'len', False)
    # Calling len(args, kwargs) (line 451)
    len_call_result_861 = invoke(stypy.reporting.localization.Localization(__file__, 451, 28), len_855, *[rstrip_call_result_859], **kwargs_860)
    
    # Applying the binary operator '+' (line 451)
    result_add_862 = python_operator(stypy.reporting.localization.Localization(__file__, 451, 7), '+', len_call_result_854, len_call_result_861)
    
    # Getting the type of 'max_line_len' (line 451)
    max_line_len_863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 50), 'max_line_len')
    # Applying the binary operator '>=' (line 451)
    result_ge_864 = python_operator(stypy.reporting.localization.Localization(__file__, 451, 7), '>=', result_add_862, max_line_len_863)
    
    # Testing the type of an if condition (line 451)
    if_condition_865 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 451, 4), result_ge_864)
    # Assigning a type to the variable 'if_condition_865' (line 451)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 451, 4), 'if_condition_865', if_condition_865)
    # SSA begins for if statement (line 451)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 's' (line 452)
    s_866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 8), 's')
    
    # Call to rstrip(...): (line 452)
    # Processing the call keyword arguments (line 452)
    kwargs_869 = {}
    # Getting the type of 'line' (line 452)
    line_867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 13), 'line', False)
    # Obtaining the member 'rstrip' of a type (line 452)
    rstrip_868 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 452, 13), line_867, 'rstrip')
    # Calling rstrip(args, kwargs) (line 452)
    rstrip_call_result_870 = invoke(stypy.reporting.localization.Localization(__file__, 452, 13), rstrip_868, *[], **kwargs_869)
    
    str_871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 452, 29), 'str', '\n')
    # Applying the binary operator '+' (line 452)
    result_add_872 = python_operator(stypy.reporting.localization.Localization(__file__, 452, 13), '+', rstrip_call_result_870, str_871)
    
    # Applying the binary operator '+=' (line 452)
    result_iadd_873 = python_operator(stypy.reporting.localization.Localization(__file__, 452, 8), '+=', s_866, result_add_872)
    # Assigning a type to the variable 's' (line 452)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 452, 8), 's', result_iadd_873)
    
    
    # Assigning a Name to a Name (line 453):
    
    # Assigning a Name to a Name (line 453):
    # Getting the type of 'next_line_prefix' (line 453)
    next_line_prefix_874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 15), 'next_line_prefix')
    # Assigning a type to the variable 'line' (line 453)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 453, 8), 'line', next_line_prefix_874)
    # SSA join for if statement (line 451)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'line' (line 454)
    line_875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 4), 'line')
    # Getting the type of 'word' (line 454)
    word_876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 12), 'word')
    # Applying the binary operator '+=' (line 454)
    result_iadd_877 = python_operator(stypy.reporting.localization.Localization(__file__, 454, 4), '+=', line_875, word_876)
    # Assigning a type to the variable 'line' (line 454)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 454, 4), 'line', result_iadd_877)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 455)
    tuple_878 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 455, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 455)
    # Adding element type (line 455)
    # Getting the type of 's' (line 455)
    s_879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 11), 's')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 455, 11), tuple_878, s_879)
    # Adding element type (line 455)
    # Getting the type of 'line' (line 455)
    line_880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 14), 'line')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 455, 11), tuple_878, line_880)
    
    # Assigning a type to the variable 'stypy_return_type' (line 455)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 455, 4), 'stypy_return_type', tuple_878)
    
    # ################# End of '_extendLine(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_extendLine' in the type store
    # Getting the type of 'stypy_return_type' (line 450)
    stypy_return_type_881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_881)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_extendLine'
    return stypy_return_type_881

# Assigning a type to the variable '_extendLine' (line 450)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 450, 0), '_extendLine', _extendLine)

@norecursion
def _formatArray(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_formatArray'
    module_type_store = module_type_store.open_function_context('_formatArray', 458, 0, False)
    
    # Passed parameters checking function
    _formatArray.stypy_localization = localization
    _formatArray.stypy_type_of_self = None
    _formatArray.stypy_type_store = module_type_store
    _formatArray.stypy_function_name = '_formatArray'
    _formatArray.stypy_param_names_list = ['a', 'format_function', 'rank', 'max_line_len', 'next_line_prefix', 'separator', 'edge_items', 'summary_insert']
    _formatArray.stypy_varargs_param_name = None
    _formatArray.stypy_kwargs_param_name = None
    _formatArray.stypy_call_defaults = defaults
    _formatArray.stypy_call_varargs = varargs
    _formatArray.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_formatArray', ['a', 'format_function', 'rank', 'max_line_len', 'next_line_prefix', 'separator', 'edge_items', 'summary_insert'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_formatArray', localization, ['a', 'format_function', 'rank', 'max_line_len', 'next_line_prefix', 'separator', 'edge_items', 'summary_insert'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_formatArray(...)' code ##################

    str_882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 466, (-1)), 'str', 'formatArray is designed for two modes of operation:\n\n    1. Full output\n\n    2. Summarized output\n\n    ')
    
    
    # Getting the type of 'rank' (line 467)
    rank_883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 7), 'rank')
    int_884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 467, 15), 'int')
    # Applying the binary operator '==' (line 467)
    result_eq_885 = python_operator(stypy.reporting.localization.Localization(__file__, 467, 7), '==', rank_883, int_884)
    
    # Testing the type of an if condition (line 467)
    if_condition_886 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 467, 4), result_eq_885)
    # Assigning a type to the variable 'if_condition_886' (line 467)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 467, 4), 'if_condition_886', if_condition_886)
    # SSA begins for if statement (line 467)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 468):
    
    # Assigning a Call to a Name (line 468):
    
    # Call to item(...): (line 468)
    # Processing the call keyword arguments (line 468)
    kwargs_889 = {}
    # Getting the type of 'a' (line 468)
    a_887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 14), 'a', False)
    # Obtaining the member 'item' of a type (line 468)
    item_888 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 468, 14), a_887, 'item')
    # Calling item(args, kwargs) (line 468)
    item_call_result_890 = invoke(stypy.reporting.localization.Localization(__file__, 468, 14), item_888, *[], **kwargs_889)
    
    # Assigning a type to the variable 'obj' (line 468)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 468, 8), 'obj', item_call_result_890)
    
    # Type idiom detected: calculating its left and rigth part (line 469)
    # Getting the type of 'tuple' (line 469)
    tuple_891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 27), 'tuple')
    # Getting the type of 'obj' (line 469)
    obj_892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 22), 'obj')
    
    (may_be_893, more_types_in_union_894) = may_be_subtype(tuple_891, obj_892)

    if may_be_893:

        if more_types_in_union_894:
            # Runtime conditional SSA (line 469)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'obj' (line 469)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 469, 8), 'obj', remove_not_subtype_from_union(obj_892, tuple))
        
        # Assigning a Call to a Name (line 470):
        
        # Assigning a Call to a Name (line 470):
        
        # Call to _convert_arrays(...): (line 470)
        # Processing the call arguments (line 470)
        # Getting the type of 'obj' (line 470)
        obj_896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 34), 'obj', False)
        # Processing the call keyword arguments (line 470)
        kwargs_897 = {}
        # Getting the type of '_convert_arrays' (line 470)
        _convert_arrays_895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 18), '_convert_arrays', False)
        # Calling _convert_arrays(args, kwargs) (line 470)
        _convert_arrays_call_result_898 = invoke(stypy.reporting.localization.Localization(__file__, 470, 18), _convert_arrays_895, *[obj_896], **kwargs_897)
        
        # Assigning a type to the variable 'obj' (line 470)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 470, 12), 'obj', _convert_arrays_call_result_898)

        if more_types_in_union_894:
            # SSA join for if statement (line 469)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Call to str(...): (line 471)
    # Processing the call arguments (line 471)
    # Getting the type of 'obj' (line 471)
    obj_900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 19), 'obj', False)
    # Processing the call keyword arguments (line 471)
    kwargs_901 = {}
    # Getting the type of 'str' (line 471)
    str_899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 15), 'str', False)
    # Calling str(args, kwargs) (line 471)
    str_call_result_902 = invoke(stypy.reporting.localization.Localization(__file__, 471, 15), str_899, *[obj_900], **kwargs_901)
    
    # Assigning a type to the variable 'stypy_return_type' (line 471)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 471, 8), 'stypy_return_type', str_call_result_902)
    # SSA join for if statement (line 467)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    # Getting the type of 'summary_insert' (line 473)
    summary_insert_903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 7), 'summary_insert')
    
    int_904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 473, 26), 'int')
    # Getting the type of 'edge_items' (line 473)
    edge_items_905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 28), 'edge_items')
    # Applying the binary operator '*' (line 473)
    result_mul_906 = python_operator(stypy.reporting.localization.Localization(__file__, 473, 26), '*', int_904, edge_items_905)
    
    
    # Call to len(...): (line 473)
    # Processing the call arguments (line 473)
    # Getting the type of 'a' (line 473)
    a_908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 45), 'a', False)
    # Processing the call keyword arguments (line 473)
    kwargs_909 = {}
    # Getting the type of 'len' (line 473)
    len_907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 41), 'len', False)
    # Calling len(args, kwargs) (line 473)
    len_call_result_910 = invoke(stypy.reporting.localization.Localization(__file__, 473, 41), len_907, *[a_908], **kwargs_909)
    
    # Applying the binary operator '<' (line 473)
    result_lt_911 = python_operator(stypy.reporting.localization.Localization(__file__, 473, 26), '<', result_mul_906, len_call_result_910)
    
    # Applying the binary operator 'and' (line 473)
    result_and_keyword_912 = python_operator(stypy.reporting.localization.Localization(__file__, 473, 7), 'and', summary_insert_903, result_lt_911)
    
    # Testing the type of an if condition (line 473)
    if_condition_913 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 473, 4), result_and_keyword_912)
    # Assigning a type to the variable 'if_condition_913' (line 473)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 473, 4), 'if_condition_913', if_condition_913)
    # SSA begins for if statement (line 473)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 474):
    
    # Assigning a Name to a Name (line 474):
    # Getting the type of 'edge_items' (line 474)
    edge_items_914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 24), 'edge_items')
    # Assigning a type to the variable 'leading_items' (line 474)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 474, 8), 'leading_items', edge_items_914)
    
    # Assigning a Name to a Name (line 475):
    
    # Assigning a Name to a Name (line 475):
    # Getting the type of 'edge_items' (line 475)
    edge_items_915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 25), 'edge_items')
    # Assigning a type to the variable 'trailing_items' (line 475)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 475, 8), 'trailing_items', edge_items_915)
    
    # Assigning a Name to a Name (line 476):
    
    # Assigning a Name to a Name (line 476):
    # Getting the type of 'summary_insert' (line 476)
    summary_insert_916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 26), 'summary_insert')
    # Assigning a type to the variable 'summary_insert1' (line 476)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 476, 8), 'summary_insert1', summary_insert_916)
    # SSA branch for the else part of an if statement (line 473)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Num to a Name (line 478):
    
    # Assigning a Num to a Name (line 478):
    int_917 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 478, 24), 'int')
    # Assigning a type to the variable 'leading_items' (line 478)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 478, 8), 'leading_items', int_917)
    
    # Assigning a Call to a Name (line 479):
    
    # Assigning a Call to a Name (line 479):
    
    # Call to len(...): (line 479)
    # Processing the call arguments (line 479)
    # Getting the type of 'a' (line 479)
    a_919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 29), 'a', False)
    # Processing the call keyword arguments (line 479)
    kwargs_920 = {}
    # Getting the type of 'len' (line 479)
    len_918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 25), 'len', False)
    # Calling len(args, kwargs) (line 479)
    len_call_result_921 = invoke(stypy.reporting.localization.Localization(__file__, 479, 25), len_918, *[a_919], **kwargs_920)
    
    # Assigning a type to the variable 'trailing_items' (line 479)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 479, 8), 'trailing_items', len_call_result_921)
    
    # Assigning a Str to a Name (line 480):
    
    # Assigning a Str to a Name (line 480):
    str_922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 480, 26), 'str', '')
    # Assigning a type to the variable 'summary_insert1' (line 480)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 480, 8), 'summary_insert1', str_922)
    # SSA join for if statement (line 473)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'rank' (line 482)
    rank_923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 7), 'rank')
    int_924 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 482, 15), 'int')
    # Applying the binary operator '==' (line 482)
    result_eq_925 = python_operator(stypy.reporting.localization.Localization(__file__, 482, 7), '==', rank_923, int_924)
    
    # Testing the type of an if condition (line 482)
    if_condition_926 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 482, 4), result_eq_925)
    # Assigning a type to the variable 'if_condition_926' (line 482)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 482, 4), 'if_condition_926', if_condition_926)
    # SSA begins for if statement (line 482)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 483):
    
    # Assigning a Str to a Name (line 483):
    str_927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 483, 12), 'str', '')
    # Assigning a type to the variable 's' (line 483)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 483, 8), 's', str_927)
    
    # Assigning a Name to a Name (line 484):
    
    # Assigning a Name to a Name (line 484):
    # Getting the type of 'next_line_prefix' (line 484)
    next_line_prefix_928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 15), 'next_line_prefix')
    # Assigning a type to the variable 'line' (line 484)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 484, 8), 'line', next_line_prefix_928)
    
    
    # Call to range(...): (line 485)
    # Processing the call arguments (line 485)
    # Getting the type of 'leading_items' (line 485)
    leading_items_930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 23), 'leading_items', False)
    # Processing the call keyword arguments (line 485)
    kwargs_931 = {}
    # Getting the type of 'range' (line 485)
    range_929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 17), 'range', False)
    # Calling range(args, kwargs) (line 485)
    range_call_result_932 = invoke(stypy.reporting.localization.Localization(__file__, 485, 17), range_929, *[leading_items_930], **kwargs_931)
    
    # Testing the type of a for loop iterable (line 485)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 485, 8), range_call_result_932)
    # Getting the type of the for loop variable (line 485)
    for_loop_var_933 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 485, 8), range_call_result_932)
    # Assigning a type to the variable 'i' (line 485)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 485, 8), 'i', for_loop_var_933)
    # SSA begins for a for statement (line 485)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a BinOp to a Name (line 486):
    
    # Assigning a BinOp to a Name (line 486):
    
    # Call to format_function(...): (line 486)
    # Processing the call arguments (line 486)
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 486)
    i_935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 37), 'i', False)
    # Getting the type of 'a' (line 486)
    a_936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 35), 'a', False)
    # Obtaining the member '__getitem__' of a type (line 486)
    getitem___937 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 486, 35), a_936, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 486)
    subscript_call_result_938 = invoke(stypy.reporting.localization.Localization(__file__, 486, 35), getitem___937, i_935)
    
    # Processing the call keyword arguments (line 486)
    kwargs_939 = {}
    # Getting the type of 'format_function' (line 486)
    format_function_934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 19), 'format_function', False)
    # Calling format_function(args, kwargs) (line 486)
    format_function_call_result_940 = invoke(stypy.reporting.localization.Localization(__file__, 486, 19), format_function_934, *[subscript_call_result_938], **kwargs_939)
    
    # Getting the type of 'separator' (line 486)
    separator_941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 43), 'separator')
    # Applying the binary operator '+' (line 486)
    result_add_942 = python_operator(stypy.reporting.localization.Localization(__file__, 486, 19), '+', format_function_call_result_940, separator_941)
    
    # Assigning a type to the variable 'word' (line 486)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 486, 12), 'word', result_add_942)
    
    # Assigning a Call to a Tuple (line 487):
    
    # Assigning a Call to a Name:
    
    # Call to _extendLine(...): (line 487)
    # Processing the call arguments (line 487)
    # Getting the type of 's' (line 487)
    s_944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 34), 's', False)
    # Getting the type of 'line' (line 487)
    line_945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 37), 'line', False)
    # Getting the type of 'word' (line 487)
    word_946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 43), 'word', False)
    # Getting the type of 'max_line_len' (line 487)
    max_line_len_947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 49), 'max_line_len', False)
    # Getting the type of 'next_line_prefix' (line 487)
    next_line_prefix_948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 63), 'next_line_prefix', False)
    # Processing the call keyword arguments (line 487)
    kwargs_949 = {}
    # Getting the type of '_extendLine' (line 487)
    _extendLine_943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 22), '_extendLine', False)
    # Calling _extendLine(args, kwargs) (line 487)
    _extendLine_call_result_950 = invoke(stypy.reporting.localization.Localization(__file__, 487, 22), _extendLine_943, *[s_944, line_945, word_946, max_line_len_947, next_line_prefix_948], **kwargs_949)
    
    # Assigning a type to the variable 'call_assignment_176' (line 487)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 487, 12), 'call_assignment_176', _extendLine_call_result_950)
    
    # Assigning a Call to a Name (line 487):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 487, 12), 'int')
    # Processing the call keyword arguments
    kwargs_954 = {}
    # Getting the type of 'call_assignment_176' (line 487)
    call_assignment_176_951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 12), 'call_assignment_176', False)
    # Obtaining the member '__getitem__' of a type (line 487)
    getitem___952 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 487, 12), call_assignment_176_951, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_955 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___952, *[int_953], **kwargs_954)
    
    # Assigning a type to the variable 'call_assignment_177' (line 487)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 487, 12), 'call_assignment_177', getitem___call_result_955)
    
    # Assigning a Name to a Name (line 487):
    # Getting the type of 'call_assignment_177' (line 487)
    call_assignment_177_956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 12), 'call_assignment_177')
    # Assigning a type to the variable 's' (line 487)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 487, 12), 's', call_assignment_177_956)
    
    # Assigning a Call to a Name (line 487):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 487, 12), 'int')
    # Processing the call keyword arguments
    kwargs_960 = {}
    # Getting the type of 'call_assignment_176' (line 487)
    call_assignment_176_957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 12), 'call_assignment_176', False)
    # Obtaining the member '__getitem__' of a type (line 487)
    getitem___958 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 487, 12), call_assignment_176_957, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_961 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___958, *[int_959], **kwargs_960)
    
    # Assigning a type to the variable 'call_assignment_178' (line 487)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 487, 12), 'call_assignment_178', getitem___call_result_961)
    
    # Assigning a Name to a Name (line 487):
    # Getting the type of 'call_assignment_178' (line 487)
    call_assignment_178_962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 12), 'call_assignment_178')
    # Assigning a type to the variable 'line' (line 487)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 487, 15), 'line', call_assignment_178_962)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'summary_insert1' (line 489)
    summary_insert1_963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 11), 'summary_insert1')
    # Testing the type of an if condition (line 489)
    if_condition_964 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 489, 8), summary_insert1_963)
    # Assigning a type to the variable 'if_condition_964' (line 489)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 489, 8), 'if_condition_964', if_condition_964)
    # SSA begins for if statement (line 489)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Tuple (line 490):
    
    # Assigning a Call to a Name:
    
    # Call to _extendLine(...): (line 490)
    # Processing the call arguments (line 490)
    # Getting the type of 's' (line 490)
    s_966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 34), 's', False)
    # Getting the type of 'line' (line 490)
    line_967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 37), 'line', False)
    # Getting the type of 'summary_insert1' (line 490)
    summary_insert1_968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 43), 'summary_insert1', False)
    # Getting the type of 'max_line_len' (line 490)
    max_line_len_969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 60), 'max_line_len', False)
    # Getting the type of 'next_line_prefix' (line 490)
    next_line_prefix_970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 74), 'next_line_prefix', False)
    # Processing the call keyword arguments (line 490)
    kwargs_971 = {}
    # Getting the type of '_extendLine' (line 490)
    _extendLine_965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 22), '_extendLine', False)
    # Calling _extendLine(args, kwargs) (line 490)
    _extendLine_call_result_972 = invoke(stypy.reporting.localization.Localization(__file__, 490, 22), _extendLine_965, *[s_966, line_967, summary_insert1_968, max_line_len_969, next_line_prefix_970], **kwargs_971)
    
    # Assigning a type to the variable 'call_assignment_179' (line 490)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 490, 12), 'call_assignment_179', _extendLine_call_result_972)
    
    # Assigning a Call to a Name (line 490):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 490, 12), 'int')
    # Processing the call keyword arguments
    kwargs_976 = {}
    # Getting the type of 'call_assignment_179' (line 490)
    call_assignment_179_973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 12), 'call_assignment_179', False)
    # Obtaining the member '__getitem__' of a type (line 490)
    getitem___974 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 490, 12), call_assignment_179_973, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_977 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___974, *[int_975], **kwargs_976)
    
    # Assigning a type to the variable 'call_assignment_180' (line 490)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 490, 12), 'call_assignment_180', getitem___call_result_977)
    
    # Assigning a Name to a Name (line 490):
    # Getting the type of 'call_assignment_180' (line 490)
    call_assignment_180_978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 12), 'call_assignment_180')
    # Assigning a type to the variable 's' (line 490)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 490, 12), 's', call_assignment_180_978)
    
    # Assigning a Call to a Name (line 490):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 490, 12), 'int')
    # Processing the call keyword arguments
    kwargs_982 = {}
    # Getting the type of 'call_assignment_179' (line 490)
    call_assignment_179_979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 12), 'call_assignment_179', False)
    # Obtaining the member '__getitem__' of a type (line 490)
    getitem___980 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 490, 12), call_assignment_179_979, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_983 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___980, *[int_981], **kwargs_982)
    
    # Assigning a type to the variable 'call_assignment_181' (line 490)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 490, 12), 'call_assignment_181', getitem___call_result_983)
    
    # Assigning a Name to a Name (line 490):
    # Getting the type of 'call_assignment_181' (line 490)
    call_assignment_181_984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 12), 'call_assignment_181')
    # Assigning a type to the variable 'line' (line 490)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 490, 15), 'line', call_assignment_181_984)
    # SSA join for if statement (line 489)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to range(...): (line 492)
    # Processing the call arguments (line 492)
    # Getting the type of 'trailing_items' (line 492)
    trailing_items_986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 23), 'trailing_items', False)
    int_987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 492, 39), 'int')
    int_988 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 492, 42), 'int')
    # Processing the call keyword arguments (line 492)
    kwargs_989 = {}
    # Getting the type of 'range' (line 492)
    range_985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 17), 'range', False)
    # Calling range(args, kwargs) (line 492)
    range_call_result_990 = invoke(stypy.reporting.localization.Localization(__file__, 492, 17), range_985, *[trailing_items_986, int_987, int_988], **kwargs_989)
    
    # Testing the type of a for loop iterable (line 492)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 492, 8), range_call_result_990)
    # Getting the type of the for loop variable (line 492)
    for_loop_var_991 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 492, 8), range_call_result_990)
    # Assigning a type to the variable 'i' (line 492)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 492, 8), 'i', for_loop_var_991)
    # SSA begins for a for statement (line 492)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a BinOp to a Name (line 493):
    
    # Assigning a BinOp to a Name (line 493):
    
    # Call to format_function(...): (line 493)
    # Processing the call arguments (line 493)
    
    # Obtaining the type of the subscript
    
    # Getting the type of 'i' (line 493)
    i_993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 38), 'i', False)
    # Applying the 'usub' unary operator (line 493)
    result___neg___994 = python_operator(stypy.reporting.localization.Localization(__file__, 493, 37), 'usub', i_993)
    
    # Getting the type of 'a' (line 493)
    a_995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 35), 'a', False)
    # Obtaining the member '__getitem__' of a type (line 493)
    getitem___996 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 493, 35), a_995, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 493)
    subscript_call_result_997 = invoke(stypy.reporting.localization.Localization(__file__, 493, 35), getitem___996, result___neg___994)
    
    # Processing the call keyword arguments (line 493)
    kwargs_998 = {}
    # Getting the type of 'format_function' (line 493)
    format_function_992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 19), 'format_function', False)
    # Calling format_function(args, kwargs) (line 493)
    format_function_call_result_999 = invoke(stypy.reporting.localization.Localization(__file__, 493, 19), format_function_992, *[subscript_call_result_997], **kwargs_998)
    
    # Getting the type of 'separator' (line 493)
    separator_1000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 44), 'separator')
    # Applying the binary operator '+' (line 493)
    result_add_1001 = python_operator(stypy.reporting.localization.Localization(__file__, 493, 19), '+', format_function_call_result_999, separator_1000)
    
    # Assigning a type to the variable 'word' (line 493)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 493, 12), 'word', result_add_1001)
    
    # Assigning a Call to a Tuple (line 494):
    
    # Assigning a Call to a Name:
    
    # Call to _extendLine(...): (line 494)
    # Processing the call arguments (line 494)
    # Getting the type of 's' (line 494)
    s_1003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 34), 's', False)
    # Getting the type of 'line' (line 494)
    line_1004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 37), 'line', False)
    # Getting the type of 'word' (line 494)
    word_1005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 43), 'word', False)
    # Getting the type of 'max_line_len' (line 494)
    max_line_len_1006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 49), 'max_line_len', False)
    # Getting the type of 'next_line_prefix' (line 494)
    next_line_prefix_1007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 63), 'next_line_prefix', False)
    # Processing the call keyword arguments (line 494)
    kwargs_1008 = {}
    # Getting the type of '_extendLine' (line 494)
    _extendLine_1002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 22), '_extendLine', False)
    # Calling _extendLine(args, kwargs) (line 494)
    _extendLine_call_result_1009 = invoke(stypy.reporting.localization.Localization(__file__, 494, 22), _extendLine_1002, *[s_1003, line_1004, word_1005, max_line_len_1006, next_line_prefix_1007], **kwargs_1008)
    
    # Assigning a type to the variable 'call_assignment_182' (line 494)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 494, 12), 'call_assignment_182', _extendLine_call_result_1009)
    
    # Assigning a Call to a Name (line 494):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_1012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 494, 12), 'int')
    # Processing the call keyword arguments
    kwargs_1013 = {}
    # Getting the type of 'call_assignment_182' (line 494)
    call_assignment_182_1010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 12), 'call_assignment_182', False)
    # Obtaining the member '__getitem__' of a type (line 494)
    getitem___1011 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 494, 12), call_assignment_182_1010, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_1014 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___1011, *[int_1012], **kwargs_1013)
    
    # Assigning a type to the variable 'call_assignment_183' (line 494)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 494, 12), 'call_assignment_183', getitem___call_result_1014)
    
    # Assigning a Name to a Name (line 494):
    # Getting the type of 'call_assignment_183' (line 494)
    call_assignment_183_1015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 12), 'call_assignment_183')
    # Assigning a type to the variable 's' (line 494)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 494, 12), 's', call_assignment_183_1015)
    
    # Assigning a Call to a Name (line 494):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_1018 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 494, 12), 'int')
    # Processing the call keyword arguments
    kwargs_1019 = {}
    # Getting the type of 'call_assignment_182' (line 494)
    call_assignment_182_1016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 12), 'call_assignment_182', False)
    # Obtaining the member '__getitem__' of a type (line 494)
    getitem___1017 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 494, 12), call_assignment_182_1016, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_1020 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___1017, *[int_1018], **kwargs_1019)
    
    # Assigning a type to the variable 'call_assignment_184' (line 494)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 494, 12), 'call_assignment_184', getitem___call_result_1020)
    
    # Assigning a Name to a Name (line 494):
    # Getting the type of 'call_assignment_184' (line 494)
    call_assignment_184_1021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 12), 'call_assignment_184')
    # Assigning a type to the variable 'line' (line 494)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 494, 15), 'line', call_assignment_184_1021)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 496):
    
    # Assigning a Call to a Name (line 496):
    
    # Call to format_function(...): (line 496)
    # Processing the call arguments (line 496)
    
    # Obtaining the type of the subscript
    int_1023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 496, 33), 'int')
    # Getting the type of 'a' (line 496)
    a_1024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 31), 'a', False)
    # Obtaining the member '__getitem__' of a type (line 496)
    getitem___1025 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 496, 31), a_1024, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 496)
    subscript_call_result_1026 = invoke(stypy.reporting.localization.Localization(__file__, 496, 31), getitem___1025, int_1023)
    
    # Processing the call keyword arguments (line 496)
    kwargs_1027 = {}
    # Getting the type of 'format_function' (line 496)
    format_function_1022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 15), 'format_function', False)
    # Calling format_function(args, kwargs) (line 496)
    format_function_call_result_1028 = invoke(stypy.reporting.localization.Localization(__file__, 496, 15), format_function_1022, *[subscript_call_result_1026], **kwargs_1027)
    
    # Assigning a type to the variable 'word' (line 496)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 496, 8), 'word', format_function_call_result_1028)
    
    # Assigning a Call to a Tuple (line 497):
    
    # Assigning a Call to a Name:
    
    # Call to _extendLine(...): (line 497)
    # Processing the call arguments (line 497)
    # Getting the type of 's' (line 497)
    s_1030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 30), 's', False)
    # Getting the type of 'line' (line 497)
    line_1031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 33), 'line', False)
    # Getting the type of 'word' (line 497)
    word_1032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 39), 'word', False)
    # Getting the type of 'max_line_len' (line 497)
    max_line_len_1033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 45), 'max_line_len', False)
    # Getting the type of 'next_line_prefix' (line 497)
    next_line_prefix_1034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 59), 'next_line_prefix', False)
    # Processing the call keyword arguments (line 497)
    kwargs_1035 = {}
    # Getting the type of '_extendLine' (line 497)
    _extendLine_1029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 18), '_extendLine', False)
    # Calling _extendLine(args, kwargs) (line 497)
    _extendLine_call_result_1036 = invoke(stypy.reporting.localization.Localization(__file__, 497, 18), _extendLine_1029, *[s_1030, line_1031, word_1032, max_line_len_1033, next_line_prefix_1034], **kwargs_1035)
    
    # Assigning a type to the variable 'call_assignment_185' (line 497)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 497, 8), 'call_assignment_185', _extendLine_call_result_1036)
    
    # Assigning a Call to a Name (line 497):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_1039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 497, 8), 'int')
    # Processing the call keyword arguments
    kwargs_1040 = {}
    # Getting the type of 'call_assignment_185' (line 497)
    call_assignment_185_1037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 8), 'call_assignment_185', False)
    # Obtaining the member '__getitem__' of a type (line 497)
    getitem___1038 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 497, 8), call_assignment_185_1037, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_1041 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___1038, *[int_1039], **kwargs_1040)
    
    # Assigning a type to the variable 'call_assignment_186' (line 497)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 497, 8), 'call_assignment_186', getitem___call_result_1041)
    
    # Assigning a Name to a Name (line 497):
    # Getting the type of 'call_assignment_186' (line 497)
    call_assignment_186_1042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 8), 'call_assignment_186')
    # Assigning a type to the variable 's' (line 497)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 497, 8), 's', call_assignment_186_1042)
    
    # Assigning a Call to a Name (line 497):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_1045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 497, 8), 'int')
    # Processing the call keyword arguments
    kwargs_1046 = {}
    # Getting the type of 'call_assignment_185' (line 497)
    call_assignment_185_1043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 8), 'call_assignment_185', False)
    # Obtaining the member '__getitem__' of a type (line 497)
    getitem___1044 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 497, 8), call_assignment_185_1043, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_1047 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___1044, *[int_1045], **kwargs_1046)
    
    # Assigning a type to the variable 'call_assignment_187' (line 497)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 497, 8), 'call_assignment_187', getitem___call_result_1047)
    
    # Assigning a Name to a Name (line 497):
    # Getting the type of 'call_assignment_187' (line 497)
    call_assignment_187_1048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 8), 'call_assignment_187')
    # Assigning a type to the variable 'line' (line 497)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 497, 11), 'line', call_assignment_187_1048)
    
    # Getting the type of 's' (line 498)
    s_1049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 8), 's')
    # Getting the type of 'line' (line 498)
    line_1050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 13), 'line')
    str_1051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 498, 20), 'str', ']\n')
    # Applying the binary operator '+' (line 498)
    result_add_1052 = python_operator(stypy.reporting.localization.Localization(__file__, 498, 13), '+', line_1050, str_1051)
    
    # Applying the binary operator '+=' (line 498)
    result_iadd_1053 = python_operator(stypy.reporting.localization.Localization(__file__, 498, 8), '+=', s_1049, result_add_1052)
    # Assigning a type to the variable 's' (line 498)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 498, 8), 's', result_iadd_1053)
    
    
    # Assigning a BinOp to a Name (line 499):
    
    # Assigning a BinOp to a Name (line 499):
    str_1054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 499, 12), 'str', '[')
    
    # Obtaining the type of the subscript
    
    # Call to len(...): (line 499)
    # Processing the call arguments (line 499)
    # Getting the type of 'next_line_prefix' (line 499)
    next_line_prefix_1056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 24), 'next_line_prefix', False)
    # Processing the call keyword arguments (line 499)
    kwargs_1057 = {}
    # Getting the type of 'len' (line 499)
    len_1055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 20), 'len', False)
    # Calling len(args, kwargs) (line 499)
    len_call_result_1058 = invoke(stypy.reporting.localization.Localization(__file__, 499, 20), len_1055, *[next_line_prefix_1056], **kwargs_1057)
    
    slice_1059 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 499, 18), len_call_result_1058, None, None)
    # Getting the type of 's' (line 499)
    s_1060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 18), 's')
    # Obtaining the member '__getitem__' of a type (line 499)
    getitem___1061 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 499, 18), s_1060, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 499)
    subscript_call_result_1062 = invoke(stypy.reporting.localization.Localization(__file__, 499, 18), getitem___1061, slice_1059)
    
    # Applying the binary operator '+' (line 499)
    result_add_1063 = python_operator(stypy.reporting.localization.Localization(__file__, 499, 12), '+', str_1054, subscript_call_result_1062)
    
    # Assigning a type to the variable 's' (line 499)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 499, 8), 's', result_add_1063)
    # SSA branch for the else part of an if statement (line 482)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Str to a Name (line 501):
    
    # Assigning a Str to a Name (line 501):
    str_1064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 501, 12), 'str', '[')
    # Assigning a type to the variable 's' (line 501)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 501, 8), 's', str_1064)
    
    # Assigning a Call to a Name (line 502):
    
    # Assigning a Call to a Name (line 502):
    
    # Call to rstrip(...): (line 502)
    # Processing the call keyword arguments (line 502)
    kwargs_1067 = {}
    # Getting the type of 'separator' (line 502)
    separator_1065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 14), 'separator', False)
    # Obtaining the member 'rstrip' of a type (line 502)
    rstrip_1066 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 502, 14), separator_1065, 'rstrip')
    # Calling rstrip(args, kwargs) (line 502)
    rstrip_call_result_1068 = invoke(stypy.reporting.localization.Localization(__file__, 502, 14), rstrip_1066, *[], **kwargs_1067)
    
    # Assigning a type to the variable 'sep' (line 502)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 502, 8), 'sep', rstrip_call_result_1068)
    
    
    # Call to range(...): (line 503)
    # Processing the call arguments (line 503)
    # Getting the type of 'leading_items' (line 503)
    leading_items_1070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 23), 'leading_items', False)
    # Processing the call keyword arguments (line 503)
    kwargs_1071 = {}
    # Getting the type of 'range' (line 503)
    range_1069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 17), 'range', False)
    # Calling range(args, kwargs) (line 503)
    range_call_result_1072 = invoke(stypy.reporting.localization.Localization(__file__, 503, 17), range_1069, *[leading_items_1070], **kwargs_1071)
    
    # Testing the type of a for loop iterable (line 503)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 503, 8), range_call_result_1072)
    # Getting the type of the for loop variable (line 503)
    for_loop_var_1073 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 503, 8), range_call_result_1072)
    # Assigning a type to the variable 'i' (line 503)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 503, 8), 'i', for_loop_var_1073)
    # SSA begins for a for statement (line 503)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Getting the type of 'i' (line 504)
    i_1074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 15), 'i')
    int_1075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 504, 19), 'int')
    # Applying the binary operator '>' (line 504)
    result_gt_1076 = python_operator(stypy.reporting.localization.Localization(__file__, 504, 15), '>', i_1074, int_1075)
    
    # Testing the type of an if condition (line 504)
    if_condition_1077 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 504, 12), result_gt_1076)
    # Assigning a type to the variable 'if_condition_1077' (line 504)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 504, 12), 'if_condition_1077', if_condition_1077)
    # SSA begins for if statement (line 504)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 's' (line 505)
    s_1078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 16), 's')
    # Getting the type of 'next_line_prefix' (line 505)
    next_line_prefix_1079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 21), 'next_line_prefix')
    # Applying the binary operator '+=' (line 505)
    result_iadd_1080 = python_operator(stypy.reporting.localization.Localization(__file__, 505, 16), '+=', s_1078, next_line_prefix_1079)
    # Assigning a type to the variable 's' (line 505)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 505, 16), 's', result_iadd_1080)
    
    # SSA join for if statement (line 504)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 's' (line 506)
    s_1081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 12), 's')
    
    # Call to _formatArray(...): (line 506)
    # Processing the call arguments (line 506)
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 506)
    i_1083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 32), 'i', False)
    # Getting the type of 'a' (line 506)
    a_1084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 30), 'a', False)
    # Obtaining the member '__getitem__' of a type (line 506)
    getitem___1085 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 506, 30), a_1084, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 506)
    subscript_call_result_1086 = invoke(stypy.reporting.localization.Localization(__file__, 506, 30), getitem___1085, i_1083)
    
    # Getting the type of 'format_function' (line 506)
    format_function_1087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 36), 'format_function', False)
    # Getting the type of 'rank' (line 506)
    rank_1088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 53), 'rank', False)
    int_1089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 506, 58), 'int')
    # Applying the binary operator '-' (line 506)
    result_sub_1090 = python_operator(stypy.reporting.localization.Localization(__file__, 506, 53), '-', rank_1088, int_1089)
    
    # Getting the type of 'max_line_len' (line 506)
    max_line_len_1091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 61), 'max_line_len', False)
    str_1092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 507, 30), 'str', ' ')
    # Getting the type of 'next_line_prefix' (line 507)
    next_line_prefix_1093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 36), 'next_line_prefix', False)
    # Applying the binary operator '+' (line 507)
    result_add_1094 = python_operator(stypy.reporting.localization.Localization(__file__, 507, 30), '+', str_1092, next_line_prefix_1093)
    
    # Getting the type of 'separator' (line 507)
    separator_1095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 54), 'separator', False)
    # Getting the type of 'edge_items' (line 507)
    edge_items_1096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 65), 'edge_items', False)
    # Getting the type of 'summary_insert' (line 508)
    summary_insert_1097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 30), 'summary_insert', False)
    # Processing the call keyword arguments (line 506)
    kwargs_1098 = {}
    # Getting the type of '_formatArray' (line 506)
    _formatArray_1082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 17), '_formatArray', False)
    # Calling _formatArray(args, kwargs) (line 506)
    _formatArray_call_result_1099 = invoke(stypy.reporting.localization.Localization(__file__, 506, 17), _formatArray_1082, *[subscript_call_result_1086, format_function_1087, result_sub_1090, max_line_len_1091, result_add_1094, separator_1095, edge_items_1096, summary_insert_1097], **kwargs_1098)
    
    # Applying the binary operator '+=' (line 506)
    result_iadd_1100 = python_operator(stypy.reporting.localization.Localization(__file__, 506, 12), '+=', s_1081, _formatArray_call_result_1099)
    # Assigning a type to the variable 's' (line 506)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 506, 12), 's', result_iadd_1100)
    
    
    # Assigning a BinOp to a Name (line 509):
    
    # Assigning a BinOp to a Name (line 509):
    
    # Call to rstrip(...): (line 509)
    # Processing the call keyword arguments (line 509)
    kwargs_1103 = {}
    # Getting the type of 's' (line 509)
    s_1101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 16), 's', False)
    # Obtaining the member 'rstrip' of a type (line 509)
    rstrip_1102 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 509, 16), s_1101, 'rstrip')
    # Calling rstrip(args, kwargs) (line 509)
    rstrip_call_result_1104 = invoke(stypy.reporting.localization.Localization(__file__, 509, 16), rstrip_1102, *[], **kwargs_1103)
    
    
    # Call to rstrip(...): (line 509)
    # Processing the call keyword arguments (line 509)
    kwargs_1107 = {}
    # Getting the type of 'sep' (line 509)
    sep_1105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 29), 'sep', False)
    # Obtaining the member 'rstrip' of a type (line 509)
    rstrip_1106 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 509, 29), sep_1105, 'rstrip')
    # Calling rstrip(args, kwargs) (line 509)
    rstrip_call_result_1108 = invoke(stypy.reporting.localization.Localization(__file__, 509, 29), rstrip_1106, *[], **kwargs_1107)
    
    # Applying the binary operator '+' (line 509)
    result_add_1109 = python_operator(stypy.reporting.localization.Localization(__file__, 509, 16), '+', rstrip_call_result_1104, rstrip_call_result_1108)
    
    str_1110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 509, 44), 'str', '\n')
    
    # Call to max(...): (line 509)
    # Processing the call arguments (line 509)
    # Getting the type of 'rank' (line 509)
    rank_1112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 53), 'rank', False)
    int_1113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 509, 58), 'int')
    # Applying the binary operator '-' (line 509)
    result_sub_1114 = python_operator(stypy.reporting.localization.Localization(__file__, 509, 53), '-', rank_1112, int_1113)
    
    int_1115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 509, 61), 'int')
    # Processing the call keyword arguments (line 509)
    kwargs_1116 = {}
    # Getting the type of 'max' (line 509)
    max_1111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 49), 'max', False)
    # Calling max(args, kwargs) (line 509)
    max_call_result_1117 = invoke(stypy.reporting.localization.Localization(__file__, 509, 49), max_1111, *[result_sub_1114, int_1115], **kwargs_1116)
    
    # Applying the binary operator '*' (line 509)
    result_mul_1118 = python_operator(stypy.reporting.localization.Localization(__file__, 509, 44), '*', str_1110, max_call_result_1117)
    
    # Applying the binary operator '+' (line 509)
    result_add_1119 = python_operator(stypy.reporting.localization.Localization(__file__, 509, 42), '+', result_add_1109, result_mul_1118)
    
    # Assigning a type to the variable 's' (line 509)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 509, 12), 's', result_add_1119)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'summary_insert1' (line 511)
    summary_insert1_1120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 11), 'summary_insert1')
    # Testing the type of an if condition (line 511)
    if_condition_1121 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 511, 8), summary_insert1_1120)
    # Assigning a type to the variable 'if_condition_1121' (line 511)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 511, 8), 'if_condition_1121', if_condition_1121)
    # SSA begins for if statement (line 511)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 's' (line 512)
    s_1122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 12), 's')
    # Getting the type of 'next_line_prefix' (line 512)
    next_line_prefix_1123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 17), 'next_line_prefix')
    # Getting the type of 'summary_insert1' (line 512)
    summary_insert1_1124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 36), 'summary_insert1')
    # Applying the binary operator '+' (line 512)
    result_add_1125 = python_operator(stypy.reporting.localization.Localization(__file__, 512, 17), '+', next_line_prefix_1123, summary_insert1_1124)
    
    str_1126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 512, 54), 'str', '\n')
    # Applying the binary operator '+' (line 512)
    result_add_1127 = python_operator(stypy.reporting.localization.Localization(__file__, 512, 52), '+', result_add_1125, str_1126)
    
    # Applying the binary operator '+=' (line 512)
    result_iadd_1128 = python_operator(stypy.reporting.localization.Localization(__file__, 512, 12), '+=', s_1122, result_add_1127)
    # Assigning a type to the variable 's' (line 512)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 512, 12), 's', result_iadd_1128)
    
    # SSA join for if statement (line 511)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to range(...): (line 514)
    # Processing the call arguments (line 514)
    # Getting the type of 'trailing_items' (line 514)
    trailing_items_1130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 23), 'trailing_items', False)
    int_1131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 514, 39), 'int')
    int_1132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 514, 42), 'int')
    # Processing the call keyword arguments (line 514)
    kwargs_1133 = {}
    # Getting the type of 'range' (line 514)
    range_1129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 17), 'range', False)
    # Calling range(args, kwargs) (line 514)
    range_call_result_1134 = invoke(stypy.reporting.localization.Localization(__file__, 514, 17), range_1129, *[trailing_items_1130, int_1131, int_1132], **kwargs_1133)
    
    # Testing the type of a for loop iterable (line 514)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 514, 8), range_call_result_1134)
    # Getting the type of the for loop variable (line 514)
    for_loop_var_1135 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 514, 8), range_call_result_1134)
    # Assigning a type to the variable 'i' (line 514)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 514, 8), 'i', for_loop_var_1135)
    # SSA begins for a for statement (line 514)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Evaluating a boolean operation
    # Getting the type of 'leading_items' (line 515)
    leading_items_1136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 15), 'leading_items')
    
    # Getting the type of 'i' (line 515)
    i_1137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 32), 'i')
    # Getting the type of 'trailing_items' (line 515)
    trailing_items_1138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 37), 'trailing_items')
    # Applying the binary operator '!=' (line 515)
    result_ne_1139 = python_operator(stypy.reporting.localization.Localization(__file__, 515, 32), '!=', i_1137, trailing_items_1138)
    
    # Applying the binary operator 'or' (line 515)
    result_or_keyword_1140 = python_operator(stypy.reporting.localization.Localization(__file__, 515, 15), 'or', leading_items_1136, result_ne_1139)
    
    # Testing the type of an if condition (line 515)
    if_condition_1141 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 515, 12), result_or_keyword_1140)
    # Assigning a type to the variable 'if_condition_1141' (line 515)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 515, 12), 'if_condition_1141', if_condition_1141)
    # SSA begins for if statement (line 515)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 's' (line 516)
    s_1142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 16), 's')
    # Getting the type of 'next_line_prefix' (line 516)
    next_line_prefix_1143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 21), 'next_line_prefix')
    # Applying the binary operator '+=' (line 516)
    result_iadd_1144 = python_operator(stypy.reporting.localization.Localization(__file__, 516, 16), '+=', s_1142, next_line_prefix_1143)
    # Assigning a type to the variable 's' (line 516)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 516, 16), 's', result_iadd_1144)
    
    # SSA join for if statement (line 515)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 's' (line 517)
    s_1145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 12), 's')
    
    # Call to _formatArray(...): (line 517)
    # Processing the call arguments (line 517)
    
    # Obtaining the type of the subscript
    
    # Getting the type of 'i' (line 517)
    i_1147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 33), 'i', False)
    # Applying the 'usub' unary operator (line 517)
    result___neg___1148 = python_operator(stypy.reporting.localization.Localization(__file__, 517, 32), 'usub', i_1147)
    
    # Getting the type of 'a' (line 517)
    a_1149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 30), 'a', False)
    # Obtaining the member '__getitem__' of a type (line 517)
    getitem___1150 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 517, 30), a_1149, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 517)
    subscript_call_result_1151 = invoke(stypy.reporting.localization.Localization(__file__, 517, 30), getitem___1150, result___neg___1148)
    
    # Getting the type of 'format_function' (line 517)
    format_function_1152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 37), 'format_function', False)
    # Getting the type of 'rank' (line 517)
    rank_1153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 54), 'rank', False)
    int_1154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 517, 59), 'int')
    # Applying the binary operator '-' (line 517)
    result_sub_1155 = python_operator(stypy.reporting.localization.Localization(__file__, 517, 54), '-', rank_1153, int_1154)
    
    # Getting the type of 'max_line_len' (line 517)
    max_line_len_1156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 62), 'max_line_len', False)
    str_1157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 518, 30), 'str', ' ')
    # Getting the type of 'next_line_prefix' (line 518)
    next_line_prefix_1158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 36), 'next_line_prefix', False)
    # Applying the binary operator '+' (line 518)
    result_add_1159 = python_operator(stypy.reporting.localization.Localization(__file__, 518, 30), '+', str_1157, next_line_prefix_1158)
    
    # Getting the type of 'separator' (line 518)
    separator_1160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 54), 'separator', False)
    # Getting the type of 'edge_items' (line 518)
    edge_items_1161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 65), 'edge_items', False)
    # Getting the type of 'summary_insert' (line 519)
    summary_insert_1162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 30), 'summary_insert', False)
    # Processing the call keyword arguments (line 517)
    kwargs_1163 = {}
    # Getting the type of '_formatArray' (line 517)
    _formatArray_1146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 17), '_formatArray', False)
    # Calling _formatArray(args, kwargs) (line 517)
    _formatArray_call_result_1164 = invoke(stypy.reporting.localization.Localization(__file__, 517, 17), _formatArray_1146, *[subscript_call_result_1151, format_function_1152, result_sub_1155, max_line_len_1156, result_add_1159, separator_1160, edge_items_1161, summary_insert_1162], **kwargs_1163)
    
    # Applying the binary operator '+=' (line 517)
    result_iadd_1165 = python_operator(stypy.reporting.localization.Localization(__file__, 517, 12), '+=', s_1145, _formatArray_call_result_1164)
    # Assigning a type to the variable 's' (line 517)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 517, 12), 's', result_iadd_1165)
    
    
    # Assigning a BinOp to a Name (line 520):
    
    # Assigning a BinOp to a Name (line 520):
    
    # Call to rstrip(...): (line 520)
    # Processing the call keyword arguments (line 520)
    kwargs_1168 = {}
    # Getting the type of 's' (line 520)
    s_1166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 16), 's', False)
    # Obtaining the member 'rstrip' of a type (line 520)
    rstrip_1167 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 520, 16), s_1166, 'rstrip')
    # Calling rstrip(args, kwargs) (line 520)
    rstrip_call_result_1169 = invoke(stypy.reporting.localization.Localization(__file__, 520, 16), rstrip_1167, *[], **kwargs_1168)
    
    
    # Call to rstrip(...): (line 520)
    # Processing the call keyword arguments (line 520)
    kwargs_1172 = {}
    # Getting the type of 'sep' (line 520)
    sep_1170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 29), 'sep', False)
    # Obtaining the member 'rstrip' of a type (line 520)
    rstrip_1171 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 520, 29), sep_1170, 'rstrip')
    # Calling rstrip(args, kwargs) (line 520)
    rstrip_call_result_1173 = invoke(stypy.reporting.localization.Localization(__file__, 520, 29), rstrip_1171, *[], **kwargs_1172)
    
    # Applying the binary operator '+' (line 520)
    result_add_1174 = python_operator(stypy.reporting.localization.Localization(__file__, 520, 16), '+', rstrip_call_result_1169, rstrip_call_result_1173)
    
    str_1175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 520, 44), 'str', '\n')
    
    # Call to max(...): (line 520)
    # Processing the call arguments (line 520)
    # Getting the type of 'rank' (line 520)
    rank_1177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 53), 'rank', False)
    int_1178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 520, 58), 'int')
    # Applying the binary operator '-' (line 520)
    result_sub_1179 = python_operator(stypy.reporting.localization.Localization(__file__, 520, 53), '-', rank_1177, int_1178)
    
    int_1180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 520, 61), 'int')
    # Processing the call keyword arguments (line 520)
    kwargs_1181 = {}
    # Getting the type of 'max' (line 520)
    max_1176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 49), 'max', False)
    # Calling max(args, kwargs) (line 520)
    max_call_result_1182 = invoke(stypy.reporting.localization.Localization(__file__, 520, 49), max_1176, *[result_sub_1179, int_1180], **kwargs_1181)
    
    # Applying the binary operator '*' (line 520)
    result_mul_1183 = python_operator(stypy.reporting.localization.Localization(__file__, 520, 44), '*', str_1175, max_call_result_1182)
    
    # Applying the binary operator '+' (line 520)
    result_add_1184 = python_operator(stypy.reporting.localization.Localization(__file__, 520, 42), '+', result_add_1174, result_mul_1183)
    
    # Assigning a type to the variable 's' (line 520)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 520, 12), 's', result_add_1184)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    # Getting the type of 'leading_items' (line 521)
    leading_items_1185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 11), 'leading_items')
    
    # Getting the type of 'trailing_items' (line 521)
    trailing_items_1186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 28), 'trailing_items')
    int_1187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 521, 45), 'int')
    # Applying the binary operator '>' (line 521)
    result_gt_1188 = python_operator(stypy.reporting.localization.Localization(__file__, 521, 28), '>', trailing_items_1186, int_1187)
    
    # Applying the binary operator 'or' (line 521)
    result_or_keyword_1189 = python_operator(stypy.reporting.localization.Localization(__file__, 521, 11), 'or', leading_items_1185, result_gt_1188)
    
    # Testing the type of an if condition (line 521)
    if_condition_1190 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 521, 8), result_or_keyword_1189)
    # Assigning a type to the variable 'if_condition_1190' (line 521)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 521, 8), 'if_condition_1190', if_condition_1190)
    # SSA begins for if statement (line 521)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 's' (line 522)
    s_1191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 12), 's')
    # Getting the type of 'next_line_prefix' (line 522)
    next_line_prefix_1192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 17), 'next_line_prefix')
    # Applying the binary operator '+=' (line 522)
    result_iadd_1193 = python_operator(stypy.reporting.localization.Localization(__file__, 522, 12), '+=', s_1191, next_line_prefix_1192)
    # Assigning a type to the variable 's' (line 522)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 522, 12), 's', result_iadd_1193)
    
    # SSA join for if statement (line 521)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 's' (line 523)
    s_1194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 8), 's')
    
    # Call to rstrip(...): (line 523)
    # Processing the call keyword arguments (line 523)
    kwargs_1214 = {}
    
    # Call to _formatArray(...): (line 523)
    # Processing the call arguments (line 523)
    
    # Obtaining the type of the subscript
    int_1196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 523, 28), 'int')
    # Getting the type of 'a' (line 523)
    a_1197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 26), 'a', False)
    # Obtaining the member '__getitem__' of a type (line 523)
    getitem___1198 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 523, 26), a_1197, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 523)
    subscript_call_result_1199 = invoke(stypy.reporting.localization.Localization(__file__, 523, 26), getitem___1198, int_1196)
    
    # Getting the type of 'format_function' (line 523)
    format_function_1200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 33), 'format_function', False)
    # Getting the type of 'rank' (line 523)
    rank_1201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 50), 'rank', False)
    int_1202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 523, 55), 'int')
    # Applying the binary operator '-' (line 523)
    result_sub_1203 = python_operator(stypy.reporting.localization.Localization(__file__, 523, 50), '-', rank_1201, int_1202)
    
    # Getting the type of 'max_line_len' (line 523)
    max_line_len_1204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 58), 'max_line_len', False)
    str_1205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 524, 26), 'str', ' ')
    # Getting the type of 'next_line_prefix' (line 524)
    next_line_prefix_1206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 32), 'next_line_prefix', False)
    # Applying the binary operator '+' (line 524)
    result_add_1207 = python_operator(stypy.reporting.localization.Localization(__file__, 524, 26), '+', str_1205, next_line_prefix_1206)
    
    # Getting the type of 'separator' (line 524)
    separator_1208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 50), 'separator', False)
    # Getting the type of 'edge_items' (line 524)
    edge_items_1209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 61), 'edge_items', False)
    # Getting the type of 'summary_insert' (line 525)
    summary_insert_1210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 26), 'summary_insert', False)
    # Processing the call keyword arguments (line 523)
    kwargs_1211 = {}
    # Getting the type of '_formatArray' (line 523)
    _formatArray_1195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 13), '_formatArray', False)
    # Calling _formatArray(args, kwargs) (line 523)
    _formatArray_call_result_1212 = invoke(stypy.reporting.localization.Localization(__file__, 523, 13), _formatArray_1195, *[subscript_call_result_1199, format_function_1200, result_sub_1203, max_line_len_1204, result_add_1207, separator_1208, edge_items_1209, summary_insert_1210], **kwargs_1211)
    
    # Obtaining the member 'rstrip' of a type (line 523)
    rstrip_1213 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 523, 13), _formatArray_call_result_1212, 'rstrip')
    # Calling rstrip(args, kwargs) (line 523)
    rstrip_call_result_1215 = invoke(stypy.reporting.localization.Localization(__file__, 523, 13), rstrip_1213, *[], **kwargs_1214)
    
    str_1216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 525, 51), 'str', ']\n')
    # Applying the binary operator '+' (line 523)
    result_add_1217 = python_operator(stypy.reporting.localization.Localization(__file__, 523, 13), '+', rstrip_call_result_1215, str_1216)
    
    # Applying the binary operator '+=' (line 523)
    result_iadd_1218 = python_operator(stypy.reporting.localization.Localization(__file__, 523, 8), '+=', s_1194, result_add_1217)
    # Assigning a type to the variable 's' (line 523)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 523, 8), 's', result_iadd_1218)
    
    # SSA join for if statement (line 482)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 's' (line 526)
    s_1219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 11), 's')
    # Assigning a type to the variable 'stypy_return_type' (line 526)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 526, 4), 'stypy_return_type', s_1219)
    
    # ################# End of '_formatArray(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_formatArray' in the type store
    # Getting the type of 'stypy_return_type' (line 458)
    stypy_return_type_1220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_1220)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_formatArray'
    return stypy_return_type_1220

# Assigning a type to the variable '_formatArray' (line 458)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 458, 0), '_formatArray', _formatArray)
# Declaration of the 'FloatFormat' class

class FloatFormat(object, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'False' (line 529)
        False_1221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 61), 'False')
        defaults = [False_1221]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 529, 4, False)
        # Assigning a type to the variable 'self' (line 530)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 530, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FloatFormat.__init__', ['data', 'precision', 'suppress_small', 'sign'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['data', 'precision', 'suppress_small', 'sign'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 530):
        
        # Assigning a Name to a Attribute (line 530):
        # Getting the type of 'precision' (line 530)
        precision_1222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 25), 'precision')
        # Getting the type of 'self' (line 530)
        self_1223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 8), 'self')
        # Setting the type of the member 'precision' of a type (line 530)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 530, 8), self_1223, 'precision', precision_1222)
        
        # Assigning a Name to a Attribute (line 531):
        
        # Assigning a Name to a Attribute (line 531):
        # Getting the type of 'suppress_small' (line 531)
        suppress_small_1224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 30), 'suppress_small')
        # Getting the type of 'self' (line 531)
        self_1225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 8), 'self')
        # Setting the type of the member 'suppress_small' of a type (line 531)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 531, 8), self_1225, 'suppress_small', suppress_small_1224)
        
        # Assigning a Name to a Attribute (line 532):
        
        # Assigning a Name to a Attribute (line 532):
        # Getting the type of 'sign' (line 532)
        sign_1226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 20), 'sign')
        # Getting the type of 'self' (line 532)
        self_1227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 8), 'self')
        # Setting the type of the member 'sign' of a type (line 532)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 532, 8), self_1227, 'sign', sign_1226)
        
        # Assigning a Name to a Attribute (line 533):
        
        # Assigning a Name to a Attribute (line 533):
        # Getting the type of 'False' (line 533)
        False_1228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 26), 'False')
        # Getting the type of 'self' (line 533)
        self_1229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 8), 'self')
        # Setting the type of the member 'exp_format' of a type (line 533)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 533, 8), self_1229, 'exp_format', False_1228)
        
        # Assigning a Name to a Attribute (line 534):
        
        # Assigning a Name to a Attribute (line 534):
        # Getting the type of 'False' (line 534)
        False_1230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 30), 'False')
        # Getting the type of 'self' (line 534)
        self_1231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 8), 'self')
        # Setting the type of the member 'large_exponent' of a type (line 534)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 534, 8), self_1231, 'large_exponent', False_1230)
        
        # Assigning a Num to a Attribute (line 535):
        
        # Assigning a Num to a Attribute (line 535):
        int_1232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 535, 27), 'int')
        # Getting the type of 'self' (line 535)
        self_1233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 8), 'self')
        # Setting the type of the member 'max_str_len' of a type (line 535)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 535, 8), self_1233, 'max_str_len', int_1232)
        
        
        # SSA begins for try-except statement (line 536)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to fillFormat(...): (line 537)
        # Processing the call arguments (line 537)
        # Getting the type of 'data' (line 537)
        data_1236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 28), 'data', False)
        # Processing the call keyword arguments (line 537)
        kwargs_1237 = {}
        # Getting the type of 'self' (line 537)
        self_1234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 12), 'self', False)
        # Obtaining the member 'fillFormat' of a type (line 537)
        fillFormat_1235 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 537, 12), self_1234, 'fillFormat')
        # Calling fillFormat(args, kwargs) (line 537)
        fillFormat_call_result_1238 = invoke(stypy.reporting.localization.Localization(__file__, 537, 12), fillFormat_1235, *[data_1236], **kwargs_1237)
        
        # SSA branch for the except part of a try statement (line 536)
        # SSA branch for the except 'Tuple' branch of a try statement (line 536)
        module_type_store.open_ssa_branch('except')
        pass
        # SSA join for try-except statement (line 536)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def fillFormat(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'fillFormat'
        module_type_store = module_type_store.open_function_context('fillFormat', 543, 4, False)
        # Assigning a type to the variable 'self' (line 544)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 544, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FloatFormat.fillFormat.__dict__.__setitem__('stypy_localization', localization)
        FloatFormat.fillFormat.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FloatFormat.fillFormat.__dict__.__setitem__('stypy_type_store', module_type_store)
        FloatFormat.fillFormat.__dict__.__setitem__('stypy_function_name', 'FloatFormat.fillFormat')
        FloatFormat.fillFormat.__dict__.__setitem__('stypy_param_names_list', ['data'])
        FloatFormat.fillFormat.__dict__.__setitem__('stypy_varargs_param_name', None)
        FloatFormat.fillFormat.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FloatFormat.fillFormat.__dict__.__setitem__('stypy_call_defaults', defaults)
        FloatFormat.fillFormat.__dict__.__setitem__('stypy_call_varargs', varargs)
        FloatFormat.fillFormat.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FloatFormat.fillFormat.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FloatFormat.fillFormat', ['data'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'fillFormat', localization, ['data'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'fillFormat(...)' code ##################

        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 544, 8))
        
        # 'from numpy.core import _nc' statement (line 544)
        update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/core/')
        import_1239 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 544, 8), 'numpy.core')

        if (type(import_1239) is not StypyTypeError):

            if (import_1239 != 'pyd_module'):
                __import__(import_1239)
                sys_modules_1240 = sys.modules[import_1239]
                import_from_module(stypy.reporting.localization.Localization(__file__, 544, 8), 'numpy.core', sys_modules_1240.module_type_store, module_type_store, ['numeric'])
                nest_module(stypy.reporting.localization.Localization(__file__, 544, 8), __file__, sys_modules_1240, sys_modules_1240.module_type_store, module_type_store)
            else:
                from numpy.core import numeric as _nc

                import_from_module(stypy.reporting.localization.Localization(__file__, 544, 8), 'numpy.core', None, module_type_store, ['numeric'], [_nc])

        else:
            # Assigning a type to the variable 'numpy.core' (line 544)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 544, 8), 'numpy.core', import_1239)

        # Adding an alias
        module_type_store.add_alias('_nc', 'numeric')
        remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/core/')
        
        
        # Call to errstate(...): (line 546)
        # Processing the call keyword arguments (line 546)
        str_1243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 546, 30), 'str', 'ignore')
        keyword_1244 = str_1243
        kwargs_1245 = {'all': keyword_1244}
        # Getting the type of '_nc' (line 546)
        _nc_1241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 13), '_nc', False)
        # Obtaining the member 'errstate' of a type (line 546)
        errstate_1242 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 546, 13), _nc_1241, 'errstate')
        # Calling errstate(args, kwargs) (line 546)
        errstate_call_result_1246 = invoke(stypy.reporting.localization.Localization(__file__, 546, 13), errstate_1242, *[], **kwargs_1245)
        
        with_1247 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 546, 13), errstate_call_result_1246, 'with parameter', '__enter__', '__exit__')

        if with_1247:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 546)
            enter___1248 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 546, 13), errstate_call_result_1246, '__enter__')
            with_enter_1249 = invoke(stypy.reporting.localization.Localization(__file__, 546, 13), enter___1248)
            
            # Assigning a BinOp to a Name (line 547):
            
            # Assigning a BinOp to a Name (line 547):
            
            # Call to isnan(...): (line 547)
            # Processing the call arguments (line 547)
            # Getting the type of 'data' (line 547)
            data_1251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 28), 'data', False)
            # Processing the call keyword arguments (line 547)
            kwargs_1252 = {}
            # Getting the type of 'isnan' (line 547)
            isnan_1250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 22), 'isnan', False)
            # Calling isnan(args, kwargs) (line 547)
            isnan_call_result_1253 = invoke(stypy.reporting.localization.Localization(__file__, 547, 22), isnan_1250, *[data_1251], **kwargs_1252)
            
            
            # Call to isinf(...): (line 547)
            # Processing the call arguments (line 547)
            # Getting the type of 'data' (line 547)
            data_1255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 42), 'data', False)
            # Processing the call keyword arguments (line 547)
            kwargs_1256 = {}
            # Getting the type of 'isinf' (line 547)
            isinf_1254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 36), 'isinf', False)
            # Calling isinf(args, kwargs) (line 547)
            isinf_call_result_1257 = invoke(stypy.reporting.localization.Localization(__file__, 547, 36), isinf_1254, *[data_1255], **kwargs_1256)
            
            # Applying the binary operator '|' (line 547)
            result_or__1258 = python_operator(stypy.reporting.localization.Localization(__file__, 547, 22), '|', isnan_call_result_1253, isinf_call_result_1257)
            
            # Assigning a type to the variable 'special' (line 547)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 547, 12), 'special', result_or__1258)
            
            # Assigning a BinOp to a Name (line 548):
            
            # Assigning a BinOp to a Name (line 548):
            
            # Call to not_equal(...): (line 548)
            # Processing the call arguments (line 548)
            # Getting the type of 'data' (line 548)
            data_1260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 30), 'data', False)
            int_1261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 548, 36), 'int')
            # Processing the call keyword arguments (line 548)
            kwargs_1262 = {}
            # Getting the type of 'not_equal' (line 548)
            not_equal_1259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 20), 'not_equal', False)
            # Calling not_equal(args, kwargs) (line 548)
            not_equal_call_result_1263 = invoke(stypy.reporting.localization.Localization(__file__, 548, 20), not_equal_1259, *[data_1260, int_1261], **kwargs_1262)
            
            
            # Getting the type of 'special' (line 548)
            special_1264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 42), 'special')
            # Applying the '~' unary operator (line 548)
            result_inv_1265 = python_operator(stypy.reporting.localization.Localization(__file__, 548, 41), '~', special_1264)
            
            # Applying the binary operator '&' (line 548)
            result_and__1266 = python_operator(stypy.reporting.localization.Localization(__file__, 548, 20), '&', not_equal_call_result_1263, result_inv_1265)
            
            # Assigning a type to the variable 'valid' (line 548)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 548, 12), 'valid', result_and__1266)
            
            # Assigning a Call to a Name (line 549):
            
            # Assigning a Call to a Name (line 549):
            
            # Call to absolute(...): (line 549)
            # Processing the call arguments (line 549)
            
            # Call to compress(...): (line 549)
            # Processing the call arguments (line 549)
            # Getting the type of 'valid' (line 549)
            valid_1270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 46), 'valid', False)
            # Processing the call keyword arguments (line 549)
            kwargs_1271 = {}
            # Getting the type of 'data' (line 549)
            data_1268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 32), 'data', False)
            # Obtaining the member 'compress' of a type (line 549)
            compress_1269 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 549, 32), data_1268, 'compress')
            # Calling compress(args, kwargs) (line 549)
            compress_call_result_1272 = invoke(stypy.reporting.localization.Localization(__file__, 549, 32), compress_1269, *[valid_1270], **kwargs_1271)
            
            # Processing the call keyword arguments (line 549)
            kwargs_1273 = {}
            # Getting the type of 'absolute' (line 549)
            absolute_1267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 23), 'absolute', False)
            # Calling absolute(args, kwargs) (line 549)
            absolute_call_result_1274 = invoke(stypy.reporting.localization.Localization(__file__, 549, 23), absolute_1267, *[compress_call_result_1272], **kwargs_1273)
            
            # Assigning a type to the variable 'non_zero' (line 549)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 549, 12), 'non_zero', absolute_call_result_1274)
            
            
            
            # Call to len(...): (line 550)
            # Processing the call arguments (line 550)
            # Getting the type of 'non_zero' (line 550)
            non_zero_1276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 19), 'non_zero', False)
            # Processing the call keyword arguments (line 550)
            kwargs_1277 = {}
            # Getting the type of 'len' (line 550)
            len_1275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 15), 'len', False)
            # Calling len(args, kwargs) (line 550)
            len_call_result_1278 = invoke(stypy.reporting.localization.Localization(__file__, 550, 15), len_1275, *[non_zero_1276], **kwargs_1277)
            
            int_1279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 550, 32), 'int')
            # Applying the binary operator '==' (line 550)
            result_eq_1280 = python_operator(stypy.reporting.localization.Localization(__file__, 550, 15), '==', len_call_result_1278, int_1279)
            
            # Testing the type of an if condition (line 550)
            if_condition_1281 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 550, 12), result_eq_1280)
            # Assigning a type to the variable 'if_condition_1281' (line 550)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 550, 12), 'if_condition_1281', if_condition_1281)
            # SSA begins for if statement (line 550)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Num to a Name (line 551):
            
            # Assigning a Num to a Name (line 551):
            float_1282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 551, 26), 'float')
            # Assigning a type to the variable 'max_val' (line 551)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 551, 16), 'max_val', float_1282)
            
            # Assigning a Num to a Name (line 552):
            
            # Assigning a Num to a Name (line 552):
            float_1283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 552, 26), 'float')
            # Assigning a type to the variable 'min_val' (line 552)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 552, 16), 'min_val', float_1283)
            # SSA branch for the else part of an if statement (line 550)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Call to a Name (line 554):
            
            # Assigning a Call to a Name (line 554):
            
            # Call to reduce(...): (line 554)
            # Processing the call arguments (line 554)
            # Getting the type of 'non_zero' (line 554)
            non_zero_1286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 41), 'non_zero', False)
            # Processing the call keyword arguments (line 554)
            kwargs_1287 = {}
            # Getting the type of 'maximum' (line 554)
            maximum_1284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 26), 'maximum', False)
            # Obtaining the member 'reduce' of a type (line 554)
            reduce_1285 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 554, 26), maximum_1284, 'reduce')
            # Calling reduce(args, kwargs) (line 554)
            reduce_call_result_1288 = invoke(stypy.reporting.localization.Localization(__file__, 554, 26), reduce_1285, *[non_zero_1286], **kwargs_1287)
            
            # Assigning a type to the variable 'max_val' (line 554)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 554, 16), 'max_val', reduce_call_result_1288)
            
            # Assigning a Call to a Name (line 555):
            
            # Assigning a Call to a Name (line 555):
            
            # Call to reduce(...): (line 555)
            # Processing the call arguments (line 555)
            # Getting the type of 'non_zero' (line 555)
            non_zero_1291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 41), 'non_zero', False)
            # Processing the call keyword arguments (line 555)
            kwargs_1292 = {}
            # Getting the type of 'minimum' (line 555)
            minimum_1289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 26), 'minimum', False)
            # Obtaining the member 'reduce' of a type (line 555)
            reduce_1290 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 555, 26), minimum_1289, 'reduce')
            # Calling reduce(args, kwargs) (line 555)
            reduce_call_result_1293 = invoke(stypy.reporting.localization.Localization(__file__, 555, 26), reduce_1290, *[non_zero_1291], **kwargs_1292)
            
            # Assigning a type to the variable 'min_val' (line 555)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 555, 16), 'min_val', reduce_call_result_1293)
            
            
            # Getting the type of 'max_val' (line 556)
            max_val_1294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 19), 'max_val')
            float_1295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 556, 30), 'float')
            # Applying the binary operator '>=' (line 556)
            result_ge_1296 = python_operator(stypy.reporting.localization.Localization(__file__, 556, 19), '>=', max_val_1294, float_1295)
            
            # Testing the type of an if condition (line 556)
            if_condition_1297 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 556, 16), result_ge_1296)
            # Assigning a type to the variable 'if_condition_1297' (line 556)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 556, 16), 'if_condition_1297', if_condition_1297)
            # SSA begins for if statement (line 556)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Name to a Attribute (line 557):
            
            # Assigning a Name to a Attribute (line 557):
            # Getting the type of 'True' (line 557)
            True_1298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 38), 'True')
            # Getting the type of 'self' (line 557)
            self_1299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 20), 'self')
            # Setting the type of the member 'exp_format' of a type (line 557)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 557, 20), self_1299, 'exp_format', True_1298)
            # SSA join for if statement (line 556)
            module_type_store = module_type_store.join_ssa_context()
            
            
            
            # Evaluating a boolean operation
            
            # Getting the type of 'self' (line 558)
            self_1300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 23), 'self')
            # Obtaining the member 'suppress_small' of a type (line 558)
            suppress_small_1301 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 558, 23), self_1300, 'suppress_small')
            # Applying the 'not' unary operator (line 558)
            result_not__1302 = python_operator(stypy.reporting.localization.Localization(__file__, 558, 19), 'not', suppress_small_1301)
            
            
            # Evaluating a boolean operation
            
            # Getting the type of 'min_val' (line 558)
            min_val_1303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 48), 'min_val')
            float_1304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 558, 58), 'float')
            # Applying the binary operator '<' (line 558)
            result_lt_1305 = python_operator(stypy.reporting.localization.Localization(__file__, 558, 48), '<', min_val_1303, float_1304)
            
            
            # Getting the type of 'max_val' (line 559)
            max_val_1306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 46), 'max_val')
            # Getting the type of 'min_val' (line 559)
            min_val_1307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 54), 'min_val')
            # Applying the binary operator 'div' (line 559)
            result_div_1308 = python_operator(stypy.reporting.localization.Localization(__file__, 559, 46), 'div', max_val_1306, min_val_1307)
            
            float_1309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 559, 64), 'float')
            # Applying the binary operator '>' (line 559)
            result_gt_1310 = python_operator(stypy.reporting.localization.Localization(__file__, 559, 46), '>', result_div_1308, float_1309)
            
            # Applying the binary operator 'or' (line 558)
            result_or_keyword_1311 = python_operator(stypy.reporting.localization.Localization(__file__, 558, 48), 'or', result_lt_1305, result_gt_1310)
            
            # Applying the binary operator 'and' (line 558)
            result_and_keyword_1312 = python_operator(stypy.reporting.localization.Localization(__file__, 558, 19), 'and', result_not__1302, result_or_keyword_1311)
            
            # Testing the type of an if condition (line 558)
            if_condition_1313 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 558, 16), result_and_keyword_1312)
            # Assigning a type to the variable 'if_condition_1313' (line 558)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 558, 16), 'if_condition_1313', if_condition_1313)
            # SSA begins for if statement (line 558)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Name to a Attribute (line 560):
            
            # Assigning a Name to a Attribute (line 560):
            # Getting the type of 'True' (line 560)
            True_1314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 38), 'True')
            # Getting the type of 'self' (line 560)
            self_1315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 20), 'self')
            # Setting the type of the member 'exp_format' of a type (line 560)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 560, 20), self_1315, 'exp_format', True_1314)
            # SSA join for if statement (line 558)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 550)
            module_type_store = module_type_store.join_ssa_context()
            
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 546)
            exit___1316 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 546, 13), errstate_call_result_1246, '__exit__')
            with_exit_1317 = invoke(stypy.reporting.localization.Localization(__file__, 546, 13), exit___1316, None, None, None)

        
        # Getting the type of 'self' (line 562)
        self_1318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 11), 'self')
        # Obtaining the member 'exp_format' of a type (line 562)
        exp_format_1319 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 562, 11), self_1318, 'exp_format')
        # Testing the type of an if condition (line 562)
        if_condition_1320 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 562, 8), exp_format_1319)
        # Assigning a type to the variable 'if_condition_1320' (line 562)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 562, 8), 'if_condition_1320', if_condition_1320)
        # SSA begins for if statement (line 562)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BoolOp to a Attribute (line 563):
        
        # Assigning a BoolOp to a Attribute (line 563):
        
        # Evaluating a boolean operation
        
        int_1321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 563, 34), 'int')
        # Getting the type of 'min_val' (line 563)
        min_val_1322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 38), 'min_val')
        # Applying the binary operator '<' (line 563)
        result_lt_1323 = python_operator(stypy.reporting.localization.Localization(__file__, 563, 34), '<', int_1321, min_val_1322)
        float_1324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 563, 48), 'float')
        # Applying the binary operator '<' (line 563)
        result_lt_1325 = python_operator(stypy.reporting.localization.Localization(__file__, 563, 34), '<', min_val_1322, float_1324)
        # Applying the binary operator '&' (line 563)
        result_and__1326 = python_operator(stypy.reporting.localization.Localization(__file__, 563, 34), '&', result_lt_1323, result_lt_1325)
        
        
        # Getting the type of 'max_val' (line 563)
        max_val_1327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 57), 'max_val')
        float_1328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 563, 68), 'float')
        # Applying the binary operator '>=' (line 563)
        result_ge_1329 = python_operator(stypy.reporting.localization.Localization(__file__, 563, 57), '>=', max_val_1327, float_1328)
        
        # Applying the binary operator 'or' (line 563)
        result_or_keyword_1330 = python_operator(stypy.reporting.localization.Localization(__file__, 563, 34), 'or', result_and__1326, result_ge_1329)
        
        # Getting the type of 'self' (line 563)
        self_1331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 12), 'self')
        # Setting the type of the member 'large_exponent' of a type (line 563)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 563, 12), self_1331, 'large_exponent', result_or_keyword_1330)
        
        # Assigning a BinOp to a Attribute (line 564):
        
        # Assigning a BinOp to a Attribute (line 564):
        int_1332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 564, 31), 'int')
        # Getting the type of 'self' (line 564)
        self_1333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 35), 'self')
        # Obtaining the member 'precision' of a type (line 564)
        precision_1334 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 564, 35), self_1333, 'precision')
        # Applying the binary operator '+' (line 564)
        result_add_1335 = python_operator(stypy.reporting.localization.Localization(__file__, 564, 31), '+', int_1332, precision_1334)
        
        # Getting the type of 'self' (line 564)
        self_1336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 12), 'self')
        # Setting the type of the member 'max_str_len' of a type (line 564)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 564, 12), self_1336, 'max_str_len', result_add_1335)
        
        # Getting the type of 'self' (line 565)
        self_1337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 15), 'self')
        # Obtaining the member 'large_exponent' of a type (line 565)
        large_exponent_1338 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 565, 15), self_1337, 'large_exponent')
        # Testing the type of an if condition (line 565)
        if_condition_1339 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 565, 12), large_exponent_1338)
        # Assigning a type to the variable 'if_condition_1339' (line 565)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 565, 12), 'if_condition_1339', if_condition_1339)
        # SSA begins for if statement (line 565)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'self' (line 566)
        self_1340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 16), 'self')
        # Obtaining the member 'max_str_len' of a type (line 566)
        max_str_len_1341 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 566, 16), self_1340, 'max_str_len')
        int_1342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 566, 36), 'int')
        # Applying the binary operator '+=' (line 566)
        result_iadd_1343 = python_operator(stypy.reporting.localization.Localization(__file__, 566, 16), '+=', max_str_len_1341, int_1342)
        # Getting the type of 'self' (line 566)
        self_1344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 16), 'self')
        # Setting the type of the member 'max_str_len' of a type (line 566)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 566, 16), self_1344, 'max_str_len', result_iadd_1343)
        
        # SSA join for if statement (line 565)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'self' (line 567)
        self_1345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 15), 'self')
        # Obtaining the member 'sign' of a type (line 567)
        sign_1346 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 567, 15), self_1345, 'sign')
        # Testing the type of an if condition (line 567)
        if_condition_1347 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 567, 12), sign_1346)
        # Assigning a type to the variable 'if_condition_1347' (line 567)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 567, 12), 'if_condition_1347', if_condition_1347)
        # SSA begins for if statement (line 567)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Str to a Name (line 568):
        
        # Assigning a Str to a Name (line 568):
        str_1348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 568, 25), 'str', '%+')
        # Assigning a type to the variable 'format' (line 568)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 568, 16), 'format', str_1348)
        # SSA branch for the else part of an if statement (line 567)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Str to a Name (line 570):
        
        # Assigning a Str to a Name (line 570):
        str_1349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 570, 25), 'str', '%')
        # Assigning a type to the variable 'format' (line 570)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 570, 16), 'format', str_1349)
        # SSA join for if statement (line 567)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 571):
        
        # Assigning a BinOp to a Name (line 571):
        # Getting the type of 'format' (line 571)
        format_1350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 21), 'format')
        str_1351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 571, 30), 'str', '%d.%de')
        
        # Obtaining an instance of the builtin type 'tuple' (line 571)
        tuple_1352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 571, 42), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 571)
        # Adding element type (line 571)
        # Getting the type of 'self' (line 571)
        self_1353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 42), 'self')
        # Obtaining the member 'max_str_len' of a type (line 571)
        max_str_len_1354 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 571, 42), self_1353, 'max_str_len')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 571, 42), tuple_1352, max_str_len_1354)
        # Adding element type (line 571)
        # Getting the type of 'self' (line 571)
        self_1355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 60), 'self')
        # Obtaining the member 'precision' of a type (line 571)
        precision_1356 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 571, 60), self_1355, 'precision')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 571, 42), tuple_1352, precision_1356)
        
        # Applying the binary operator '%' (line 571)
        result_mod_1357 = python_operator(stypy.reporting.localization.Localization(__file__, 571, 30), '%', str_1351, tuple_1352)
        
        # Applying the binary operator '+' (line 571)
        result_add_1358 = python_operator(stypy.reporting.localization.Localization(__file__, 571, 21), '+', format_1350, result_mod_1357)
        
        # Assigning a type to the variable 'format' (line 571)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 571, 12), 'format', result_add_1358)
        # SSA branch for the else part of an if statement (line 562)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a BinOp to a Name (line 573):
        
        # Assigning a BinOp to a Name (line 573):
        str_1359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 573, 21), 'str', '%%.%df')
        
        # Obtaining an instance of the builtin type 'tuple' (line 573)
        tuple_1360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 573, 33), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 573)
        # Adding element type (line 573)
        # Getting the type of 'self' (line 573)
        self_1361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 33), 'self')
        # Obtaining the member 'precision' of a type (line 573)
        precision_1362 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 573, 33), self_1361, 'precision')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 573, 33), tuple_1360, precision_1362)
        
        # Applying the binary operator '%' (line 573)
        result_mod_1363 = python_operator(stypy.reporting.localization.Localization(__file__, 573, 21), '%', str_1359, tuple_1360)
        
        # Assigning a type to the variable 'format' (line 573)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 573, 12), 'format', result_mod_1363)
        
        
        # Call to len(...): (line 574)
        # Processing the call arguments (line 574)
        # Getting the type of 'non_zero' (line 574)
        non_zero_1365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 19), 'non_zero', False)
        # Processing the call keyword arguments (line 574)
        kwargs_1366 = {}
        # Getting the type of 'len' (line 574)
        len_1364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 15), 'len', False)
        # Calling len(args, kwargs) (line 574)
        len_call_result_1367 = invoke(stypy.reporting.localization.Localization(__file__, 574, 15), len_1364, *[non_zero_1365], **kwargs_1366)
        
        # Testing the type of an if condition (line 574)
        if_condition_1368 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 574, 12), len_call_result_1367)
        # Assigning a type to the variable 'if_condition_1368' (line 574)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 574, 12), 'if_condition_1368', if_condition_1368)
        # SSA begins for if statement (line 574)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 575):
        
        # Assigning a Call to a Name (line 575):
        
        # Call to max(...): (line 575)
        # Processing the call arguments (line 575)
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'non_zero' (line 576)
        non_zero_1377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 42), 'non_zero', False)
        comprehension_1378 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 575, 33), non_zero_1377)
        # Assigning a type to the variable 'x' (line 575)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 575, 33), 'x', comprehension_1378)
        
        # Call to _digits(...): (line 575)
        # Processing the call arguments (line 575)
        # Getting the type of 'x' (line 575)
        x_1371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 41), 'x', False)
        # Getting the type of 'self' (line 575)
        self_1372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 44), 'self', False)
        # Obtaining the member 'precision' of a type (line 575)
        precision_1373 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 575, 44), self_1372, 'precision')
        # Getting the type of 'format' (line 575)
        format_1374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 60), 'format', False)
        # Processing the call keyword arguments (line 575)
        kwargs_1375 = {}
        # Getting the type of '_digits' (line 575)
        _digits_1370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 33), '_digits', False)
        # Calling _digits(args, kwargs) (line 575)
        _digits_call_result_1376 = invoke(stypy.reporting.localization.Localization(__file__, 575, 33), _digits_1370, *[x_1371, precision_1373, format_1374], **kwargs_1375)
        
        list_1379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 575, 33), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 575, 33), list_1379, _digits_call_result_1376)
        # Processing the call keyword arguments (line 575)
        kwargs_1380 = {}
        # Getting the type of 'max' (line 575)
        max_1369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 28), 'max', False)
        # Calling max(args, kwargs) (line 575)
        max_call_result_1381 = invoke(stypy.reporting.localization.Localization(__file__, 575, 28), max_1369, *[list_1379], **kwargs_1380)
        
        # Assigning a type to the variable 'precision' (line 575)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 575, 16), 'precision', max_call_result_1381)
        # SSA branch for the else part of an if statement (line 574)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Num to a Name (line 578):
        
        # Assigning a Num to a Name (line 578):
        int_1382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 578, 28), 'int')
        # Assigning a type to the variable 'precision' (line 578)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 578, 16), 'precision', int_1382)
        # SSA join for if statement (line 574)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 579):
        
        # Assigning a Call to a Name (line 579):
        
        # Call to min(...): (line 579)
        # Processing the call arguments (line 579)
        # Getting the type of 'self' (line 579)
        self_1384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 28), 'self', False)
        # Obtaining the member 'precision' of a type (line 579)
        precision_1385 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 579, 28), self_1384, 'precision')
        # Getting the type of 'precision' (line 579)
        precision_1386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 44), 'precision', False)
        # Processing the call keyword arguments (line 579)
        kwargs_1387 = {}
        # Getting the type of 'min' (line 579)
        min_1383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 24), 'min', False)
        # Calling min(args, kwargs) (line 579)
        min_call_result_1388 = invoke(stypy.reporting.localization.Localization(__file__, 579, 24), min_1383, *[precision_1385, precision_1386], **kwargs_1387)
        
        # Assigning a type to the variable 'precision' (line 579)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 579, 12), 'precision', min_call_result_1388)
        
        # Assigning a BinOp to a Attribute (line 580):
        
        # Assigning a BinOp to a Attribute (line 580):
        
        # Call to len(...): (line 580)
        # Processing the call arguments (line 580)
        
        # Call to str(...): (line 580)
        # Processing the call arguments (line 580)
        
        # Call to int(...): (line 580)
        # Processing the call arguments (line 580)
        # Getting the type of 'max_val' (line 580)
        max_val_1392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 43), 'max_val', False)
        # Processing the call keyword arguments (line 580)
        kwargs_1393 = {}
        # Getting the type of 'int' (line 580)
        int_1391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 39), 'int', False)
        # Calling int(args, kwargs) (line 580)
        int_call_result_1394 = invoke(stypy.reporting.localization.Localization(__file__, 580, 39), int_1391, *[max_val_1392], **kwargs_1393)
        
        # Processing the call keyword arguments (line 580)
        kwargs_1395 = {}
        # Getting the type of 'str' (line 580)
        str_1390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 35), 'str', False)
        # Calling str(args, kwargs) (line 580)
        str_call_result_1396 = invoke(stypy.reporting.localization.Localization(__file__, 580, 35), str_1390, *[int_call_result_1394], **kwargs_1395)
        
        # Processing the call keyword arguments (line 580)
        kwargs_1397 = {}
        # Getting the type of 'len' (line 580)
        len_1389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 31), 'len', False)
        # Calling len(args, kwargs) (line 580)
        len_call_result_1398 = invoke(stypy.reporting.localization.Localization(__file__, 580, 31), len_1389, *[str_call_result_1396], **kwargs_1397)
        
        # Getting the type of 'precision' (line 580)
        precision_1399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 56), 'precision')
        # Applying the binary operator '+' (line 580)
        result_add_1400 = python_operator(stypy.reporting.localization.Localization(__file__, 580, 31), '+', len_call_result_1398, precision_1399)
        
        int_1401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 580, 68), 'int')
        # Applying the binary operator '+' (line 580)
        result_add_1402 = python_operator(stypy.reporting.localization.Localization(__file__, 580, 66), '+', result_add_1400, int_1401)
        
        # Getting the type of 'self' (line 580)
        self_1403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 12), 'self')
        # Setting the type of the member 'max_str_len' of a type (line 580)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 580, 12), self_1403, 'max_str_len', result_add_1402)
        
        
        # Call to any(...): (line 581)
        # Processing the call arguments (line 581)
        # Getting the type of 'special' (line 581)
        special_1406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 23), 'special', False)
        # Processing the call keyword arguments (line 581)
        kwargs_1407 = {}
        # Getting the type of '_nc' (line 581)
        _nc_1404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 15), '_nc', False)
        # Obtaining the member 'any' of a type (line 581)
        any_1405 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 581, 15), _nc_1404, 'any')
        # Calling any(args, kwargs) (line 581)
        any_call_result_1408 = invoke(stypy.reporting.localization.Localization(__file__, 581, 15), any_1405, *[special_1406], **kwargs_1407)
        
        # Testing the type of an if condition (line 581)
        if_condition_1409 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 581, 12), any_call_result_1408)
        # Assigning a type to the variable 'if_condition_1409' (line 581)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 581, 12), 'if_condition_1409', if_condition_1409)
        # SSA begins for if statement (line 581)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Attribute (line 582):
        
        # Assigning a Call to a Attribute (line 582):
        
        # Call to max(...): (line 582)
        # Processing the call arguments (line 582)
        # Getting the type of 'self' (line 582)
        self_1411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 39), 'self', False)
        # Obtaining the member 'max_str_len' of a type (line 582)
        max_str_len_1412 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 582, 39), self_1411, 'max_str_len')
        
        # Call to len(...): (line 583)
        # Processing the call arguments (line 583)
        # Getting the type of '_nan_str' (line 583)
        _nan_str_1414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 43), '_nan_str', False)
        # Processing the call keyword arguments (line 583)
        kwargs_1415 = {}
        # Getting the type of 'len' (line 583)
        len_1413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 39), 'len', False)
        # Calling len(args, kwargs) (line 583)
        len_call_result_1416 = invoke(stypy.reporting.localization.Localization(__file__, 583, 39), len_1413, *[_nan_str_1414], **kwargs_1415)
        
        
        # Call to len(...): (line 584)
        # Processing the call arguments (line 584)
        # Getting the type of '_inf_str' (line 584)
        _inf_str_1418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 43), '_inf_str', False)
        # Processing the call keyword arguments (line 584)
        kwargs_1419 = {}
        # Getting the type of 'len' (line 584)
        len_1417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 39), 'len', False)
        # Calling len(args, kwargs) (line 584)
        len_call_result_1420 = invoke(stypy.reporting.localization.Localization(__file__, 584, 39), len_1417, *[_inf_str_1418], **kwargs_1419)
        
        int_1421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 584, 53), 'int')
        # Applying the binary operator '+' (line 584)
        result_add_1422 = python_operator(stypy.reporting.localization.Localization(__file__, 584, 39), '+', len_call_result_1420, int_1421)
        
        # Processing the call keyword arguments (line 582)
        kwargs_1423 = {}
        # Getting the type of 'max' (line 582)
        max_1410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 35), 'max', False)
        # Calling max(args, kwargs) (line 582)
        max_call_result_1424 = invoke(stypy.reporting.localization.Localization(__file__, 582, 35), max_1410, *[max_str_len_1412, len_call_result_1416, result_add_1422], **kwargs_1423)
        
        # Getting the type of 'self' (line 582)
        self_1425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 16), 'self')
        # Setting the type of the member 'max_str_len' of a type (line 582)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 582, 16), self_1425, 'max_str_len', max_call_result_1424)
        # SSA join for if statement (line 581)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'self' (line 585)
        self_1426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 15), 'self')
        # Obtaining the member 'sign' of a type (line 585)
        sign_1427 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 585, 15), self_1426, 'sign')
        # Testing the type of an if condition (line 585)
        if_condition_1428 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 585, 12), sign_1427)
        # Assigning a type to the variable 'if_condition_1428' (line 585)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 585, 12), 'if_condition_1428', if_condition_1428)
        # SSA begins for if statement (line 585)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Str to a Name (line 586):
        
        # Assigning a Str to a Name (line 586):
        str_1429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 586, 25), 'str', '%#+')
        # Assigning a type to the variable 'format' (line 586)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 586, 16), 'format', str_1429)
        # SSA branch for the else part of an if statement (line 585)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Str to a Name (line 588):
        
        # Assigning a Str to a Name (line 588):
        str_1430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 588, 25), 'str', '%#')
        # Assigning a type to the variable 'format' (line 588)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 588, 16), 'format', str_1430)
        # SSA join for if statement (line 585)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 589):
        
        # Assigning a BinOp to a Name (line 589):
        # Getting the type of 'format' (line 589)
        format_1431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 21), 'format')
        str_1432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 589, 30), 'str', '%d.%df')
        
        # Obtaining an instance of the builtin type 'tuple' (line 589)
        tuple_1433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 589, 42), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 589)
        # Adding element type (line 589)
        # Getting the type of 'self' (line 589)
        self_1434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 42), 'self')
        # Obtaining the member 'max_str_len' of a type (line 589)
        max_str_len_1435 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 589, 42), self_1434, 'max_str_len')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 589, 42), tuple_1433, max_str_len_1435)
        # Adding element type (line 589)
        # Getting the type of 'precision' (line 589)
        precision_1436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 60), 'precision')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 589, 42), tuple_1433, precision_1436)
        
        # Applying the binary operator '%' (line 589)
        result_mod_1437 = python_operator(stypy.reporting.localization.Localization(__file__, 589, 30), '%', str_1432, tuple_1433)
        
        # Applying the binary operator '+' (line 589)
        result_add_1438 = python_operator(stypy.reporting.localization.Localization(__file__, 589, 21), '+', format_1431, result_mod_1437)
        
        # Assigning a type to the variable 'format' (line 589)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 589, 12), 'format', result_add_1438)
        # SSA join for if statement (line 562)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Attribute (line 591):
        
        # Assigning a BinOp to a Attribute (line 591):
        str_1439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 591, 27), 'str', '%%%ds')
        
        # Obtaining an instance of the builtin type 'tuple' (line 591)
        tuple_1440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 591, 38), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 591)
        # Adding element type (line 591)
        # Getting the type of 'self' (line 591)
        self_1441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 38), 'self')
        # Obtaining the member 'max_str_len' of a type (line 591)
        max_str_len_1442 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 591, 38), self_1441, 'max_str_len')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 591, 38), tuple_1440, max_str_len_1442)
        
        # Applying the binary operator '%' (line 591)
        result_mod_1443 = python_operator(stypy.reporting.localization.Localization(__file__, 591, 27), '%', str_1439, tuple_1440)
        
        # Getting the type of 'self' (line 591)
        self_1444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 8), 'self')
        # Setting the type of the member 'special_fmt' of a type (line 591)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 591, 8), self_1444, 'special_fmt', result_mod_1443)
        
        # Assigning a Name to a Attribute (line 592):
        
        # Assigning a Name to a Attribute (line 592):
        # Getting the type of 'format' (line 592)
        format_1445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 22), 'format')
        # Getting the type of 'self' (line 592)
        self_1446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 8), 'self')
        # Setting the type of the member 'format' of a type (line 592)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 592, 8), self_1446, 'format', format_1445)
        
        # ################# End of 'fillFormat(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'fillFormat' in the type store
        # Getting the type of 'stypy_return_type' (line 543)
        stypy_return_type_1447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1447)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'fillFormat'
        return stypy_return_type_1447


    @norecursion
    def __call__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'True' (line 594)
        True_1448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 38), 'True')
        defaults = [True_1448]
        # Create a new context for function '__call__'
        module_type_store = module_type_store.open_function_context('__call__', 594, 4, False)
        # Assigning a type to the variable 'self' (line 595)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 595, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FloatFormat.__call__.__dict__.__setitem__('stypy_localization', localization)
        FloatFormat.__call__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FloatFormat.__call__.__dict__.__setitem__('stypy_type_store', module_type_store)
        FloatFormat.__call__.__dict__.__setitem__('stypy_function_name', 'FloatFormat.__call__')
        FloatFormat.__call__.__dict__.__setitem__('stypy_param_names_list', ['x', 'strip_zeros'])
        FloatFormat.__call__.__dict__.__setitem__('stypy_varargs_param_name', None)
        FloatFormat.__call__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FloatFormat.__call__.__dict__.__setitem__('stypy_call_defaults', defaults)
        FloatFormat.__call__.__dict__.__setitem__('stypy_call_varargs', varargs)
        FloatFormat.__call__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FloatFormat.__call__.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FloatFormat.__call__', ['x', 'strip_zeros'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__call__', localization, ['x', 'strip_zeros'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__call__(...)' code ##################

        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 595, 8))
        
        # 'from numpy.core import _nc' statement (line 595)
        update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/core/')
        import_1449 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 595, 8), 'numpy.core')

        if (type(import_1449) is not StypyTypeError):

            if (import_1449 != 'pyd_module'):
                __import__(import_1449)
                sys_modules_1450 = sys.modules[import_1449]
                import_from_module(stypy.reporting.localization.Localization(__file__, 595, 8), 'numpy.core', sys_modules_1450.module_type_store, module_type_store, ['numeric'])
                nest_module(stypy.reporting.localization.Localization(__file__, 595, 8), __file__, sys_modules_1450, sys_modules_1450.module_type_store, module_type_store)
            else:
                from numpy.core import numeric as _nc

                import_from_module(stypy.reporting.localization.Localization(__file__, 595, 8), 'numpy.core', None, module_type_store, ['numeric'], [_nc])

        else:
            # Assigning a type to the variable 'numpy.core' (line 595)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 595, 8), 'numpy.core', import_1449)

        # Adding an alias
        module_type_store.add_alias('_nc', 'numeric')
        remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/core/')
        
        
        # Call to errstate(...): (line 597)
        # Processing the call keyword arguments (line 597)
        str_1453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 597, 34), 'str', 'ignore')
        keyword_1454 = str_1453
        kwargs_1455 = {'invalid': keyword_1454}
        # Getting the type of '_nc' (line 597)
        _nc_1451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 13), '_nc', False)
        # Obtaining the member 'errstate' of a type (line 597)
        errstate_1452 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 597, 13), _nc_1451, 'errstate')
        # Calling errstate(args, kwargs) (line 597)
        errstate_call_result_1456 = invoke(stypy.reporting.localization.Localization(__file__, 597, 13), errstate_1452, *[], **kwargs_1455)
        
        with_1457 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 597, 13), errstate_call_result_1456, 'with parameter', '__enter__', '__exit__')

        if with_1457:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 597)
            enter___1458 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 597, 13), errstate_call_result_1456, '__enter__')
            with_enter_1459 = invoke(stypy.reporting.localization.Localization(__file__, 597, 13), enter___1458)
            
            
            # Call to isnan(...): (line 598)
            # Processing the call arguments (line 598)
            # Getting the type of 'x' (line 598)
            x_1461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 21), 'x', False)
            # Processing the call keyword arguments (line 598)
            kwargs_1462 = {}
            # Getting the type of 'isnan' (line 598)
            isnan_1460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 15), 'isnan', False)
            # Calling isnan(args, kwargs) (line 598)
            isnan_call_result_1463 = invoke(stypy.reporting.localization.Localization(__file__, 598, 15), isnan_1460, *[x_1461], **kwargs_1462)
            
            # Testing the type of an if condition (line 598)
            if_condition_1464 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 598, 12), isnan_call_result_1463)
            # Assigning a type to the variable 'if_condition_1464' (line 598)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 598, 12), 'if_condition_1464', if_condition_1464)
            # SSA begins for if statement (line 598)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Getting the type of 'self' (line 599)
            self_1465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 599, 19), 'self')
            # Obtaining the member 'sign' of a type (line 599)
            sign_1466 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 599, 19), self_1465, 'sign')
            # Testing the type of an if condition (line 599)
            if_condition_1467 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 599, 16), sign_1466)
            # Assigning a type to the variable 'if_condition_1467' (line 599)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 599, 16), 'if_condition_1467', if_condition_1467)
            # SSA begins for if statement (line 599)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'self' (line 600)
            self_1468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 27), 'self')
            # Obtaining the member 'special_fmt' of a type (line 600)
            special_fmt_1469 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 600, 27), self_1468, 'special_fmt')
            
            # Obtaining an instance of the builtin type 'tuple' (line 600)
            tuple_1470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 600, 47), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 600)
            # Adding element type (line 600)
            str_1471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 600, 47), 'str', '+')
            # Getting the type of '_nan_str' (line 600)
            _nan_str_1472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 53), '_nan_str')
            # Applying the binary operator '+' (line 600)
            result_add_1473 = python_operator(stypy.reporting.localization.Localization(__file__, 600, 47), '+', str_1471, _nan_str_1472)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 600, 47), tuple_1470, result_add_1473)
            
            # Applying the binary operator '%' (line 600)
            result_mod_1474 = python_operator(stypy.reporting.localization.Localization(__file__, 600, 27), '%', special_fmt_1469, tuple_1470)
            
            # Assigning a type to the variable 'stypy_return_type' (line 600)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 600, 20), 'stypy_return_type', result_mod_1474)
            # SSA branch for the else part of an if statement (line 599)
            module_type_store.open_ssa_branch('else')
            # Getting the type of 'self' (line 602)
            self_1475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 27), 'self')
            # Obtaining the member 'special_fmt' of a type (line 602)
            special_fmt_1476 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 602, 27), self_1475, 'special_fmt')
            
            # Obtaining an instance of the builtin type 'tuple' (line 602)
            tuple_1477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 602, 47), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 602)
            # Adding element type (line 602)
            # Getting the type of '_nan_str' (line 602)
            _nan_str_1478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 47), '_nan_str')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 602, 47), tuple_1477, _nan_str_1478)
            
            # Applying the binary operator '%' (line 602)
            result_mod_1479 = python_operator(stypy.reporting.localization.Localization(__file__, 602, 27), '%', special_fmt_1476, tuple_1477)
            
            # Assigning a type to the variable 'stypy_return_type' (line 602)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 602, 20), 'stypy_return_type', result_mod_1479)
            # SSA join for if statement (line 599)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA branch for the else part of an if statement (line 598)
            module_type_store.open_ssa_branch('else')
            
            
            # Call to isinf(...): (line 603)
            # Processing the call arguments (line 603)
            # Getting the type of 'x' (line 603)
            x_1481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 23), 'x', False)
            # Processing the call keyword arguments (line 603)
            kwargs_1482 = {}
            # Getting the type of 'isinf' (line 603)
            isinf_1480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 17), 'isinf', False)
            # Calling isinf(args, kwargs) (line 603)
            isinf_call_result_1483 = invoke(stypy.reporting.localization.Localization(__file__, 603, 17), isinf_1480, *[x_1481], **kwargs_1482)
            
            # Testing the type of an if condition (line 603)
            if_condition_1484 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 603, 17), isinf_call_result_1483)
            # Assigning a type to the variable 'if_condition_1484' (line 603)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 603, 17), 'if_condition_1484', if_condition_1484)
            # SSA begins for if statement (line 603)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            
            # Getting the type of 'x' (line 604)
            x_1485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 19), 'x')
            int_1486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 604, 23), 'int')
            # Applying the binary operator '>' (line 604)
            result_gt_1487 = python_operator(stypy.reporting.localization.Localization(__file__, 604, 19), '>', x_1485, int_1486)
            
            # Testing the type of an if condition (line 604)
            if_condition_1488 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 604, 16), result_gt_1487)
            # Assigning a type to the variable 'if_condition_1488' (line 604)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 604, 16), 'if_condition_1488', if_condition_1488)
            # SSA begins for if statement (line 604)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Getting the type of 'self' (line 605)
            self_1489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 23), 'self')
            # Obtaining the member 'sign' of a type (line 605)
            sign_1490 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 605, 23), self_1489, 'sign')
            # Testing the type of an if condition (line 605)
            if_condition_1491 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 605, 20), sign_1490)
            # Assigning a type to the variable 'if_condition_1491' (line 605)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 605, 20), 'if_condition_1491', if_condition_1491)
            # SSA begins for if statement (line 605)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'self' (line 606)
            self_1492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 606, 31), 'self')
            # Obtaining the member 'special_fmt' of a type (line 606)
            special_fmt_1493 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 606, 31), self_1492, 'special_fmt')
            
            # Obtaining an instance of the builtin type 'tuple' (line 606)
            tuple_1494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 606, 51), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 606)
            # Adding element type (line 606)
            str_1495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 606, 51), 'str', '+')
            # Getting the type of '_inf_str' (line 606)
            _inf_str_1496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 606, 57), '_inf_str')
            # Applying the binary operator '+' (line 606)
            result_add_1497 = python_operator(stypy.reporting.localization.Localization(__file__, 606, 51), '+', str_1495, _inf_str_1496)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 606, 51), tuple_1494, result_add_1497)
            
            # Applying the binary operator '%' (line 606)
            result_mod_1498 = python_operator(stypy.reporting.localization.Localization(__file__, 606, 31), '%', special_fmt_1493, tuple_1494)
            
            # Assigning a type to the variable 'stypy_return_type' (line 606)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 606, 24), 'stypy_return_type', result_mod_1498)
            # SSA branch for the else part of an if statement (line 605)
            module_type_store.open_ssa_branch('else')
            # Getting the type of 'self' (line 608)
            self_1499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 608, 31), 'self')
            # Obtaining the member 'special_fmt' of a type (line 608)
            special_fmt_1500 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 608, 31), self_1499, 'special_fmt')
            
            # Obtaining an instance of the builtin type 'tuple' (line 608)
            tuple_1501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 608, 51), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 608)
            # Adding element type (line 608)
            # Getting the type of '_inf_str' (line 608)
            _inf_str_1502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 608, 51), '_inf_str')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 608, 51), tuple_1501, _inf_str_1502)
            
            # Applying the binary operator '%' (line 608)
            result_mod_1503 = python_operator(stypy.reporting.localization.Localization(__file__, 608, 31), '%', special_fmt_1500, tuple_1501)
            
            # Assigning a type to the variable 'stypy_return_type' (line 608)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 608, 24), 'stypy_return_type', result_mod_1503)
            # SSA join for if statement (line 605)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA branch for the else part of an if statement (line 604)
            module_type_store.open_ssa_branch('else')
            # Getting the type of 'self' (line 610)
            self_1504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 27), 'self')
            # Obtaining the member 'special_fmt' of a type (line 610)
            special_fmt_1505 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 610, 27), self_1504, 'special_fmt')
            
            # Obtaining an instance of the builtin type 'tuple' (line 610)
            tuple_1506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 610, 47), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 610)
            # Adding element type (line 610)
            str_1507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 610, 47), 'str', '-')
            # Getting the type of '_inf_str' (line 610)
            _inf_str_1508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 53), '_inf_str')
            # Applying the binary operator '+' (line 610)
            result_add_1509 = python_operator(stypy.reporting.localization.Localization(__file__, 610, 47), '+', str_1507, _inf_str_1508)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 610, 47), tuple_1506, result_add_1509)
            
            # Applying the binary operator '%' (line 610)
            result_mod_1510 = python_operator(stypy.reporting.localization.Localization(__file__, 610, 27), '%', special_fmt_1505, tuple_1506)
            
            # Assigning a type to the variable 'stypy_return_type' (line 610)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 610, 20), 'stypy_return_type', result_mod_1510)
            # SSA join for if statement (line 604)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 603)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 598)
            module_type_store = module_type_store.join_ssa_context()
            
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 597)
            exit___1511 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 597, 13), errstate_call_result_1456, '__exit__')
            with_exit_1512 = invoke(stypy.reporting.localization.Localization(__file__, 597, 13), exit___1511, None, None, None)

        
        # Assigning a BinOp to a Name (line 612):
        
        # Assigning a BinOp to a Name (line 612):
        # Getting the type of 'self' (line 612)
        self_1513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 612, 12), 'self')
        # Obtaining the member 'format' of a type (line 612)
        format_1514 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 612, 12), self_1513, 'format')
        # Getting the type of 'x' (line 612)
        x_1515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 612, 26), 'x')
        # Applying the binary operator '%' (line 612)
        result_mod_1516 = python_operator(stypy.reporting.localization.Localization(__file__, 612, 12), '%', format_1514, x_1515)
        
        # Assigning a type to the variable 's' (line 612)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 612, 8), 's', result_mod_1516)
        
        # Getting the type of 'self' (line 613)
        self_1517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 11), 'self')
        # Obtaining the member 'large_exponent' of a type (line 613)
        large_exponent_1518 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 613, 11), self_1517, 'large_exponent')
        # Testing the type of an if condition (line 613)
        if_condition_1519 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 613, 8), large_exponent_1518)
        # Assigning a type to the variable 'if_condition_1519' (line 613)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 613, 8), 'if_condition_1519', if_condition_1519)
        # SSA begins for if statement (line 613)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Name (line 615):
        
        # Assigning a Subscript to a Name (line 615):
        
        # Obtaining the type of the subscript
        int_1520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 615, 24), 'int')
        # Getting the type of 's' (line 615)
        s_1521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 22), 's')
        # Obtaining the member '__getitem__' of a type (line 615)
        getitem___1522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 615, 22), s_1521, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 615)
        subscript_call_result_1523 = invoke(stypy.reporting.localization.Localization(__file__, 615, 22), getitem___1522, int_1520)
        
        # Assigning a type to the variable 'expsign' (line 615)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 615, 12), 'expsign', subscript_call_result_1523)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'expsign' (line 616)
        expsign_1524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 15), 'expsign')
        str_1525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 616, 26), 'str', '+')
        # Applying the binary operator '==' (line 616)
        result_eq_1526 = python_operator(stypy.reporting.localization.Localization(__file__, 616, 15), '==', expsign_1524, str_1525)
        
        
        # Getting the type of 'expsign' (line 616)
        expsign_1527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 33), 'expsign')
        str_1528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 616, 44), 'str', '-')
        # Applying the binary operator '==' (line 616)
        result_eq_1529 = python_operator(stypy.reporting.localization.Localization(__file__, 616, 33), '==', expsign_1527, str_1528)
        
        # Applying the binary operator 'or' (line 616)
        result_or_keyword_1530 = python_operator(stypy.reporting.localization.Localization(__file__, 616, 15), 'or', result_eq_1526, result_eq_1529)
        
        # Testing the type of an if condition (line 616)
        if_condition_1531 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 616, 12), result_or_keyword_1530)
        # Assigning a type to the variable 'if_condition_1531' (line 616)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 616, 12), 'if_condition_1531', if_condition_1531)
        # SSA begins for if statement (line 616)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 617):
        
        # Assigning a BinOp to a Name (line 617):
        
        # Obtaining the type of the subscript
        int_1532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 617, 22), 'int')
        int_1533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 617, 24), 'int')
        slice_1534 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 617, 20), int_1532, int_1533, None)
        # Getting the type of 's' (line 617)
        s_1535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 20), 's')
        # Obtaining the member '__getitem__' of a type (line 617)
        getitem___1536 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 617, 20), s_1535, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 617)
        subscript_call_result_1537 = invoke(stypy.reporting.localization.Localization(__file__, 617, 20), getitem___1536, slice_1534)
        
        str_1538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 617, 30), 'str', '0')
        # Applying the binary operator '+' (line 617)
        result_add_1539 = python_operator(stypy.reporting.localization.Localization(__file__, 617, 20), '+', subscript_call_result_1537, str_1538)
        
        
        # Obtaining the type of the subscript
        int_1540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 617, 38), 'int')
        slice_1541 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 617, 36), int_1540, None, None)
        # Getting the type of 's' (line 617)
        s_1542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 36), 's')
        # Obtaining the member '__getitem__' of a type (line 617)
        getitem___1543 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 617, 36), s_1542, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 617)
        subscript_call_result_1544 = invoke(stypy.reporting.localization.Localization(__file__, 617, 36), getitem___1543, slice_1541)
        
        # Applying the binary operator '+' (line 617)
        result_add_1545 = python_operator(stypy.reporting.localization.Localization(__file__, 617, 34), '+', result_add_1539, subscript_call_result_1544)
        
        # Assigning a type to the variable 's' (line 617)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 617, 16), 's', result_add_1545)
        # SSA join for if statement (line 616)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 613)
        module_type_store.open_ssa_branch('else')
        
        # Getting the type of 'self' (line 618)
        self_1546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 13), 'self')
        # Obtaining the member 'exp_format' of a type (line 618)
        exp_format_1547 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 618, 13), self_1546, 'exp_format')
        # Testing the type of an if condition (line 618)
        if_condition_1548 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 618, 13), exp_format_1547)
        # Assigning a type to the variable 'if_condition_1548' (line 618)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 618, 13), 'if_condition_1548', if_condition_1548)
        # SSA begins for if statement (line 618)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        
        # Obtaining the type of the subscript
        int_1549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 620, 17), 'int')
        # Getting the type of 's' (line 620)
        s_1550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 15), 's')
        # Obtaining the member '__getitem__' of a type (line 620)
        getitem___1551 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 620, 15), s_1550, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 620)
        subscript_call_result_1552 = invoke(stypy.reporting.localization.Localization(__file__, 620, 15), getitem___1551, int_1549)
        
        str_1553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 620, 24), 'str', '0')
        # Applying the binary operator '==' (line 620)
        result_eq_1554 = python_operator(stypy.reporting.localization.Localization(__file__, 620, 15), '==', subscript_call_result_1552, str_1553)
        
        # Testing the type of an if condition (line 620)
        if_condition_1555 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 620, 12), result_eq_1554)
        # Assigning a type to the variable 'if_condition_1555' (line 620)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 620, 12), 'if_condition_1555', if_condition_1555)
        # SSA begins for if statement (line 620)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 621):
        
        # Assigning a BinOp to a Name (line 621):
        str_1556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 621, 20), 'str', ' ')
        
        # Obtaining the type of the subscript
        int_1557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 621, 29), 'int')
        slice_1558 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 621, 26), None, int_1557, None)
        # Getting the type of 's' (line 621)
        s_1559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 26), 's')
        # Obtaining the member '__getitem__' of a type (line 621)
        getitem___1560 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 621, 26), s_1559, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 621)
        subscript_call_result_1561 = invoke(stypy.reporting.localization.Localization(__file__, 621, 26), getitem___1560, slice_1558)
        
        # Applying the binary operator '+' (line 621)
        result_add_1562 = python_operator(stypy.reporting.localization.Localization(__file__, 621, 20), '+', str_1556, subscript_call_result_1561)
        
        
        # Obtaining the type of the subscript
        int_1563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 621, 37), 'int')
        slice_1564 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 621, 35), int_1563, None, None)
        # Getting the type of 's' (line 621)
        s_1565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 35), 's')
        # Obtaining the member '__getitem__' of a type (line 621)
        getitem___1566 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 621, 35), s_1565, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 621)
        subscript_call_result_1567 = invoke(stypy.reporting.localization.Localization(__file__, 621, 35), getitem___1566, slice_1564)
        
        # Applying the binary operator '+' (line 621)
        result_add_1568 = python_operator(stypy.reporting.localization.Localization(__file__, 621, 33), '+', result_add_1562, subscript_call_result_1567)
        
        # Assigning a type to the variable 's' (line 621)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 621, 16), 's', result_add_1568)
        # SSA join for if statement (line 620)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 618)
        module_type_store.open_ssa_branch('else')
        
        # Getting the type of 'strip_zeros' (line 622)
        strip_zeros_1569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 13), 'strip_zeros')
        # Testing the type of an if condition (line 622)
        if_condition_1570 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 622, 13), strip_zeros_1569)
        # Assigning a type to the variable 'if_condition_1570' (line 622)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 622, 13), 'if_condition_1570', if_condition_1570)
        # SSA begins for if statement (line 622)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 623):
        
        # Assigning a Call to a Name (line 623):
        
        # Call to rstrip(...): (line 623)
        # Processing the call arguments (line 623)
        str_1573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 623, 25), 'str', '0')
        # Processing the call keyword arguments (line 623)
        kwargs_1574 = {}
        # Getting the type of 's' (line 623)
        s_1571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 16), 's', False)
        # Obtaining the member 'rstrip' of a type (line 623)
        rstrip_1572 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 623, 16), s_1571, 'rstrip')
        # Calling rstrip(args, kwargs) (line 623)
        rstrip_call_result_1575 = invoke(stypy.reporting.localization.Localization(__file__, 623, 16), rstrip_1572, *[str_1573], **kwargs_1574)
        
        # Assigning a type to the variable 'z' (line 623)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 623, 12), 'z', rstrip_call_result_1575)
        
        # Assigning a BinOp to a Name (line 624):
        
        # Assigning a BinOp to a Name (line 624):
        # Getting the type of 'z' (line 624)
        z_1576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 624, 16), 'z')
        str_1577 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 624, 20), 'str', ' ')
        
        # Call to len(...): (line 624)
        # Processing the call arguments (line 624)
        # Getting the type of 's' (line 624)
        s_1579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 624, 29), 's', False)
        # Processing the call keyword arguments (line 624)
        kwargs_1580 = {}
        # Getting the type of 'len' (line 624)
        len_1578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 624, 25), 'len', False)
        # Calling len(args, kwargs) (line 624)
        len_call_result_1581 = invoke(stypy.reporting.localization.Localization(__file__, 624, 25), len_1578, *[s_1579], **kwargs_1580)
        
        
        # Call to len(...): (line 624)
        # Processing the call arguments (line 624)
        # Getting the type of 'z' (line 624)
        z_1583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 624, 36), 'z', False)
        # Processing the call keyword arguments (line 624)
        kwargs_1584 = {}
        # Getting the type of 'len' (line 624)
        len_1582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 624, 32), 'len', False)
        # Calling len(args, kwargs) (line 624)
        len_call_result_1585 = invoke(stypy.reporting.localization.Localization(__file__, 624, 32), len_1582, *[z_1583], **kwargs_1584)
        
        # Applying the binary operator '-' (line 624)
        result_sub_1586 = python_operator(stypy.reporting.localization.Localization(__file__, 624, 25), '-', len_call_result_1581, len_call_result_1585)
        
        # Applying the binary operator '*' (line 624)
        result_mul_1587 = python_operator(stypy.reporting.localization.Localization(__file__, 624, 20), '*', str_1577, result_sub_1586)
        
        # Applying the binary operator '+' (line 624)
        result_add_1588 = python_operator(stypy.reporting.localization.Localization(__file__, 624, 16), '+', z_1576, result_mul_1587)
        
        # Assigning a type to the variable 's' (line 624)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 624, 12), 's', result_add_1588)
        # SSA join for if statement (line 622)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 618)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 613)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 's' (line 625)
        s_1589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 15), 's')
        # Assigning a type to the variable 'stypy_return_type' (line 625)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 625, 8), 'stypy_return_type', s_1589)
        
        # ################# End of '__call__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__call__' in the type store
        # Getting the type of 'stypy_return_type' (line 594)
        stypy_return_type_1590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1590)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__call__'
        return stypy_return_type_1590


# Assigning a type to the variable 'FloatFormat' (line 528)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 528, 0), 'FloatFormat', FloatFormat)

@norecursion
def _digits(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_digits'
    module_type_store = module_type_store.open_function_context('_digits', 628, 0, False)
    
    # Passed parameters checking function
    _digits.stypy_localization = localization
    _digits.stypy_type_of_self = None
    _digits.stypy_type_store = module_type_store
    _digits.stypy_function_name = '_digits'
    _digits.stypy_param_names_list = ['x', 'precision', 'format']
    _digits.stypy_varargs_param_name = None
    _digits.stypy_kwargs_param_name = None
    _digits.stypy_call_defaults = defaults
    _digits.stypy_call_varargs = varargs
    _digits.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_digits', ['x', 'precision', 'format'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_digits', localization, ['x', 'precision', 'format'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_digits(...)' code ##################

    
    # Assigning a BinOp to a Name (line 629):
    
    # Assigning a BinOp to a Name (line 629):
    # Getting the type of 'format' (line 629)
    format_1591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 629, 8), 'format')
    # Getting the type of 'x' (line 629)
    x_1592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 629, 17), 'x')
    # Applying the binary operator '%' (line 629)
    result_mod_1593 = python_operator(stypy.reporting.localization.Localization(__file__, 629, 8), '%', format_1591, x_1592)
    
    # Assigning a type to the variable 's' (line 629)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 629, 4), 's', result_mod_1593)
    
    # Assigning a Call to a Name (line 630):
    
    # Assigning a Call to a Name (line 630):
    
    # Call to rstrip(...): (line 630)
    # Processing the call arguments (line 630)
    str_1596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 630, 17), 'str', '0')
    # Processing the call keyword arguments (line 630)
    kwargs_1597 = {}
    # Getting the type of 's' (line 630)
    s_1594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 630, 8), 's', False)
    # Obtaining the member 'rstrip' of a type (line 630)
    rstrip_1595 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 630, 8), s_1594, 'rstrip')
    # Calling rstrip(args, kwargs) (line 630)
    rstrip_call_result_1598 = invoke(stypy.reporting.localization.Localization(__file__, 630, 8), rstrip_1595, *[str_1596], **kwargs_1597)
    
    # Assigning a type to the variable 'z' (line 630)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 630, 4), 'z', rstrip_call_result_1598)
    # Getting the type of 'precision' (line 631)
    precision_1599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 631, 11), 'precision')
    
    # Call to len(...): (line 631)
    # Processing the call arguments (line 631)
    # Getting the type of 's' (line 631)
    s_1601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 631, 27), 's', False)
    # Processing the call keyword arguments (line 631)
    kwargs_1602 = {}
    # Getting the type of 'len' (line 631)
    len_1600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 631, 23), 'len', False)
    # Calling len(args, kwargs) (line 631)
    len_call_result_1603 = invoke(stypy.reporting.localization.Localization(__file__, 631, 23), len_1600, *[s_1601], **kwargs_1602)
    
    # Applying the binary operator '-' (line 631)
    result_sub_1604 = python_operator(stypy.reporting.localization.Localization(__file__, 631, 11), '-', precision_1599, len_call_result_1603)
    
    
    # Call to len(...): (line 631)
    # Processing the call arguments (line 631)
    # Getting the type of 'z' (line 631)
    z_1606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 631, 36), 'z', False)
    # Processing the call keyword arguments (line 631)
    kwargs_1607 = {}
    # Getting the type of 'len' (line 631)
    len_1605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 631, 32), 'len', False)
    # Calling len(args, kwargs) (line 631)
    len_call_result_1608 = invoke(stypy.reporting.localization.Localization(__file__, 631, 32), len_1605, *[z_1606], **kwargs_1607)
    
    # Applying the binary operator '+' (line 631)
    result_add_1609 = python_operator(stypy.reporting.localization.Localization(__file__, 631, 30), '+', result_sub_1604, len_call_result_1608)
    
    # Assigning a type to the variable 'stypy_return_type' (line 631)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 631, 4), 'stypy_return_type', result_add_1609)
    
    # ################# End of '_digits(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_digits' in the type store
    # Getting the type of 'stypy_return_type' (line 628)
    stypy_return_type_1610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 628, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_1610)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_digits'
    return stypy_return_type_1610

# Assigning a type to the variable '_digits' (line 628)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 628, 0), '_digits', _digits)
# Declaration of the 'IntegerFormat' class

class IntegerFormat(object, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 635, 4, False)
        # Assigning a type to the variable 'self' (line 636)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 636, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'IntegerFormat.__init__', ['data'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['data'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        
        # SSA begins for try-except statement (line 636)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Name (line 637):
        
        # Assigning a Call to a Name (line 637):
        
        # Call to max(...): (line 637)
        # Processing the call arguments (line 637)
        
        # Call to len(...): (line 637)
        # Processing the call arguments (line 637)
        
        # Call to str(...): (line 637)
        # Processing the call arguments (line 637)
        
        # Call to reduce(...): (line 637)
        # Processing the call arguments (line 637)
        # Getting the type of 'data' (line 637)
        data_1616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 53), 'data', False)
        # Processing the call keyword arguments (line 637)
        kwargs_1617 = {}
        # Getting the type of 'maximum' (line 637)
        maximum_1614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 38), 'maximum', False)
        # Obtaining the member 'reduce' of a type (line 637)
        reduce_1615 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 637, 38), maximum_1614, 'reduce')
        # Calling reduce(args, kwargs) (line 637)
        reduce_call_result_1618 = invoke(stypy.reporting.localization.Localization(__file__, 637, 38), reduce_1615, *[data_1616], **kwargs_1617)
        
        # Processing the call keyword arguments (line 637)
        kwargs_1619 = {}
        # Getting the type of 'str' (line 637)
        str_1613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 34), 'str', False)
        # Calling str(args, kwargs) (line 637)
        str_call_result_1620 = invoke(stypy.reporting.localization.Localization(__file__, 637, 34), str_1613, *[reduce_call_result_1618], **kwargs_1619)
        
        # Processing the call keyword arguments (line 637)
        kwargs_1621 = {}
        # Getting the type of 'len' (line 637)
        len_1612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 30), 'len', False)
        # Calling len(args, kwargs) (line 637)
        len_call_result_1622 = invoke(stypy.reporting.localization.Localization(__file__, 637, 30), len_1612, *[str_call_result_1620], **kwargs_1621)
        
        
        # Call to len(...): (line 638)
        # Processing the call arguments (line 638)
        
        # Call to str(...): (line 638)
        # Processing the call arguments (line 638)
        
        # Call to reduce(...): (line 638)
        # Processing the call arguments (line 638)
        # Getting the type of 'data' (line 638)
        data_1627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 638, 53), 'data', False)
        # Processing the call keyword arguments (line 638)
        kwargs_1628 = {}
        # Getting the type of 'minimum' (line 638)
        minimum_1625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 638, 38), 'minimum', False)
        # Obtaining the member 'reduce' of a type (line 638)
        reduce_1626 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 638, 38), minimum_1625, 'reduce')
        # Calling reduce(args, kwargs) (line 638)
        reduce_call_result_1629 = invoke(stypy.reporting.localization.Localization(__file__, 638, 38), reduce_1626, *[data_1627], **kwargs_1628)
        
        # Processing the call keyword arguments (line 638)
        kwargs_1630 = {}
        # Getting the type of 'str' (line 638)
        str_1624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 638, 34), 'str', False)
        # Calling str(args, kwargs) (line 638)
        str_call_result_1631 = invoke(stypy.reporting.localization.Localization(__file__, 638, 34), str_1624, *[reduce_call_result_1629], **kwargs_1630)
        
        # Processing the call keyword arguments (line 638)
        kwargs_1632 = {}
        # Getting the type of 'len' (line 638)
        len_1623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 638, 30), 'len', False)
        # Calling len(args, kwargs) (line 638)
        len_call_result_1633 = invoke(stypy.reporting.localization.Localization(__file__, 638, 30), len_1623, *[str_call_result_1631], **kwargs_1632)
        
        # Processing the call keyword arguments (line 637)
        kwargs_1634 = {}
        # Getting the type of 'max' (line 637)
        max_1611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 26), 'max', False)
        # Calling max(args, kwargs) (line 637)
        max_call_result_1635 = invoke(stypy.reporting.localization.Localization(__file__, 637, 26), max_1611, *[len_call_result_1622, len_call_result_1633], **kwargs_1634)
        
        # Assigning a type to the variable 'max_str_len' (line 637)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 637, 12), 'max_str_len', max_call_result_1635)
        
        # Assigning a BinOp to a Attribute (line 639):
        
        # Assigning a BinOp to a Attribute (line 639):
        str_1636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 639, 26), 'str', '%')
        
        # Call to str(...): (line 639)
        # Processing the call arguments (line 639)
        # Getting the type of 'max_str_len' (line 639)
        max_str_len_1638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 639, 36), 'max_str_len', False)
        # Processing the call keyword arguments (line 639)
        kwargs_1639 = {}
        # Getting the type of 'str' (line 639)
        str_1637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 639, 32), 'str', False)
        # Calling str(args, kwargs) (line 639)
        str_call_result_1640 = invoke(stypy.reporting.localization.Localization(__file__, 639, 32), str_1637, *[max_str_len_1638], **kwargs_1639)
        
        # Applying the binary operator '+' (line 639)
        result_add_1641 = python_operator(stypy.reporting.localization.Localization(__file__, 639, 26), '+', str_1636, str_call_result_1640)
        
        str_1642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 639, 51), 'str', 'd')
        # Applying the binary operator '+' (line 639)
        result_add_1643 = python_operator(stypy.reporting.localization.Localization(__file__, 639, 49), '+', result_add_1641, str_1642)
        
        # Getting the type of 'self' (line 639)
        self_1644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 639, 12), 'self')
        # Setting the type of the member 'format' of a type (line 639)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 639, 12), self_1644, 'format', result_add_1643)
        # SSA branch for the except part of a try statement (line 636)
        # SSA branch for the except 'Tuple' branch of a try statement (line 636)
        module_type_store.open_ssa_branch('except')
        pass
        # SSA branch for the except 'ValueError' branch of a try statement (line 636)
        module_type_store.open_ssa_branch('except')
        pass
        # SSA join for try-except statement (line 636)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def __call__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__call__'
        module_type_store = module_type_store.open_function_context('__call__', 648, 4, False)
        # Assigning a type to the variable 'self' (line 649)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 649, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        IntegerFormat.__call__.__dict__.__setitem__('stypy_localization', localization)
        IntegerFormat.__call__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        IntegerFormat.__call__.__dict__.__setitem__('stypy_type_store', module_type_store)
        IntegerFormat.__call__.__dict__.__setitem__('stypy_function_name', 'IntegerFormat.__call__')
        IntegerFormat.__call__.__dict__.__setitem__('stypy_param_names_list', ['x'])
        IntegerFormat.__call__.__dict__.__setitem__('stypy_varargs_param_name', None)
        IntegerFormat.__call__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        IntegerFormat.__call__.__dict__.__setitem__('stypy_call_defaults', defaults)
        IntegerFormat.__call__.__dict__.__setitem__('stypy_call_varargs', varargs)
        IntegerFormat.__call__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        IntegerFormat.__call__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'IntegerFormat.__call__', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__call__', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__call__(...)' code ##################

        
        
        # Getting the type of '_MININT' (line 649)
        _MININT_1645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 11), '_MININT')
        # Getting the type of 'x' (line 649)
        x_1646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 21), 'x')
        # Applying the binary operator '<' (line 649)
        result_lt_1647 = python_operator(stypy.reporting.localization.Localization(__file__, 649, 11), '<', _MININT_1645, x_1646)
        # Getting the type of '_MAXINT' (line 649)
        _MAXINT_1648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 25), '_MAXINT')
        # Applying the binary operator '<' (line 649)
        result_lt_1649 = python_operator(stypy.reporting.localization.Localization(__file__, 649, 11), '<', x_1646, _MAXINT_1648)
        # Applying the binary operator '&' (line 649)
        result_and__1650 = python_operator(stypy.reporting.localization.Localization(__file__, 649, 11), '&', result_lt_1647, result_lt_1649)
        
        # Testing the type of an if condition (line 649)
        if_condition_1651 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 649, 8), result_and__1650)
        # Assigning a type to the variable 'if_condition_1651' (line 649)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 649, 8), 'if_condition_1651', if_condition_1651)
        # SSA begins for if statement (line 649)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'self' (line 650)
        self_1652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 19), 'self')
        # Obtaining the member 'format' of a type (line 650)
        format_1653 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 650, 19), self_1652, 'format')
        # Getting the type of 'x' (line 650)
        x_1654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 33), 'x')
        # Applying the binary operator '%' (line 650)
        result_mod_1655 = python_operator(stypy.reporting.localization.Localization(__file__, 650, 19), '%', format_1653, x_1654)
        
        # Assigning a type to the variable 'stypy_return_type' (line 650)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 650, 12), 'stypy_return_type', result_mod_1655)
        # SSA branch for the else part of an if statement (line 649)
        module_type_store.open_ssa_branch('else')
        str_1656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 652, 19), 'str', '%s')
        # Getting the type of 'x' (line 652)
        x_1657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 652, 26), 'x')
        # Applying the binary operator '%' (line 652)
        result_mod_1658 = python_operator(stypy.reporting.localization.Localization(__file__, 652, 19), '%', str_1656, x_1657)
        
        # Assigning a type to the variable 'stypy_return_type' (line 652)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 652, 12), 'stypy_return_type', result_mod_1658)
        # SSA join for if statement (line 649)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__call__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__call__' in the type store
        # Getting the type of 'stypy_return_type' (line 648)
        stypy_return_type_1659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 648, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1659)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__call__'
        return stypy_return_type_1659


# Assigning a type to the variable 'IntegerFormat' (line 634)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 634, 0), 'IntegerFormat', IntegerFormat)
# Declaration of the 'LongFloatFormat' class

class LongFloatFormat(object, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'False' (line 657)
        False_1660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 657, 39), 'False')
        defaults = [False_1660]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 657, 4, False)
        # Assigning a type to the variable 'self' (line 658)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 658, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LongFloatFormat.__init__', ['precision', 'sign'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['precision', 'sign'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 658):
        
        # Assigning a Name to a Attribute (line 658):
        # Getting the type of 'precision' (line 658)
        precision_1661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 658, 25), 'precision')
        # Getting the type of 'self' (line 658)
        self_1662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 658, 8), 'self')
        # Setting the type of the member 'precision' of a type (line 658)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 658, 8), self_1662, 'precision', precision_1661)
        
        # Assigning a Name to a Attribute (line 659):
        
        # Assigning a Name to a Attribute (line 659):
        # Getting the type of 'sign' (line 659)
        sign_1663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 20), 'sign')
        # Getting the type of 'self' (line 659)
        self_1664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 8), 'self')
        # Setting the type of the member 'sign' of a type (line 659)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 659, 8), self_1664, 'sign', sign_1663)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def __call__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__call__'
        module_type_store = module_type_store.open_function_context('__call__', 661, 4, False)
        # Assigning a type to the variable 'self' (line 662)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 662, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        LongFloatFormat.__call__.__dict__.__setitem__('stypy_localization', localization)
        LongFloatFormat.__call__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        LongFloatFormat.__call__.__dict__.__setitem__('stypy_type_store', module_type_store)
        LongFloatFormat.__call__.__dict__.__setitem__('stypy_function_name', 'LongFloatFormat.__call__')
        LongFloatFormat.__call__.__dict__.__setitem__('stypy_param_names_list', ['x'])
        LongFloatFormat.__call__.__dict__.__setitem__('stypy_varargs_param_name', None)
        LongFloatFormat.__call__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        LongFloatFormat.__call__.__dict__.__setitem__('stypy_call_defaults', defaults)
        LongFloatFormat.__call__.__dict__.__setitem__('stypy_call_varargs', varargs)
        LongFloatFormat.__call__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        LongFloatFormat.__call__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LongFloatFormat.__call__', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__call__', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__call__(...)' code ##################

        
        
        # Call to isnan(...): (line 662)
        # Processing the call arguments (line 662)
        # Getting the type of 'x' (line 662)
        x_1666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 17), 'x', False)
        # Processing the call keyword arguments (line 662)
        kwargs_1667 = {}
        # Getting the type of 'isnan' (line 662)
        isnan_1665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 11), 'isnan', False)
        # Calling isnan(args, kwargs) (line 662)
        isnan_call_result_1668 = invoke(stypy.reporting.localization.Localization(__file__, 662, 11), isnan_1665, *[x_1666], **kwargs_1667)
        
        # Testing the type of an if condition (line 662)
        if_condition_1669 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 662, 8), isnan_call_result_1668)
        # Assigning a type to the variable 'if_condition_1669' (line 662)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 662, 8), 'if_condition_1669', if_condition_1669)
        # SSA begins for if statement (line 662)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'self' (line 663)
        self_1670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 15), 'self')
        # Obtaining the member 'sign' of a type (line 663)
        sign_1671 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 663, 15), self_1670, 'sign')
        # Testing the type of an if condition (line 663)
        if_condition_1672 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 663, 12), sign_1671)
        # Assigning a type to the variable 'if_condition_1672' (line 663)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 663, 12), 'if_condition_1672', if_condition_1672)
        # SSA begins for if statement (line 663)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        str_1673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 664, 23), 'str', '+')
        # Getting the type of '_nan_str' (line 664)
        _nan_str_1674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 664, 29), '_nan_str')
        # Applying the binary operator '+' (line 664)
        result_add_1675 = python_operator(stypy.reporting.localization.Localization(__file__, 664, 23), '+', str_1673, _nan_str_1674)
        
        # Assigning a type to the variable 'stypy_return_type' (line 664)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 664, 16), 'stypy_return_type', result_add_1675)
        # SSA branch for the else part of an if statement (line 663)
        module_type_store.open_ssa_branch('else')
        str_1676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 666, 23), 'str', ' ')
        # Getting the type of '_nan_str' (line 666)
        _nan_str_1677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 29), '_nan_str')
        # Applying the binary operator '+' (line 666)
        result_add_1678 = python_operator(stypy.reporting.localization.Localization(__file__, 666, 23), '+', str_1676, _nan_str_1677)
        
        # Assigning a type to the variable 'stypy_return_type' (line 666)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 666, 16), 'stypy_return_type', result_add_1678)
        # SSA join for if statement (line 663)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 662)
        module_type_store.open_ssa_branch('else')
        
        
        # Call to isinf(...): (line 667)
        # Processing the call arguments (line 667)
        # Getting the type of 'x' (line 667)
        x_1680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 19), 'x', False)
        # Processing the call keyword arguments (line 667)
        kwargs_1681 = {}
        # Getting the type of 'isinf' (line 667)
        isinf_1679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 13), 'isinf', False)
        # Calling isinf(args, kwargs) (line 667)
        isinf_call_result_1682 = invoke(stypy.reporting.localization.Localization(__file__, 667, 13), isinf_1679, *[x_1680], **kwargs_1681)
        
        # Testing the type of an if condition (line 667)
        if_condition_1683 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 667, 13), isinf_call_result_1682)
        # Assigning a type to the variable 'if_condition_1683' (line 667)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 667, 13), 'if_condition_1683', if_condition_1683)
        # SSA begins for if statement (line 667)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Getting the type of 'x' (line 668)
        x_1684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 15), 'x')
        int_1685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 668, 19), 'int')
        # Applying the binary operator '>' (line 668)
        result_gt_1686 = python_operator(stypy.reporting.localization.Localization(__file__, 668, 15), '>', x_1684, int_1685)
        
        # Testing the type of an if condition (line 668)
        if_condition_1687 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 668, 12), result_gt_1686)
        # Assigning a type to the variable 'if_condition_1687' (line 668)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 668, 12), 'if_condition_1687', if_condition_1687)
        # SSA begins for if statement (line 668)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'self' (line 669)
        self_1688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 669, 19), 'self')
        # Obtaining the member 'sign' of a type (line 669)
        sign_1689 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 669, 19), self_1688, 'sign')
        # Testing the type of an if condition (line 669)
        if_condition_1690 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 669, 16), sign_1689)
        # Assigning a type to the variable 'if_condition_1690' (line 669)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 669, 16), 'if_condition_1690', if_condition_1690)
        # SSA begins for if statement (line 669)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        str_1691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 670, 27), 'str', '+')
        # Getting the type of '_inf_str' (line 670)
        _inf_str_1692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 670, 33), '_inf_str')
        # Applying the binary operator '+' (line 670)
        result_add_1693 = python_operator(stypy.reporting.localization.Localization(__file__, 670, 27), '+', str_1691, _inf_str_1692)
        
        # Assigning a type to the variable 'stypy_return_type' (line 670)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 670, 20), 'stypy_return_type', result_add_1693)
        # SSA branch for the else part of an if statement (line 669)
        module_type_store.open_ssa_branch('else')
        str_1694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 672, 27), 'str', ' ')
        # Getting the type of '_inf_str' (line 672)
        _inf_str_1695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 672, 33), '_inf_str')
        # Applying the binary operator '+' (line 672)
        result_add_1696 = python_operator(stypy.reporting.localization.Localization(__file__, 672, 27), '+', str_1694, _inf_str_1695)
        
        # Assigning a type to the variable 'stypy_return_type' (line 672)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 672, 20), 'stypy_return_type', result_add_1696)
        # SSA join for if statement (line 669)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 668)
        module_type_store.open_ssa_branch('else')
        str_1697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 674, 23), 'str', '-')
        # Getting the type of '_inf_str' (line 674)
        _inf_str_1698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 674, 29), '_inf_str')
        # Applying the binary operator '+' (line 674)
        result_add_1699 = python_operator(stypy.reporting.localization.Localization(__file__, 674, 23), '+', str_1697, _inf_str_1698)
        
        # Assigning a type to the variable 'stypy_return_type' (line 674)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 674, 16), 'stypy_return_type', result_add_1699)
        # SSA join for if statement (line 668)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 667)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'x' (line 675)
        x_1700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 13), 'x')
        int_1701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 675, 18), 'int')
        # Applying the binary operator '>=' (line 675)
        result_ge_1702 = python_operator(stypy.reporting.localization.Localization(__file__, 675, 13), '>=', x_1700, int_1701)
        
        # Testing the type of an if condition (line 675)
        if_condition_1703 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 675, 13), result_ge_1702)
        # Assigning a type to the variable 'if_condition_1703' (line 675)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 675, 13), 'if_condition_1703', if_condition_1703)
        # SSA begins for if statement (line 675)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'self' (line 676)
        self_1704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 676, 15), 'self')
        # Obtaining the member 'sign' of a type (line 676)
        sign_1705 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 676, 15), self_1704, 'sign')
        # Testing the type of an if condition (line 676)
        if_condition_1706 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 676, 12), sign_1705)
        # Assigning a type to the variable 'if_condition_1706' (line 676)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 676, 12), 'if_condition_1706', if_condition_1706)
        # SSA begins for if statement (line 676)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        str_1707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 677, 23), 'str', '+')
        
        # Call to format_longfloat(...): (line 677)
        # Processing the call arguments (line 677)
        # Getting the type of 'x' (line 677)
        x_1709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 677, 46), 'x', False)
        # Getting the type of 'self' (line 677)
        self_1710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 677, 49), 'self', False)
        # Obtaining the member 'precision' of a type (line 677)
        precision_1711 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 677, 49), self_1710, 'precision')
        # Processing the call keyword arguments (line 677)
        kwargs_1712 = {}
        # Getting the type of 'format_longfloat' (line 677)
        format_longfloat_1708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 677, 29), 'format_longfloat', False)
        # Calling format_longfloat(args, kwargs) (line 677)
        format_longfloat_call_result_1713 = invoke(stypy.reporting.localization.Localization(__file__, 677, 29), format_longfloat_1708, *[x_1709, precision_1711], **kwargs_1712)
        
        # Applying the binary operator '+' (line 677)
        result_add_1714 = python_operator(stypy.reporting.localization.Localization(__file__, 677, 23), '+', str_1707, format_longfloat_call_result_1713)
        
        # Assigning a type to the variable 'stypy_return_type' (line 677)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 677, 16), 'stypy_return_type', result_add_1714)
        # SSA branch for the else part of an if statement (line 676)
        module_type_store.open_ssa_branch('else')
        str_1715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 679, 23), 'str', ' ')
        
        # Call to format_longfloat(...): (line 679)
        # Processing the call arguments (line 679)
        # Getting the type of 'x' (line 679)
        x_1717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 679, 46), 'x', False)
        # Getting the type of 'self' (line 679)
        self_1718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 679, 49), 'self', False)
        # Obtaining the member 'precision' of a type (line 679)
        precision_1719 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 679, 49), self_1718, 'precision')
        # Processing the call keyword arguments (line 679)
        kwargs_1720 = {}
        # Getting the type of 'format_longfloat' (line 679)
        format_longfloat_1716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 679, 29), 'format_longfloat', False)
        # Calling format_longfloat(args, kwargs) (line 679)
        format_longfloat_call_result_1721 = invoke(stypy.reporting.localization.Localization(__file__, 679, 29), format_longfloat_1716, *[x_1717, precision_1719], **kwargs_1720)
        
        # Applying the binary operator '+' (line 679)
        result_add_1722 = python_operator(stypy.reporting.localization.Localization(__file__, 679, 23), '+', str_1715, format_longfloat_call_result_1721)
        
        # Assigning a type to the variable 'stypy_return_type' (line 679)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 679, 16), 'stypy_return_type', result_add_1722)
        # SSA join for if statement (line 676)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 675)
        module_type_store.open_ssa_branch('else')
        
        # Call to format_longfloat(...): (line 681)
        # Processing the call arguments (line 681)
        # Getting the type of 'x' (line 681)
        x_1724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 36), 'x', False)
        # Getting the type of 'self' (line 681)
        self_1725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 39), 'self', False)
        # Obtaining the member 'precision' of a type (line 681)
        precision_1726 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 681, 39), self_1725, 'precision')
        # Processing the call keyword arguments (line 681)
        kwargs_1727 = {}
        # Getting the type of 'format_longfloat' (line 681)
        format_longfloat_1723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 19), 'format_longfloat', False)
        # Calling format_longfloat(args, kwargs) (line 681)
        format_longfloat_call_result_1728 = invoke(stypy.reporting.localization.Localization(__file__, 681, 19), format_longfloat_1723, *[x_1724, precision_1726], **kwargs_1727)
        
        # Assigning a type to the variable 'stypy_return_type' (line 681)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 681, 12), 'stypy_return_type', format_longfloat_call_result_1728)
        # SSA join for if statement (line 675)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 667)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 662)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__call__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__call__' in the type store
        # Getting the type of 'stypy_return_type' (line 661)
        stypy_return_type_1729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1729)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__call__'
        return stypy_return_type_1729


# Assigning a type to the variable 'LongFloatFormat' (line 654)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 654, 0), 'LongFloatFormat', LongFloatFormat)
# Declaration of the 'LongComplexFormat' class

class LongComplexFormat(object, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 685, 4, False)
        # Assigning a type to the variable 'self' (line 686)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 686, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LongComplexFormat.__init__', ['precision'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['precision'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Call to a Attribute (line 686):
        
        # Assigning a Call to a Attribute (line 686):
        
        # Call to LongFloatFormat(...): (line 686)
        # Processing the call arguments (line 686)
        # Getting the type of 'precision' (line 686)
        precision_1731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 686, 43), 'precision', False)
        # Processing the call keyword arguments (line 686)
        kwargs_1732 = {}
        # Getting the type of 'LongFloatFormat' (line 686)
        LongFloatFormat_1730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 686, 27), 'LongFloatFormat', False)
        # Calling LongFloatFormat(args, kwargs) (line 686)
        LongFloatFormat_call_result_1733 = invoke(stypy.reporting.localization.Localization(__file__, 686, 27), LongFloatFormat_1730, *[precision_1731], **kwargs_1732)
        
        # Getting the type of 'self' (line 686)
        self_1734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 686, 8), 'self')
        # Setting the type of the member 'real_format' of a type (line 686)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 686, 8), self_1734, 'real_format', LongFloatFormat_call_result_1733)
        
        # Assigning a Call to a Attribute (line 687):
        
        # Assigning a Call to a Attribute (line 687):
        
        # Call to LongFloatFormat(...): (line 687)
        # Processing the call arguments (line 687)
        # Getting the type of 'precision' (line 687)
        precision_1736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 687, 43), 'precision', False)
        # Processing the call keyword arguments (line 687)
        # Getting the type of 'True' (line 687)
        True_1737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 687, 59), 'True', False)
        keyword_1738 = True_1737
        kwargs_1739 = {'sign': keyword_1738}
        # Getting the type of 'LongFloatFormat' (line 687)
        LongFloatFormat_1735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 687, 27), 'LongFloatFormat', False)
        # Calling LongFloatFormat(args, kwargs) (line 687)
        LongFloatFormat_call_result_1740 = invoke(stypy.reporting.localization.Localization(__file__, 687, 27), LongFloatFormat_1735, *[precision_1736], **kwargs_1739)
        
        # Getting the type of 'self' (line 687)
        self_1741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 687, 8), 'self')
        # Setting the type of the member 'imag_format' of a type (line 687)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 687, 8), self_1741, 'imag_format', LongFloatFormat_call_result_1740)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def __call__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__call__'
        module_type_store = module_type_store.open_function_context('__call__', 689, 4, False)
        # Assigning a type to the variable 'self' (line 690)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 690, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        LongComplexFormat.__call__.__dict__.__setitem__('stypy_localization', localization)
        LongComplexFormat.__call__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        LongComplexFormat.__call__.__dict__.__setitem__('stypy_type_store', module_type_store)
        LongComplexFormat.__call__.__dict__.__setitem__('stypy_function_name', 'LongComplexFormat.__call__')
        LongComplexFormat.__call__.__dict__.__setitem__('stypy_param_names_list', ['x'])
        LongComplexFormat.__call__.__dict__.__setitem__('stypy_varargs_param_name', None)
        LongComplexFormat.__call__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        LongComplexFormat.__call__.__dict__.__setitem__('stypy_call_defaults', defaults)
        LongComplexFormat.__call__.__dict__.__setitem__('stypy_call_varargs', varargs)
        LongComplexFormat.__call__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        LongComplexFormat.__call__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LongComplexFormat.__call__', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__call__', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__call__(...)' code ##################

        
        # Assigning a Call to a Name (line 690):
        
        # Assigning a Call to a Name (line 690):
        
        # Call to real_format(...): (line 690)
        # Processing the call arguments (line 690)
        # Getting the type of 'x' (line 690)
        x_1744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 29), 'x', False)
        # Obtaining the member 'real' of a type (line 690)
        real_1745 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 690, 29), x_1744, 'real')
        # Processing the call keyword arguments (line 690)
        kwargs_1746 = {}
        # Getting the type of 'self' (line 690)
        self_1742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 12), 'self', False)
        # Obtaining the member 'real_format' of a type (line 690)
        real_format_1743 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 690, 12), self_1742, 'real_format')
        # Calling real_format(args, kwargs) (line 690)
        real_format_call_result_1747 = invoke(stypy.reporting.localization.Localization(__file__, 690, 12), real_format_1743, *[real_1745], **kwargs_1746)
        
        # Assigning a type to the variable 'r' (line 690)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 690, 8), 'r', real_format_call_result_1747)
        
        # Assigning a Call to a Name (line 691):
        
        # Assigning a Call to a Name (line 691):
        
        # Call to imag_format(...): (line 691)
        # Processing the call arguments (line 691)
        # Getting the type of 'x' (line 691)
        x_1750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 691, 29), 'x', False)
        # Obtaining the member 'imag' of a type (line 691)
        imag_1751 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 691, 29), x_1750, 'imag')
        # Processing the call keyword arguments (line 691)
        kwargs_1752 = {}
        # Getting the type of 'self' (line 691)
        self_1748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 691, 12), 'self', False)
        # Obtaining the member 'imag_format' of a type (line 691)
        imag_format_1749 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 691, 12), self_1748, 'imag_format')
        # Calling imag_format(args, kwargs) (line 691)
        imag_format_call_result_1753 = invoke(stypy.reporting.localization.Localization(__file__, 691, 12), imag_format_1749, *[imag_1751], **kwargs_1752)
        
        # Assigning a type to the variable 'i' (line 691)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 691, 8), 'i', imag_format_call_result_1753)
        # Getting the type of 'r' (line 692)
        r_1754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 15), 'r')
        # Getting the type of 'i' (line 692)
        i_1755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 19), 'i')
        # Applying the binary operator '+' (line 692)
        result_add_1756 = python_operator(stypy.reporting.localization.Localization(__file__, 692, 15), '+', r_1754, i_1755)
        
        str_1757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 692, 23), 'str', 'j')
        # Applying the binary operator '+' (line 692)
        result_add_1758 = python_operator(stypy.reporting.localization.Localization(__file__, 692, 21), '+', result_add_1756, str_1757)
        
        # Assigning a type to the variable 'stypy_return_type' (line 692)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 692, 8), 'stypy_return_type', result_add_1758)
        
        # ################# End of '__call__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__call__' in the type store
        # Getting the type of 'stypy_return_type' (line 689)
        stypy_return_type_1759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1759)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__call__'
        return stypy_return_type_1759


# Assigning a type to the variable 'LongComplexFormat' (line 684)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 684, 0), 'LongComplexFormat', LongComplexFormat)
# Declaration of the 'ComplexFormat' class

class ComplexFormat(object, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 696, 4, False)
        # Assigning a type to the variable 'self' (line 697)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 697, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ComplexFormat.__init__', ['x', 'precision', 'suppress_small'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['x', 'precision', 'suppress_small'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Call to a Attribute (line 697):
        
        # Assigning a Call to a Attribute (line 697):
        
        # Call to FloatFormat(...): (line 697)
        # Processing the call arguments (line 697)
        # Getting the type of 'x' (line 697)
        x_1761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 39), 'x', False)
        # Obtaining the member 'real' of a type (line 697)
        real_1762 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 697, 39), x_1761, 'real')
        # Getting the type of 'precision' (line 697)
        precision_1763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 47), 'precision', False)
        # Getting the type of 'suppress_small' (line 697)
        suppress_small_1764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 58), 'suppress_small', False)
        # Processing the call keyword arguments (line 697)
        kwargs_1765 = {}
        # Getting the type of 'FloatFormat' (line 697)
        FloatFormat_1760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 27), 'FloatFormat', False)
        # Calling FloatFormat(args, kwargs) (line 697)
        FloatFormat_call_result_1766 = invoke(stypy.reporting.localization.Localization(__file__, 697, 27), FloatFormat_1760, *[real_1762, precision_1763, suppress_small_1764], **kwargs_1765)
        
        # Getting the type of 'self' (line 697)
        self_1767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 8), 'self')
        # Setting the type of the member 'real_format' of a type (line 697)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 697, 8), self_1767, 'real_format', FloatFormat_call_result_1766)
        
        # Assigning a Call to a Attribute (line 698):
        
        # Assigning a Call to a Attribute (line 698):
        
        # Call to FloatFormat(...): (line 698)
        # Processing the call arguments (line 698)
        # Getting the type of 'x' (line 698)
        x_1769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 39), 'x', False)
        # Obtaining the member 'imag' of a type (line 698)
        imag_1770 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 698, 39), x_1769, 'imag')
        # Getting the type of 'precision' (line 698)
        precision_1771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 47), 'precision', False)
        # Getting the type of 'suppress_small' (line 698)
        suppress_small_1772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 58), 'suppress_small', False)
        # Processing the call keyword arguments (line 698)
        # Getting the type of 'True' (line 699)
        True_1773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 699, 44), 'True', False)
        keyword_1774 = True_1773
        kwargs_1775 = {'sign': keyword_1774}
        # Getting the type of 'FloatFormat' (line 698)
        FloatFormat_1768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 27), 'FloatFormat', False)
        # Calling FloatFormat(args, kwargs) (line 698)
        FloatFormat_call_result_1776 = invoke(stypy.reporting.localization.Localization(__file__, 698, 27), FloatFormat_1768, *[imag_1770, precision_1771, suppress_small_1772], **kwargs_1775)
        
        # Getting the type of 'self' (line 698)
        self_1777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 8), 'self')
        # Setting the type of the member 'imag_format' of a type (line 698)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 698, 8), self_1777, 'imag_format', FloatFormat_call_result_1776)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def __call__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__call__'
        module_type_store = module_type_store.open_function_context('__call__', 701, 4, False)
        # Assigning a type to the variable 'self' (line 702)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 702, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ComplexFormat.__call__.__dict__.__setitem__('stypy_localization', localization)
        ComplexFormat.__call__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ComplexFormat.__call__.__dict__.__setitem__('stypy_type_store', module_type_store)
        ComplexFormat.__call__.__dict__.__setitem__('stypy_function_name', 'ComplexFormat.__call__')
        ComplexFormat.__call__.__dict__.__setitem__('stypy_param_names_list', ['x'])
        ComplexFormat.__call__.__dict__.__setitem__('stypy_varargs_param_name', None)
        ComplexFormat.__call__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ComplexFormat.__call__.__dict__.__setitem__('stypy_call_defaults', defaults)
        ComplexFormat.__call__.__dict__.__setitem__('stypy_call_varargs', varargs)
        ComplexFormat.__call__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ComplexFormat.__call__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ComplexFormat.__call__', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__call__', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__call__(...)' code ##################

        
        # Assigning a Call to a Name (line 702):
        
        # Assigning a Call to a Name (line 702):
        
        # Call to real_format(...): (line 702)
        # Processing the call arguments (line 702)
        # Getting the type of 'x' (line 702)
        x_1780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 29), 'x', False)
        # Obtaining the member 'real' of a type (line 702)
        real_1781 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 702, 29), x_1780, 'real')
        # Processing the call keyword arguments (line 702)
        # Getting the type of 'False' (line 702)
        False_1782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 49), 'False', False)
        keyword_1783 = False_1782
        kwargs_1784 = {'strip_zeros': keyword_1783}
        # Getting the type of 'self' (line 702)
        self_1778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 12), 'self', False)
        # Obtaining the member 'real_format' of a type (line 702)
        real_format_1779 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 702, 12), self_1778, 'real_format')
        # Calling real_format(args, kwargs) (line 702)
        real_format_call_result_1785 = invoke(stypy.reporting.localization.Localization(__file__, 702, 12), real_format_1779, *[real_1781], **kwargs_1784)
        
        # Assigning a type to the variable 'r' (line 702)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 702, 8), 'r', real_format_call_result_1785)
        
        # Assigning a Call to a Name (line 703):
        
        # Assigning a Call to a Name (line 703):
        
        # Call to imag_format(...): (line 703)
        # Processing the call arguments (line 703)
        # Getting the type of 'x' (line 703)
        x_1788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 703, 29), 'x', False)
        # Obtaining the member 'imag' of a type (line 703)
        imag_1789 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 703, 29), x_1788, 'imag')
        # Processing the call keyword arguments (line 703)
        # Getting the type of 'False' (line 703)
        False_1790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 703, 49), 'False', False)
        keyword_1791 = False_1790
        kwargs_1792 = {'strip_zeros': keyword_1791}
        # Getting the type of 'self' (line 703)
        self_1786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 703, 12), 'self', False)
        # Obtaining the member 'imag_format' of a type (line 703)
        imag_format_1787 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 703, 12), self_1786, 'imag_format')
        # Calling imag_format(args, kwargs) (line 703)
        imag_format_call_result_1793 = invoke(stypy.reporting.localization.Localization(__file__, 703, 12), imag_format_1787, *[imag_1789], **kwargs_1792)
        
        # Assigning a type to the variable 'i' (line 703)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 703, 8), 'i', imag_format_call_result_1793)
        
        
        # Getting the type of 'self' (line 704)
        self_1794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 704, 15), 'self')
        # Obtaining the member 'imag_format' of a type (line 704)
        imag_format_1795 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 704, 15), self_1794, 'imag_format')
        # Obtaining the member 'exp_format' of a type (line 704)
        exp_format_1796 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 704, 15), imag_format_1795, 'exp_format')
        # Applying the 'not' unary operator (line 704)
        result_not__1797 = python_operator(stypy.reporting.localization.Localization(__file__, 704, 11), 'not', exp_format_1796)
        
        # Testing the type of an if condition (line 704)
        if_condition_1798 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 704, 8), result_not__1797)
        # Assigning a type to the variable 'if_condition_1798' (line 704)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 704, 8), 'if_condition_1798', if_condition_1798)
        # SSA begins for if statement (line 704)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 705):
        
        # Assigning a Call to a Name (line 705):
        
        # Call to rstrip(...): (line 705)
        # Processing the call arguments (line 705)
        str_1801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 705, 25), 'str', '0')
        # Processing the call keyword arguments (line 705)
        kwargs_1802 = {}
        # Getting the type of 'i' (line 705)
        i_1799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 16), 'i', False)
        # Obtaining the member 'rstrip' of a type (line 705)
        rstrip_1800 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 705, 16), i_1799, 'rstrip')
        # Calling rstrip(args, kwargs) (line 705)
        rstrip_call_result_1803 = invoke(stypy.reporting.localization.Localization(__file__, 705, 16), rstrip_1800, *[str_1801], **kwargs_1802)
        
        # Assigning a type to the variable 'z' (line 705)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 705, 12), 'z', rstrip_call_result_1803)
        
        # Assigning a BinOp to a Name (line 706):
        
        # Assigning a BinOp to a Name (line 706):
        # Getting the type of 'z' (line 706)
        z_1804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 706, 16), 'z')
        str_1805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 706, 20), 'str', 'j')
        # Applying the binary operator '+' (line 706)
        result_add_1806 = python_operator(stypy.reporting.localization.Localization(__file__, 706, 16), '+', z_1804, str_1805)
        
        str_1807 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 706, 26), 'str', ' ')
        
        # Call to len(...): (line 706)
        # Processing the call arguments (line 706)
        # Getting the type of 'i' (line 706)
        i_1809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 706, 35), 'i', False)
        # Processing the call keyword arguments (line 706)
        kwargs_1810 = {}
        # Getting the type of 'len' (line 706)
        len_1808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 706, 31), 'len', False)
        # Calling len(args, kwargs) (line 706)
        len_call_result_1811 = invoke(stypy.reporting.localization.Localization(__file__, 706, 31), len_1808, *[i_1809], **kwargs_1810)
        
        
        # Call to len(...): (line 706)
        # Processing the call arguments (line 706)
        # Getting the type of 'z' (line 706)
        z_1813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 706, 42), 'z', False)
        # Processing the call keyword arguments (line 706)
        kwargs_1814 = {}
        # Getting the type of 'len' (line 706)
        len_1812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 706, 38), 'len', False)
        # Calling len(args, kwargs) (line 706)
        len_call_result_1815 = invoke(stypy.reporting.localization.Localization(__file__, 706, 38), len_1812, *[z_1813], **kwargs_1814)
        
        # Applying the binary operator '-' (line 706)
        result_sub_1816 = python_operator(stypy.reporting.localization.Localization(__file__, 706, 31), '-', len_call_result_1811, len_call_result_1815)
        
        # Applying the binary operator '*' (line 706)
        result_mul_1817 = python_operator(stypy.reporting.localization.Localization(__file__, 706, 26), '*', str_1807, result_sub_1816)
        
        # Applying the binary operator '+' (line 706)
        result_add_1818 = python_operator(stypy.reporting.localization.Localization(__file__, 706, 24), '+', result_add_1806, result_mul_1817)
        
        # Assigning a type to the variable 'i' (line 706)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 706, 12), 'i', result_add_1818)
        # SSA branch for the else part of an if statement (line 704)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a BinOp to a Name (line 708):
        
        # Assigning a BinOp to a Name (line 708):
        # Getting the type of 'i' (line 708)
        i_1819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 708, 16), 'i')
        str_1820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 708, 20), 'str', 'j')
        # Applying the binary operator '+' (line 708)
        result_add_1821 = python_operator(stypy.reporting.localization.Localization(__file__, 708, 16), '+', i_1819, str_1820)
        
        # Assigning a type to the variable 'i' (line 708)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 708, 12), 'i', result_add_1821)
        # SSA join for if statement (line 704)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'r' (line 709)
        r_1822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 709, 15), 'r')
        # Getting the type of 'i' (line 709)
        i_1823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 709, 19), 'i')
        # Applying the binary operator '+' (line 709)
        result_add_1824 = python_operator(stypy.reporting.localization.Localization(__file__, 709, 15), '+', r_1822, i_1823)
        
        # Assigning a type to the variable 'stypy_return_type' (line 709)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 709, 8), 'stypy_return_type', result_add_1824)
        
        # ################# End of '__call__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__call__' in the type store
        # Getting the type of 'stypy_return_type' (line 701)
        stypy_return_type_1825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 701, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1825)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__call__'
        return stypy_return_type_1825


# Assigning a type to the variable 'ComplexFormat' (line 695)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 695, 0), 'ComplexFormat', ComplexFormat)
# Declaration of the 'DatetimeFormat' class

class DatetimeFormat(object, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 713)
        None_1826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 713, 31), 'None')
        # Getting the type of 'None' (line 713)
        None_1827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 713, 46), 'None')
        str_1828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 713, 60), 'str', 'same_kind')
        defaults = [None_1826, None_1827, str_1828]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 713, 4, False)
        # Assigning a type to the variable 'self' (line 714)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 714, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DatetimeFormat.__init__', ['x', 'unit', 'timezone', 'casting'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['x', 'unit', 'timezone', 'casting'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Type idiom detected: calculating its left and rigth part (line 715)
        # Getting the type of 'unit' (line 715)
        unit_1829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 715, 11), 'unit')
        # Getting the type of 'None' (line 715)
        None_1830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 715, 19), 'None')
        
        (may_be_1831, more_types_in_union_1832) = may_be_none(unit_1829, None_1830)

        if may_be_1831:

            if more_types_in_union_1832:
                # Runtime conditional SSA (line 715)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            
            # Getting the type of 'x' (line 716)
            x_1833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 716, 15), 'x')
            # Obtaining the member 'dtype' of a type (line 716)
            dtype_1834 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 716, 15), x_1833, 'dtype')
            # Obtaining the member 'kind' of a type (line 716)
            kind_1835 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 716, 15), dtype_1834, 'kind')
            str_1836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 716, 31), 'str', 'M')
            # Applying the binary operator '==' (line 716)
            result_eq_1837 = python_operator(stypy.reporting.localization.Localization(__file__, 716, 15), '==', kind_1835, str_1836)
            
            # Testing the type of an if condition (line 716)
            if_condition_1838 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 716, 12), result_eq_1837)
            # Assigning a type to the variable 'if_condition_1838' (line 716)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 716, 12), 'if_condition_1838', if_condition_1838)
            # SSA begins for if statement (line 716)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Subscript to a Name (line 717):
            
            # Assigning a Subscript to a Name (line 717):
            
            # Obtaining the type of the subscript
            int_1839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 717, 46), 'int')
            
            # Call to datetime_data(...): (line 717)
            # Processing the call arguments (line 717)
            # Getting the type of 'x' (line 717)
            x_1841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 717, 37), 'x', False)
            # Obtaining the member 'dtype' of a type (line 717)
            dtype_1842 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 717, 37), x_1841, 'dtype')
            # Processing the call keyword arguments (line 717)
            kwargs_1843 = {}
            # Getting the type of 'datetime_data' (line 717)
            datetime_data_1840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 717, 23), 'datetime_data', False)
            # Calling datetime_data(args, kwargs) (line 717)
            datetime_data_call_result_1844 = invoke(stypy.reporting.localization.Localization(__file__, 717, 23), datetime_data_1840, *[dtype_1842], **kwargs_1843)
            
            # Obtaining the member '__getitem__' of a type (line 717)
            getitem___1845 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 717, 23), datetime_data_call_result_1844, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 717)
            subscript_call_result_1846 = invoke(stypy.reporting.localization.Localization(__file__, 717, 23), getitem___1845, int_1839)
            
            # Assigning a type to the variable 'unit' (line 717)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 717, 16), 'unit', subscript_call_result_1846)
            # SSA branch for the else part of an if statement (line 716)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Str to a Name (line 719):
            
            # Assigning a Str to a Name (line 719):
            str_1847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 719, 23), 'str', 's')
            # Assigning a type to the variable 'unit' (line 719)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 719, 16), 'unit', str_1847)
            # SSA join for if statement (line 716)
            module_type_store = module_type_store.join_ssa_context()
            

            if more_types_in_union_1832:
                # SSA join for if statement (line 715)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 721)
        # Getting the type of 'timezone' (line 721)
        timezone_1848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 721, 11), 'timezone')
        # Getting the type of 'None' (line 721)
        None_1849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 721, 23), 'None')
        
        (may_be_1850, more_types_in_union_1851) = may_be_none(timezone_1848, None_1849)

        if may_be_1850:

            if more_types_in_union_1851:
                # Runtime conditional SSA (line 721)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Str to a Name (line 722):
            
            # Assigning a Str to a Name (line 722):
            str_1852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 722, 23), 'str', 'naive')
            # Assigning a type to the variable 'timezone' (line 722)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 722, 12), 'timezone', str_1852)

            if more_types_in_union_1851:
                # SSA join for if statement (line 721)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Name to a Attribute (line 723):
        
        # Assigning a Name to a Attribute (line 723):
        # Getting the type of 'timezone' (line 723)
        timezone_1853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 723, 24), 'timezone')
        # Getting the type of 'self' (line 723)
        self_1854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 723, 8), 'self')
        # Setting the type of the member 'timezone' of a type (line 723)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 723, 8), self_1854, 'timezone', timezone_1853)
        
        # Assigning a Name to a Attribute (line 724):
        
        # Assigning a Name to a Attribute (line 724):
        # Getting the type of 'unit' (line 724)
        unit_1855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 724, 20), 'unit')
        # Getting the type of 'self' (line 724)
        self_1856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 724, 8), 'self')
        # Setting the type of the member 'unit' of a type (line 724)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 724, 8), self_1856, 'unit', unit_1855)
        
        # Assigning a Name to a Attribute (line 725):
        
        # Assigning a Name to a Attribute (line 725):
        # Getting the type of 'casting' (line 725)
        casting_1857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 725, 23), 'casting')
        # Getting the type of 'self' (line 725)
        self_1858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 725, 8), 'self')
        # Setting the type of the member 'casting' of a type (line 725)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 725, 8), self_1858, 'casting', casting_1857)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def __call__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__call__'
        module_type_store = module_type_store.open_function_context('__call__', 727, 4, False)
        # Assigning a type to the variable 'self' (line 728)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 728, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DatetimeFormat.__call__.__dict__.__setitem__('stypy_localization', localization)
        DatetimeFormat.__call__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DatetimeFormat.__call__.__dict__.__setitem__('stypy_type_store', module_type_store)
        DatetimeFormat.__call__.__dict__.__setitem__('stypy_function_name', 'DatetimeFormat.__call__')
        DatetimeFormat.__call__.__dict__.__setitem__('stypy_param_names_list', ['x'])
        DatetimeFormat.__call__.__dict__.__setitem__('stypy_varargs_param_name', None)
        DatetimeFormat.__call__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DatetimeFormat.__call__.__dict__.__setitem__('stypy_call_defaults', defaults)
        DatetimeFormat.__call__.__dict__.__setitem__('stypy_call_varargs', varargs)
        DatetimeFormat.__call__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DatetimeFormat.__call__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DatetimeFormat.__call__', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__call__', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__call__(...)' code ##################

        str_1859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 728, 15), 'str', "'%s'")
        
        # Call to datetime_as_string(...): (line 728)
        # Processing the call arguments (line 728)
        # Getting the type of 'x' (line 728)
        x_1861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 728, 43), 'x', False)
        # Processing the call keyword arguments (line 728)
        # Getting the type of 'self' (line 729)
        self_1862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 729, 41), 'self', False)
        # Obtaining the member 'unit' of a type (line 729)
        unit_1863 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 729, 41), self_1862, 'unit')
        keyword_1864 = unit_1863
        # Getting the type of 'self' (line 730)
        self_1865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 730, 45), 'self', False)
        # Obtaining the member 'timezone' of a type (line 730)
        timezone_1866 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 730, 45), self_1865, 'timezone')
        keyword_1867 = timezone_1866
        # Getting the type of 'self' (line 731)
        self_1868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 731, 44), 'self', False)
        # Obtaining the member 'casting' of a type (line 731)
        casting_1869 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 731, 44), self_1868, 'casting')
        keyword_1870 = casting_1869
        kwargs_1871 = {'timezone': keyword_1867, 'casting': keyword_1870, 'unit': keyword_1864}
        # Getting the type of 'datetime_as_string' (line 728)
        datetime_as_string_1860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 728, 24), 'datetime_as_string', False)
        # Calling datetime_as_string(args, kwargs) (line 728)
        datetime_as_string_call_result_1872 = invoke(stypy.reporting.localization.Localization(__file__, 728, 24), datetime_as_string_1860, *[x_1861], **kwargs_1871)
        
        # Applying the binary operator '%' (line 728)
        result_mod_1873 = python_operator(stypy.reporting.localization.Localization(__file__, 728, 15), '%', str_1859, datetime_as_string_call_result_1872)
        
        # Assigning a type to the variable 'stypy_return_type' (line 728)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 728, 8), 'stypy_return_type', result_mod_1873)
        
        # ################# End of '__call__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__call__' in the type store
        # Getting the type of 'stypy_return_type' (line 727)
        stypy_return_type_1874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 727, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1874)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__call__'
        return stypy_return_type_1874


# Assigning a type to the variable 'DatetimeFormat' (line 712)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 712, 0), 'DatetimeFormat', DatetimeFormat)
# Declaration of the 'TimedeltaFormat' class

class TimedeltaFormat(object, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 734, 4, False)
        # Assigning a type to the variable 'self' (line 735)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 735, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TimedeltaFormat.__init__', ['data'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['data'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        
        # Getting the type of 'data' (line 735)
        data_1875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 735, 11), 'data')
        # Obtaining the member 'dtype' of a type (line 735)
        dtype_1876 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 735, 11), data_1875, 'dtype')
        # Obtaining the member 'kind' of a type (line 735)
        kind_1877 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 735, 11), dtype_1876, 'kind')
        str_1878 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 735, 30), 'str', 'm')
        # Applying the binary operator '==' (line 735)
        result_eq_1879 = python_operator(stypy.reporting.localization.Localization(__file__, 735, 11), '==', kind_1877, str_1878)
        
        # Testing the type of an if condition (line 735)
        if_condition_1880 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 735, 8), result_eq_1879)
        # Assigning a type to the variable 'if_condition_1880' (line 735)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 735, 8), 'if_condition_1880', if_condition_1880)
        # SSA begins for if statement (line 735)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Name (line 736):
        
        # Assigning a Subscript to a Name (line 736):
        
        # Obtaining the type of the subscript
        int_1881 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 736, 57), 'int')
        
        # Call to array(...): (line 736)
        # Processing the call arguments (line 736)
        
        # Obtaining an instance of the builtin type 'list' (line 736)
        list_1883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 736, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 736)
        # Adding element type (line 736)
        str_1884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 736, 31), 'str', 'NaT')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 736, 30), list_1883, str_1884)
        
        # Processing the call keyword arguments (line 736)
        # Getting the type of 'data' (line 736)
        data_1885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 736, 45), 'data', False)
        # Obtaining the member 'dtype' of a type (line 736)
        dtype_1886 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 736, 45), data_1885, 'dtype')
        keyword_1887 = dtype_1886
        kwargs_1888 = {'dtype': keyword_1887}
        # Getting the type of 'array' (line 736)
        array_1882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 736, 24), 'array', False)
        # Calling array(args, kwargs) (line 736)
        array_call_result_1889 = invoke(stypy.reporting.localization.Localization(__file__, 736, 24), array_1882, *[list_1883], **kwargs_1888)
        
        # Obtaining the member '__getitem__' of a type (line 736)
        getitem___1890 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 736, 24), array_call_result_1889, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 736)
        subscript_call_result_1891 = invoke(stypy.reporting.localization.Localization(__file__, 736, 24), getitem___1890, int_1881)
        
        # Assigning a type to the variable 'nat_value' (line 736)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 736, 12), 'nat_value', subscript_call_result_1891)
        
        # Assigning a Call to a Name (line 737):
        
        # Assigning a Call to a Name (line 737):
        
        # Call to view(...): (line 737)
        # Processing the call arguments (line 737)
        str_1901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 737, 54), 'str', 'i8')
        # Processing the call keyword arguments (line 737)
        kwargs_1902 = {}
        
        # Obtaining the type of the subscript
        
        # Call to not_equal(...): (line 737)
        # Processing the call arguments (line 737)
        # Getting the type of 'data' (line 737)
        data_1893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 737, 31), 'data', False)
        # Getting the type of 'nat_value' (line 737)
        nat_value_1894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 737, 37), 'nat_value', False)
        # Processing the call keyword arguments (line 737)
        kwargs_1895 = {}
        # Getting the type of 'not_equal' (line 737)
        not_equal_1892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 737, 21), 'not_equal', False)
        # Calling not_equal(args, kwargs) (line 737)
        not_equal_call_result_1896 = invoke(stypy.reporting.localization.Localization(__file__, 737, 21), not_equal_1892, *[data_1893, nat_value_1894], **kwargs_1895)
        
        # Getting the type of 'data' (line 737)
        data_1897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 737, 16), 'data', False)
        # Obtaining the member '__getitem__' of a type (line 737)
        getitem___1898 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 737, 16), data_1897, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 737)
        subscript_call_result_1899 = invoke(stypy.reporting.localization.Localization(__file__, 737, 16), getitem___1898, not_equal_call_result_1896)
        
        # Obtaining the member 'view' of a type (line 737)
        view_1900 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 737, 16), subscript_call_result_1899, 'view')
        # Calling view(args, kwargs) (line 737)
        view_call_result_1903 = invoke(stypy.reporting.localization.Localization(__file__, 737, 16), view_1900, *[str_1901], **kwargs_1902)
        
        # Assigning a type to the variable 'v' (line 737)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 737, 12), 'v', view_call_result_1903)
        
        
        
        # Call to len(...): (line 738)
        # Processing the call arguments (line 738)
        # Getting the type of 'v' (line 738)
        v_1905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 738, 19), 'v', False)
        # Processing the call keyword arguments (line 738)
        kwargs_1906 = {}
        # Getting the type of 'len' (line 738)
        len_1904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 738, 15), 'len', False)
        # Calling len(args, kwargs) (line 738)
        len_call_result_1907 = invoke(stypy.reporting.localization.Localization(__file__, 738, 15), len_1904, *[v_1905], **kwargs_1906)
        
        int_1908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 738, 24), 'int')
        # Applying the binary operator '>' (line 738)
        result_gt_1909 = python_operator(stypy.reporting.localization.Localization(__file__, 738, 15), '>', len_call_result_1907, int_1908)
        
        # Testing the type of an if condition (line 738)
        if_condition_1910 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 738, 12), result_gt_1909)
        # Assigning a type to the variable 'if_condition_1910' (line 738)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 738, 12), 'if_condition_1910', if_condition_1910)
        # SSA begins for if statement (line 738)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 740):
        
        # Assigning a Call to a Name (line 740):
        
        # Call to max(...): (line 740)
        # Processing the call arguments (line 740)
        
        # Call to len(...): (line 740)
        # Processing the call arguments (line 740)
        
        # Call to str(...): (line 740)
        # Processing the call arguments (line 740)
        
        # Call to reduce(...): (line 740)
        # Processing the call arguments (line 740)
        # Getting the type of 'v' (line 740)
        v_1916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 740, 57), 'v', False)
        # Processing the call keyword arguments (line 740)
        kwargs_1917 = {}
        # Getting the type of 'maximum' (line 740)
        maximum_1914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 740, 42), 'maximum', False)
        # Obtaining the member 'reduce' of a type (line 740)
        reduce_1915 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 740, 42), maximum_1914, 'reduce')
        # Calling reduce(args, kwargs) (line 740)
        reduce_call_result_1918 = invoke(stypy.reporting.localization.Localization(__file__, 740, 42), reduce_1915, *[v_1916], **kwargs_1917)
        
        # Processing the call keyword arguments (line 740)
        kwargs_1919 = {}
        # Getting the type of 'str' (line 740)
        str_1913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 740, 38), 'str', False)
        # Calling str(args, kwargs) (line 740)
        str_call_result_1920 = invoke(stypy.reporting.localization.Localization(__file__, 740, 38), str_1913, *[reduce_call_result_1918], **kwargs_1919)
        
        # Processing the call keyword arguments (line 740)
        kwargs_1921 = {}
        # Getting the type of 'len' (line 740)
        len_1912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 740, 34), 'len', False)
        # Calling len(args, kwargs) (line 740)
        len_call_result_1922 = invoke(stypy.reporting.localization.Localization(__file__, 740, 34), len_1912, *[str_call_result_1920], **kwargs_1921)
        
        
        # Call to len(...): (line 741)
        # Processing the call arguments (line 741)
        
        # Call to str(...): (line 741)
        # Processing the call arguments (line 741)
        
        # Call to reduce(...): (line 741)
        # Processing the call arguments (line 741)
        # Getting the type of 'v' (line 741)
        v_1927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 741, 57), 'v', False)
        # Processing the call keyword arguments (line 741)
        kwargs_1928 = {}
        # Getting the type of 'minimum' (line 741)
        minimum_1925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 741, 42), 'minimum', False)
        # Obtaining the member 'reduce' of a type (line 741)
        reduce_1926 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 741, 42), minimum_1925, 'reduce')
        # Calling reduce(args, kwargs) (line 741)
        reduce_call_result_1929 = invoke(stypy.reporting.localization.Localization(__file__, 741, 42), reduce_1926, *[v_1927], **kwargs_1928)
        
        # Processing the call keyword arguments (line 741)
        kwargs_1930 = {}
        # Getting the type of 'str' (line 741)
        str_1924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 741, 38), 'str', False)
        # Calling str(args, kwargs) (line 741)
        str_call_result_1931 = invoke(stypy.reporting.localization.Localization(__file__, 741, 38), str_1924, *[reduce_call_result_1929], **kwargs_1930)
        
        # Processing the call keyword arguments (line 741)
        kwargs_1932 = {}
        # Getting the type of 'len' (line 741)
        len_1923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 741, 34), 'len', False)
        # Calling len(args, kwargs) (line 741)
        len_call_result_1933 = invoke(stypy.reporting.localization.Localization(__file__, 741, 34), len_1923, *[str_call_result_1931], **kwargs_1932)
        
        # Processing the call keyword arguments (line 740)
        kwargs_1934 = {}
        # Getting the type of 'max' (line 740)
        max_1911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 740, 30), 'max', False)
        # Calling max(args, kwargs) (line 740)
        max_call_result_1935 = invoke(stypy.reporting.localization.Localization(__file__, 740, 30), max_1911, *[len_call_result_1922, len_call_result_1933], **kwargs_1934)
        
        # Assigning a type to the variable 'max_str_len' (line 740)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 740, 16), 'max_str_len', max_call_result_1935)
        # SSA branch for the else part of an if statement (line 738)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Num to a Name (line 743):
        
        # Assigning a Num to a Name (line 743):
        int_1936 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 743, 30), 'int')
        # Assigning a type to the variable 'max_str_len' (line 743)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 743, 16), 'max_str_len', int_1936)
        # SSA join for if statement (line 738)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Call to len(...): (line 744)
        # Processing the call arguments (line 744)
        # Getting the type of 'v' (line 744)
        v_1938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 744, 19), 'v', False)
        # Processing the call keyword arguments (line 744)
        kwargs_1939 = {}
        # Getting the type of 'len' (line 744)
        len_1937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 744, 15), 'len', False)
        # Calling len(args, kwargs) (line 744)
        len_call_result_1940 = invoke(stypy.reporting.localization.Localization(__file__, 744, 15), len_1937, *[v_1938], **kwargs_1939)
        
        
        # Call to len(...): (line 744)
        # Processing the call arguments (line 744)
        # Getting the type of 'data' (line 744)
        data_1942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 744, 28), 'data', False)
        # Processing the call keyword arguments (line 744)
        kwargs_1943 = {}
        # Getting the type of 'len' (line 744)
        len_1941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 744, 24), 'len', False)
        # Calling len(args, kwargs) (line 744)
        len_call_result_1944 = invoke(stypy.reporting.localization.Localization(__file__, 744, 24), len_1941, *[data_1942], **kwargs_1943)
        
        # Applying the binary operator '<' (line 744)
        result_lt_1945 = python_operator(stypy.reporting.localization.Localization(__file__, 744, 15), '<', len_call_result_1940, len_call_result_1944)
        
        # Testing the type of an if condition (line 744)
        if_condition_1946 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 744, 12), result_lt_1945)
        # Assigning a type to the variable 'if_condition_1946' (line 744)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 744, 12), 'if_condition_1946', if_condition_1946)
        # SSA begins for if statement (line 744)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 746):
        
        # Assigning a Call to a Name (line 746):
        
        # Call to max(...): (line 746)
        # Processing the call arguments (line 746)
        # Getting the type of 'max_str_len' (line 746)
        max_str_len_1948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 746, 34), 'max_str_len', False)
        int_1949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 746, 47), 'int')
        # Processing the call keyword arguments (line 746)
        kwargs_1950 = {}
        # Getting the type of 'max' (line 746)
        max_1947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 746, 30), 'max', False)
        # Calling max(args, kwargs) (line 746)
        max_call_result_1951 = invoke(stypy.reporting.localization.Localization(__file__, 746, 30), max_1947, *[max_str_len_1948, int_1949], **kwargs_1950)
        
        # Assigning a type to the variable 'max_str_len' (line 746)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 746, 16), 'max_str_len', max_call_result_1951)
        # SSA join for if statement (line 744)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Attribute (line 747):
        
        # Assigning a BinOp to a Attribute (line 747):
        str_1952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 747, 26), 'str', '%')
        
        # Call to str(...): (line 747)
        # Processing the call arguments (line 747)
        # Getting the type of 'max_str_len' (line 747)
        max_str_len_1954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 747, 36), 'max_str_len', False)
        # Processing the call keyword arguments (line 747)
        kwargs_1955 = {}
        # Getting the type of 'str' (line 747)
        str_1953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 747, 32), 'str', False)
        # Calling str(args, kwargs) (line 747)
        str_call_result_1956 = invoke(stypy.reporting.localization.Localization(__file__, 747, 32), str_1953, *[max_str_len_1954], **kwargs_1955)
        
        # Applying the binary operator '+' (line 747)
        result_add_1957 = python_operator(stypy.reporting.localization.Localization(__file__, 747, 26), '+', str_1952, str_call_result_1956)
        
        str_1958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 747, 51), 'str', 'd')
        # Applying the binary operator '+' (line 747)
        result_add_1959 = python_operator(stypy.reporting.localization.Localization(__file__, 747, 49), '+', result_add_1957, str_1958)
        
        # Getting the type of 'self' (line 747)
        self_1960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 747, 12), 'self')
        # Setting the type of the member 'format' of a type (line 747)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 747, 12), self_1960, 'format', result_add_1959)
        
        # Assigning a Call to a Attribute (line 748):
        
        # Assigning a Call to a Attribute (line 748):
        
        # Call to rjust(...): (line 748)
        # Processing the call arguments (line 748)
        # Getting the type of 'max_str_len' (line 748)
        max_str_len_1963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 748, 38), 'max_str_len', False)
        # Processing the call keyword arguments (line 748)
        kwargs_1964 = {}
        str_1961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 748, 24), 'str', "'NaT'")
        # Obtaining the member 'rjust' of a type (line 748)
        rjust_1962 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 748, 24), str_1961, 'rjust')
        # Calling rjust(args, kwargs) (line 748)
        rjust_call_result_1965 = invoke(stypy.reporting.localization.Localization(__file__, 748, 24), rjust_1962, *[max_str_len_1963], **kwargs_1964)
        
        # Getting the type of 'self' (line 748)
        self_1966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 748, 12), 'self')
        # Setting the type of the member '_nat' of a type (line 748)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 748, 12), self_1966, '_nat', rjust_call_result_1965)
        # SSA join for if statement (line 735)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def __call__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__call__'
        module_type_store = module_type_store.open_function_context('__call__', 750, 4, False)
        # Assigning a type to the variable 'self' (line 751)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 751, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TimedeltaFormat.__call__.__dict__.__setitem__('stypy_localization', localization)
        TimedeltaFormat.__call__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TimedeltaFormat.__call__.__dict__.__setitem__('stypy_type_store', module_type_store)
        TimedeltaFormat.__call__.__dict__.__setitem__('stypy_function_name', 'TimedeltaFormat.__call__')
        TimedeltaFormat.__call__.__dict__.__setitem__('stypy_param_names_list', ['x'])
        TimedeltaFormat.__call__.__dict__.__setitem__('stypy_varargs_param_name', None)
        TimedeltaFormat.__call__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TimedeltaFormat.__call__.__dict__.__setitem__('stypy_call_defaults', defaults)
        TimedeltaFormat.__call__.__dict__.__setitem__('stypy_call_varargs', varargs)
        TimedeltaFormat.__call__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TimedeltaFormat.__call__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TimedeltaFormat.__call__', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__call__', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__call__(...)' code ##################

        
        
        # Getting the type of 'x' (line 751)
        x_1967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 751, 11), 'x')
        int_1968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 751, 15), 'int')
        # Applying the binary operator '+' (line 751)
        result_add_1969 = python_operator(stypy.reporting.localization.Localization(__file__, 751, 11), '+', x_1967, int_1968)
        
        # Getting the type of 'x' (line 751)
        x_1970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 751, 20), 'x')
        # Applying the binary operator '==' (line 751)
        result_eq_1971 = python_operator(stypy.reporting.localization.Localization(__file__, 751, 11), '==', result_add_1969, x_1970)
        
        # Testing the type of an if condition (line 751)
        if_condition_1972 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 751, 8), result_eq_1971)
        # Assigning a type to the variable 'if_condition_1972' (line 751)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 751, 8), 'if_condition_1972', if_condition_1972)
        # SSA begins for if statement (line 751)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'self' (line 752)
        self_1973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 752, 19), 'self')
        # Obtaining the member '_nat' of a type (line 752)
        _nat_1974 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 752, 19), self_1973, '_nat')
        # Assigning a type to the variable 'stypy_return_type' (line 752)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 752, 12), 'stypy_return_type', _nat_1974)
        # SSA branch for the else part of an if statement (line 751)
        module_type_store.open_ssa_branch('else')
        # Getting the type of 'self' (line 754)
        self_1975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 754, 19), 'self')
        # Obtaining the member 'format' of a type (line 754)
        format_1976 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 754, 19), self_1975, 'format')
        
        # Call to astype(...): (line 754)
        # Processing the call arguments (line 754)
        str_1979 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 754, 42), 'str', 'i8')
        # Processing the call keyword arguments (line 754)
        kwargs_1980 = {}
        # Getting the type of 'x' (line 754)
        x_1977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 754, 33), 'x', False)
        # Obtaining the member 'astype' of a type (line 754)
        astype_1978 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 754, 33), x_1977, 'astype')
        # Calling astype(args, kwargs) (line 754)
        astype_call_result_1981 = invoke(stypy.reporting.localization.Localization(__file__, 754, 33), astype_1978, *[str_1979], **kwargs_1980)
        
        # Applying the binary operator '%' (line 754)
        result_mod_1982 = python_operator(stypy.reporting.localization.Localization(__file__, 754, 19), '%', format_1976, astype_call_result_1981)
        
        # Assigning a type to the variable 'stypy_return_type' (line 754)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 754, 12), 'stypy_return_type', result_mod_1982)
        # SSA join for if statement (line 751)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__call__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__call__' in the type store
        # Getting the type of 'stypy_return_type' (line 750)
        stypy_return_type_1983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 750, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1983)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__call__'
        return stypy_return_type_1983


# Assigning a type to the variable 'TimedeltaFormat' (line 733)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 733, 0), 'TimedeltaFormat', TimedeltaFormat)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
