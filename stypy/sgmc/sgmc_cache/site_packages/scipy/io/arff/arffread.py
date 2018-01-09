
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # Last Change: Mon Aug 20 08:00 PM 2007 J
2: from __future__ import division, print_function, absolute_import
3: 
4: import re
5: import itertools
6: import datetime
7: from functools import partial
8: 
9: import numpy as np
10: 
11: from scipy._lib.six import next
12: 
13: '''A module to read arff files.'''
14: 
15: __all__ = ['MetaData', 'loadarff', 'ArffError', 'ParseArffError']
16: 
17: # An Arff file is basically two parts:
18: #   - header
19: #   - data
20: #
21: # A header has each of its components starting by @META where META is one of
22: # the keyword (attribute of relation, for now).
23: 
24: # TODO:
25: #   - both integer and reals are treated as numeric -> the integer info
26: #    is lost!
27: #   - Replace ValueError by ParseError or something
28: 
29: # We know can handle the following:
30: #   - numeric and nominal attributes
31: #   - missing values for numeric attributes
32: 
33: r_meta = re.compile(r'^\s*@')
34: # Match a comment
35: r_comment = re.compile(r'^%')
36: # Match an empty line
37: r_empty = re.compile(r'^\s+$')
38: # Match a header line, that is a line which starts by @ + a word
39: r_headerline = re.compile(r'^@\S*')
40: r_datameta = re.compile(r'^@[Dd][Aa][Tt][Aa]')
41: r_relation = re.compile(r'^@[Rr][Ee][Ll][Aa][Tt][Ii][Oo][Nn]\s*(\S*)')
42: r_attribute = re.compile(r'^@[Aa][Tt][Tt][Rr][Ii][Bb][Uu][Tt][Ee]\s*(..*$)')
43: 
44: # To get attributes name enclosed with ''
45: r_comattrval = re.compile(r"'(..+)'\s+(..+$)")
46: # To get normal attributes
47: r_wcomattrval = re.compile(r"(\S+)\s+(..+$)")
48: 
49: #-------------------------
50: # Module defined exception
51: #-------------------------
52: 
53: 
54: class ArffError(IOError):
55:     pass
56: 
57: 
58: class ParseArffError(ArffError):
59:     pass
60: 
61: #------------------
62: # Various utilities
63: #------------------
64: 
65: # An attribute  is defined as @attribute name value
66: 
67: 
68: def parse_type(attrtype):
69:     '''Given an arff attribute value (meta data), returns its type.
70: 
71:     Expect the value to be a name.'''
72:     uattribute = attrtype.lower().strip()
73:     if uattribute[0] == '{':
74:         return 'nominal'
75:     elif uattribute[:len('real')] == 'real':
76:         return 'numeric'
77:     elif uattribute[:len('integer')] == 'integer':
78:         return 'numeric'
79:     elif uattribute[:len('numeric')] == 'numeric':
80:         return 'numeric'
81:     elif uattribute[:len('string')] == 'string':
82:         return 'string'
83:     elif uattribute[:len('relational')] == 'relational':
84:         return 'relational'
85:     elif uattribute[:len('date')] == 'date':
86:         return 'date'
87:     else:
88:         raise ParseArffError("unknown attribute %s" % uattribute)
89: 
90: 
91: def get_nominal(attribute):
92:     '''If attribute is nominal, returns a list of the values'''
93:     return attribute.split(',')
94: 
95: 
96: def read_data_list(ofile):
97:     '''Read each line of the iterable and put it in a list.'''
98:     data = [next(ofile)]
99:     if data[0].strip()[0] == '{':
100:         raise ValueError("This looks like a sparse ARFF: not supported yet")
101:     data.extend([i for i in ofile])
102:     return data
103: 
104: 
105: def get_ndata(ofile):
106:     '''Read the whole file to get number of data attributes.'''
107:     data = [next(ofile)]
108:     loc = 1
109:     if data[0].strip()[0] == '{':
110:         raise ValueError("This looks like a sparse ARFF: not supported yet")
111:     for i in ofile:
112:         loc += 1
113:     return loc
114: 
115: 
116: def maxnomlen(atrv):
117:     '''Given a string containing a nominal type definition, returns the
118:     string len of the biggest component.
119: 
120:     A nominal type is defined as seomthing framed between brace ({}).
121: 
122:     Parameters
123:     ----------
124:     atrv : str
125:        Nominal type definition
126: 
127:     Returns
128:     -------
129:     slen : int
130:        length of longest component
131: 
132:     Examples
133:     --------
134:     maxnomlen("{floup, bouga, fl, ratata}") returns 6 (the size of
135:     ratata, the longest nominal value).
136: 
137:     >>> maxnomlen("{floup, bouga, fl, ratata}")
138:     6
139:     '''
140:     nomtp = get_nom_val(atrv)
141:     return max(len(i) for i in nomtp)
142: 
143: 
144: def get_nom_val(atrv):
145:     '''Given a string containing a nominal type, returns a tuple of the
146:     possible values.
147: 
148:     A nominal type is defined as something framed between braces ({}).
149: 
150:     Parameters
151:     ----------
152:     atrv : str
153:        Nominal type definition
154: 
155:     Returns
156:     -------
157:     poss_vals : tuple
158:        possible values
159: 
160:     Examples
161:     --------
162:     >>> get_nom_val("{floup, bouga, fl, ratata}")
163:     ('floup', 'bouga', 'fl', 'ratata')
164:     '''
165:     r_nominal = re.compile('{(.+)}')
166:     m = r_nominal.match(atrv)
167:     if m:
168:         return tuple(i.strip() for i in m.group(1).split(','))
169:     else:
170:         raise ValueError("This does not look like a nominal string")
171: 
172: 
173: def get_date_format(atrv):
174:     r_date = re.compile(r"[Dd][Aa][Tt][Ee]\s+[\"']?(.+?)[\"']?$")
175:     m = r_date.match(atrv)
176:     if m:
177:         pattern = m.group(1).strip()
178:         # convert time pattern from Java's SimpleDateFormat to C's format
179:         datetime_unit = None
180:         if "yyyy" in pattern:
181:             pattern = pattern.replace("yyyy", "%Y")
182:             datetime_unit = "Y"
183:         elif "yy":
184:             pattern = pattern.replace("yy", "%y")
185:             datetime_unit = "Y"
186:         if "MM" in pattern:
187:             pattern = pattern.replace("MM", "%m")
188:             datetime_unit = "M"
189:         if "dd" in pattern:
190:             pattern = pattern.replace("dd", "%d")
191:             datetime_unit = "D"
192:         if "HH" in pattern:
193:             pattern = pattern.replace("HH", "%H")
194:             datetime_unit = "h"
195:         if "mm" in pattern:
196:             pattern = pattern.replace("mm", "%M")
197:             datetime_unit = "m"
198:         if "ss" in pattern:
199:             pattern = pattern.replace("ss", "%S")
200:             datetime_unit = "s"
201:         if "z" in pattern or "Z" in pattern:
202:             raise ValueError("Date type attributes with time zone not "
203:                              "supported, yet")
204: 
205:         if datetime_unit is None:
206:             raise ValueError("Invalid or unsupported date format")
207: 
208:         return pattern, datetime_unit
209:     else:
210:         raise ValueError("Invalid or no date format")
211: 
212: 
213: def go_data(ofile):
214:     '''Skip header.
215: 
216:     the first next() call of the returned iterator will be the @data line'''
217:     return itertools.dropwhile(lambda x: not r_datameta.match(x), ofile)
218: 
219: 
220: #----------------
221: # Parsing header
222: #----------------
223: def tokenize_attribute(iterable, attribute):
224:     '''Parse a raw string in header (eg starts by @attribute).
225: 
226:     Given a raw string attribute, try to get the name and type of the
227:     attribute. Constraints:
228: 
229:     * The first line must start with @attribute (case insensitive, and
230:       space like characters before @attribute are allowed)
231:     * Works also if the attribute is spread on multilines.
232:     * Works if empty lines or comments are in between
233: 
234:     Parameters
235:     ----------
236:     attribute : str
237:        the attribute string.
238: 
239:     Returns
240:     -------
241:     name : str
242:        name of the attribute
243:     value : str
244:        value of the attribute
245:     next : str
246:        next line to be parsed
247: 
248:     Examples
249:     --------
250:     If attribute is a string defined in python as r"floupi real", will
251:     return floupi as name, and real as value.
252: 
253:     >>> iterable = iter([0] * 10) # dummy iterator
254:     >>> tokenize_attribute(iterable, r"@attribute floupi real")
255:     ('floupi', 'real', 0)
256: 
257:     If attribute is r"'floupi 2' real", will return 'floupi 2' as name,
258:     and real as value.
259: 
260:     >>> tokenize_attribute(iterable, r"  @attribute 'floupi 2' real   ")
261:     ('floupi 2', 'real', 0)
262: 
263:     '''
264:     sattr = attribute.strip()
265:     mattr = r_attribute.match(sattr)
266:     if mattr:
267:         # atrv is everything after @attribute
268:         atrv = mattr.group(1)
269:         if r_comattrval.match(atrv):
270:             name, type = tokenize_single_comma(atrv)
271:             next_item = next(iterable)
272:         elif r_wcomattrval.match(atrv):
273:             name, type = tokenize_single_wcomma(atrv)
274:             next_item = next(iterable)
275:         else:
276:             # Not sure we should support this, as it does not seem supported by
277:             # weka.
278:             raise ValueError("multi line not supported yet")
279:             #name, type, next_item = tokenize_multilines(iterable, atrv)
280:     else:
281:         raise ValueError("First line unparsable: %s" % sattr)
282: 
283:     if type == 'relational':
284:         raise ValueError("relational attributes not supported yet")
285:     return name, type, next_item
286: 
287: 
288: def tokenize_single_comma(val):
289:     # XXX we match twice the same string (here and at the caller level). It is
290:     # stupid, but it is easier for now...
291:     m = r_comattrval.match(val)
292:     if m:
293:         try:
294:             name = m.group(1).strip()
295:             type = m.group(2).strip()
296:         except IndexError:
297:             raise ValueError("Error while tokenizing attribute")
298:     else:
299:         raise ValueError("Error while tokenizing single %s" % val)
300:     return name, type
301: 
302: 
303: def tokenize_single_wcomma(val):
304:     # XXX we match twice the same string (here and at the caller level). It is
305:     # stupid, but it is easier for now...
306:     m = r_wcomattrval.match(val)
307:     if m:
308:         try:
309:             name = m.group(1).strip()
310:             type = m.group(2).strip()
311:         except IndexError:
312:             raise ValueError("Error while tokenizing attribute")
313:     else:
314:         raise ValueError("Error while tokenizing single %s" % val)
315:     return name, type
316: 
317: 
318: def read_header(ofile):
319:     '''Read the header of the iterable ofile.'''
320:     i = next(ofile)
321: 
322:     # Pass first comments
323:     while r_comment.match(i):
324:         i = next(ofile)
325: 
326:     # Header is everything up to DATA attribute ?
327:     relation = None
328:     attributes = []
329:     while not r_datameta.match(i):
330:         m = r_headerline.match(i)
331:         if m:
332:             isattr = r_attribute.match(i)
333:             if isattr:
334:                 name, type, i = tokenize_attribute(ofile, i)
335:                 attributes.append((name, type))
336:             else:
337:                 isrel = r_relation.match(i)
338:                 if isrel:
339:                     relation = isrel.group(1)
340:                 else:
341:                     raise ValueError("Error parsing line %s" % i)
342:                 i = next(ofile)
343:         else:
344:             i = next(ofile)
345: 
346:     return relation, attributes
347: 
348: 
349: #--------------------
350: # Parsing actual data
351: #--------------------
352: def safe_float(x):
353:     '''given a string x, convert it to a float. If the stripped string is a ?,
354:     return a Nan (missing value).
355: 
356:     Parameters
357:     ----------
358:     x : str
359:        string to convert
360: 
361:     Returns
362:     -------
363:     f : float
364:        where float can be nan
365: 
366:     Examples
367:     --------
368:     >>> safe_float('1')
369:     1.0
370:     >>> safe_float('1\\n')
371:     1.0
372:     >>> safe_float('?\\n')
373:     nan
374:     '''
375:     if '?' in x:
376:         return np.nan
377:     else:
378:         return float(x)
379: 
380: 
381: def safe_nominal(value, pvalue):
382:     svalue = value.strip()
383:     if svalue in pvalue:
384:         return svalue
385:     elif svalue == '?':
386:         return svalue
387:     else:
388:         raise ValueError("%s value not in %s" % (str(svalue), str(pvalue)))
389: 
390: 
391: def safe_date(value, date_format, datetime_unit):
392:     date_str = value.strip().strip("'").strip('"')
393:     if date_str == '?':
394:         return np.datetime64('NaT', datetime_unit)
395:     else:
396:         dt = datetime.datetime.strptime(date_str, date_format)
397:         return np.datetime64(dt).astype("datetime64[%s]" % datetime_unit)
398: 
399: 
400: class MetaData(object):
401:     '''Small container to keep useful informations on a ARFF dataset.
402: 
403:     Knows about attributes names and types.
404: 
405:     Examples
406:     --------
407:     ::
408: 
409:         data, meta = loadarff('iris.arff')
410:         # This will print the attributes names of the iris.arff dataset
411:         for i in meta:
412:             print(i)
413:         # This works too
414:         meta.names()
415:         # Getting attribute type
416:         types = meta.types()
417: 
418:     Notes
419:     -----
420:     Also maintains the list of attributes in order, i.e. doing for i in
421:     meta, where meta is an instance of MetaData, will return the
422:     different attribute names in the order they were defined.
423:     '''
424:     def __init__(self, rel, attr):
425:         self.name = rel
426:         # We need the dictionary to be ordered
427:         # XXX: may be better to implement an ordered dictionary
428:         self._attributes = {}
429:         self._attrnames = []
430:         for name, value in attr:
431:             tp = parse_type(value)
432:             self._attrnames.append(name)
433:             if tp == 'nominal':
434:                 self._attributes[name] = (tp, get_nom_val(value))
435:             elif tp == 'date':
436:                 self._attributes[name] = (tp, get_date_format(value)[0])
437:             else:
438:                 self._attributes[name] = (tp, None)
439: 
440:     def __repr__(self):
441:         msg = ""
442:         msg += "Dataset: %s\n" % self.name
443:         for i in self._attrnames:
444:             msg += "\t%s's type is %s" % (i, self._attributes[i][0])
445:             if self._attributes[i][1]:
446:                 msg += ", range is %s" % str(self._attributes[i][1])
447:             msg += '\n'
448:         return msg
449: 
450:     def __iter__(self):
451:         return iter(self._attrnames)
452: 
453:     def __getitem__(self, key):
454:         return self._attributes[key]
455: 
456:     def names(self):
457:         '''Return the list of attribute names.'''
458:         return self._attrnames
459: 
460:     def types(self):
461:         '''Return the list of attribute types.'''
462:         attr_types = [self._attributes[name][0] for name in self._attrnames]
463:         return attr_types
464: 
465: 
466: def loadarff(f):
467:     '''
468:     Read an arff file.
469: 
470:     The data is returned as a record array, which can be accessed much like
471:     a dictionary of numpy arrays.  For example, if one of the attributes is
472:     called 'pressure', then its first 10 data points can be accessed from the
473:     ``data`` record array like so: ``data['pressure'][0:10]``
474: 
475: 
476:     Parameters
477:     ----------
478:     f : file-like or str
479:        File-like object to read from, or filename to open.
480: 
481:     Returns
482:     -------
483:     data : record array
484:        The data of the arff file, accessible by attribute names.
485:     meta : `MetaData`
486:        Contains information about the arff file such as name and
487:        type of attributes, the relation (name of the dataset), etc...
488: 
489:     Raises
490:     ------
491:     ParseArffError
492:         This is raised if the given file is not ARFF-formatted.
493:     NotImplementedError
494:         The ARFF file has an attribute which is not supported yet.
495: 
496:     Notes
497:     -----
498: 
499:     This function should be able to read most arff files. Not
500:     implemented functionality include:
501: 
502:     * date type attributes
503:     * string type attributes
504: 
505:     It can read files with numeric and nominal attributes.  It cannot read
506:     files with sparse data ({} in the file).  However, this function can
507:     read files with missing data (? in the file), representing the data
508:     points as NaNs.
509: 
510:     Examples
511:     --------
512:     >>> from scipy.io import arff
513:     >>> from io import StringIO
514:     >>> content = \"\"\"
515:     ... @relation foo
516:     ... @attribute width  numeric
517:     ... @attribute height numeric
518:     ... @attribute color  {red,green,blue,yellow,black}
519:     ... @data
520:     ... 5.0,3.25,blue
521:     ... 4.5,3.75,green
522:     ... 3.0,4.00,red
523:     ... \"\"\"
524:     >>> f = StringIO(content)
525:     >>> data, meta = arff.loadarff(f)
526:     >>> data
527:     array([(5.0, 3.25, 'blue'), (4.5, 3.75, 'green'), (3.0, 4.0, 'red')],
528:           dtype=[('width', '<f8'), ('height', '<f8'), ('color', '|S6')])
529:     >>> meta
530:     Dataset: foo
531:     \twidth's type is numeric
532:     \theight's type is numeric
533:     \tcolor's type is nominal, range is ('red', 'green', 'blue', 'yellow', 'black')
534: 
535:     '''
536:     if hasattr(f, 'read'):
537:         ofile = f
538:     else:
539:         ofile = open(f, 'rt')
540:     try:
541:         return _loadarff(ofile)
542:     finally:
543:         if ofile is not f:  # only close what we opened
544:             ofile.close()
545: 
546: 
547: def _loadarff(ofile):
548:     # Parse the header file
549:     try:
550:         rel, attr = read_header(ofile)
551:     except ValueError as e:
552:         msg = "Error while parsing header, error was: " + str(e)
553:         raise ParseArffError(msg)
554: 
555:     # Check whether we have a string attribute (not supported yet)
556:     hasstr = False
557:     for name, value in attr:
558:         type = parse_type(value)
559:         if type == 'string':
560:             hasstr = True
561: 
562:     meta = MetaData(rel, attr)
563: 
564:     # XXX The following code is not great
565:     # Build the type descriptor descr and the list of convertors to convert
566:     # each attribute to the suitable type (which should match the one in
567:     # descr).
568: 
569:     # This can be used once we want to support integer as integer values and
570:     # not as numeric anymore (using masked arrays ?).
571:     acls2dtype = {'real': float, 'integer': float, 'numeric': float}
572:     acls2conv = {'real': safe_float,
573:                  'integer': safe_float,
574:                  'numeric': safe_float}
575:     descr = []
576:     convertors = []
577:     if not hasstr:
578:         for name, value in attr:
579:             type = parse_type(value)
580:             if type == 'date':
581:                 date_format, datetime_unit = get_date_format(value)
582:                 descr.append((name, "datetime64[%s]" % datetime_unit))
583:                 convertors.append(partial(safe_date, date_format=date_format,
584:                                           datetime_unit=datetime_unit))
585:             elif type == 'nominal':
586:                 n = maxnomlen(value)
587:                 descr.append((name, 'S%d' % n))
588:                 pvalue = get_nom_val(value)
589:                 convertors.append(partial(safe_nominal, pvalue=pvalue))
590:             else:
591:                 descr.append((name, acls2dtype[type]))
592:                 convertors.append(safe_float)
593:                 #dc.append(acls2conv[type])
594:                 #sdescr.append((name, acls2sdtype[type]))
595:     else:
596:         # How to support string efficiently ? Ideally, we should know the max
597:         # size of the string before allocating the numpy array.
598:         raise NotImplementedError("String attributes not supported yet, sorry")
599: 
600:     ni = len(convertors)
601: 
602:     def generator(row_iter, delim=','):
603:         # TODO: this is where we are spending times (~80%). I think things
604:         # could be made more efficiently:
605:         #   - We could for example "compile" the function, because some values
606:         #   do not change here.
607:         #   - The function to convert a line to dtyped values could also be
608:         #   generated on the fly from a string and be executed instead of
609:         #   looping.
610:         #   - The regex are overkill: for comments, checking that a line starts
611:         #   by % should be enough and faster, and for empty lines, same thing
612:         #   --> this does not seem to change anything.
613: 
614:         # 'compiling' the range since it does not change
615:         # Note, I have already tried zipping the converters and
616:         # row elements and got slightly worse performance.
617:         elems = list(range(ni))
618: 
619:         for raw in row_iter:
620:             # We do not abstract skipping comments and empty lines for
621:             # performance reasons.
622:             if r_comment.match(raw) or r_empty.match(raw):
623:                 continue
624:             row = raw.split(delim)
625:             yield tuple([convertors[i](row[i]) for i in elems])
626: 
627:     a = generator(ofile)
628:     # No error should happen here: it is a bug otherwise
629:     data = np.fromiter(a, descr)
630:     return data, meta
631: 
632: 
633: #-----
634: # Misc
635: #-----
636: def basic_stats(data):
637:     nbfac = data.size * 1. / (data.size - 1)
638:     return np.nanmin(data), np.nanmax(data), np.mean(data), np.std(data) * nbfac
639: 
640: 
641: def print_attribute(name, tp, data):
642:     type = tp[0]
643:     if type == 'numeric' or type == 'real' or type == 'integer':
644:         min, max, mean, std = basic_stats(data)
645:         print("%s,%s,%f,%f,%f,%f" % (name, type, min, max, mean, std))
646:     else:
647:         msg = name + ",{"
648:         for i in range(len(tp[1])-1):
649:             msg += tp[1][i] + ","
650:         msg += tp[1][-1]
651:         msg += "}"
652:         print(msg)
653: 
654: 
655: def test_weka(filename):
656:     data, meta = loadarff(filename)
657:     print(len(data.dtype))
658:     print(data.size)
659:     for i in meta:
660:         print_attribute(i, meta[i], data[i])
661: 
662: # make sure nose does not find this as a test
663: test_weka.__test__ = False
664: 
665: 
666: if __name__ == '__main__':
667:     import sys
668:     filename = sys.argv[1]
669:     test_weka(filename)
670: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import re' statement (line 4)
import re

import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 're', re, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import itertools' statement (line 5)
import itertools

import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'itertools', itertools, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import datetime' statement (line 6)
import datetime

import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'datetime', datetime, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from functools import partial' statement (line 7)
try:
    from functools import partial

except:
    partial = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'functools', None, module_type_store, ['partial'], [partial])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'import numpy' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/arff/')
import_128260 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy')

if (type(import_128260) is not StypyTypeError):

    if (import_128260 != 'pyd_module'):
        __import__(import_128260)
        sys_modules_128261 = sys.modules[import_128260]
        import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'np', sys_modules_128261.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy', import_128260)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/arff/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from scipy._lib.six import next' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/arff/')
import_128262 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy._lib.six')

if (type(import_128262) is not StypyTypeError):

    if (import_128262 != 'pyd_module'):
        __import__(import_128262)
        sys_modules_128263 = sys.modules[import_128262]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy._lib.six', sys_modules_128263.module_type_store, module_type_store, ['next'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_128263, sys_modules_128263.module_type_store, module_type_store)
    else:
        from scipy._lib.six import next

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy._lib.six', None, module_type_store, ['next'], [next])

else:
    # Assigning a type to the variable 'scipy._lib.six' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy._lib.six', import_128262)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/arff/')

str_128264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 0), 'str', 'A module to read arff files.')

# Assigning a List to a Name (line 15):

# Assigning a List to a Name (line 15):
__all__ = ['MetaData', 'loadarff', 'ArffError', 'ParseArffError']
module_type_store.set_exportable_members(['MetaData', 'loadarff', 'ArffError', 'ParseArffError'])

# Obtaining an instance of the builtin type 'list' (line 15)
list_128265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 15)
# Adding element type (line 15)
str_128266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 11), 'str', 'MetaData')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 10), list_128265, str_128266)
# Adding element type (line 15)
str_128267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 23), 'str', 'loadarff')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 10), list_128265, str_128267)
# Adding element type (line 15)
str_128268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 35), 'str', 'ArffError')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 10), list_128265, str_128268)
# Adding element type (line 15)
str_128269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 48), 'str', 'ParseArffError')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 10), list_128265, str_128269)

# Assigning a type to the variable '__all__' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), '__all__', list_128265)

# Assigning a Call to a Name (line 33):

# Assigning a Call to a Name (line 33):

# Call to compile(...): (line 33)
# Processing the call arguments (line 33)
str_128272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 20), 'str', '^\\s*@')
# Processing the call keyword arguments (line 33)
kwargs_128273 = {}
# Getting the type of 're' (line 33)
re_128270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 9), 're', False)
# Obtaining the member 'compile' of a type (line 33)
compile_128271 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 9), re_128270, 'compile')
# Calling compile(args, kwargs) (line 33)
compile_call_result_128274 = invoke(stypy.reporting.localization.Localization(__file__, 33, 9), compile_128271, *[str_128272], **kwargs_128273)

# Assigning a type to the variable 'r_meta' (line 33)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 0), 'r_meta', compile_call_result_128274)

# Assigning a Call to a Name (line 35):

# Assigning a Call to a Name (line 35):

# Call to compile(...): (line 35)
# Processing the call arguments (line 35)
str_128277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 23), 'str', '^%')
# Processing the call keyword arguments (line 35)
kwargs_128278 = {}
# Getting the type of 're' (line 35)
re_128275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 12), 're', False)
# Obtaining the member 'compile' of a type (line 35)
compile_128276 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 12), re_128275, 'compile')
# Calling compile(args, kwargs) (line 35)
compile_call_result_128279 = invoke(stypy.reporting.localization.Localization(__file__, 35, 12), compile_128276, *[str_128277], **kwargs_128278)

# Assigning a type to the variable 'r_comment' (line 35)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 0), 'r_comment', compile_call_result_128279)

# Assigning a Call to a Name (line 37):

# Assigning a Call to a Name (line 37):

# Call to compile(...): (line 37)
# Processing the call arguments (line 37)
str_128282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 21), 'str', '^\\s+$')
# Processing the call keyword arguments (line 37)
kwargs_128283 = {}
# Getting the type of 're' (line 37)
re_128280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 10), 're', False)
# Obtaining the member 'compile' of a type (line 37)
compile_128281 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 10), re_128280, 'compile')
# Calling compile(args, kwargs) (line 37)
compile_call_result_128284 = invoke(stypy.reporting.localization.Localization(__file__, 37, 10), compile_128281, *[str_128282], **kwargs_128283)

# Assigning a type to the variable 'r_empty' (line 37)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 0), 'r_empty', compile_call_result_128284)

# Assigning a Call to a Name (line 39):

# Assigning a Call to a Name (line 39):

# Call to compile(...): (line 39)
# Processing the call arguments (line 39)
str_128287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 26), 'str', '^@\\S*')
# Processing the call keyword arguments (line 39)
kwargs_128288 = {}
# Getting the type of 're' (line 39)
re_128285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 15), 're', False)
# Obtaining the member 'compile' of a type (line 39)
compile_128286 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 15), re_128285, 'compile')
# Calling compile(args, kwargs) (line 39)
compile_call_result_128289 = invoke(stypy.reporting.localization.Localization(__file__, 39, 15), compile_128286, *[str_128287], **kwargs_128288)

# Assigning a type to the variable 'r_headerline' (line 39)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 0), 'r_headerline', compile_call_result_128289)

# Assigning a Call to a Name (line 40):

# Assigning a Call to a Name (line 40):

# Call to compile(...): (line 40)
# Processing the call arguments (line 40)
str_128292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 24), 'str', '^@[Dd][Aa][Tt][Aa]')
# Processing the call keyword arguments (line 40)
kwargs_128293 = {}
# Getting the type of 're' (line 40)
re_128290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 13), 're', False)
# Obtaining the member 'compile' of a type (line 40)
compile_128291 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 13), re_128290, 'compile')
# Calling compile(args, kwargs) (line 40)
compile_call_result_128294 = invoke(stypy.reporting.localization.Localization(__file__, 40, 13), compile_128291, *[str_128292], **kwargs_128293)

# Assigning a type to the variable 'r_datameta' (line 40)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 0), 'r_datameta', compile_call_result_128294)

# Assigning a Call to a Name (line 41):

# Assigning a Call to a Name (line 41):

# Call to compile(...): (line 41)
# Processing the call arguments (line 41)
str_128297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 24), 'str', '^@[Rr][Ee][Ll][Aa][Tt][Ii][Oo][Nn]\\s*(\\S*)')
# Processing the call keyword arguments (line 41)
kwargs_128298 = {}
# Getting the type of 're' (line 41)
re_128295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 13), 're', False)
# Obtaining the member 'compile' of a type (line 41)
compile_128296 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 13), re_128295, 'compile')
# Calling compile(args, kwargs) (line 41)
compile_call_result_128299 = invoke(stypy.reporting.localization.Localization(__file__, 41, 13), compile_128296, *[str_128297], **kwargs_128298)

# Assigning a type to the variable 'r_relation' (line 41)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 0), 'r_relation', compile_call_result_128299)

# Assigning a Call to a Name (line 42):

# Assigning a Call to a Name (line 42):

# Call to compile(...): (line 42)
# Processing the call arguments (line 42)
str_128302 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 25), 'str', '^@[Aa][Tt][Tt][Rr][Ii][Bb][Uu][Tt][Ee]\\s*(..*$)')
# Processing the call keyword arguments (line 42)
kwargs_128303 = {}
# Getting the type of 're' (line 42)
re_128300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 14), 're', False)
# Obtaining the member 'compile' of a type (line 42)
compile_128301 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 14), re_128300, 'compile')
# Calling compile(args, kwargs) (line 42)
compile_call_result_128304 = invoke(stypy.reporting.localization.Localization(__file__, 42, 14), compile_128301, *[str_128302], **kwargs_128303)

# Assigning a type to the variable 'r_attribute' (line 42)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 0), 'r_attribute', compile_call_result_128304)

# Assigning a Call to a Name (line 45):

# Assigning a Call to a Name (line 45):

# Call to compile(...): (line 45)
# Processing the call arguments (line 45)
str_128307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 26), 'str', "'(..+)'\\s+(..+$)")
# Processing the call keyword arguments (line 45)
kwargs_128308 = {}
# Getting the type of 're' (line 45)
re_128305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 15), 're', False)
# Obtaining the member 'compile' of a type (line 45)
compile_128306 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 15), re_128305, 'compile')
# Calling compile(args, kwargs) (line 45)
compile_call_result_128309 = invoke(stypy.reporting.localization.Localization(__file__, 45, 15), compile_128306, *[str_128307], **kwargs_128308)

# Assigning a type to the variable 'r_comattrval' (line 45)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 0), 'r_comattrval', compile_call_result_128309)

# Assigning a Call to a Name (line 47):

# Assigning a Call to a Name (line 47):

# Call to compile(...): (line 47)
# Processing the call arguments (line 47)
str_128312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 27), 'str', '(\\S+)\\s+(..+$)')
# Processing the call keyword arguments (line 47)
kwargs_128313 = {}
# Getting the type of 're' (line 47)
re_128310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 16), 're', False)
# Obtaining the member 'compile' of a type (line 47)
compile_128311 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 16), re_128310, 'compile')
# Calling compile(args, kwargs) (line 47)
compile_call_result_128314 = invoke(stypy.reporting.localization.Localization(__file__, 47, 16), compile_128311, *[str_128312], **kwargs_128313)

# Assigning a type to the variable 'r_wcomattrval' (line 47)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 0), 'r_wcomattrval', compile_call_result_128314)
# Declaration of the 'ArffError' class
# Getting the type of 'IOError' (line 54)
IOError_128315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 16), 'IOError')

class ArffError(IOError_128315, ):
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 54, 0, False)
        # Assigning a type to the variable 'self' (line 55)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ArffError.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'ArffError' (line 54)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 0), 'ArffError', ArffError)
# Declaration of the 'ParseArffError' class
# Getting the type of 'ArffError' (line 58)
ArffError_128316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 21), 'ArffError')

class ParseArffError(ArffError_128316, ):
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 58, 0, False)
        # Assigning a type to the variable 'self' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ParseArffError.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'ParseArffError' (line 58)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 0), 'ParseArffError', ParseArffError)

@norecursion
def parse_type(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'parse_type'
    module_type_store = module_type_store.open_function_context('parse_type', 68, 0, False)
    
    # Passed parameters checking function
    parse_type.stypy_localization = localization
    parse_type.stypy_type_of_self = None
    parse_type.stypy_type_store = module_type_store
    parse_type.stypy_function_name = 'parse_type'
    parse_type.stypy_param_names_list = ['attrtype']
    parse_type.stypy_varargs_param_name = None
    parse_type.stypy_kwargs_param_name = None
    parse_type.stypy_call_defaults = defaults
    parse_type.stypy_call_varargs = varargs
    parse_type.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'parse_type', ['attrtype'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'parse_type', localization, ['attrtype'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'parse_type(...)' code ##################

    str_128317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, (-1)), 'str', 'Given an arff attribute value (meta data), returns its type.\n\n    Expect the value to be a name.')
    
    # Assigning a Call to a Name (line 72):
    
    # Assigning a Call to a Name (line 72):
    
    # Call to strip(...): (line 72)
    # Processing the call keyword arguments (line 72)
    kwargs_128323 = {}
    
    # Call to lower(...): (line 72)
    # Processing the call keyword arguments (line 72)
    kwargs_128320 = {}
    # Getting the type of 'attrtype' (line 72)
    attrtype_128318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 17), 'attrtype', False)
    # Obtaining the member 'lower' of a type (line 72)
    lower_128319 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 17), attrtype_128318, 'lower')
    # Calling lower(args, kwargs) (line 72)
    lower_call_result_128321 = invoke(stypy.reporting.localization.Localization(__file__, 72, 17), lower_128319, *[], **kwargs_128320)
    
    # Obtaining the member 'strip' of a type (line 72)
    strip_128322 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 17), lower_call_result_128321, 'strip')
    # Calling strip(args, kwargs) (line 72)
    strip_call_result_128324 = invoke(stypy.reporting.localization.Localization(__file__, 72, 17), strip_128322, *[], **kwargs_128323)
    
    # Assigning a type to the variable 'uattribute' (line 72)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'uattribute', strip_call_result_128324)
    
    
    
    # Obtaining the type of the subscript
    int_128325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 18), 'int')
    # Getting the type of 'uattribute' (line 73)
    uattribute_128326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 7), 'uattribute')
    # Obtaining the member '__getitem__' of a type (line 73)
    getitem___128327 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 7), uattribute_128326, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 73)
    subscript_call_result_128328 = invoke(stypy.reporting.localization.Localization(__file__, 73, 7), getitem___128327, int_128325)
    
    str_128329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 24), 'str', '{')
    # Applying the binary operator '==' (line 73)
    result_eq_128330 = python_operator(stypy.reporting.localization.Localization(__file__, 73, 7), '==', subscript_call_result_128328, str_128329)
    
    # Testing the type of an if condition (line 73)
    if_condition_128331 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 73, 4), result_eq_128330)
    # Assigning a type to the variable 'if_condition_128331' (line 73)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 4), 'if_condition_128331', if_condition_128331)
    # SSA begins for if statement (line 73)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    str_128332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 15), 'str', 'nominal')
    # Assigning a type to the variable 'stypy_return_type' (line 74)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'stypy_return_type', str_128332)
    # SSA branch for the else part of an if statement (line 73)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Obtaining the type of the subscript
    
    # Call to len(...): (line 75)
    # Processing the call arguments (line 75)
    str_128334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 25), 'str', 'real')
    # Processing the call keyword arguments (line 75)
    kwargs_128335 = {}
    # Getting the type of 'len' (line 75)
    len_128333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 21), 'len', False)
    # Calling len(args, kwargs) (line 75)
    len_call_result_128336 = invoke(stypy.reporting.localization.Localization(__file__, 75, 21), len_128333, *[str_128334], **kwargs_128335)
    
    slice_128337 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 75, 9), None, len_call_result_128336, None)
    # Getting the type of 'uattribute' (line 75)
    uattribute_128338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 9), 'uattribute')
    # Obtaining the member '__getitem__' of a type (line 75)
    getitem___128339 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 9), uattribute_128338, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 75)
    subscript_call_result_128340 = invoke(stypy.reporting.localization.Localization(__file__, 75, 9), getitem___128339, slice_128337)
    
    str_128341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 37), 'str', 'real')
    # Applying the binary operator '==' (line 75)
    result_eq_128342 = python_operator(stypy.reporting.localization.Localization(__file__, 75, 9), '==', subscript_call_result_128340, str_128341)
    
    # Testing the type of an if condition (line 75)
    if_condition_128343 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 75, 9), result_eq_128342)
    # Assigning a type to the variable 'if_condition_128343' (line 75)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 9), 'if_condition_128343', if_condition_128343)
    # SSA begins for if statement (line 75)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    str_128344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 15), 'str', 'numeric')
    # Assigning a type to the variable 'stypy_return_type' (line 76)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'stypy_return_type', str_128344)
    # SSA branch for the else part of an if statement (line 75)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Obtaining the type of the subscript
    
    # Call to len(...): (line 77)
    # Processing the call arguments (line 77)
    str_128346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 25), 'str', 'integer')
    # Processing the call keyword arguments (line 77)
    kwargs_128347 = {}
    # Getting the type of 'len' (line 77)
    len_128345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 21), 'len', False)
    # Calling len(args, kwargs) (line 77)
    len_call_result_128348 = invoke(stypy.reporting.localization.Localization(__file__, 77, 21), len_128345, *[str_128346], **kwargs_128347)
    
    slice_128349 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 77, 9), None, len_call_result_128348, None)
    # Getting the type of 'uattribute' (line 77)
    uattribute_128350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 9), 'uattribute')
    # Obtaining the member '__getitem__' of a type (line 77)
    getitem___128351 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 9), uattribute_128350, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 77)
    subscript_call_result_128352 = invoke(stypy.reporting.localization.Localization(__file__, 77, 9), getitem___128351, slice_128349)
    
    str_128353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 40), 'str', 'integer')
    # Applying the binary operator '==' (line 77)
    result_eq_128354 = python_operator(stypy.reporting.localization.Localization(__file__, 77, 9), '==', subscript_call_result_128352, str_128353)
    
    # Testing the type of an if condition (line 77)
    if_condition_128355 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 77, 9), result_eq_128354)
    # Assigning a type to the variable 'if_condition_128355' (line 77)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 9), 'if_condition_128355', if_condition_128355)
    # SSA begins for if statement (line 77)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    str_128356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 15), 'str', 'numeric')
    # Assigning a type to the variable 'stypy_return_type' (line 78)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'stypy_return_type', str_128356)
    # SSA branch for the else part of an if statement (line 77)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Obtaining the type of the subscript
    
    # Call to len(...): (line 79)
    # Processing the call arguments (line 79)
    str_128358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 25), 'str', 'numeric')
    # Processing the call keyword arguments (line 79)
    kwargs_128359 = {}
    # Getting the type of 'len' (line 79)
    len_128357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 21), 'len', False)
    # Calling len(args, kwargs) (line 79)
    len_call_result_128360 = invoke(stypy.reporting.localization.Localization(__file__, 79, 21), len_128357, *[str_128358], **kwargs_128359)
    
    slice_128361 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 79, 9), None, len_call_result_128360, None)
    # Getting the type of 'uattribute' (line 79)
    uattribute_128362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 9), 'uattribute')
    # Obtaining the member '__getitem__' of a type (line 79)
    getitem___128363 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 9), uattribute_128362, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 79)
    subscript_call_result_128364 = invoke(stypy.reporting.localization.Localization(__file__, 79, 9), getitem___128363, slice_128361)
    
    str_128365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 40), 'str', 'numeric')
    # Applying the binary operator '==' (line 79)
    result_eq_128366 = python_operator(stypy.reporting.localization.Localization(__file__, 79, 9), '==', subscript_call_result_128364, str_128365)
    
    # Testing the type of an if condition (line 79)
    if_condition_128367 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 79, 9), result_eq_128366)
    # Assigning a type to the variable 'if_condition_128367' (line 79)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 9), 'if_condition_128367', if_condition_128367)
    # SSA begins for if statement (line 79)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    str_128368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 15), 'str', 'numeric')
    # Assigning a type to the variable 'stypy_return_type' (line 80)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'stypy_return_type', str_128368)
    # SSA branch for the else part of an if statement (line 79)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Obtaining the type of the subscript
    
    # Call to len(...): (line 81)
    # Processing the call arguments (line 81)
    str_128370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 25), 'str', 'string')
    # Processing the call keyword arguments (line 81)
    kwargs_128371 = {}
    # Getting the type of 'len' (line 81)
    len_128369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 21), 'len', False)
    # Calling len(args, kwargs) (line 81)
    len_call_result_128372 = invoke(stypy.reporting.localization.Localization(__file__, 81, 21), len_128369, *[str_128370], **kwargs_128371)
    
    slice_128373 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 81, 9), None, len_call_result_128372, None)
    # Getting the type of 'uattribute' (line 81)
    uattribute_128374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 9), 'uattribute')
    # Obtaining the member '__getitem__' of a type (line 81)
    getitem___128375 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 9), uattribute_128374, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 81)
    subscript_call_result_128376 = invoke(stypy.reporting.localization.Localization(__file__, 81, 9), getitem___128375, slice_128373)
    
    str_128377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 39), 'str', 'string')
    # Applying the binary operator '==' (line 81)
    result_eq_128378 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 9), '==', subscript_call_result_128376, str_128377)
    
    # Testing the type of an if condition (line 81)
    if_condition_128379 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 81, 9), result_eq_128378)
    # Assigning a type to the variable 'if_condition_128379' (line 81)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 9), 'if_condition_128379', if_condition_128379)
    # SSA begins for if statement (line 81)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    str_128380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 15), 'str', 'string')
    # Assigning a type to the variable 'stypy_return_type' (line 82)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'stypy_return_type', str_128380)
    # SSA branch for the else part of an if statement (line 81)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Obtaining the type of the subscript
    
    # Call to len(...): (line 83)
    # Processing the call arguments (line 83)
    str_128382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 25), 'str', 'relational')
    # Processing the call keyword arguments (line 83)
    kwargs_128383 = {}
    # Getting the type of 'len' (line 83)
    len_128381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 21), 'len', False)
    # Calling len(args, kwargs) (line 83)
    len_call_result_128384 = invoke(stypy.reporting.localization.Localization(__file__, 83, 21), len_128381, *[str_128382], **kwargs_128383)
    
    slice_128385 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 83, 9), None, len_call_result_128384, None)
    # Getting the type of 'uattribute' (line 83)
    uattribute_128386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 9), 'uattribute')
    # Obtaining the member '__getitem__' of a type (line 83)
    getitem___128387 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 9), uattribute_128386, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 83)
    subscript_call_result_128388 = invoke(stypy.reporting.localization.Localization(__file__, 83, 9), getitem___128387, slice_128385)
    
    str_128389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 43), 'str', 'relational')
    # Applying the binary operator '==' (line 83)
    result_eq_128390 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 9), '==', subscript_call_result_128388, str_128389)
    
    # Testing the type of an if condition (line 83)
    if_condition_128391 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 83, 9), result_eq_128390)
    # Assigning a type to the variable 'if_condition_128391' (line 83)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 9), 'if_condition_128391', if_condition_128391)
    # SSA begins for if statement (line 83)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    str_128392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 15), 'str', 'relational')
    # Assigning a type to the variable 'stypy_return_type' (line 84)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'stypy_return_type', str_128392)
    # SSA branch for the else part of an if statement (line 83)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Obtaining the type of the subscript
    
    # Call to len(...): (line 85)
    # Processing the call arguments (line 85)
    str_128394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 25), 'str', 'date')
    # Processing the call keyword arguments (line 85)
    kwargs_128395 = {}
    # Getting the type of 'len' (line 85)
    len_128393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 21), 'len', False)
    # Calling len(args, kwargs) (line 85)
    len_call_result_128396 = invoke(stypy.reporting.localization.Localization(__file__, 85, 21), len_128393, *[str_128394], **kwargs_128395)
    
    slice_128397 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 85, 9), None, len_call_result_128396, None)
    # Getting the type of 'uattribute' (line 85)
    uattribute_128398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 9), 'uattribute')
    # Obtaining the member '__getitem__' of a type (line 85)
    getitem___128399 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 9), uattribute_128398, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 85)
    subscript_call_result_128400 = invoke(stypy.reporting.localization.Localization(__file__, 85, 9), getitem___128399, slice_128397)
    
    str_128401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 37), 'str', 'date')
    # Applying the binary operator '==' (line 85)
    result_eq_128402 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 9), '==', subscript_call_result_128400, str_128401)
    
    # Testing the type of an if condition (line 85)
    if_condition_128403 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 85, 9), result_eq_128402)
    # Assigning a type to the variable 'if_condition_128403' (line 85)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 9), 'if_condition_128403', if_condition_128403)
    # SSA begins for if statement (line 85)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    str_128404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 15), 'str', 'date')
    # Assigning a type to the variable 'stypy_return_type' (line 86)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'stypy_return_type', str_128404)
    # SSA branch for the else part of an if statement (line 85)
    module_type_store.open_ssa_branch('else')
    
    # Call to ParseArffError(...): (line 88)
    # Processing the call arguments (line 88)
    str_128406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 29), 'str', 'unknown attribute %s')
    # Getting the type of 'uattribute' (line 88)
    uattribute_128407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 54), 'uattribute', False)
    # Applying the binary operator '%' (line 88)
    result_mod_128408 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 29), '%', str_128406, uattribute_128407)
    
    # Processing the call keyword arguments (line 88)
    kwargs_128409 = {}
    # Getting the type of 'ParseArffError' (line 88)
    ParseArffError_128405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 14), 'ParseArffError', False)
    # Calling ParseArffError(args, kwargs) (line 88)
    ParseArffError_call_result_128410 = invoke(stypy.reporting.localization.Localization(__file__, 88, 14), ParseArffError_128405, *[result_mod_128408], **kwargs_128409)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 88, 8), ParseArffError_call_result_128410, 'raise parameter', BaseException)
    # SSA join for if statement (line 85)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 83)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 81)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 79)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 77)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 75)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 73)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'parse_type(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'parse_type' in the type store
    # Getting the type of 'stypy_return_type' (line 68)
    stypy_return_type_128411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_128411)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'parse_type'
    return stypy_return_type_128411

# Assigning a type to the variable 'parse_type' (line 68)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 0), 'parse_type', parse_type)

@norecursion
def get_nominal(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'get_nominal'
    module_type_store = module_type_store.open_function_context('get_nominal', 91, 0, False)
    
    # Passed parameters checking function
    get_nominal.stypy_localization = localization
    get_nominal.stypy_type_of_self = None
    get_nominal.stypy_type_store = module_type_store
    get_nominal.stypy_function_name = 'get_nominal'
    get_nominal.stypy_param_names_list = ['attribute']
    get_nominal.stypy_varargs_param_name = None
    get_nominal.stypy_kwargs_param_name = None
    get_nominal.stypy_call_defaults = defaults
    get_nominal.stypy_call_varargs = varargs
    get_nominal.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'get_nominal', ['attribute'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'get_nominal', localization, ['attribute'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'get_nominal(...)' code ##################

    str_128412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 4), 'str', 'If attribute is nominal, returns a list of the values')
    
    # Call to split(...): (line 93)
    # Processing the call arguments (line 93)
    str_128415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 27), 'str', ',')
    # Processing the call keyword arguments (line 93)
    kwargs_128416 = {}
    # Getting the type of 'attribute' (line 93)
    attribute_128413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 11), 'attribute', False)
    # Obtaining the member 'split' of a type (line 93)
    split_128414 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 11), attribute_128413, 'split')
    # Calling split(args, kwargs) (line 93)
    split_call_result_128417 = invoke(stypy.reporting.localization.Localization(__file__, 93, 11), split_128414, *[str_128415], **kwargs_128416)
    
    # Assigning a type to the variable 'stypy_return_type' (line 93)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 4), 'stypy_return_type', split_call_result_128417)
    
    # ################# End of 'get_nominal(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_nominal' in the type store
    # Getting the type of 'stypy_return_type' (line 91)
    stypy_return_type_128418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_128418)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_nominal'
    return stypy_return_type_128418

# Assigning a type to the variable 'get_nominal' (line 91)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 0), 'get_nominal', get_nominal)

@norecursion
def read_data_list(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'read_data_list'
    module_type_store = module_type_store.open_function_context('read_data_list', 96, 0, False)
    
    # Passed parameters checking function
    read_data_list.stypy_localization = localization
    read_data_list.stypy_type_of_self = None
    read_data_list.stypy_type_store = module_type_store
    read_data_list.stypy_function_name = 'read_data_list'
    read_data_list.stypy_param_names_list = ['ofile']
    read_data_list.stypy_varargs_param_name = None
    read_data_list.stypy_kwargs_param_name = None
    read_data_list.stypy_call_defaults = defaults
    read_data_list.stypy_call_varargs = varargs
    read_data_list.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'read_data_list', ['ofile'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'read_data_list', localization, ['ofile'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'read_data_list(...)' code ##################

    str_128419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 4), 'str', 'Read each line of the iterable and put it in a list.')
    
    # Assigning a List to a Name (line 98):
    
    # Assigning a List to a Name (line 98):
    
    # Obtaining an instance of the builtin type 'list' (line 98)
    list_128420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 11), 'list')
    # Adding type elements to the builtin type 'list' instance (line 98)
    # Adding element type (line 98)
    
    # Call to next(...): (line 98)
    # Processing the call arguments (line 98)
    # Getting the type of 'ofile' (line 98)
    ofile_128422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 17), 'ofile', False)
    # Processing the call keyword arguments (line 98)
    kwargs_128423 = {}
    # Getting the type of 'next' (line 98)
    next_128421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 12), 'next', False)
    # Calling next(args, kwargs) (line 98)
    next_call_result_128424 = invoke(stypy.reporting.localization.Localization(__file__, 98, 12), next_128421, *[ofile_128422], **kwargs_128423)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 11), list_128420, next_call_result_128424)
    
    # Assigning a type to the variable 'data' (line 98)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'data', list_128420)
    
    
    
    # Obtaining the type of the subscript
    int_128425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 23), 'int')
    
    # Call to strip(...): (line 99)
    # Processing the call keyword arguments (line 99)
    kwargs_128431 = {}
    
    # Obtaining the type of the subscript
    int_128426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 12), 'int')
    # Getting the type of 'data' (line 99)
    data_128427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 7), 'data', False)
    # Obtaining the member '__getitem__' of a type (line 99)
    getitem___128428 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 7), data_128427, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 99)
    subscript_call_result_128429 = invoke(stypy.reporting.localization.Localization(__file__, 99, 7), getitem___128428, int_128426)
    
    # Obtaining the member 'strip' of a type (line 99)
    strip_128430 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 7), subscript_call_result_128429, 'strip')
    # Calling strip(args, kwargs) (line 99)
    strip_call_result_128432 = invoke(stypy.reporting.localization.Localization(__file__, 99, 7), strip_128430, *[], **kwargs_128431)
    
    # Obtaining the member '__getitem__' of a type (line 99)
    getitem___128433 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 7), strip_call_result_128432, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 99)
    subscript_call_result_128434 = invoke(stypy.reporting.localization.Localization(__file__, 99, 7), getitem___128433, int_128425)
    
    str_128435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 29), 'str', '{')
    # Applying the binary operator '==' (line 99)
    result_eq_128436 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 7), '==', subscript_call_result_128434, str_128435)
    
    # Testing the type of an if condition (line 99)
    if_condition_128437 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 99, 4), result_eq_128436)
    # Assigning a type to the variable 'if_condition_128437' (line 99)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'if_condition_128437', if_condition_128437)
    # SSA begins for if statement (line 99)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 100)
    # Processing the call arguments (line 100)
    str_128439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 25), 'str', 'This looks like a sparse ARFF: not supported yet')
    # Processing the call keyword arguments (line 100)
    kwargs_128440 = {}
    # Getting the type of 'ValueError' (line 100)
    ValueError_128438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 100)
    ValueError_call_result_128441 = invoke(stypy.reporting.localization.Localization(__file__, 100, 14), ValueError_128438, *[str_128439], **kwargs_128440)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 100, 8), ValueError_call_result_128441, 'raise parameter', BaseException)
    # SSA join for if statement (line 99)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to extend(...): (line 101)
    # Processing the call arguments (line 101)
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'ofile' (line 101)
    ofile_128445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 28), 'ofile', False)
    comprehension_128446 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 17), ofile_128445)
    # Assigning a type to the variable 'i' (line 101)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 17), 'i', comprehension_128446)
    # Getting the type of 'i' (line 101)
    i_128444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 17), 'i', False)
    list_128447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 17), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 17), list_128447, i_128444)
    # Processing the call keyword arguments (line 101)
    kwargs_128448 = {}
    # Getting the type of 'data' (line 101)
    data_128442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'data', False)
    # Obtaining the member 'extend' of a type (line 101)
    extend_128443 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 4), data_128442, 'extend')
    # Calling extend(args, kwargs) (line 101)
    extend_call_result_128449 = invoke(stypy.reporting.localization.Localization(__file__, 101, 4), extend_128443, *[list_128447], **kwargs_128448)
    
    # Getting the type of 'data' (line 102)
    data_128450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 11), 'data')
    # Assigning a type to the variable 'stypy_return_type' (line 102)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 4), 'stypy_return_type', data_128450)
    
    # ################# End of 'read_data_list(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'read_data_list' in the type store
    # Getting the type of 'stypy_return_type' (line 96)
    stypy_return_type_128451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_128451)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'read_data_list'
    return stypy_return_type_128451

# Assigning a type to the variable 'read_data_list' (line 96)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 0), 'read_data_list', read_data_list)

@norecursion
def get_ndata(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'get_ndata'
    module_type_store = module_type_store.open_function_context('get_ndata', 105, 0, False)
    
    # Passed parameters checking function
    get_ndata.stypy_localization = localization
    get_ndata.stypy_type_of_self = None
    get_ndata.stypy_type_store = module_type_store
    get_ndata.stypy_function_name = 'get_ndata'
    get_ndata.stypy_param_names_list = ['ofile']
    get_ndata.stypy_varargs_param_name = None
    get_ndata.stypy_kwargs_param_name = None
    get_ndata.stypy_call_defaults = defaults
    get_ndata.stypy_call_varargs = varargs
    get_ndata.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'get_ndata', ['ofile'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'get_ndata', localization, ['ofile'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'get_ndata(...)' code ##################

    str_128452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 4), 'str', 'Read the whole file to get number of data attributes.')
    
    # Assigning a List to a Name (line 107):
    
    # Assigning a List to a Name (line 107):
    
    # Obtaining an instance of the builtin type 'list' (line 107)
    list_128453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 11), 'list')
    # Adding type elements to the builtin type 'list' instance (line 107)
    # Adding element type (line 107)
    
    # Call to next(...): (line 107)
    # Processing the call arguments (line 107)
    # Getting the type of 'ofile' (line 107)
    ofile_128455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 17), 'ofile', False)
    # Processing the call keyword arguments (line 107)
    kwargs_128456 = {}
    # Getting the type of 'next' (line 107)
    next_128454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 12), 'next', False)
    # Calling next(args, kwargs) (line 107)
    next_call_result_128457 = invoke(stypy.reporting.localization.Localization(__file__, 107, 12), next_128454, *[ofile_128455], **kwargs_128456)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 11), list_128453, next_call_result_128457)
    
    # Assigning a type to the variable 'data' (line 107)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 4), 'data', list_128453)
    
    # Assigning a Num to a Name (line 108):
    
    # Assigning a Num to a Name (line 108):
    int_128458 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 10), 'int')
    # Assigning a type to the variable 'loc' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), 'loc', int_128458)
    
    
    
    # Obtaining the type of the subscript
    int_128459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 23), 'int')
    
    # Call to strip(...): (line 109)
    # Processing the call keyword arguments (line 109)
    kwargs_128465 = {}
    
    # Obtaining the type of the subscript
    int_128460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 12), 'int')
    # Getting the type of 'data' (line 109)
    data_128461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 7), 'data', False)
    # Obtaining the member '__getitem__' of a type (line 109)
    getitem___128462 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 7), data_128461, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 109)
    subscript_call_result_128463 = invoke(stypy.reporting.localization.Localization(__file__, 109, 7), getitem___128462, int_128460)
    
    # Obtaining the member 'strip' of a type (line 109)
    strip_128464 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 7), subscript_call_result_128463, 'strip')
    # Calling strip(args, kwargs) (line 109)
    strip_call_result_128466 = invoke(stypy.reporting.localization.Localization(__file__, 109, 7), strip_128464, *[], **kwargs_128465)
    
    # Obtaining the member '__getitem__' of a type (line 109)
    getitem___128467 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 7), strip_call_result_128466, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 109)
    subscript_call_result_128468 = invoke(stypy.reporting.localization.Localization(__file__, 109, 7), getitem___128467, int_128459)
    
    str_128469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 29), 'str', '{')
    # Applying the binary operator '==' (line 109)
    result_eq_128470 = python_operator(stypy.reporting.localization.Localization(__file__, 109, 7), '==', subscript_call_result_128468, str_128469)
    
    # Testing the type of an if condition (line 109)
    if_condition_128471 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 109, 4), result_eq_128470)
    # Assigning a type to the variable 'if_condition_128471' (line 109)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 4), 'if_condition_128471', if_condition_128471)
    # SSA begins for if statement (line 109)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 110)
    # Processing the call arguments (line 110)
    str_128473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 25), 'str', 'This looks like a sparse ARFF: not supported yet')
    # Processing the call keyword arguments (line 110)
    kwargs_128474 = {}
    # Getting the type of 'ValueError' (line 110)
    ValueError_128472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 110)
    ValueError_call_result_128475 = invoke(stypy.reporting.localization.Localization(__file__, 110, 14), ValueError_128472, *[str_128473], **kwargs_128474)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 110, 8), ValueError_call_result_128475, 'raise parameter', BaseException)
    # SSA join for if statement (line 109)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'ofile' (line 111)
    ofile_128476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 13), 'ofile')
    # Testing the type of a for loop iterable (line 111)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 111, 4), ofile_128476)
    # Getting the type of the for loop variable (line 111)
    for_loop_var_128477 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 111, 4), ofile_128476)
    # Assigning a type to the variable 'i' (line 111)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 4), 'i', for_loop_var_128477)
    # SSA begins for a for statement (line 111)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Getting the type of 'loc' (line 112)
    loc_128478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'loc')
    int_128479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 15), 'int')
    # Applying the binary operator '+=' (line 112)
    result_iadd_128480 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 8), '+=', loc_128478, int_128479)
    # Assigning a type to the variable 'loc' (line 112)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'loc', result_iadd_128480)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'loc' (line 113)
    loc_128481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 11), 'loc')
    # Assigning a type to the variable 'stypy_return_type' (line 113)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 4), 'stypy_return_type', loc_128481)
    
    # ################# End of 'get_ndata(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_ndata' in the type store
    # Getting the type of 'stypy_return_type' (line 105)
    stypy_return_type_128482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_128482)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_ndata'
    return stypy_return_type_128482

# Assigning a type to the variable 'get_ndata' (line 105)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 0), 'get_ndata', get_ndata)

@norecursion
def maxnomlen(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'maxnomlen'
    module_type_store = module_type_store.open_function_context('maxnomlen', 116, 0, False)
    
    # Passed parameters checking function
    maxnomlen.stypy_localization = localization
    maxnomlen.stypy_type_of_self = None
    maxnomlen.stypy_type_store = module_type_store
    maxnomlen.stypy_function_name = 'maxnomlen'
    maxnomlen.stypy_param_names_list = ['atrv']
    maxnomlen.stypy_varargs_param_name = None
    maxnomlen.stypy_kwargs_param_name = None
    maxnomlen.stypy_call_defaults = defaults
    maxnomlen.stypy_call_varargs = varargs
    maxnomlen.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'maxnomlen', ['atrv'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'maxnomlen', localization, ['atrv'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'maxnomlen(...)' code ##################

    str_128483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, (-1)), 'str', 'Given a string containing a nominal type definition, returns the\n    string len of the biggest component.\n\n    A nominal type is defined as seomthing framed between brace ({}).\n\n    Parameters\n    ----------\n    atrv : str\n       Nominal type definition\n\n    Returns\n    -------\n    slen : int\n       length of longest component\n\n    Examples\n    --------\n    maxnomlen("{floup, bouga, fl, ratata}") returns 6 (the size of\n    ratata, the longest nominal value).\n\n    >>> maxnomlen("{floup, bouga, fl, ratata}")\n    6\n    ')
    
    # Assigning a Call to a Name (line 140):
    
    # Assigning a Call to a Name (line 140):
    
    # Call to get_nom_val(...): (line 140)
    # Processing the call arguments (line 140)
    # Getting the type of 'atrv' (line 140)
    atrv_128485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 24), 'atrv', False)
    # Processing the call keyword arguments (line 140)
    kwargs_128486 = {}
    # Getting the type of 'get_nom_val' (line 140)
    get_nom_val_128484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 12), 'get_nom_val', False)
    # Calling get_nom_val(args, kwargs) (line 140)
    get_nom_val_call_result_128487 = invoke(stypy.reporting.localization.Localization(__file__, 140, 12), get_nom_val_128484, *[atrv_128485], **kwargs_128486)
    
    # Assigning a type to the variable 'nomtp' (line 140)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 4), 'nomtp', get_nom_val_call_result_128487)
    
    # Call to max(...): (line 141)
    # Processing the call arguments (line 141)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 141, 15, True)
    # Calculating comprehension expression
    # Getting the type of 'nomtp' (line 141)
    nomtp_128493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 31), 'nomtp', False)
    comprehension_128494 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 141, 15), nomtp_128493)
    # Assigning a type to the variable 'i' (line 141)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 15), 'i', comprehension_128494)
    
    # Call to len(...): (line 141)
    # Processing the call arguments (line 141)
    # Getting the type of 'i' (line 141)
    i_128490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 19), 'i', False)
    # Processing the call keyword arguments (line 141)
    kwargs_128491 = {}
    # Getting the type of 'len' (line 141)
    len_128489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 15), 'len', False)
    # Calling len(args, kwargs) (line 141)
    len_call_result_128492 = invoke(stypy.reporting.localization.Localization(__file__, 141, 15), len_128489, *[i_128490], **kwargs_128491)
    
    list_128495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 15), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 141, 15), list_128495, len_call_result_128492)
    # Processing the call keyword arguments (line 141)
    kwargs_128496 = {}
    # Getting the type of 'max' (line 141)
    max_128488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 11), 'max', False)
    # Calling max(args, kwargs) (line 141)
    max_call_result_128497 = invoke(stypy.reporting.localization.Localization(__file__, 141, 11), max_128488, *[list_128495], **kwargs_128496)
    
    # Assigning a type to the variable 'stypy_return_type' (line 141)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 4), 'stypy_return_type', max_call_result_128497)
    
    # ################# End of 'maxnomlen(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'maxnomlen' in the type store
    # Getting the type of 'stypy_return_type' (line 116)
    stypy_return_type_128498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_128498)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'maxnomlen'
    return stypy_return_type_128498

# Assigning a type to the variable 'maxnomlen' (line 116)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 0), 'maxnomlen', maxnomlen)

@norecursion
def get_nom_val(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'get_nom_val'
    module_type_store = module_type_store.open_function_context('get_nom_val', 144, 0, False)
    
    # Passed parameters checking function
    get_nom_val.stypy_localization = localization
    get_nom_val.stypy_type_of_self = None
    get_nom_val.stypy_type_store = module_type_store
    get_nom_val.stypy_function_name = 'get_nom_val'
    get_nom_val.stypy_param_names_list = ['atrv']
    get_nom_val.stypy_varargs_param_name = None
    get_nom_val.stypy_kwargs_param_name = None
    get_nom_val.stypy_call_defaults = defaults
    get_nom_val.stypy_call_varargs = varargs
    get_nom_val.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'get_nom_val', ['atrv'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'get_nom_val', localization, ['atrv'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'get_nom_val(...)' code ##################

    str_128499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, (-1)), 'str', 'Given a string containing a nominal type, returns a tuple of the\n    possible values.\n\n    A nominal type is defined as something framed between braces ({}).\n\n    Parameters\n    ----------\n    atrv : str\n       Nominal type definition\n\n    Returns\n    -------\n    poss_vals : tuple\n       possible values\n\n    Examples\n    --------\n    >>> get_nom_val("{floup, bouga, fl, ratata}")\n    (\'floup\', \'bouga\', \'fl\', \'ratata\')\n    ')
    
    # Assigning a Call to a Name (line 165):
    
    # Assigning a Call to a Name (line 165):
    
    # Call to compile(...): (line 165)
    # Processing the call arguments (line 165)
    str_128502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 27), 'str', '{(.+)}')
    # Processing the call keyword arguments (line 165)
    kwargs_128503 = {}
    # Getting the type of 're' (line 165)
    re_128500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 16), 're', False)
    # Obtaining the member 'compile' of a type (line 165)
    compile_128501 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 16), re_128500, 'compile')
    # Calling compile(args, kwargs) (line 165)
    compile_call_result_128504 = invoke(stypy.reporting.localization.Localization(__file__, 165, 16), compile_128501, *[str_128502], **kwargs_128503)
    
    # Assigning a type to the variable 'r_nominal' (line 165)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 4), 'r_nominal', compile_call_result_128504)
    
    # Assigning a Call to a Name (line 166):
    
    # Assigning a Call to a Name (line 166):
    
    # Call to match(...): (line 166)
    # Processing the call arguments (line 166)
    # Getting the type of 'atrv' (line 166)
    atrv_128507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 24), 'atrv', False)
    # Processing the call keyword arguments (line 166)
    kwargs_128508 = {}
    # Getting the type of 'r_nominal' (line 166)
    r_nominal_128505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'r_nominal', False)
    # Obtaining the member 'match' of a type (line 166)
    match_128506 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 8), r_nominal_128505, 'match')
    # Calling match(args, kwargs) (line 166)
    match_call_result_128509 = invoke(stypy.reporting.localization.Localization(__file__, 166, 8), match_128506, *[atrv_128507], **kwargs_128508)
    
    # Assigning a type to the variable 'm' (line 166)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 4), 'm', match_call_result_128509)
    
    # Getting the type of 'm' (line 167)
    m_128510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 7), 'm')
    # Testing the type of an if condition (line 167)
    if_condition_128511 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 167, 4), m_128510)
    # Assigning a type to the variable 'if_condition_128511' (line 167)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 4), 'if_condition_128511', if_condition_128511)
    # SSA begins for if statement (line 167)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to tuple(...): (line 168)
    # Processing the call arguments (line 168)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 168, 21, True)
    # Calculating comprehension expression
    
    # Call to split(...): (line 168)
    # Processing the call arguments (line 168)
    str_128523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 57), 'str', ',')
    # Processing the call keyword arguments (line 168)
    kwargs_128524 = {}
    
    # Call to group(...): (line 168)
    # Processing the call arguments (line 168)
    int_128519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 48), 'int')
    # Processing the call keyword arguments (line 168)
    kwargs_128520 = {}
    # Getting the type of 'm' (line 168)
    m_128517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 40), 'm', False)
    # Obtaining the member 'group' of a type (line 168)
    group_128518 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 40), m_128517, 'group')
    # Calling group(args, kwargs) (line 168)
    group_call_result_128521 = invoke(stypy.reporting.localization.Localization(__file__, 168, 40), group_128518, *[int_128519], **kwargs_128520)
    
    # Obtaining the member 'split' of a type (line 168)
    split_128522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 40), group_call_result_128521, 'split')
    # Calling split(args, kwargs) (line 168)
    split_call_result_128525 = invoke(stypy.reporting.localization.Localization(__file__, 168, 40), split_128522, *[str_128523], **kwargs_128524)
    
    comprehension_128526 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 21), split_call_result_128525)
    # Assigning a type to the variable 'i' (line 168)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 21), 'i', comprehension_128526)
    
    # Call to strip(...): (line 168)
    # Processing the call keyword arguments (line 168)
    kwargs_128515 = {}
    # Getting the type of 'i' (line 168)
    i_128513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 21), 'i', False)
    # Obtaining the member 'strip' of a type (line 168)
    strip_128514 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 21), i_128513, 'strip')
    # Calling strip(args, kwargs) (line 168)
    strip_call_result_128516 = invoke(stypy.reporting.localization.Localization(__file__, 168, 21), strip_128514, *[], **kwargs_128515)
    
    list_128527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 21), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 21), list_128527, strip_call_result_128516)
    # Processing the call keyword arguments (line 168)
    kwargs_128528 = {}
    # Getting the type of 'tuple' (line 168)
    tuple_128512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 15), 'tuple', False)
    # Calling tuple(args, kwargs) (line 168)
    tuple_call_result_128529 = invoke(stypy.reporting.localization.Localization(__file__, 168, 15), tuple_128512, *[list_128527], **kwargs_128528)
    
    # Assigning a type to the variable 'stypy_return_type' (line 168)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 8), 'stypy_return_type', tuple_call_result_128529)
    # SSA branch for the else part of an if statement (line 167)
    module_type_store.open_ssa_branch('else')
    
    # Call to ValueError(...): (line 170)
    # Processing the call arguments (line 170)
    str_128531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 25), 'str', 'This does not look like a nominal string')
    # Processing the call keyword arguments (line 170)
    kwargs_128532 = {}
    # Getting the type of 'ValueError' (line 170)
    ValueError_128530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 170)
    ValueError_call_result_128533 = invoke(stypy.reporting.localization.Localization(__file__, 170, 14), ValueError_128530, *[str_128531], **kwargs_128532)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 170, 8), ValueError_call_result_128533, 'raise parameter', BaseException)
    # SSA join for if statement (line 167)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'get_nom_val(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_nom_val' in the type store
    # Getting the type of 'stypy_return_type' (line 144)
    stypy_return_type_128534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_128534)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_nom_val'
    return stypy_return_type_128534

# Assigning a type to the variable 'get_nom_val' (line 144)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 0), 'get_nom_val', get_nom_val)

@norecursion
def get_date_format(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'get_date_format'
    module_type_store = module_type_store.open_function_context('get_date_format', 173, 0, False)
    
    # Passed parameters checking function
    get_date_format.stypy_localization = localization
    get_date_format.stypy_type_of_self = None
    get_date_format.stypy_type_store = module_type_store
    get_date_format.stypy_function_name = 'get_date_format'
    get_date_format.stypy_param_names_list = ['atrv']
    get_date_format.stypy_varargs_param_name = None
    get_date_format.stypy_kwargs_param_name = None
    get_date_format.stypy_call_defaults = defaults
    get_date_format.stypy_call_varargs = varargs
    get_date_format.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'get_date_format', ['atrv'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'get_date_format', localization, ['atrv'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'get_date_format(...)' code ##################

    
    # Assigning a Call to a Name (line 174):
    
    # Assigning a Call to a Name (line 174):
    
    # Call to compile(...): (line 174)
    # Processing the call arguments (line 174)
    str_128537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 24), 'str', '[Dd][Aa][Tt][Ee]\\s+[\\"\']?(.+?)[\\"\']?$')
    # Processing the call keyword arguments (line 174)
    kwargs_128538 = {}
    # Getting the type of 're' (line 174)
    re_128535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 13), 're', False)
    # Obtaining the member 'compile' of a type (line 174)
    compile_128536 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 13), re_128535, 'compile')
    # Calling compile(args, kwargs) (line 174)
    compile_call_result_128539 = invoke(stypy.reporting.localization.Localization(__file__, 174, 13), compile_128536, *[str_128537], **kwargs_128538)
    
    # Assigning a type to the variable 'r_date' (line 174)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 4), 'r_date', compile_call_result_128539)
    
    # Assigning a Call to a Name (line 175):
    
    # Assigning a Call to a Name (line 175):
    
    # Call to match(...): (line 175)
    # Processing the call arguments (line 175)
    # Getting the type of 'atrv' (line 175)
    atrv_128542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 21), 'atrv', False)
    # Processing the call keyword arguments (line 175)
    kwargs_128543 = {}
    # Getting the type of 'r_date' (line 175)
    r_date_128540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'r_date', False)
    # Obtaining the member 'match' of a type (line 175)
    match_128541 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 8), r_date_128540, 'match')
    # Calling match(args, kwargs) (line 175)
    match_call_result_128544 = invoke(stypy.reporting.localization.Localization(__file__, 175, 8), match_128541, *[atrv_128542], **kwargs_128543)
    
    # Assigning a type to the variable 'm' (line 175)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 4), 'm', match_call_result_128544)
    
    # Getting the type of 'm' (line 176)
    m_128545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 7), 'm')
    # Testing the type of an if condition (line 176)
    if_condition_128546 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 176, 4), m_128545)
    # Assigning a type to the variable 'if_condition_128546' (line 176)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 4), 'if_condition_128546', if_condition_128546)
    # SSA begins for if statement (line 176)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 177):
    
    # Assigning a Call to a Name (line 177):
    
    # Call to strip(...): (line 177)
    # Processing the call keyword arguments (line 177)
    kwargs_128553 = {}
    
    # Call to group(...): (line 177)
    # Processing the call arguments (line 177)
    int_128549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 26), 'int')
    # Processing the call keyword arguments (line 177)
    kwargs_128550 = {}
    # Getting the type of 'm' (line 177)
    m_128547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 18), 'm', False)
    # Obtaining the member 'group' of a type (line 177)
    group_128548 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 18), m_128547, 'group')
    # Calling group(args, kwargs) (line 177)
    group_call_result_128551 = invoke(stypy.reporting.localization.Localization(__file__, 177, 18), group_128548, *[int_128549], **kwargs_128550)
    
    # Obtaining the member 'strip' of a type (line 177)
    strip_128552 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 18), group_call_result_128551, 'strip')
    # Calling strip(args, kwargs) (line 177)
    strip_call_result_128554 = invoke(stypy.reporting.localization.Localization(__file__, 177, 18), strip_128552, *[], **kwargs_128553)
    
    # Assigning a type to the variable 'pattern' (line 177)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), 'pattern', strip_call_result_128554)
    
    # Assigning a Name to a Name (line 179):
    
    # Assigning a Name to a Name (line 179):
    # Getting the type of 'None' (line 179)
    None_128555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 24), 'None')
    # Assigning a type to the variable 'datetime_unit' (line 179)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 8), 'datetime_unit', None_128555)
    
    
    str_128556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 11), 'str', 'yyyy')
    # Getting the type of 'pattern' (line 180)
    pattern_128557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 21), 'pattern')
    # Applying the binary operator 'in' (line 180)
    result_contains_128558 = python_operator(stypy.reporting.localization.Localization(__file__, 180, 11), 'in', str_128556, pattern_128557)
    
    # Testing the type of an if condition (line 180)
    if_condition_128559 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 180, 8), result_contains_128558)
    # Assigning a type to the variable 'if_condition_128559' (line 180)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 8), 'if_condition_128559', if_condition_128559)
    # SSA begins for if statement (line 180)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 181):
    
    # Assigning a Call to a Name (line 181):
    
    # Call to replace(...): (line 181)
    # Processing the call arguments (line 181)
    str_128562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 38), 'str', 'yyyy')
    str_128563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 46), 'str', '%Y')
    # Processing the call keyword arguments (line 181)
    kwargs_128564 = {}
    # Getting the type of 'pattern' (line 181)
    pattern_128560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 22), 'pattern', False)
    # Obtaining the member 'replace' of a type (line 181)
    replace_128561 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 22), pattern_128560, 'replace')
    # Calling replace(args, kwargs) (line 181)
    replace_call_result_128565 = invoke(stypy.reporting.localization.Localization(__file__, 181, 22), replace_128561, *[str_128562, str_128563], **kwargs_128564)
    
    # Assigning a type to the variable 'pattern' (line 181)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 12), 'pattern', replace_call_result_128565)
    
    # Assigning a Str to a Name (line 182):
    
    # Assigning a Str to a Name (line 182):
    str_128566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 28), 'str', 'Y')
    # Assigning a type to the variable 'datetime_unit' (line 182)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 12), 'datetime_unit', str_128566)
    # SSA branch for the else part of an if statement (line 180)
    module_type_store.open_ssa_branch('else')
    
    str_128567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 13), 'str', 'yy')
    # Testing the type of an if condition (line 183)
    if_condition_128568 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 183, 13), str_128567)
    # Assigning a type to the variable 'if_condition_128568' (line 183)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 13), 'if_condition_128568', if_condition_128568)
    # SSA begins for if statement (line 183)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 184):
    
    # Assigning a Call to a Name (line 184):
    
    # Call to replace(...): (line 184)
    # Processing the call arguments (line 184)
    str_128571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 38), 'str', 'yy')
    str_128572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 44), 'str', '%y')
    # Processing the call keyword arguments (line 184)
    kwargs_128573 = {}
    # Getting the type of 'pattern' (line 184)
    pattern_128569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 22), 'pattern', False)
    # Obtaining the member 'replace' of a type (line 184)
    replace_128570 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 22), pattern_128569, 'replace')
    # Calling replace(args, kwargs) (line 184)
    replace_call_result_128574 = invoke(stypy.reporting.localization.Localization(__file__, 184, 22), replace_128570, *[str_128571, str_128572], **kwargs_128573)
    
    # Assigning a type to the variable 'pattern' (line 184)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 12), 'pattern', replace_call_result_128574)
    
    # Assigning a Str to a Name (line 185):
    
    # Assigning a Str to a Name (line 185):
    str_128575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 28), 'str', 'Y')
    # Assigning a type to the variable 'datetime_unit' (line 185)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 12), 'datetime_unit', str_128575)
    # SSA join for if statement (line 183)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 180)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    str_128576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 11), 'str', 'MM')
    # Getting the type of 'pattern' (line 186)
    pattern_128577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 19), 'pattern')
    # Applying the binary operator 'in' (line 186)
    result_contains_128578 = python_operator(stypy.reporting.localization.Localization(__file__, 186, 11), 'in', str_128576, pattern_128577)
    
    # Testing the type of an if condition (line 186)
    if_condition_128579 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 186, 8), result_contains_128578)
    # Assigning a type to the variable 'if_condition_128579' (line 186)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 8), 'if_condition_128579', if_condition_128579)
    # SSA begins for if statement (line 186)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 187):
    
    # Assigning a Call to a Name (line 187):
    
    # Call to replace(...): (line 187)
    # Processing the call arguments (line 187)
    str_128582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 38), 'str', 'MM')
    str_128583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 44), 'str', '%m')
    # Processing the call keyword arguments (line 187)
    kwargs_128584 = {}
    # Getting the type of 'pattern' (line 187)
    pattern_128580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 22), 'pattern', False)
    # Obtaining the member 'replace' of a type (line 187)
    replace_128581 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 22), pattern_128580, 'replace')
    # Calling replace(args, kwargs) (line 187)
    replace_call_result_128585 = invoke(stypy.reporting.localization.Localization(__file__, 187, 22), replace_128581, *[str_128582, str_128583], **kwargs_128584)
    
    # Assigning a type to the variable 'pattern' (line 187)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 12), 'pattern', replace_call_result_128585)
    
    # Assigning a Str to a Name (line 188):
    
    # Assigning a Str to a Name (line 188):
    str_128586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 28), 'str', 'M')
    # Assigning a type to the variable 'datetime_unit' (line 188)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 12), 'datetime_unit', str_128586)
    # SSA join for if statement (line 186)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    str_128587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 11), 'str', 'dd')
    # Getting the type of 'pattern' (line 189)
    pattern_128588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 19), 'pattern')
    # Applying the binary operator 'in' (line 189)
    result_contains_128589 = python_operator(stypy.reporting.localization.Localization(__file__, 189, 11), 'in', str_128587, pattern_128588)
    
    # Testing the type of an if condition (line 189)
    if_condition_128590 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 189, 8), result_contains_128589)
    # Assigning a type to the variable 'if_condition_128590' (line 189)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 8), 'if_condition_128590', if_condition_128590)
    # SSA begins for if statement (line 189)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 190):
    
    # Assigning a Call to a Name (line 190):
    
    # Call to replace(...): (line 190)
    # Processing the call arguments (line 190)
    str_128593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 38), 'str', 'dd')
    str_128594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 44), 'str', '%d')
    # Processing the call keyword arguments (line 190)
    kwargs_128595 = {}
    # Getting the type of 'pattern' (line 190)
    pattern_128591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 22), 'pattern', False)
    # Obtaining the member 'replace' of a type (line 190)
    replace_128592 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 22), pattern_128591, 'replace')
    # Calling replace(args, kwargs) (line 190)
    replace_call_result_128596 = invoke(stypy.reporting.localization.Localization(__file__, 190, 22), replace_128592, *[str_128593, str_128594], **kwargs_128595)
    
    # Assigning a type to the variable 'pattern' (line 190)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 12), 'pattern', replace_call_result_128596)
    
    # Assigning a Str to a Name (line 191):
    
    # Assigning a Str to a Name (line 191):
    str_128597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 28), 'str', 'D')
    # Assigning a type to the variable 'datetime_unit' (line 191)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 12), 'datetime_unit', str_128597)
    # SSA join for if statement (line 189)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    str_128598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 11), 'str', 'HH')
    # Getting the type of 'pattern' (line 192)
    pattern_128599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 19), 'pattern')
    # Applying the binary operator 'in' (line 192)
    result_contains_128600 = python_operator(stypy.reporting.localization.Localization(__file__, 192, 11), 'in', str_128598, pattern_128599)
    
    # Testing the type of an if condition (line 192)
    if_condition_128601 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 192, 8), result_contains_128600)
    # Assigning a type to the variable 'if_condition_128601' (line 192)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 8), 'if_condition_128601', if_condition_128601)
    # SSA begins for if statement (line 192)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 193):
    
    # Assigning a Call to a Name (line 193):
    
    # Call to replace(...): (line 193)
    # Processing the call arguments (line 193)
    str_128604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 38), 'str', 'HH')
    str_128605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 44), 'str', '%H')
    # Processing the call keyword arguments (line 193)
    kwargs_128606 = {}
    # Getting the type of 'pattern' (line 193)
    pattern_128602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 22), 'pattern', False)
    # Obtaining the member 'replace' of a type (line 193)
    replace_128603 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 22), pattern_128602, 'replace')
    # Calling replace(args, kwargs) (line 193)
    replace_call_result_128607 = invoke(stypy.reporting.localization.Localization(__file__, 193, 22), replace_128603, *[str_128604, str_128605], **kwargs_128606)
    
    # Assigning a type to the variable 'pattern' (line 193)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 12), 'pattern', replace_call_result_128607)
    
    # Assigning a Str to a Name (line 194):
    
    # Assigning a Str to a Name (line 194):
    str_128608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 28), 'str', 'h')
    # Assigning a type to the variable 'datetime_unit' (line 194)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 12), 'datetime_unit', str_128608)
    # SSA join for if statement (line 192)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    str_128609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 11), 'str', 'mm')
    # Getting the type of 'pattern' (line 195)
    pattern_128610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 19), 'pattern')
    # Applying the binary operator 'in' (line 195)
    result_contains_128611 = python_operator(stypy.reporting.localization.Localization(__file__, 195, 11), 'in', str_128609, pattern_128610)
    
    # Testing the type of an if condition (line 195)
    if_condition_128612 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 195, 8), result_contains_128611)
    # Assigning a type to the variable 'if_condition_128612' (line 195)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 8), 'if_condition_128612', if_condition_128612)
    # SSA begins for if statement (line 195)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 196):
    
    # Assigning a Call to a Name (line 196):
    
    # Call to replace(...): (line 196)
    # Processing the call arguments (line 196)
    str_128615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 38), 'str', 'mm')
    str_128616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 44), 'str', '%M')
    # Processing the call keyword arguments (line 196)
    kwargs_128617 = {}
    # Getting the type of 'pattern' (line 196)
    pattern_128613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 22), 'pattern', False)
    # Obtaining the member 'replace' of a type (line 196)
    replace_128614 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 22), pattern_128613, 'replace')
    # Calling replace(args, kwargs) (line 196)
    replace_call_result_128618 = invoke(stypy.reporting.localization.Localization(__file__, 196, 22), replace_128614, *[str_128615, str_128616], **kwargs_128617)
    
    # Assigning a type to the variable 'pattern' (line 196)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 12), 'pattern', replace_call_result_128618)
    
    # Assigning a Str to a Name (line 197):
    
    # Assigning a Str to a Name (line 197):
    str_128619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 28), 'str', 'm')
    # Assigning a type to the variable 'datetime_unit' (line 197)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 12), 'datetime_unit', str_128619)
    # SSA join for if statement (line 195)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    str_128620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 11), 'str', 'ss')
    # Getting the type of 'pattern' (line 198)
    pattern_128621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 19), 'pattern')
    # Applying the binary operator 'in' (line 198)
    result_contains_128622 = python_operator(stypy.reporting.localization.Localization(__file__, 198, 11), 'in', str_128620, pattern_128621)
    
    # Testing the type of an if condition (line 198)
    if_condition_128623 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 198, 8), result_contains_128622)
    # Assigning a type to the variable 'if_condition_128623' (line 198)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 8), 'if_condition_128623', if_condition_128623)
    # SSA begins for if statement (line 198)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 199):
    
    # Assigning a Call to a Name (line 199):
    
    # Call to replace(...): (line 199)
    # Processing the call arguments (line 199)
    str_128626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 38), 'str', 'ss')
    str_128627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 44), 'str', '%S')
    # Processing the call keyword arguments (line 199)
    kwargs_128628 = {}
    # Getting the type of 'pattern' (line 199)
    pattern_128624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 22), 'pattern', False)
    # Obtaining the member 'replace' of a type (line 199)
    replace_128625 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 22), pattern_128624, 'replace')
    # Calling replace(args, kwargs) (line 199)
    replace_call_result_128629 = invoke(stypy.reporting.localization.Localization(__file__, 199, 22), replace_128625, *[str_128626, str_128627], **kwargs_128628)
    
    # Assigning a type to the variable 'pattern' (line 199)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 12), 'pattern', replace_call_result_128629)
    
    # Assigning a Str to a Name (line 200):
    
    # Assigning a Str to a Name (line 200):
    str_128630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 28), 'str', 's')
    # Assigning a type to the variable 'datetime_unit' (line 200)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 12), 'datetime_unit', str_128630)
    # SSA join for if statement (line 198)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    str_128631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 11), 'str', 'z')
    # Getting the type of 'pattern' (line 201)
    pattern_128632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 18), 'pattern')
    # Applying the binary operator 'in' (line 201)
    result_contains_128633 = python_operator(stypy.reporting.localization.Localization(__file__, 201, 11), 'in', str_128631, pattern_128632)
    
    
    str_128634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 29), 'str', 'Z')
    # Getting the type of 'pattern' (line 201)
    pattern_128635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 36), 'pattern')
    # Applying the binary operator 'in' (line 201)
    result_contains_128636 = python_operator(stypy.reporting.localization.Localization(__file__, 201, 29), 'in', str_128634, pattern_128635)
    
    # Applying the binary operator 'or' (line 201)
    result_or_keyword_128637 = python_operator(stypy.reporting.localization.Localization(__file__, 201, 11), 'or', result_contains_128633, result_contains_128636)
    
    # Testing the type of an if condition (line 201)
    if_condition_128638 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 201, 8), result_or_keyword_128637)
    # Assigning a type to the variable 'if_condition_128638' (line 201)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 8), 'if_condition_128638', if_condition_128638)
    # SSA begins for if statement (line 201)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 202)
    # Processing the call arguments (line 202)
    str_128640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 29), 'str', 'Date type attributes with time zone not supported, yet')
    # Processing the call keyword arguments (line 202)
    kwargs_128641 = {}
    # Getting the type of 'ValueError' (line 202)
    ValueError_128639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 202)
    ValueError_call_result_128642 = invoke(stypy.reporting.localization.Localization(__file__, 202, 18), ValueError_128639, *[str_128640], **kwargs_128641)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 202, 12), ValueError_call_result_128642, 'raise parameter', BaseException)
    # SSA join for if statement (line 201)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 205)
    # Getting the type of 'datetime_unit' (line 205)
    datetime_unit_128643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 11), 'datetime_unit')
    # Getting the type of 'None' (line 205)
    None_128644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 28), 'None')
    
    (may_be_128645, more_types_in_union_128646) = may_be_none(datetime_unit_128643, None_128644)

    if may_be_128645:

        if more_types_in_union_128646:
            # Runtime conditional SSA (line 205)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Call to ValueError(...): (line 206)
        # Processing the call arguments (line 206)
        str_128648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 29), 'str', 'Invalid or unsupported date format')
        # Processing the call keyword arguments (line 206)
        kwargs_128649 = {}
        # Getting the type of 'ValueError' (line 206)
        ValueError_128647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 206)
        ValueError_call_result_128650 = invoke(stypy.reporting.localization.Localization(__file__, 206, 18), ValueError_128647, *[str_128648], **kwargs_128649)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 206, 12), ValueError_call_result_128650, 'raise parameter', BaseException)

        if more_types_in_union_128646:
            # SSA join for if statement (line 205)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Obtaining an instance of the builtin type 'tuple' (line 208)
    tuple_128651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 208)
    # Adding element type (line 208)
    # Getting the type of 'pattern' (line 208)
    pattern_128652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 15), 'pattern')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 15), tuple_128651, pattern_128652)
    # Adding element type (line 208)
    # Getting the type of 'datetime_unit' (line 208)
    datetime_unit_128653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 24), 'datetime_unit')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 15), tuple_128651, datetime_unit_128653)
    
    # Assigning a type to the variable 'stypy_return_type' (line 208)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 8), 'stypy_return_type', tuple_128651)
    # SSA branch for the else part of an if statement (line 176)
    module_type_store.open_ssa_branch('else')
    
    # Call to ValueError(...): (line 210)
    # Processing the call arguments (line 210)
    str_128655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 25), 'str', 'Invalid or no date format')
    # Processing the call keyword arguments (line 210)
    kwargs_128656 = {}
    # Getting the type of 'ValueError' (line 210)
    ValueError_128654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 210)
    ValueError_call_result_128657 = invoke(stypy.reporting.localization.Localization(__file__, 210, 14), ValueError_128654, *[str_128655], **kwargs_128656)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 210, 8), ValueError_call_result_128657, 'raise parameter', BaseException)
    # SSA join for if statement (line 176)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'get_date_format(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_date_format' in the type store
    # Getting the type of 'stypy_return_type' (line 173)
    stypy_return_type_128658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_128658)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_date_format'
    return stypy_return_type_128658

# Assigning a type to the variable 'get_date_format' (line 173)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 0), 'get_date_format', get_date_format)

@norecursion
def go_data(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'go_data'
    module_type_store = module_type_store.open_function_context('go_data', 213, 0, False)
    
    # Passed parameters checking function
    go_data.stypy_localization = localization
    go_data.stypy_type_of_self = None
    go_data.stypy_type_store = module_type_store
    go_data.stypy_function_name = 'go_data'
    go_data.stypy_param_names_list = ['ofile']
    go_data.stypy_varargs_param_name = None
    go_data.stypy_kwargs_param_name = None
    go_data.stypy_call_defaults = defaults
    go_data.stypy_call_varargs = varargs
    go_data.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'go_data', ['ofile'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'go_data', localization, ['ofile'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'go_data(...)' code ##################

    str_128659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, (-1)), 'str', 'Skip header.\n\n    the first next() call of the returned iterator will be the @data line')
    
    # Call to dropwhile(...): (line 217)
    # Processing the call arguments (line 217)

    @norecursion
    def _stypy_temp_lambda_86(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_86'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_86', 217, 31, True)
        # Passed parameters checking function
        _stypy_temp_lambda_86.stypy_localization = localization
        _stypy_temp_lambda_86.stypy_type_of_self = None
        _stypy_temp_lambda_86.stypy_type_store = module_type_store
        _stypy_temp_lambda_86.stypy_function_name = '_stypy_temp_lambda_86'
        _stypy_temp_lambda_86.stypy_param_names_list = ['x']
        _stypy_temp_lambda_86.stypy_varargs_param_name = None
        _stypy_temp_lambda_86.stypy_kwargs_param_name = None
        _stypy_temp_lambda_86.stypy_call_defaults = defaults
        _stypy_temp_lambda_86.stypy_call_varargs = varargs
        _stypy_temp_lambda_86.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_86', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Stacktrace push for error reporting
        localization.set_stack_trace('_stypy_temp_lambda_86', ['x'], arguments)
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of the lambda function code ##################

        
        
        # Call to match(...): (line 217)
        # Processing the call arguments (line 217)
        # Getting the type of 'x' (line 217)
        x_128664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 62), 'x', False)
        # Processing the call keyword arguments (line 217)
        kwargs_128665 = {}
        # Getting the type of 'r_datameta' (line 217)
        r_datameta_128662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 45), 'r_datameta', False)
        # Obtaining the member 'match' of a type (line 217)
        match_128663 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 45), r_datameta_128662, 'match')
        # Calling match(args, kwargs) (line 217)
        match_call_result_128666 = invoke(stypy.reporting.localization.Localization(__file__, 217, 45), match_128663, *[x_128664], **kwargs_128665)
        
        # Applying the 'not' unary operator (line 217)
        result_not__128667 = python_operator(stypy.reporting.localization.Localization(__file__, 217, 41), 'not', match_call_result_128666)
        
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 217)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 31), 'stypy_return_type', result_not__128667)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_86' in the type store
        # Getting the type of 'stypy_return_type' (line 217)
        stypy_return_type_128668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 31), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_128668)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_86'
        return stypy_return_type_128668

    # Assigning a type to the variable '_stypy_temp_lambda_86' (line 217)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 31), '_stypy_temp_lambda_86', _stypy_temp_lambda_86)
    # Getting the type of '_stypy_temp_lambda_86' (line 217)
    _stypy_temp_lambda_86_128669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 31), '_stypy_temp_lambda_86')
    # Getting the type of 'ofile' (line 217)
    ofile_128670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 66), 'ofile', False)
    # Processing the call keyword arguments (line 217)
    kwargs_128671 = {}
    # Getting the type of 'itertools' (line 217)
    itertools_128660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 11), 'itertools', False)
    # Obtaining the member 'dropwhile' of a type (line 217)
    dropwhile_128661 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 11), itertools_128660, 'dropwhile')
    # Calling dropwhile(args, kwargs) (line 217)
    dropwhile_call_result_128672 = invoke(stypy.reporting.localization.Localization(__file__, 217, 11), dropwhile_128661, *[_stypy_temp_lambda_86_128669, ofile_128670], **kwargs_128671)
    
    # Assigning a type to the variable 'stypy_return_type' (line 217)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 4), 'stypy_return_type', dropwhile_call_result_128672)
    
    # ################# End of 'go_data(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'go_data' in the type store
    # Getting the type of 'stypy_return_type' (line 213)
    stypy_return_type_128673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_128673)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'go_data'
    return stypy_return_type_128673

# Assigning a type to the variable 'go_data' (line 213)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 0), 'go_data', go_data)

@norecursion
def tokenize_attribute(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'tokenize_attribute'
    module_type_store = module_type_store.open_function_context('tokenize_attribute', 223, 0, False)
    
    # Passed parameters checking function
    tokenize_attribute.stypy_localization = localization
    tokenize_attribute.stypy_type_of_self = None
    tokenize_attribute.stypy_type_store = module_type_store
    tokenize_attribute.stypy_function_name = 'tokenize_attribute'
    tokenize_attribute.stypy_param_names_list = ['iterable', 'attribute']
    tokenize_attribute.stypy_varargs_param_name = None
    tokenize_attribute.stypy_kwargs_param_name = None
    tokenize_attribute.stypy_call_defaults = defaults
    tokenize_attribute.stypy_call_varargs = varargs
    tokenize_attribute.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'tokenize_attribute', ['iterable', 'attribute'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'tokenize_attribute', localization, ['iterable', 'attribute'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'tokenize_attribute(...)' code ##################

    str_128674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, (-1)), 'str', 'Parse a raw string in header (eg starts by @attribute).\n\n    Given a raw string attribute, try to get the name and type of the\n    attribute. Constraints:\n\n    * The first line must start with @attribute (case insensitive, and\n      space like characters before @attribute are allowed)\n    * Works also if the attribute is spread on multilines.\n    * Works if empty lines or comments are in between\n\n    Parameters\n    ----------\n    attribute : str\n       the attribute string.\n\n    Returns\n    -------\n    name : str\n       name of the attribute\n    value : str\n       value of the attribute\n    next : str\n       next line to be parsed\n\n    Examples\n    --------\n    If attribute is a string defined in python as r"floupi real", will\n    return floupi as name, and real as value.\n\n    >>> iterable = iter([0] * 10) # dummy iterator\n    >>> tokenize_attribute(iterable, r"@attribute floupi real")\n    (\'floupi\', \'real\', 0)\n\n    If attribute is r"\'floupi 2\' real", will return \'floupi 2\' as name,\n    and real as value.\n\n    >>> tokenize_attribute(iterable, r"  @attribute \'floupi 2\' real   ")\n    (\'floupi 2\', \'real\', 0)\n\n    ')
    
    # Assigning a Call to a Name (line 264):
    
    # Assigning a Call to a Name (line 264):
    
    # Call to strip(...): (line 264)
    # Processing the call keyword arguments (line 264)
    kwargs_128677 = {}
    # Getting the type of 'attribute' (line 264)
    attribute_128675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 12), 'attribute', False)
    # Obtaining the member 'strip' of a type (line 264)
    strip_128676 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 12), attribute_128675, 'strip')
    # Calling strip(args, kwargs) (line 264)
    strip_call_result_128678 = invoke(stypy.reporting.localization.Localization(__file__, 264, 12), strip_128676, *[], **kwargs_128677)
    
    # Assigning a type to the variable 'sattr' (line 264)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 4), 'sattr', strip_call_result_128678)
    
    # Assigning a Call to a Name (line 265):
    
    # Assigning a Call to a Name (line 265):
    
    # Call to match(...): (line 265)
    # Processing the call arguments (line 265)
    # Getting the type of 'sattr' (line 265)
    sattr_128681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 30), 'sattr', False)
    # Processing the call keyword arguments (line 265)
    kwargs_128682 = {}
    # Getting the type of 'r_attribute' (line 265)
    r_attribute_128679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 12), 'r_attribute', False)
    # Obtaining the member 'match' of a type (line 265)
    match_128680 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 12), r_attribute_128679, 'match')
    # Calling match(args, kwargs) (line 265)
    match_call_result_128683 = invoke(stypy.reporting.localization.Localization(__file__, 265, 12), match_128680, *[sattr_128681], **kwargs_128682)
    
    # Assigning a type to the variable 'mattr' (line 265)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 4), 'mattr', match_call_result_128683)
    
    # Getting the type of 'mattr' (line 266)
    mattr_128684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 7), 'mattr')
    # Testing the type of an if condition (line 266)
    if_condition_128685 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 266, 4), mattr_128684)
    # Assigning a type to the variable 'if_condition_128685' (line 266)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 266, 4), 'if_condition_128685', if_condition_128685)
    # SSA begins for if statement (line 266)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 268):
    
    # Assigning a Call to a Name (line 268):
    
    # Call to group(...): (line 268)
    # Processing the call arguments (line 268)
    int_128688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 27), 'int')
    # Processing the call keyword arguments (line 268)
    kwargs_128689 = {}
    # Getting the type of 'mattr' (line 268)
    mattr_128686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 15), 'mattr', False)
    # Obtaining the member 'group' of a type (line 268)
    group_128687 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 15), mattr_128686, 'group')
    # Calling group(args, kwargs) (line 268)
    group_call_result_128690 = invoke(stypy.reporting.localization.Localization(__file__, 268, 15), group_128687, *[int_128688], **kwargs_128689)
    
    # Assigning a type to the variable 'atrv' (line 268)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 8), 'atrv', group_call_result_128690)
    
    
    # Call to match(...): (line 269)
    # Processing the call arguments (line 269)
    # Getting the type of 'atrv' (line 269)
    atrv_128693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 30), 'atrv', False)
    # Processing the call keyword arguments (line 269)
    kwargs_128694 = {}
    # Getting the type of 'r_comattrval' (line 269)
    r_comattrval_128691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 11), 'r_comattrval', False)
    # Obtaining the member 'match' of a type (line 269)
    match_128692 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 11), r_comattrval_128691, 'match')
    # Calling match(args, kwargs) (line 269)
    match_call_result_128695 = invoke(stypy.reporting.localization.Localization(__file__, 269, 11), match_128692, *[atrv_128693], **kwargs_128694)
    
    # Testing the type of an if condition (line 269)
    if_condition_128696 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 269, 8), match_call_result_128695)
    # Assigning a type to the variable 'if_condition_128696' (line 269)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 8), 'if_condition_128696', if_condition_128696)
    # SSA begins for if statement (line 269)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Tuple (line 270):
    
    # Assigning a Subscript to a Name (line 270):
    
    # Obtaining the type of the subscript
    int_128697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 12), 'int')
    
    # Call to tokenize_single_comma(...): (line 270)
    # Processing the call arguments (line 270)
    # Getting the type of 'atrv' (line 270)
    atrv_128699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 47), 'atrv', False)
    # Processing the call keyword arguments (line 270)
    kwargs_128700 = {}
    # Getting the type of 'tokenize_single_comma' (line 270)
    tokenize_single_comma_128698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 25), 'tokenize_single_comma', False)
    # Calling tokenize_single_comma(args, kwargs) (line 270)
    tokenize_single_comma_call_result_128701 = invoke(stypy.reporting.localization.Localization(__file__, 270, 25), tokenize_single_comma_128698, *[atrv_128699], **kwargs_128700)
    
    # Obtaining the member '__getitem__' of a type (line 270)
    getitem___128702 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 12), tokenize_single_comma_call_result_128701, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 270)
    subscript_call_result_128703 = invoke(stypy.reporting.localization.Localization(__file__, 270, 12), getitem___128702, int_128697)
    
    # Assigning a type to the variable 'tuple_var_assignment_128243' (line 270)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 270, 12), 'tuple_var_assignment_128243', subscript_call_result_128703)
    
    # Assigning a Subscript to a Name (line 270):
    
    # Obtaining the type of the subscript
    int_128704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 12), 'int')
    
    # Call to tokenize_single_comma(...): (line 270)
    # Processing the call arguments (line 270)
    # Getting the type of 'atrv' (line 270)
    atrv_128706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 47), 'atrv', False)
    # Processing the call keyword arguments (line 270)
    kwargs_128707 = {}
    # Getting the type of 'tokenize_single_comma' (line 270)
    tokenize_single_comma_128705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 25), 'tokenize_single_comma', False)
    # Calling tokenize_single_comma(args, kwargs) (line 270)
    tokenize_single_comma_call_result_128708 = invoke(stypy.reporting.localization.Localization(__file__, 270, 25), tokenize_single_comma_128705, *[atrv_128706], **kwargs_128707)
    
    # Obtaining the member '__getitem__' of a type (line 270)
    getitem___128709 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 12), tokenize_single_comma_call_result_128708, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 270)
    subscript_call_result_128710 = invoke(stypy.reporting.localization.Localization(__file__, 270, 12), getitem___128709, int_128704)
    
    # Assigning a type to the variable 'tuple_var_assignment_128244' (line 270)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 270, 12), 'tuple_var_assignment_128244', subscript_call_result_128710)
    
    # Assigning a Name to a Name (line 270):
    # Getting the type of 'tuple_var_assignment_128243' (line 270)
    tuple_var_assignment_128243_128711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 12), 'tuple_var_assignment_128243')
    # Assigning a type to the variable 'name' (line 270)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 270, 12), 'name', tuple_var_assignment_128243_128711)
    
    # Assigning a Name to a Name (line 270):
    # Getting the type of 'tuple_var_assignment_128244' (line 270)
    tuple_var_assignment_128244_128712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 12), 'tuple_var_assignment_128244')
    # Assigning a type to the variable 'type' (line 270)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 270, 18), 'type', tuple_var_assignment_128244_128712)
    
    # Assigning a Call to a Name (line 271):
    
    # Assigning a Call to a Name (line 271):
    
    # Call to next(...): (line 271)
    # Processing the call arguments (line 271)
    # Getting the type of 'iterable' (line 271)
    iterable_128714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 29), 'iterable', False)
    # Processing the call keyword arguments (line 271)
    kwargs_128715 = {}
    # Getting the type of 'next' (line 271)
    next_128713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 24), 'next', False)
    # Calling next(args, kwargs) (line 271)
    next_call_result_128716 = invoke(stypy.reporting.localization.Localization(__file__, 271, 24), next_128713, *[iterable_128714], **kwargs_128715)
    
    # Assigning a type to the variable 'next_item' (line 271)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 12), 'next_item', next_call_result_128716)
    # SSA branch for the else part of an if statement (line 269)
    module_type_store.open_ssa_branch('else')
    
    
    # Call to match(...): (line 272)
    # Processing the call arguments (line 272)
    # Getting the type of 'atrv' (line 272)
    atrv_128719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 33), 'atrv', False)
    # Processing the call keyword arguments (line 272)
    kwargs_128720 = {}
    # Getting the type of 'r_wcomattrval' (line 272)
    r_wcomattrval_128717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 13), 'r_wcomattrval', False)
    # Obtaining the member 'match' of a type (line 272)
    match_128718 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 13), r_wcomattrval_128717, 'match')
    # Calling match(args, kwargs) (line 272)
    match_call_result_128721 = invoke(stypy.reporting.localization.Localization(__file__, 272, 13), match_128718, *[atrv_128719], **kwargs_128720)
    
    # Testing the type of an if condition (line 272)
    if_condition_128722 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 272, 13), match_call_result_128721)
    # Assigning a type to the variable 'if_condition_128722' (line 272)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 13), 'if_condition_128722', if_condition_128722)
    # SSA begins for if statement (line 272)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Tuple (line 273):
    
    # Assigning a Subscript to a Name (line 273):
    
    # Obtaining the type of the subscript
    int_128723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 12), 'int')
    
    # Call to tokenize_single_wcomma(...): (line 273)
    # Processing the call arguments (line 273)
    # Getting the type of 'atrv' (line 273)
    atrv_128725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 48), 'atrv', False)
    # Processing the call keyword arguments (line 273)
    kwargs_128726 = {}
    # Getting the type of 'tokenize_single_wcomma' (line 273)
    tokenize_single_wcomma_128724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 25), 'tokenize_single_wcomma', False)
    # Calling tokenize_single_wcomma(args, kwargs) (line 273)
    tokenize_single_wcomma_call_result_128727 = invoke(stypy.reporting.localization.Localization(__file__, 273, 25), tokenize_single_wcomma_128724, *[atrv_128725], **kwargs_128726)
    
    # Obtaining the member '__getitem__' of a type (line 273)
    getitem___128728 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 12), tokenize_single_wcomma_call_result_128727, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 273)
    subscript_call_result_128729 = invoke(stypy.reporting.localization.Localization(__file__, 273, 12), getitem___128728, int_128723)
    
    # Assigning a type to the variable 'tuple_var_assignment_128245' (line 273)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 12), 'tuple_var_assignment_128245', subscript_call_result_128729)
    
    # Assigning a Subscript to a Name (line 273):
    
    # Obtaining the type of the subscript
    int_128730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 12), 'int')
    
    # Call to tokenize_single_wcomma(...): (line 273)
    # Processing the call arguments (line 273)
    # Getting the type of 'atrv' (line 273)
    atrv_128732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 48), 'atrv', False)
    # Processing the call keyword arguments (line 273)
    kwargs_128733 = {}
    # Getting the type of 'tokenize_single_wcomma' (line 273)
    tokenize_single_wcomma_128731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 25), 'tokenize_single_wcomma', False)
    # Calling tokenize_single_wcomma(args, kwargs) (line 273)
    tokenize_single_wcomma_call_result_128734 = invoke(stypy.reporting.localization.Localization(__file__, 273, 25), tokenize_single_wcomma_128731, *[atrv_128732], **kwargs_128733)
    
    # Obtaining the member '__getitem__' of a type (line 273)
    getitem___128735 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 12), tokenize_single_wcomma_call_result_128734, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 273)
    subscript_call_result_128736 = invoke(stypy.reporting.localization.Localization(__file__, 273, 12), getitem___128735, int_128730)
    
    # Assigning a type to the variable 'tuple_var_assignment_128246' (line 273)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 12), 'tuple_var_assignment_128246', subscript_call_result_128736)
    
    # Assigning a Name to a Name (line 273):
    # Getting the type of 'tuple_var_assignment_128245' (line 273)
    tuple_var_assignment_128245_128737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 12), 'tuple_var_assignment_128245')
    # Assigning a type to the variable 'name' (line 273)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 12), 'name', tuple_var_assignment_128245_128737)
    
    # Assigning a Name to a Name (line 273):
    # Getting the type of 'tuple_var_assignment_128246' (line 273)
    tuple_var_assignment_128246_128738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 12), 'tuple_var_assignment_128246')
    # Assigning a type to the variable 'type' (line 273)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 18), 'type', tuple_var_assignment_128246_128738)
    
    # Assigning a Call to a Name (line 274):
    
    # Assigning a Call to a Name (line 274):
    
    # Call to next(...): (line 274)
    # Processing the call arguments (line 274)
    # Getting the type of 'iterable' (line 274)
    iterable_128740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 29), 'iterable', False)
    # Processing the call keyword arguments (line 274)
    kwargs_128741 = {}
    # Getting the type of 'next' (line 274)
    next_128739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 24), 'next', False)
    # Calling next(args, kwargs) (line 274)
    next_call_result_128742 = invoke(stypy.reporting.localization.Localization(__file__, 274, 24), next_128739, *[iterable_128740], **kwargs_128741)
    
    # Assigning a type to the variable 'next_item' (line 274)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 12), 'next_item', next_call_result_128742)
    # SSA branch for the else part of an if statement (line 272)
    module_type_store.open_ssa_branch('else')
    
    # Call to ValueError(...): (line 278)
    # Processing the call arguments (line 278)
    str_128744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 29), 'str', 'multi line not supported yet')
    # Processing the call keyword arguments (line 278)
    kwargs_128745 = {}
    # Getting the type of 'ValueError' (line 278)
    ValueError_128743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 278)
    ValueError_call_result_128746 = invoke(stypy.reporting.localization.Localization(__file__, 278, 18), ValueError_128743, *[str_128744], **kwargs_128745)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 278, 12), ValueError_call_result_128746, 'raise parameter', BaseException)
    # SSA join for if statement (line 272)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 269)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 266)
    module_type_store.open_ssa_branch('else')
    
    # Call to ValueError(...): (line 281)
    # Processing the call arguments (line 281)
    str_128748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 25), 'str', 'First line unparsable: %s')
    # Getting the type of 'sattr' (line 281)
    sattr_128749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 55), 'sattr', False)
    # Applying the binary operator '%' (line 281)
    result_mod_128750 = python_operator(stypy.reporting.localization.Localization(__file__, 281, 25), '%', str_128748, sattr_128749)
    
    # Processing the call keyword arguments (line 281)
    kwargs_128751 = {}
    # Getting the type of 'ValueError' (line 281)
    ValueError_128747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 281)
    ValueError_call_result_128752 = invoke(stypy.reporting.localization.Localization(__file__, 281, 14), ValueError_128747, *[result_mod_128750], **kwargs_128751)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 281, 8), ValueError_call_result_128752, 'raise parameter', BaseException)
    # SSA join for if statement (line 266)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'type' (line 283)
    type_128753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 7), 'type')
    str_128754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 15), 'str', 'relational')
    # Applying the binary operator '==' (line 283)
    result_eq_128755 = python_operator(stypy.reporting.localization.Localization(__file__, 283, 7), '==', type_128753, str_128754)
    
    # Testing the type of an if condition (line 283)
    if_condition_128756 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 283, 4), result_eq_128755)
    # Assigning a type to the variable 'if_condition_128756' (line 283)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 4), 'if_condition_128756', if_condition_128756)
    # SSA begins for if statement (line 283)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 284)
    # Processing the call arguments (line 284)
    str_128758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 25), 'str', 'relational attributes not supported yet')
    # Processing the call keyword arguments (line 284)
    kwargs_128759 = {}
    # Getting the type of 'ValueError' (line 284)
    ValueError_128757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 284)
    ValueError_call_result_128760 = invoke(stypy.reporting.localization.Localization(__file__, 284, 14), ValueError_128757, *[str_128758], **kwargs_128759)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 284, 8), ValueError_call_result_128760, 'raise parameter', BaseException)
    # SSA join for if statement (line 283)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 285)
    tuple_128761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 285)
    # Adding element type (line 285)
    # Getting the type of 'name' (line 285)
    name_128762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 11), 'name')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 285, 11), tuple_128761, name_128762)
    # Adding element type (line 285)
    # Getting the type of 'type' (line 285)
    type_128763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 17), 'type')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 285, 11), tuple_128761, type_128763)
    # Adding element type (line 285)
    # Getting the type of 'next_item' (line 285)
    next_item_128764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 23), 'next_item')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 285, 11), tuple_128761, next_item_128764)
    
    # Assigning a type to the variable 'stypy_return_type' (line 285)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 4), 'stypy_return_type', tuple_128761)
    
    # ################# End of 'tokenize_attribute(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'tokenize_attribute' in the type store
    # Getting the type of 'stypy_return_type' (line 223)
    stypy_return_type_128765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_128765)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'tokenize_attribute'
    return stypy_return_type_128765

# Assigning a type to the variable 'tokenize_attribute' (line 223)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 0), 'tokenize_attribute', tokenize_attribute)

@norecursion
def tokenize_single_comma(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'tokenize_single_comma'
    module_type_store = module_type_store.open_function_context('tokenize_single_comma', 288, 0, False)
    
    # Passed parameters checking function
    tokenize_single_comma.stypy_localization = localization
    tokenize_single_comma.stypy_type_of_self = None
    tokenize_single_comma.stypy_type_store = module_type_store
    tokenize_single_comma.stypy_function_name = 'tokenize_single_comma'
    tokenize_single_comma.stypy_param_names_list = ['val']
    tokenize_single_comma.stypy_varargs_param_name = None
    tokenize_single_comma.stypy_kwargs_param_name = None
    tokenize_single_comma.stypy_call_defaults = defaults
    tokenize_single_comma.stypy_call_varargs = varargs
    tokenize_single_comma.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'tokenize_single_comma', ['val'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'tokenize_single_comma', localization, ['val'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'tokenize_single_comma(...)' code ##################

    
    # Assigning a Call to a Name (line 291):
    
    # Assigning a Call to a Name (line 291):
    
    # Call to match(...): (line 291)
    # Processing the call arguments (line 291)
    # Getting the type of 'val' (line 291)
    val_128768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 27), 'val', False)
    # Processing the call keyword arguments (line 291)
    kwargs_128769 = {}
    # Getting the type of 'r_comattrval' (line 291)
    r_comattrval_128766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 8), 'r_comattrval', False)
    # Obtaining the member 'match' of a type (line 291)
    match_128767 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 8), r_comattrval_128766, 'match')
    # Calling match(args, kwargs) (line 291)
    match_call_result_128770 = invoke(stypy.reporting.localization.Localization(__file__, 291, 8), match_128767, *[val_128768], **kwargs_128769)
    
    # Assigning a type to the variable 'm' (line 291)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 4), 'm', match_call_result_128770)
    
    # Getting the type of 'm' (line 292)
    m_128771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 7), 'm')
    # Testing the type of an if condition (line 292)
    if_condition_128772 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 292, 4), m_128771)
    # Assigning a type to the variable 'if_condition_128772' (line 292)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 4), 'if_condition_128772', if_condition_128772)
    # SSA begins for if statement (line 292)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # SSA begins for try-except statement (line 293)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Name (line 294):
    
    # Assigning a Call to a Name (line 294):
    
    # Call to strip(...): (line 294)
    # Processing the call keyword arguments (line 294)
    kwargs_128779 = {}
    
    # Call to group(...): (line 294)
    # Processing the call arguments (line 294)
    int_128775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 27), 'int')
    # Processing the call keyword arguments (line 294)
    kwargs_128776 = {}
    # Getting the type of 'm' (line 294)
    m_128773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 19), 'm', False)
    # Obtaining the member 'group' of a type (line 294)
    group_128774 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 19), m_128773, 'group')
    # Calling group(args, kwargs) (line 294)
    group_call_result_128777 = invoke(stypy.reporting.localization.Localization(__file__, 294, 19), group_128774, *[int_128775], **kwargs_128776)
    
    # Obtaining the member 'strip' of a type (line 294)
    strip_128778 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 19), group_call_result_128777, 'strip')
    # Calling strip(args, kwargs) (line 294)
    strip_call_result_128780 = invoke(stypy.reporting.localization.Localization(__file__, 294, 19), strip_128778, *[], **kwargs_128779)
    
    # Assigning a type to the variable 'name' (line 294)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 12), 'name', strip_call_result_128780)
    
    # Assigning a Call to a Name (line 295):
    
    # Assigning a Call to a Name (line 295):
    
    # Call to strip(...): (line 295)
    # Processing the call keyword arguments (line 295)
    kwargs_128787 = {}
    
    # Call to group(...): (line 295)
    # Processing the call arguments (line 295)
    int_128783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 27), 'int')
    # Processing the call keyword arguments (line 295)
    kwargs_128784 = {}
    # Getting the type of 'm' (line 295)
    m_128781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 19), 'm', False)
    # Obtaining the member 'group' of a type (line 295)
    group_128782 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 19), m_128781, 'group')
    # Calling group(args, kwargs) (line 295)
    group_call_result_128785 = invoke(stypy.reporting.localization.Localization(__file__, 295, 19), group_128782, *[int_128783], **kwargs_128784)
    
    # Obtaining the member 'strip' of a type (line 295)
    strip_128786 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 19), group_call_result_128785, 'strip')
    # Calling strip(args, kwargs) (line 295)
    strip_call_result_128788 = invoke(stypy.reporting.localization.Localization(__file__, 295, 19), strip_128786, *[], **kwargs_128787)
    
    # Assigning a type to the variable 'type' (line 295)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 12), 'type', strip_call_result_128788)
    # SSA branch for the except part of a try statement (line 293)
    # SSA branch for the except 'IndexError' branch of a try statement (line 293)
    module_type_store.open_ssa_branch('except')
    
    # Call to ValueError(...): (line 297)
    # Processing the call arguments (line 297)
    str_128790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, 29), 'str', 'Error while tokenizing attribute')
    # Processing the call keyword arguments (line 297)
    kwargs_128791 = {}
    # Getting the type of 'ValueError' (line 297)
    ValueError_128789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 297)
    ValueError_call_result_128792 = invoke(stypy.reporting.localization.Localization(__file__, 297, 18), ValueError_128789, *[str_128790], **kwargs_128791)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 297, 12), ValueError_call_result_128792, 'raise parameter', BaseException)
    # SSA join for try-except statement (line 293)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 292)
    module_type_store.open_ssa_branch('else')
    
    # Call to ValueError(...): (line 299)
    # Processing the call arguments (line 299)
    str_128794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 25), 'str', 'Error while tokenizing single %s')
    # Getting the type of 'val' (line 299)
    val_128795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 62), 'val', False)
    # Applying the binary operator '%' (line 299)
    result_mod_128796 = python_operator(stypy.reporting.localization.Localization(__file__, 299, 25), '%', str_128794, val_128795)
    
    # Processing the call keyword arguments (line 299)
    kwargs_128797 = {}
    # Getting the type of 'ValueError' (line 299)
    ValueError_128793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 299)
    ValueError_call_result_128798 = invoke(stypy.reporting.localization.Localization(__file__, 299, 14), ValueError_128793, *[result_mod_128796], **kwargs_128797)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 299, 8), ValueError_call_result_128798, 'raise parameter', BaseException)
    # SSA join for if statement (line 292)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 300)
    tuple_128799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 300)
    # Adding element type (line 300)
    # Getting the type of 'name' (line 300)
    name_128800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 11), 'name')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 300, 11), tuple_128799, name_128800)
    # Adding element type (line 300)
    # Getting the type of 'type' (line 300)
    type_128801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 17), 'type')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 300, 11), tuple_128799, type_128801)
    
    # Assigning a type to the variable 'stypy_return_type' (line 300)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 4), 'stypy_return_type', tuple_128799)
    
    # ################# End of 'tokenize_single_comma(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'tokenize_single_comma' in the type store
    # Getting the type of 'stypy_return_type' (line 288)
    stypy_return_type_128802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_128802)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'tokenize_single_comma'
    return stypy_return_type_128802

# Assigning a type to the variable 'tokenize_single_comma' (line 288)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 0), 'tokenize_single_comma', tokenize_single_comma)

@norecursion
def tokenize_single_wcomma(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'tokenize_single_wcomma'
    module_type_store = module_type_store.open_function_context('tokenize_single_wcomma', 303, 0, False)
    
    # Passed parameters checking function
    tokenize_single_wcomma.stypy_localization = localization
    tokenize_single_wcomma.stypy_type_of_self = None
    tokenize_single_wcomma.stypy_type_store = module_type_store
    tokenize_single_wcomma.stypy_function_name = 'tokenize_single_wcomma'
    tokenize_single_wcomma.stypy_param_names_list = ['val']
    tokenize_single_wcomma.stypy_varargs_param_name = None
    tokenize_single_wcomma.stypy_kwargs_param_name = None
    tokenize_single_wcomma.stypy_call_defaults = defaults
    tokenize_single_wcomma.stypy_call_varargs = varargs
    tokenize_single_wcomma.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'tokenize_single_wcomma', ['val'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'tokenize_single_wcomma', localization, ['val'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'tokenize_single_wcomma(...)' code ##################

    
    # Assigning a Call to a Name (line 306):
    
    # Assigning a Call to a Name (line 306):
    
    # Call to match(...): (line 306)
    # Processing the call arguments (line 306)
    # Getting the type of 'val' (line 306)
    val_128805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 28), 'val', False)
    # Processing the call keyword arguments (line 306)
    kwargs_128806 = {}
    # Getting the type of 'r_wcomattrval' (line 306)
    r_wcomattrval_128803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 8), 'r_wcomattrval', False)
    # Obtaining the member 'match' of a type (line 306)
    match_128804 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 8), r_wcomattrval_128803, 'match')
    # Calling match(args, kwargs) (line 306)
    match_call_result_128807 = invoke(stypy.reporting.localization.Localization(__file__, 306, 8), match_128804, *[val_128805], **kwargs_128806)
    
    # Assigning a type to the variable 'm' (line 306)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 4), 'm', match_call_result_128807)
    
    # Getting the type of 'm' (line 307)
    m_128808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 7), 'm')
    # Testing the type of an if condition (line 307)
    if_condition_128809 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 307, 4), m_128808)
    # Assigning a type to the variable 'if_condition_128809' (line 307)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 307, 4), 'if_condition_128809', if_condition_128809)
    # SSA begins for if statement (line 307)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # SSA begins for try-except statement (line 308)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Name (line 309):
    
    # Assigning a Call to a Name (line 309):
    
    # Call to strip(...): (line 309)
    # Processing the call keyword arguments (line 309)
    kwargs_128816 = {}
    
    # Call to group(...): (line 309)
    # Processing the call arguments (line 309)
    int_128812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 27), 'int')
    # Processing the call keyword arguments (line 309)
    kwargs_128813 = {}
    # Getting the type of 'm' (line 309)
    m_128810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 19), 'm', False)
    # Obtaining the member 'group' of a type (line 309)
    group_128811 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 19), m_128810, 'group')
    # Calling group(args, kwargs) (line 309)
    group_call_result_128814 = invoke(stypy.reporting.localization.Localization(__file__, 309, 19), group_128811, *[int_128812], **kwargs_128813)
    
    # Obtaining the member 'strip' of a type (line 309)
    strip_128815 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 19), group_call_result_128814, 'strip')
    # Calling strip(args, kwargs) (line 309)
    strip_call_result_128817 = invoke(stypy.reporting.localization.Localization(__file__, 309, 19), strip_128815, *[], **kwargs_128816)
    
    # Assigning a type to the variable 'name' (line 309)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 12), 'name', strip_call_result_128817)
    
    # Assigning a Call to a Name (line 310):
    
    # Assigning a Call to a Name (line 310):
    
    # Call to strip(...): (line 310)
    # Processing the call keyword arguments (line 310)
    kwargs_128824 = {}
    
    # Call to group(...): (line 310)
    # Processing the call arguments (line 310)
    int_128820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 27), 'int')
    # Processing the call keyword arguments (line 310)
    kwargs_128821 = {}
    # Getting the type of 'm' (line 310)
    m_128818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 19), 'm', False)
    # Obtaining the member 'group' of a type (line 310)
    group_128819 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 19), m_128818, 'group')
    # Calling group(args, kwargs) (line 310)
    group_call_result_128822 = invoke(stypy.reporting.localization.Localization(__file__, 310, 19), group_128819, *[int_128820], **kwargs_128821)
    
    # Obtaining the member 'strip' of a type (line 310)
    strip_128823 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 19), group_call_result_128822, 'strip')
    # Calling strip(args, kwargs) (line 310)
    strip_call_result_128825 = invoke(stypy.reporting.localization.Localization(__file__, 310, 19), strip_128823, *[], **kwargs_128824)
    
    # Assigning a type to the variable 'type' (line 310)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 12), 'type', strip_call_result_128825)
    # SSA branch for the except part of a try statement (line 308)
    # SSA branch for the except 'IndexError' branch of a try statement (line 308)
    module_type_store.open_ssa_branch('except')
    
    # Call to ValueError(...): (line 312)
    # Processing the call arguments (line 312)
    str_128827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 29), 'str', 'Error while tokenizing attribute')
    # Processing the call keyword arguments (line 312)
    kwargs_128828 = {}
    # Getting the type of 'ValueError' (line 312)
    ValueError_128826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 312)
    ValueError_call_result_128829 = invoke(stypy.reporting.localization.Localization(__file__, 312, 18), ValueError_128826, *[str_128827], **kwargs_128828)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 312, 12), ValueError_call_result_128829, 'raise parameter', BaseException)
    # SSA join for try-except statement (line 308)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 307)
    module_type_store.open_ssa_branch('else')
    
    # Call to ValueError(...): (line 314)
    # Processing the call arguments (line 314)
    str_128831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 25), 'str', 'Error while tokenizing single %s')
    # Getting the type of 'val' (line 314)
    val_128832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 62), 'val', False)
    # Applying the binary operator '%' (line 314)
    result_mod_128833 = python_operator(stypy.reporting.localization.Localization(__file__, 314, 25), '%', str_128831, val_128832)
    
    # Processing the call keyword arguments (line 314)
    kwargs_128834 = {}
    # Getting the type of 'ValueError' (line 314)
    ValueError_128830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 314)
    ValueError_call_result_128835 = invoke(stypy.reporting.localization.Localization(__file__, 314, 14), ValueError_128830, *[result_mod_128833], **kwargs_128834)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 314, 8), ValueError_call_result_128835, 'raise parameter', BaseException)
    # SSA join for if statement (line 307)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 315)
    tuple_128836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 315)
    # Adding element type (line 315)
    # Getting the type of 'name' (line 315)
    name_128837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 11), 'name')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 315, 11), tuple_128836, name_128837)
    # Adding element type (line 315)
    # Getting the type of 'type' (line 315)
    type_128838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 17), 'type')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 315, 11), tuple_128836, type_128838)
    
    # Assigning a type to the variable 'stypy_return_type' (line 315)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 4), 'stypy_return_type', tuple_128836)
    
    # ################# End of 'tokenize_single_wcomma(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'tokenize_single_wcomma' in the type store
    # Getting the type of 'stypy_return_type' (line 303)
    stypy_return_type_128839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_128839)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'tokenize_single_wcomma'
    return stypy_return_type_128839

# Assigning a type to the variable 'tokenize_single_wcomma' (line 303)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 0), 'tokenize_single_wcomma', tokenize_single_wcomma)

@norecursion
def read_header(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'read_header'
    module_type_store = module_type_store.open_function_context('read_header', 318, 0, False)
    
    # Passed parameters checking function
    read_header.stypy_localization = localization
    read_header.stypy_type_of_self = None
    read_header.stypy_type_store = module_type_store
    read_header.stypy_function_name = 'read_header'
    read_header.stypy_param_names_list = ['ofile']
    read_header.stypy_varargs_param_name = None
    read_header.stypy_kwargs_param_name = None
    read_header.stypy_call_defaults = defaults
    read_header.stypy_call_varargs = varargs
    read_header.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'read_header', ['ofile'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'read_header', localization, ['ofile'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'read_header(...)' code ##################

    str_128840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 4), 'str', 'Read the header of the iterable ofile.')
    
    # Assigning a Call to a Name (line 320):
    
    # Assigning a Call to a Name (line 320):
    
    # Call to next(...): (line 320)
    # Processing the call arguments (line 320)
    # Getting the type of 'ofile' (line 320)
    ofile_128842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 13), 'ofile', False)
    # Processing the call keyword arguments (line 320)
    kwargs_128843 = {}
    # Getting the type of 'next' (line 320)
    next_128841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 8), 'next', False)
    # Calling next(args, kwargs) (line 320)
    next_call_result_128844 = invoke(stypy.reporting.localization.Localization(__file__, 320, 8), next_128841, *[ofile_128842], **kwargs_128843)
    
    # Assigning a type to the variable 'i' (line 320)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 320, 4), 'i', next_call_result_128844)
    
    
    # Call to match(...): (line 323)
    # Processing the call arguments (line 323)
    # Getting the type of 'i' (line 323)
    i_128847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 26), 'i', False)
    # Processing the call keyword arguments (line 323)
    kwargs_128848 = {}
    # Getting the type of 'r_comment' (line 323)
    r_comment_128845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 10), 'r_comment', False)
    # Obtaining the member 'match' of a type (line 323)
    match_128846 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 323, 10), r_comment_128845, 'match')
    # Calling match(args, kwargs) (line 323)
    match_call_result_128849 = invoke(stypy.reporting.localization.Localization(__file__, 323, 10), match_128846, *[i_128847], **kwargs_128848)
    
    # Testing the type of an if condition (line 323)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 323, 4), match_call_result_128849)
    # SSA begins for while statement (line 323)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Assigning a Call to a Name (line 324):
    
    # Assigning a Call to a Name (line 324):
    
    # Call to next(...): (line 324)
    # Processing the call arguments (line 324)
    # Getting the type of 'ofile' (line 324)
    ofile_128851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 17), 'ofile', False)
    # Processing the call keyword arguments (line 324)
    kwargs_128852 = {}
    # Getting the type of 'next' (line 324)
    next_128850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 12), 'next', False)
    # Calling next(args, kwargs) (line 324)
    next_call_result_128853 = invoke(stypy.reporting.localization.Localization(__file__, 324, 12), next_128850, *[ofile_128851], **kwargs_128852)
    
    # Assigning a type to the variable 'i' (line 324)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 324, 8), 'i', next_call_result_128853)
    # SSA join for while statement (line 323)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Name (line 327):
    
    # Assigning a Name to a Name (line 327):
    # Getting the type of 'None' (line 327)
    None_128854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 15), 'None')
    # Assigning a type to the variable 'relation' (line 327)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 327, 4), 'relation', None_128854)
    
    # Assigning a List to a Name (line 328):
    
    # Assigning a List to a Name (line 328):
    
    # Obtaining an instance of the builtin type 'list' (line 328)
    list_128855 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 328)
    
    # Assigning a type to the variable 'attributes' (line 328)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 328, 4), 'attributes', list_128855)
    
    
    
    # Call to match(...): (line 329)
    # Processing the call arguments (line 329)
    # Getting the type of 'i' (line 329)
    i_128858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 31), 'i', False)
    # Processing the call keyword arguments (line 329)
    kwargs_128859 = {}
    # Getting the type of 'r_datameta' (line 329)
    r_datameta_128856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 14), 'r_datameta', False)
    # Obtaining the member 'match' of a type (line 329)
    match_128857 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 14), r_datameta_128856, 'match')
    # Calling match(args, kwargs) (line 329)
    match_call_result_128860 = invoke(stypy.reporting.localization.Localization(__file__, 329, 14), match_128857, *[i_128858], **kwargs_128859)
    
    # Applying the 'not' unary operator (line 329)
    result_not__128861 = python_operator(stypy.reporting.localization.Localization(__file__, 329, 10), 'not', match_call_result_128860)
    
    # Testing the type of an if condition (line 329)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 329, 4), result_not__128861)
    # SSA begins for while statement (line 329)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Assigning a Call to a Name (line 330):
    
    # Assigning a Call to a Name (line 330):
    
    # Call to match(...): (line 330)
    # Processing the call arguments (line 330)
    # Getting the type of 'i' (line 330)
    i_128864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 31), 'i', False)
    # Processing the call keyword arguments (line 330)
    kwargs_128865 = {}
    # Getting the type of 'r_headerline' (line 330)
    r_headerline_128862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 12), 'r_headerline', False)
    # Obtaining the member 'match' of a type (line 330)
    match_128863 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 12), r_headerline_128862, 'match')
    # Calling match(args, kwargs) (line 330)
    match_call_result_128866 = invoke(stypy.reporting.localization.Localization(__file__, 330, 12), match_128863, *[i_128864], **kwargs_128865)
    
    # Assigning a type to the variable 'm' (line 330)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 8), 'm', match_call_result_128866)
    
    # Getting the type of 'm' (line 331)
    m_128867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 11), 'm')
    # Testing the type of an if condition (line 331)
    if_condition_128868 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 331, 8), m_128867)
    # Assigning a type to the variable 'if_condition_128868' (line 331)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 8), 'if_condition_128868', if_condition_128868)
    # SSA begins for if statement (line 331)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 332):
    
    # Assigning a Call to a Name (line 332):
    
    # Call to match(...): (line 332)
    # Processing the call arguments (line 332)
    # Getting the type of 'i' (line 332)
    i_128871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 39), 'i', False)
    # Processing the call keyword arguments (line 332)
    kwargs_128872 = {}
    # Getting the type of 'r_attribute' (line 332)
    r_attribute_128869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 21), 'r_attribute', False)
    # Obtaining the member 'match' of a type (line 332)
    match_128870 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 21), r_attribute_128869, 'match')
    # Calling match(args, kwargs) (line 332)
    match_call_result_128873 = invoke(stypy.reporting.localization.Localization(__file__, 332, 21), match_128870, *[i_128871], **kwargs_128872)
    
    # Assigning a type to the variable 'isattr' (line 332)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 332, 12), 'isattr', match_call_result_128873)
    
    # Getting the type of 'isattr' (line 333)
    isattr_128874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 15), 'isattr')
    # Testing the type of an if condition (line 333)
    if_condition_128875 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 333, 12), isattr_128874)
    # Assigning a type to the variable 'if_condition_128875' (line 333)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 333, 12), 'if_condition_128875', if_condition_128875)
    # SSA begins for if statement (line 333)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Tuple (line 334):
    
    # Assigning a Subscript to a Name (line 334):
    
    # Obtaining the type of the subscript
    int_128876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 16), 'int')
    
    # Call to tokenize_attribute(...): (line 334)
    # Processing the call arguments (line 334)
    # Getting the type of 'ofile' (line 334)
    ofile_128878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 51), 'ofile', False)
    # Getting the type of 'i' (line 334)
    i_128879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 58), 'i', False)
    # Processing the call keyword arguments (line 334)
    kwargs_128880 = {}
    # Getting the type of 'tokenize_attribute' (line 334)
    tokenize_attribute_128877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 32), 'tokenize_attribute', False)
    # Calling tokenize_attribute(args, kwargs) (line 334)
    tokenize_attribute_call_result_128881 = invoke(stypy.reporting.localization.Localization(__file__, 334, 32), tokenize_attribute_128877, *[ofile_128878, i_128879], **kwargs_128880)
    
    # Obtaining the member '__getitem__' of a type (line 334)
    getitem___128882 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 16), tokenize_attribute_call_result_128881, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 334)
    subscript_call_result_128883 = invoke(stypy.reporting.localization.Localization(__file__, 334, 16), getitem___128882, int_128876)
    
    # Assigning a type to the variable 'tuple_var_assignment_128247' (line 334)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 16), 'tuple_var_assignment_128247', subscript_call_result_128883)
    
    # Assigning a Subscript to a Name (line 334):
    
    # Obtaining the type of the subscript
    int_128884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 16), 'int')
    
    # Call to tokenize_attribute(...): (line 334)
    # Processing the call arguments (line 334)
    # Getting the type of 'ofile' (line 334)
    ofile_128886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 51), 'ofile', False)
    # Getting the type of 'i' (line 334)
    i_128887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 58), 'i', False)
    # Processing the call keyword arguments (line 334)
    kwargs_128888 = {}
    # Getting the type of 'tokenize_attribute' (line 334)
    tokenize_attribute_128885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 32), 'tokenize_attribute', False)
    # Calling tokenize_attribute(args, kwargs) (line 334)
    tokenize_attribute_call_result_128889 = invoke(stypy.reporting.localization.Localization(__file__, 334, 32), tokenize_attribute_128885, *[ofile_128886, i_128887], **kwargs_128888)
    
    # Obtaining the member '__getitem__' of a type (line 334)
    getitem___128890 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 16), tokenize_attribute_call_result_128889, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 334)
    subscript_call_result_128891 = invoke(stypy.reporting.localization.Localization(__file__, 334, 16), getitem___128890, int_128884)
    
    # Assigning a type to the variable 'tuple_var_assignment_128248' (line 334)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 16), 'tuple_var_assignment_128248', subscript_call_result_128891)
    
    # Assigning a Subscript to a Name (line 334):
    
    # Obtaining the type of the subscript
    int_128892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 16), 'int')
    
    # Call to tokenize_attribute(...): (line 334)
    # Processing the call arguments (line 334)
    # Getting the type of 'ofile' (line 334)
    ofile_128894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 51), 'ofile', False)
    # Getting the type of 'i' (line 334)
    i_128895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 58), 'i', False)
    # Processing the call keyword arguments (line 334)
    kwargs_128896 = {}
    # Getting the type of 'tokenize_attribute' (line 334)
    tokenize_attribute_128893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 32), 'tokenize_attribute', False)
    # Calling tokenize_attribute(args, kwargs) (line 334)
    tokenize_attribute_call_result_128897 = invoke(stypy.reporting.localization.Localization(__file__, 334, 32), tokenize_attribute_128893, *[ofile_128894, i_128895], **kwargs_128896)
    
    # Obtaining the member '__getitem__' of a type (line 334)
    getitem___128898 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 16), tokenize_attribute_call_result_128897, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 334)
    subscript_call_result_128899 = invoke(stypy.reporting.localization.Localization(__file__, 334, 16), getitem___128898, int_128892)
    
    # Assigning a type to the variable 'tuple_var_assignment_128249' (line 334)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 16), 'tuple_var_assignment_128249', subscript_call_result_128899)
    
    # Assigning a Name to a Name (line 334):
    # Getting the type of 'tuple_var_assignment_128247' (line 334)
    tuple_var_assignment_128247_128900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 16), 'tuple_var_assignment_128247')
    # Assigning a type to the variable 'name' (line 334)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 16), 'name', tuple_var_assignment_128247_128900)
    
    # Assigning a Name to a Name (line 334):
    # Getting the type of 'tuple_var_assignment_128248' (line 334)
    tuple_var_assignment_128248_128901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 16), 'tuple_var_assignment_128248')
    # Assigning a type to the variable 'type' (line 334)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 22), 'type', tuple_var_assignment_128248_128901)
    
    # Assigning a Name to a Name (line 334):
    # Getting the type of 'tuple_var_assignment_128249' (line 334)
    tuple_var_assignment_128249_128902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 16), 'tuple_var_assignment_128249')
    # Assigning a type to the variable 'i' (line 334)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 28), 'i', tuple_var_assignment_128249_128902)
    
    # Call to append(...): (line 335)
    # Processing the call arguments (line 335)
    
    # Obtaining an instance of the builtin type 'tuple' (line 335)
    tuple_128905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 35), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 335)
    # Adding element type (line 335)
    # Getting the type of 'name' (line 335)
    name_128906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 35), 'name', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 335, 35), tuple_128905, name_128906)
    # Adding element type (line 335)
    # Getting the type of 'type' (line 335)
    type_128907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 41), 'type', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 335, 35), tuple_128905, type_128907)
    
    # Processing the call keyword arguments (line 335)
    kwargs_128908 = {}
    # Getting the type of 'attributes' (line 335)
    attributes_128903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 16), 'attributes', False)
    # Obtaining the member 'append' of a type (line 335)
    append_128904 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 16), attributes_128903, 'append')
    # Calling append(args, kwargs) (line 335)
    append_call_result_128909 = invoke(stypy.reporting.localization.Localization(__file__, 335, 16), append_128904, *[tuple_128905], **kwargs_128908)
    
    # SSA branch for the else part of an if statement (line 333)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 337):
    
    # Assigning a Call to a Name (line 337):
    
    # Call to match(...): (line 337)
    # Processing the call arguments (line 337)
    # Getting the type of 'i' (line 337)
    i_128912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 41), 'i', False)
    # Processing the call keyword arguments (line 337)
    kwargs_128913 = {}
    # Getting the type of 'r_relation' (line 337)
    r_relation_128910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 24), 'r_relation', False)
    # Obtaining the member 'match' of a type (line 337)
    match_128911 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 24), r_relation_128910, 'match')
    # Calling match(args, kwargs) (line 337)
    match_call_result_128914 = invoke(stypy.reporting.localization.Localization(__file__, 337, 24), match_128911, *[i_128912], **kwargs_128913)
    
    # Assigning a type to the variable 'isrel' (line 337)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 337, 16), 'isrel', match_call_result_128914)
    
    # Getting the type of 'isrel' (line 338)
    isrel_128915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 19), 'isrel')
    # Testing the type of an if condition (line 338)
    if_condition_128916 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 338, 16), isrel_128915)
    # Assigning a type to the variable 'if_condition_128916' (line 338)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 338, 16), 'if_condition_128916', if_condition_128916)
    # SSA begins for if statement (line 338)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 339):
    
    # Assigning a Call to a Name (line 339):
    
    # Call to group(...): (line 339)
    # Processing the call arguments (line 339)
    int_128919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 43), 'int')
    # Processing the call keyword arguments (line 339)
    kwargs_128920 = {}
    # Getting the type of 'isrel' (line 339)
    isrel_128917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 31), 'isrel', False)
    # Obtaining the member 'group' of a type (line 339)
    group_128918 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 339, 31), isrel_128917, 'group')
    # Calling group(args, kwargs) (line 339)
    group_call_result_128921 = invoke(stypy.reporting.localization.Localization(__file__, 339, 31), group_128918, *[int_128919], **kwargs_128920)
    
    # Assigning a type to the variable 'relation' (line 339)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 339, 20), 'relation', group_call_result_128921)
    # SSA branch for the else part of an if statement (line 338)
    module_type_store.open_ssa_branch('else')
    
    # Call to ValueError(...): (line 341)
    # Processing the call arguments (line 341)
    str_128923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 37), 'str', 'Error parsing line %s')
    # Getting the type of 'i' (line 341)
    i_128924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 63), 'i', False)
    # Applying the binary operator '%' (line 341)
    result_mod_128925 = python_operator(stypy.reporting.localization.Localization(__file__, 341, 37), '%', str_128923, i_128924)
    
    # Processing the call keyword arguments (line 341)
    kwargs_128926 = {}
    # Getting the type of 'ValueError' (line 341)
    ValueError_128922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 26), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 341)
    ValueError_call_result_128927 = invoke(stypy.reporting.localization.Localization(__file__, 341, 26), ValueError_128922, *[result_mod_128925], **kwargs_128926)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 341, 20), ValueError_call_result_128927, 'raise parameter', BaseException)
    # SSA join for if statement (line 338)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 342):
    
    # Assigning a Call to a Name (line 342):
    
    # Call to next(...): (line 342)
    # Processing the call arguments (line 342)
    # Getting the type of 'ofile' (line 342)
    ofile_128929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 25), 'ofile', False)
    # Processing the call keyword arguments (line 342)
    kwargs_128930 = {}
    # Getting the type of 'next' (line 342)
    next_128928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 20), 'next', False)
    # Calling next(args, kwargs) (line 342)
    next_call_result_128931 = invoke(stypy.reporting.localization.Localization(__file__, 342, 20), next_128928, *[ofile_128929], **kwargs_128930)
    
    # Assigning a type to the variable 'i' (line 342)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 342, 16), 'i', next_call_result_128931)
    # SSA join for if statement (line 333)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 331)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 344):
    
    # Assigning a Call to a Name (line 344):
    
    # Call to next(...): (line 344)
    # Processing the call arguments (line 344)
    # Getting the type of 'ofile' (line 344)
    ofile_128933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 21), 'ofile', False)
    # Processing the call keyword arguments (line 344)
    kwargs_128934 = {}
    # Getting the type of 'next' (line 344)
    next_128932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 16), 'next', False)
    # Calling next(args, kwargs) (line 344)
    next_call_result_128935 = invoke(stypy.reporting.localization.Localization(__file__, 344, 16), next_128932, *[ofile_128933], **kwargs_128934)
    
    # Assigning a type to the variable 'i' (line 344)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 344, 12), 'i', next_call_result_128935)
    # SSA join for if statement (line 331)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for while statement (line 329)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 346)
    tuple_128936 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 346, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 346)
    # Adding element type (line 346)
    # Getting the type of 'relation' (line 346)
    relation_128937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 11), 'relation')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 346, 11), tuple_128936, relation_128937)
    # Adding element type (line 346)
    # Getting the type of 'attributes' (line 346)
    attributes_128938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 21), 'attributes')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 346, 11), tuple_128936, attributes_128938)
    
    # Assigning a type to the variable 'stypy_return_type' (line 346)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 346, 4), 'stypy_return_type', tuple_128936)
    
    # ################# End of 'read_header(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'read_header' in the type store
    # Getting the type of 'stypy_return_type' (line 318)
    stypy_return_type_128939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_128939)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'read_header'
    return stypy_return_type_128939

# Assigning a type to the variable 'read_header' (line 318)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 0), 'read_header', read_header)

@norecursion
def safe_float(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'safe_float'
    module_type_store = module_type_store.open_function_context('safe_float', 352, 0, False)
    
    # Passed parameters checking function
    safe_float.stypy_localization = localization
    safe_float.stypy_type_of_self = None
    safe_float.stypy_type_store = module_type_store
    safe_float.stypy_function_name = 'safe_float'
    safe_float.stypy_param_names_list = ['x']
    safe_float.stypy_varargs_param_name = None
    safe_float.stypy_kwargs_param_name = None
    safe_float.stypy_call_defaults = defaults
    safe_float.stypy_call_varargs = varargs
    safe_float.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'safe_float', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'safe_float', localization, ['x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'safe_float(...)' code ##################

    str_128940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, (-1)), 'str', "given a string x, convert it to a float. If the stripped string is a ?,\n    return a Nan (missing value).\n\n    Parameters\n    ----------\n    x : str\n       string to convert\n\n    Returns\n    -------\n    f : float\n       where float can be nan\n\n    Examples\n    --------\n    >>> safe_float('1')\n    1.0\n    >>> safe_float('1\\n')\n    1.0\n    >>> safe_float('?\\n')\n    nan\n    ")
    
    
    str_128941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 7), 'str', '?')
    # Getting the type of 'x' (line 375)
    x_128942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 14), 'x')
    # Applying the binary operator 'in' (line 375)
    result_contains_128943 = python_operator(stypy.reporting.localization.Localization(__file__, 375, 7), 'in', str_128941, x_128942)
    
    # Testing the type of an if condition (line 375)
    if_condition_128944 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 375, 4), result_contains_128943)
    # Assigning a type to the variable 'if_condition_128944' (line 375)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 375, 4), 'if_condition_128944', if_condition_128944)
    # SSA begins for if statement (line 375)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'np' (line 376)
    np_128945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 15), 'np')
    # Obtaining the member 'nan' of a type (line 376)
    nan_128946 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 376, 15), np_128945, 'nan')
    # Assigning a type to the variable 'stypy_return_type' (line 376)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 376, 8), 'stypy_return_type', nan_128946)
    # SSA branch for the else part of an if statement (line 375)
    module_type_store.open_ssa_branch('else')
    
    # Call to float(...): (line 378)
    # Processing the call arguments (line 378)
    # Getting the type of 'x' (line 378)
    x_128948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 21), 'x', False)
    # Processing the call keyword arguments (line 378)
    kwargs_128949 = {}
    # Getting the type of 'float' (line 378)
    float_128947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 15), 'float', False)
    # Calling float(args, kwargs) (line 378)
    float_call_result_128950 = invoke(stypy.reporting.localization.Localization(__file__, 378, 15), float_128947, *[x_128948], **kwargs_128949)
    
    # Assigning a type to the variable 'stypy_return_type' (line 378)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 378, 8), 'stypy_return_type', float_call_result_128950)
    # SSA join for if statement (line 375)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'safe_float(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'safe_float' in the type store
    # Getting the type of 'stypy_return_type' (line 352)
    stypy_return_type_128951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_128951)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'safe_float'
    return stypy_return_type_128951

# Assigning a type to the variable 'safe_float' (line 352)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 352, 0), 'safe_float', safe_float)

@norecursion
def safe_nominal(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'safe_nominal'
    module_type_store = module_type_store.open_function_context('safe_nominal', 381, 0, False)
    
    # Passed parameters checking function
    safe_nominal.stypy_localization = localization
    safe_nominal.stypy_type_of_self = None
    safe_nominal.stypy_type_store = module_type_store
    safe_nominal.stypy_function_name = 'safe_nominal'
    safe_nominal.stypy_param_names_list = ['value', 'pvalue']
    safe_nominal.stypy_varargs_param_name = None
    safe_nominal.stypy_kwargs_param_name = None
    safe_nominal.stypy_call_defaults = defaults
    safe_nominal.stypy_call_varargs = varargs
    safe_nominal.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'safe_nominal', ['value', 'pvalue'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'safe_nominal', localization, ['value', 'pvalue'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'safe_nominal(...)' code ##################

    
    # Assigning a Call to a Name (line 382):
    
    # Assigning a Call to a Name (line 382):
    
    # Call to strip(...): (line 382)
    # Processing the call keyword arguments (line 382)
    kwargs_128954 = {}
    # Getting the type of 'value' (line 382)
    value_128952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 13), 'value', False)
    # Obtaining the member 'strip' of a type (line 382)
    strip_128953 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 382, 13), value_128952, 'strip')
    # Calling strip(args, kwargs) (line 382)
    strip_call_result_128955 = invoke(stypy.reporting.localization.Localization(__file__, 382, 13), strip_128953, *[], **kwargs_128954)
    
    # Assigning a type to the variable 'svalue' (line 382)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 382, 4), 'svalue', strip_call_result_128955)
    
    
    # Getting the type of 'svalue' (line 383)
    svalue_128956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 7), 'svalue')
    # Getting the type of 'pvalue' (line 383)
    pvalue_128957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 17), 'pvalue')
    # Applying the binary operator 'in' (line 383)
    result_contains_128958 = python_operator(stypy.reporting.localization.Localization(__file__, 383, 7), 'in', svalue_128956, pvalue_128957)
    
    # Testing the type of an if condition (line 383)
    if_condition_128959 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 383, 4), result_contains_128958)
    # Assigning a type to the variable 'if_condition_128959' (line 383)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 4), 'if_condition_128959', if_condition_128959)
    # SSA begins for if statement (line 383)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'svalue' (line 384)
    svalue_128960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 15), 'svalue')
    # Assigning a type to the variable 'stypy_return_type' (line 384)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 384, 8), 'stypy_return_type', svalue_128960)
    # SSA branch for the else part of an if statement (line 383)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'svalue' (line 385)
    svalue_128961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 9), 'svalue')
    str_128962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 385, 19), 'str', '?')
    # Applying the binary operator '==' (line 385)
    result_eq_128963 = python_operator(stypy.reporting.localization.Localization(__file__, 385, 9), '==', svalue_128961, str_128962)
    
    # Testing the type of an if condition (line 385)
    if_condition_128964 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 385, 9), result_eq_128963)
    # Assigning a type to the variable 'if_condition_128964' (line 385)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 385, 9), 'if_condition_128964', if_condition_128964)
    # SSA begins for if statement (line 385)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'svalue' (line 386)
    svalue_128965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 15), 'svalue')
    # Assigning a type to the variable 'stypy_return_type' (line 386)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 386, 8), 'stypy_return_type', svalue_128965)
    # SSA branch for the else part of an if statement (line 385)
    module_type_store.open_ssa_branch('else')
    
    # Call to ValueError(...): (line 388)
    # Processing the call arguments (line 388)
    str_128967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 388, 25), 'str', '%s value not in %s')
    
    # Obtaining an instance of the builtin type 'tuple' (line 388)
    tuple_128968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 388, 49), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 388)
    # Adding element type (line 388)
    
    # Call to str(...): (line 388)
    # Processing the call arguments (line 388)
    # Getting the type of 'svalue' (line 388)
    svalue_128970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 53), 'svalue', False)
    # Processing the call keyword arguments (line 388)
    kwargs_128971 = {}
    # Getting the type of 'str' (line 388)
    str_128969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 49), 'str', False)
    # Calling str(args, kwargs) (line 388)
    str_call_result_128972 = invoke(stypy.reporting.localization.Localization(__file__, 388, 49), str_128969, *[svalue_128970], **kwargs_128971)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 388, 49), tuple_128968, str_call_result_128972)
    # Adding element type (line 388)
    
    # Call to str(...): (line 388)
    # Processing the call arguments (line 388)
    # Getting the type of 'pvalue' (line 388)
    pvalue_128974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 66), 'pvalue', False)
    # Processing the call keyword arguments (line 388)
    kwargs_128975 = {}
    # Getting the type of 'str' (line 388)
    str_128973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 62), 'str', False)
    # Calling str(args, kwargs) (line 388)
    str_call_result_128976 = invoke(stypy.reporting.localization.Localization(__file__, 388, 62), str_128973, *[pvalue_128974], **kwargs_128975)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 388, 49), tuple_128968, str_call_result_128976)
    
    # Applying the binary operator '%' (line 388)
    result_mod_128977 = python_operator(stypy.reporting.localization.Localization(__file__, 388, 25), '%', str_128967, tuple_128968)
    
    # Processing the call keyword arguments (line 388)
    kwargs_128978 = {}
    # Getting the type of 'ValueError' (line 388)
    ValueError_128966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 388)
    ValueError_call_result_128979 = invoke(stypy.reporting.localization.Localization(__file__, 388, 14), ValueError_128966, *[result_mod_128977], **kwargs_128978)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 388, 8), ValueError_call_result_128979, 'raise parameter', BaseException)
    # SSA join for if statement (line 385)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 383)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'safe_nominal(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'safe_nominal' in the type store
    # Getting the type of 'stypy_return_type' (line 381)
    stypy_return_type_128980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_128980)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'safe_nominal'
    return stypy_return_type_128980

# Assigning a type to the variable 'safe_nominal' (line 381)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 381, 0), 'safe_nominal', safe_nominal)

@norecursion
def safe_date(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'safe_date'
    module_type_store = module_type_store.open_function_context('safe_date', 391, 0, False)
    
    # Passed parameters checking function
    safe_date.stypy_localization = localization
    safe_date.stypy_type_of_self = None
    safe_date.stypy_type_store = module_type_store
    safe_date.stypy_function_name = 'safe_date'
    safe_date.stypy_param_names_list = ['value', 'date_format', 'datetime_unit']
    safe_date.stypy_varargs_param_name = None
    safe_date.stypy_kwargs_param_name = None
    safe_date.stypy_call_defaults = defaults
    safe_date.stypy_call_varargs = varargs
    safe_date.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'safe_date', ['value', 'date_format', 'datetime_unit'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'safe_date', localization, ['value', 'date_format', 'datetime_unit'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'safe_date(...)' code ##################

    
    # Assigning a Call to a Name (line 392):
    
    # Assigning a Call to a Name (line 392):
    
    # Call to strip(...): (line 392)
    # Processing the call arguments (line 392)
    str_128990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 392, 46), 'str', '"')
    # Processing the call keyword arguments (line 392)
    kwargs_128991 = {}
    
    # Call to strip(...): (line 392)
    # Processing the call arguments (line 392)
    str_128986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 392, 35), 'str', "'")
    # Processing the call keyword arguments (line 392)
    kwargs_128987 = {}
    
    # Call to strip(...): (line 392)
    # Processing the call keyword arguments (line 392)
    kwargs_128983 = {}
    # Getting the type of 'value' (line 392)
    value_128981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 15), 'value', False)
    # Obtaining the member 'strip' of a type (line 392)
    strip_128982 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 392, 15), value_128981, 'strip')
    # Calling strip(args, kwargs) (line 392)
    strip_call_result_128984 = invoke(stypy.reporting.localization.Localization(__file__, 392, 15), strip_128982, *[], **kwargs_128983)
    
    # Obtaining the member 'strip' of a type (line 392)
    strip_128985 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 392, 15), strip_call_result_128984, 'strip')
    # Calling strip(args, kwargs) (line 392)
    strip_call_result_128988 = invoke(stypy.reporting.localization.Localization(__file__, 392, 15), strip_128985, *[str_128986], **kwargs_128987)
    
    # Obtaining the member 'strip' of a type (line 392)
    strip_128989 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 392, 15), strip_call_result_128988, 'strip')
    # Calling strip(args, kwargs) (line 392)
    strip_call_result_128992 = invoke(stypy.reporting.localization.Localization(__file__, 392, 15), strip_128989, *[str_128990], **kwargs_128991)
    
    # Assigning a type to the variable 'date_str' (line 392)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 392, 4), 'date_str', strip_call_result_128992)
    
    
    # Getting the type of 'date_str' (line 393)
    date_str_128993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 7), 'date_str')
    str_128994 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 19), 'str', '?')
    # Applying the binary operator '==' (line 393)
    result_eq_128995 = python_operator(stypy.reporting.localization.Localization(__file__, 393, 7), '==', date_str_128993, str_128994)
    
    # Testing the type of an if condition (line 393)
    if_condition_128996 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 393, 4), result_eq_128995)
    # Assigning a type to the variable 'if_condition_128996' (line 393)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 393, 4), 'if_condition_128996', if_condition_128996)
    # SSA begins for if statement (line 393)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to datetime64(...): (line 394)
    # Processing the call arguments (line 394)
    str_128999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 29), 'str', 'NaT')
    # Getting the type of 'datetime_unit' (line 394)
    datetime_unit_129000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 36), 'datetime_unit', False)
    # Processing the call keyword arguments (line 394)
    kwargs_129001 = {}
    # Getting the type of 'np' (line 394)
    np_128997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 15), 'np', False)
    # Obtaining the member 'datetime64' of a type (line 394)
    datetime64_128998 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 15), np_128997, 'datetime64')
    # Calling datetime64(args, kwargs) (line 394)
    datetime64_call_result_129002 = invoke(stypy.reporting.localization.Localization(__file__, 394, 15), datetime64_128998, *[str_128999, datetime_unit_129000], **kwargs_129001)
    
    # Assigning a type to the variable 'stypy_return_type' (line 394)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 394, 8), 'stypy_return_type', datetime64_call_result_129002)
    # SSA branch for the else part of an if statement (line 393)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 396):
    
    # Assigning a Call to a Name (line 396):
    
    # Call to strptime(...): (line 396)
    # Processing the call arguments (line 396)
    # Getting the type of 'date_str' (line 396)
    date_str_129006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 40), 'date_str', False)
    # Getting the type of 'date_format' (line 396)
    date_format_129007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 50), 'date_format', False)
    # Processing the call keyword arguments (line 396)
    kwargs_129008 = {}
    # Getting the type of 'datetime' (line 396)
    datetime_129003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 13), 'datetime', False)
    # Obtaining the member 'datetime' of a type (line 396)
    datetime_129004 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 396, 13), datetime_129003, 'datetime')
    # Obtaining the member 'strptime' of a type (line 396)
    strptime_129005 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 396, 13), datetime_129004, 'strptime')
    # Calling strptime(args, kwargs) (line 396)
    strptime_call_result_129009 = invoke(stypy.reporting.localization.Localization(__file__, 396, 13), strptime_129005, *[date_str_129006, date_format_129007], **kwargs_129008)
    
    # Assigning a type to the variable 'dt' (line 396)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 396, 8), 'dt', strptime_call_result_129009)
    
    # Call to astype(...): (line 397)
    # Processing the call arguments (line 397)
    str_129016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 397, 40), 'str', 'datetime64[%s]')
    # Getting the type of 'datetime_unit' (line 397)
    datetime_unit_129017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 59), 'datetime_unit', False)
    # Applying the binary operator '%' (line 397)
    result_mod_129018 = python_operator(stypy.reporting.localization.Localization(__file__, 397, 40), '%', str_129016, datetime_unit_129017)
    
    # Processing the call keyword arguments (line 397)
    kwargs_129019 = {}
    
    # Call to datetime64(...): (line 397)
    # Processing the call arguments (line 397)
    # Getting the type of 'dt' (line 397)
    dt_129012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 29), 'dt', False)
    # Processing the call keyword arguments (line 397)
    kwargs_129013 = {}
    # Getting the type of 'np' (line 397)
    np_129010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 15), 'np', False)
    # Obtaining the member 'datetime64' of a type (line 397)
    datetime64_129011 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 397, 15), np_129010, 'datetime64')
    # Calling datetime64(args, kwargs) (line 397)
    datetime64_call_result_129014 = invoke(stypy.reporting.localization.Localization(__file__, 397, 15), datetime64_129011, *[dt_129012], **kwargs_129013)
    
    # Obtaining the member 'astype' of a type (line 397)
    astype_129015 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 397, 15), datetime64_call_result_129014, 'astype')
    # Calling astype(args, kwargs) (line 397)
    astype_call_result_129020 = invoke(stypy.reporting.localization.Localization(__file__, 397, 15), astype_129015, *[result_mod_129018], **kwargs_129019)
    
    # Assigning a type to the variable 'stypy_return_type' (line 397)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 397, 8), 'stypy_return_type', astype_call_result_129020)
    # SSA join for if statement (line 393)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'safe_date(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'safe_date' in the type store
    # Getting the type of 'stypy_return_type' (line 391)
    stypy_return_type_129021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_129021)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'safe_date'
    return stypy_return_type_129021

# Assigning a type to the variable 'safe_date' (line 391)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 391, 0), 'safe_date', safe_date)
# Declaration of the 'MetaData' class

class MetaData(object, ):
    str_129022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, (-1)), 'str', "Small container to keep useful informations on a ARFF dataset.\n\n    Knows about attributes names and types.\n\n    Examples\n    --------\n    ::\n\n        data, meta = loadarff('iris.arff')\n        # This will print the attributes names of the iris.arff dataset\n        for i in meta:\n            print(i)\n        # This works too\n        meta.names()\n        # Getting attribute type\n        types = meta.types()\n\n    Notes\n    -----\n    Also maintains the list of attributes in order, i.e. doing for i in\n    meta, where meta is an instance of MetaData, will return the\n    different attribute names in the order they were defined.\n    ")

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 424, 4, False)
        # Assigning a type to the variable 'self' (line 425)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 425, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MetaData.__init__', ['rel', 'attr'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['rel', 'attr'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 425):
        
        # Assigning a Name to a Attribute (line 425):
        # Getting the type of 'rel' (line 425)
        rel_129023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 20), 'rel')
        # Getting the type of 'self' (line 425)
        self_129024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 8), 'self')
        # Setting the type of the member 'name' of a type (line 425)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 425, 8), self_129024, 'name', rel_129023)
        
        # Assigning a Dict to a Attribute (line 428):
        
        # Assigning a Dict to a Attribute (line 428):
        
        # Obtaining an instance of the builtin type 'dict' (line 428)
        dict_129025 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 428, 27), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 428)
        
        # Getting the type of 'self' (line 428)
        self_129026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 8), 'self')
        # Setting the type of the member '_attributes' of a type (line 428)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 428, 8), self_129026, '_attributes', dict_129025)
        
        # Assigning a List to a Attribute (line 429):
        
        # Assigning a List to a Attribute (line 429):
        
        # Obtaining an instance of the builtin type 'list' (line 429)
        list_129027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 429, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 429)
        
        # Getting the type of 'self' (line 429)
        self_129028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 8), 'self')
        # Setting the type of the member '_attrnames' of a type (line 429)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 429, 8), self_129028, '_attrnames', list_129027)
        
        # Getting the type of 'attr' (line 430)
        attr_129029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 27), 'attr')
        # Testing the type of a for loop iterable (line 430)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 430, 8), attr_129029)
        # Getting the type of the for loop variable (line 430)
        for_loop_var_129030 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 430, 8), attr_129029)
        # Assigning a type to the variable 'name' (line 430)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 430, 8), 'name', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 430, 8), for_loop_var_129030))
        # Assigning a type to the variable 'value' (line 430)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 430, 8), 'value', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 430, 8), for_loop_var_129030))
        # SSA begins for a for statement (line 430)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 431):
        
        # Assigning a Call to a Name (line 431):
        
        # Call to parse_type(...): (line 431)
        # Processing the call arguments (line 431)
        # Getting the type of 'value' (line 431)
        value_129032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 28), 'value', False)
        # Processing the call keyword arguments (line 431)
        kwargs_129033 = {}
        # Getting the type of 'parse_type' (line 431)
        parse_type_129031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 17), 'parse_type', False)
        # Calling parse_type(args, kwargs) (line 431)
        parse_type_call_result_129034 = invoke(stypy.reporting.localization.Localization(__file__, 431, 17), parse_type_129031, *[value_129032], **kwargs_129033)
        
        # Assigning a type to the variable 'tp' (line 431)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 431, 12), 'tp', parse_type_call_result_129034)
        
        # Call to append(...): (line 432)
        # Processing the call arguments (line 432)
        # Getting the type of 'name' (line 432)
        name_129038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 35), 'name', False)
        # Processing the call keyword arguments (line 432)
        kwargs_129039 = {}
        # Getting the type of 'self' (line 432)
        self_129035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 12), 'self', False)
        # Obtaining the member '_attrnames' of a type (line 432)
        _attrnames_129036 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 432, 12), self_129035, '_attrnames')
        # Obtaining the member 'append' of a type (line 432)
        append_129037 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 432, 12), _attrnames_129036, 'append')
        # Calling append(args, kwargs) (line 432)
        append_call_result_129040 = invoke(stypy.reporting.localization.Localization(__file__, 432, 12), append_129037, *[name_129038], **kwargs_129039)
        
        
        
        # Getting the type of 'tp' (line 433)
        tp_129041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 15), 'tp')
        str_129042 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 21), 'str', 'nominal')
        # Applying the binary operator '==' (line 433)
        result_eq_129043 = python_operator(stypy.reporting.localization.Localization(__file__, 433, 15), '==', tp_129041, str_129042)
        
        # Testing the type of an if condition (line 433)
        if_condition_129044 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 433, 12), result_eq_129043)
        # Assigning a type to the variable 'if_condition_129044' (line 433)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 433, 12), 'if_condition_129044', if_condition_129044)
        # SSA begins for if statement (line 433)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Tuple to a Subscript (line 434):
        
        # Assigning a Tuple to a Subscript (line 434):
        
        # Obtaining an instance of the builtin type 'tuple' (line 434)
        tuple_129045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, 42), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 434)
        # Adding element type (line 434)
        # Getting the type of 'tp' (line 434)
        tp_129046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 42), 'tp')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 434, 42), tuple_129045, tp_129046)
        # Adding element type (line 434)
        
        # Call to get_nom_val(...): (line 434)
        # Processing the call arguments (line 434)
        # Getting the type of 'value' (line 434)
        value_129048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 58), 'value', False)
        # Processing the call keyword arguments (line 434)
        kwargs_129049 = {}
        # Getting the type of 'get_nom_val' (line 434)
        get_nom_val_129047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 46), 'get_nom_val', False)
        # Calling get_nom_val(args, kwargs) (line 434)
        get_nom_val_call_result_129050 = invoke(stypy.reporting.localization.Localization(__file__, 434, 46), get_nom_val_129047, *[value_129048], **kwargs_129049)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 434, 42), tuple_129045, get_nom_val_call_result_129050)
        
        # Getting the type of 'self' (line 434)
        self_129051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 16), 'self')
        # Obtaining the member '_attributes' of a type (line 434)
        _attributes_129052 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 434, 16), self_129051, '_attributes')
        # Getting the type of 'name' (line 434)
        name_129053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 33), 'name')
        # Storing an element on a container (line 434)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 434, 16), _attributes_129052, (name_129053, tuple_129045))
        # SSA branch for the else part of an if statement (line 433)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'tp' (line 435)
        tp_129054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 17), 'tp')
        str_129055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 435, 23), 'str', 'date')
        # Applying the binary operator '==' (line 435)
        result_eq_129056 = python_operator(stypy.reporting.localization.Localization(__file__, 435, 17), '==', tp_129054, str_129055)
        
        # Testing the type of an if condition (line 435)
        if_condition_129057 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 435, 17), result_eq_129056)
        # Assigning a type to the variable 'if_condition_129057' (line 435)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 435, 17), 'if_condition_129057', if_condition_129057)
        # SSA begins for if statement (line 435)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Tuple to a Subscript (line 436):
        
        # Assigning a Tuple to a Subscript (line 436):
        
        # Obtaining an instance of the builtin type 'tuple' (line 436)
        tuple_129058 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 436, 42), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 436)
        # Adding element type (line 436)
        # Getting the type of 'tp' (line 436)
        tp_129059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 42), 'tp')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 436, 42), tuple_129058, tp_129059)
        # Adding element type (line 436)
        
        # Obtaining the type of the subscript
        int_129060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 436, 69), 'int')
        
        # Call to get_date_format(...): (line 436)
        # Processing the call arguments (line 436)
        # Getting the type of 'value' (line 436)
        value_129062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 62), 'value', False)
        # Processing the call keyword arguments (line 436)
        kwargs_129063 = {}
        # Getting the type of 'get_date_format' (line 436)
        get_date_format_129061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 46), 'get_date_format', False)
        # Calling get_date_format(args, kwargs) (line 436)
        get_date_format_call_result_129064 = invoke(stypy.reporting.localization.Localization(__file__, 436, 46), get_date_format_129061, *[value_129062], **kwargs_129063)
        
        # Obtaining the member '__getitem__' of a type (line 436)
        getitem___129065 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 436, 46), get_date_format_call_result_129064, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 436)
        subscript_call_result_129066 = invoke(stypy.reporting.localization.Localization(__file__, 436, 46), getitem___129065, int_129060)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 436, 42), tuple_129058, subscript_call_result_129066)
        
        # Getting the type of 'self' (line 436)
        self_129067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 16), 'self')
        # Obtaining the member '_attributes' of a type (line 436)
        _attributes_129068 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 436, 16), self_129067, '_attributes')
        # Getting the type of 'name' (line 436)
        name_129069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 33), 'name')
        # Storing an element on a container (line 436)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 436, 16), _attributes_129068, (name_129069, tuple_129058))
        # SSA branch for the else part of an if statement (line 435)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Tuple to a Subscript (line 438):
        
        # Assigning a Tuple to a Subscript (line 438):
        
        # Obtaining an instance of the builtin type 'tuple' (line 438)
        tuple_129070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 438, 42), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 438)
        # Adding element type (line 438)
        # Getting the type of 'tp' (line 438)
        tp_129071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 42), 'tp')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 438, 42), tuple_129070, tp_129071)
        # Adding element type (line 438)
        # Getting the type of 'None' (line 438)
        None_129072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 46), 'None')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 438, 42), tuple_129070, None_129072)
        
        # Getting the type of 'self' (line 438)
        self_129073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 16), 'self')
        # Obtaining the member '_attributes' of a type (line 438)
        _attributes_129074 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 438, 16), self_129073, '_attributes')
        # Getting the type of 'name' (line 438)
        name_129075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 33), 'name')
        # Storing an element on a container (line 438)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 438, 16), _attributes_129074, (name_129075, tuple_129070))
        # SSA join for if statement (line 435)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 433)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def stypy__repr__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__repr__'
        module_type_store = module_type_store.open_function_context('__repr__', 440, 4, False)
        # Assigning a type to the variable 'self' (line 441)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 441, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MetaData.stypy__repr__.__dict__.__setitem__('stypy_localization', localization)
        MetaData.stypy__repr__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MetaData.stypy__repr__.__dict__.__setitem__('stypy_type_store', module_type_store)
        MetaData.stypy__repr__.__dict__.__setitem__('stypy_function_name', 'MetaData.stypy__repr__')
        MetaData.stypy__repr__.__dict__.__setitem__('stypy_param_names_list', [])
        MetaData.stypy__repr__.__dict__.__setitem__('stypy_varargs_param_name', None)
        MetaData.stypy__repr__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MetaData.stypy__repr__.__dict__.__setitem__('stypy_call_defaults', defaults)
        MetaData.stypy__repr__.__dict__.__setitem__('stypy_call_varargs', varargs)
        MetaData.stypy__repr__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MetaData.stypy__repr__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MetaData.stypy__repr__', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Str to a Name (line 441):
        
        # Assigning a Str to a Name (line 441):
        str_129076 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 441, 14), 'str', '')
        # Assigning a type to the variable 'msg' (line 441)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 441, 8), 'msg', str_129076)
        
        # Getting the type of 'msg' (line 442)
        msg_129077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 8), 'msg')
        str_129078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 442, 15), 'str', 'Dataset: %s\n')
        # Getting the type of 'self' (line 442)
        self_129079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 33), 'self')
        # Obtaining the member 'name' of a type (line 442)
        name_129080 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 442, 33), self_129079, 'name')
        # Applying the binary operator '%' (line 442)
        result_mod_129081 = python_operator(stypy.reporting.localization.Localization(__file__, 442, 15), '%', str_129078, name_129080)
        
        # Applying the binary operator '+=' (line 442)
        result_iadd_129082 = python_operator(stypy.reporting.localization.Localization(__file__, 442, 8), '+=', msg_129077, result_mod_129081)
        # Assigning a type to the variable 'msg' (line 442)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 442, 8), 'msg', result_iadd_129082)
        
        
        # Getting the type of 'self' (line 443)
        self_129083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 17), 'self')
        # Obtaining the member '_attrnames' of a type (line 443)
        _attrnames_129084 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 443, 17), self_129083, '_attrnames')
        # Testing the type of a for loop iterable (line 443)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 443, 8), _attrnames_129084)
        # Getting the type of the for loop variable (line 443)
        for_loop_var_129085 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 443, 8), _attrnames_129084)
        # Assigning a type to the variable 'i' (line 443)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 443, 8), 'i', for_loop_var_129085)
        # SSA begins for a for statement (line 443)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'msg' (line 444)
        msg_129086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 12), 'msg')
        str_129087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 444, 19), 'str', "\t%s's type is %s")
        
        # Obtaining an instance of the builtin type 'tuple' (line 444)
        tuple_129088 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 444, 42), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 444)
        # Adding element type (line 444)
        # Getting the type of 'i' (line 444)
        i_129089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 42), 'i')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 444, 42), tuple_129088, i_129089)
        # Adding element type (line 444)
        
        # Obtaining the type of the subscript
        int_129090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 444, 65), 'int')
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 444)
        i_129091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 62), 'i')
        # Getting the type of 'self' (line 444)
        self_129092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 45), 'self')
        # Obtaining the member '_attributes' of a type (line 444)
        _attributes_129093 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 444, 45), self_129092, '_attributes')
        # Obtaining the member '__getitem__' of a type (line 444)
        getitem___129094 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 444, 45), _attributes_129093, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 444)
        subscript_call_result_129095 = invoke(stypy.reporting.localization.Localization(__file__, 444, 45), getitem___129094, i_129091)
        
        # Obtaining the member '__getitem__' of a type (line 444)
        getitem___129096 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 444, 45), subscript_call_result_129095, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 444)
        subscript_call_result_129097 = invoke(stypy.reporting.localization.Localization(__file__, 444, 45), getitem___129096, int_129090)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 444, 42), tuple_129088, subscript_call_result_129097)
        
        # Applying the binary operator '%' (line 444)
        result_mod_129098 = python_operator(stypy.reporting.localization.Localization(__file__, 444, 19), '%', str_129087, tuple_129088)
        
        # Applying the binary operator '+=' (line 444)
        result_iadd_129099 = python_operator(stypy.reporting.localization.Localization(__file__, 444, 12), '+=', msg_129086, result_mod_129098)
        # Assigning a type to the variable 'msg' (line 444)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 12), 'msg', result_iadd_129099)
        
        
        
        # Obtaining the type of the subscript
        int_129100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 445, 35), 'int')
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 445)
        i_129101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 32), 'i')
        # Getting the type of 'self' (line 445)
        self_129102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 15), 'self')
        # Obtaining the member '_attributes' of a type (line 445)
        _attributes_129103 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 445, 15), self_129102, '_attributes')
        # Obtaining the member '__getitem__' of a type (line 445)
        getitem___129104 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 445, 15), _attributes_129103, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 445)
        subscript_call_result_129105 = invoke(stypy.reporting.localization.Localization(__file__, 445, 15), getitem___129104, i_129101)
        
        # Obtaining the member '__getitem__' of a type (line 445)
        getitem___129106 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 445, 15), subscript_call_result_129105, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 445)
        subscript_call_result_129107 = invoke(stypy.reporting.localization.Localization(__file__, 445, 15), getitem___129106, int_129100)
        
        # Testing the type of an if condition (line 445)
        if_condition_129108 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 445, 12), subscript_call_result_129107)
        # Assigning a type to the variable 'if_condition_129108' (line 445)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 445, 12), 'if_condition_129108', if_condition_129108)
        # SSA begins for if statement (line 445)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'msg' (line 446)
        msg_129109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 16), 'msg')
        str_129110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 446, 23), 'str', ', range is %s')
        
        # Call to str(...): (line 446)
        # Processing the call arguments (line 446)
        
        # Obtaining the type of the subscript
        int_129112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 446, 65), 'int')
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 446)
        i_129113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 62), 'i', False)
        # Getting the type of 'self' (line 446)
        self_129114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 45), 'self', False)
        # Obtaining the member '_attributes' of a type (line 446)
        _attributes_129115 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 446, 45), self_129114, '_attributes')
        # Obtaining the member '__getitem__' of a type (line 446)
        getitem___129116 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 446, 45), _attributes_129115, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 446)
        subscript_call_result_129117 = invoke(stypy.reporting.localization.Localization(__file__, 446, 45), getitem___129116, i_129113)
        
        # Obtaining the member '__getitem__' of a type (line 446)
        getitem___129118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 446, 45), subscript_call_result_129117, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 446)
        subscript_call_result_129119 = invoke(stypy.reporting.localization.Localization(__file__, 446, 45), getitem___129118, int_129112)
        
        # Processing the call keyword arguments (line 446)
        kwargs_129120 = {}
        # Getting the type of 'str' (line 446)
        str_129111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 41), 'str', False)
        # Calling str(args, kwargs) (line 446)
        str_call_result_129121 = invoke(stypy.reporting.localization.Localization(__file__, 446, 41), str_129111, *[subscript_call_result_129119], **kwargs_129120)
        
        # Applying the binary operator '%' (line 446)
        result_mod_129122 = python_operator(stypy.reporting.localization.Localization(__file__, 446, 23), '%', str_129110, str_call_result_129121)
        
        # Applying the binary operator '+=' (line 446)
        result_iadd_129123 = python_operator(stypy.reporting.localization.Localization(__file__, 446, 16), '+=', msg_129109, result_mod_129122)
        # Assigning a type to the variable 'msg' (line 446)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 446, 16), 'msg', result_iadd_129123)
        
        # SSA join for if statement (line 445)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'msg' (line 447)
        msg_129124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 12), 'msg')
        str_129125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 447, 19), 'str', '\n')
        # Applying the binary operator '+=' (line 447)
        result_iadd_129126 = python_operator(stypy.reporting.localization.Localization(__file__, 447, 12), '+=', msg_129124, str_129125)
        # Assigning a type to the variable 'msg' (line 447)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 447, 12), 'msg', result_iadd_129126)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'msg' (line 448)
        msg_129127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 15), 'msg')
        # Assigning a type to the variable 'stypy_return_type' (line 448)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 448, 8), 'stypy_return_type', msg_129127)
        
        # ################# End of '__repr__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__repr__' in the type store
        # Getting the type of 'stypy_return_type' (line 440)
        stypy_return_type_129128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_129128)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__repr__'
        return stypy_return_type_129128


    @norecursion
    def __iter__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__iter__'
        module_type_store = module_type_store.open_function_context('__iter__', 450, 4, False)
        # Assigning a type to the variable 'self' (line 451)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 451, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MetaData.__iter__.__dict__.__setitem__('stypy_localization', localization)
        MetaData.__iter__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MetaData.__iter__.__dict__.__setitem__('stypy_type_store', module_type_store)
        MetaData.__iter__.__dict__.__setitem__('stypy_function_name', 'MetaData.__iter__')
        MetaData.__iter__.__dict__.__setitem__('stypy_param_names_list', [])
        MetaData.__iter__.__dict__.__setitem__('stypy_varargs_param_name', None)
        MetaData.__iter__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MetaData.__iter__.__dict__.__setitem__('stypy_call_defaults', defaults)
        MetaData.__iter__.__dict__.__setitem__('stypy_call_varargs', varargs)
        MetaData.__iter__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MetaData.__iter__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MetaData.__iter__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__iter__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__iter__(...)' code ##################

        
        # Call to iter(...): (line 451)
        # Processing the call arguments (line 451)
        # Getting the type of 'self' (line 451)
        self_129130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 20), 'self', False)
        # Obtaining the member '_attrnames' of a type (line 451)
        _attrnames_129131 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 451, 20), self_129130, '_attrnames')
        # Processing the call keyword arguments (line 451)
        kwargs_129132 = {}
        # Getting the type of 'iter' (line 451)
        iter_129129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 15), 'iter', False)
        # Calling iter(args, kwargs) (line 451)
        iter_call_result_129133 = invoke(stypy.reporting.localization.Localization(__file__, 451, 15), iter_129129, *[_attrnames_129131], **kwargs_129132)
        
        # Assigning a type to the variable 'stypy_return_type' (line 451)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 451, 8), 'stypy_return_type', iter_call_result_129133)
        
        # ################# End of '__iter__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__iter__' in the type store
        # Getting the type of 'stypy_return_type' (line 450)
        stypy_return_type_129134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_129134)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__iter__'
        return stypy_return_type_129134


    @norecursion
    def __getitem__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__getitem__'
        module_type_store = module_type_store.open_function_context('__getitem__', 453, 4, False)
        # Assigning a type to the variable 'self' (line 454)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 454, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MetaData.__getitem__.__dict__.__setitem__('stypy_localization', localization)
        MetaData.__getitem__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MetaData.__getitem__.__dict__.__setitem__('stypy_type_store', module_type_store)
        MetaData.__getitem__.__dict__.__setitem__('stypy_function_name', 'MetaData.__getitem__')
        MetaData.__getitem__.__dict__.__setitem__('stypy_param_names_list', ['key'])
        MetaData.__getitem__.__dict__.__setitem__('stypy_varargs_param_name', None)
        MetaData.__getitem__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MetaData.__getitem__.__dict__.__setitem__('stypy_call_defaults', defaults)
        MetaData.__getitem__.__dict__.__setitem__('stypy_call_varargs', varargs)
        MetaData.__getitem__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MetaData.__getitem__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MetaData.__getitem__', ['key'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__getitem__', localization, ['key'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__getitem__(...)' code ##################

        
        # Obtaining the type of the subscript
        # Getting the type of 'key' (line 454)
        key_129135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 32), 'key')
        # Getting the type of 'self' (line 454)
        self_129136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 15), 'self')
        # Obtaining the member '_attributes' of a type (line 454)
        _attributes_129137 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 454, 15), self_129136, '_attributes')
        # Obtaining the member '__getitem__' of a type (line 454)
        getitem___129138 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 454, 15), _attributes_129137, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 454)
        subscript_call_result_129139 = invoke(stypy.reporting.localization.Localization(__file__, 454, 15), getitem___129138, key_129135)
        
        # Assigning a type to the variable 'stypy_return_type' (line 454)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 454, 8), 'stypy_return_type', subscript_call_result_129139)
        
        # ################# End of '__getitem__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__getitem__' in the type store
        # Getting the type of 'stypy_return_type' (line 453)
        stypy_return_type_129140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_129140)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__getitem__'
        return stypy_return_type_129140


    @norecursion
    def names(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'names'
        module_type_store = module_type_store.open_function_context('names', 456, 4, False)
        # Assigning a type to the variable 'self' (line 457)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 457, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MetaData.names.__dict__.__setitem__('stypy_localization', localization)
        MetaData.names.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MetaData.names.__dict__.__setitem__('stypy_type_store', module_type_store)
        MetaData.names.__dict__.__setitem__('stypy_function_name', 'MetaData.names')
        MetaData.names.__dict__.__setitem__('stypy_param_names_list', [])
        MetaData.names.__dict__.__setitem__('stypy_varargs_param_name', None)
        MetaData.names.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MetaData.names.__dict__.__setitem__('stypy_call_defaults', defaults)
        MetaData.names.__dict__.__setitem__('stypy_call_varargs', varargs)
        MetaData.names.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MetaData.names.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MetaData.names', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'names', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'names(...)' code ##################

        str_129141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 457, 8), 'str', 'Return the list of attribute names.')
        # Getting the type of 'self' (line 458)
        self_129142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 15), 'self')
        # Obtaining the member '_attrnames' of a type (line 458)
        _attrnames_129143 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 458, 15), self_129142, '_attrnames')
        # Assigning a type to the variable 'stypy_return_type' (line 458)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 458, 8), 'stypy_return_type', _attrnames_129143)
        
        # ################# End of 'names(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'names' in the type store
        # Getting the type of 'stypy_return_type' (line 456)
        stypy_return_type_129144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_129144)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'names'
        return stypy_return_type_129144


    @norecursion
    def types(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'types'
        module_type_store = module_type_store.open_function_context('types', 460, 4, False)
        # Assigning a type to the variable 'self' (line 461)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 461, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MetaData.types.__dict__.__setitem__('stypy_localization', localization)
        MetaData.types.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MetaData.types.__dict__.__setitem__('stypy_type_store', module_type_store)
        MetaData.types.__dict__.__setitem__('stypy_function_name', 'MetaData.types')
        MetaData.types.__dict__.__setitem__('stypy_param_names_list', [])
        MetaData.types.__dict__.__setitem__('stypy_varargs_param_name', None)
        MetaData.types.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MetaData.types.__dict__.__setitem__('stypy_call_defaults', defaults)
        MetaData.types.__dict__.__setitem__('stypy_call_varargs', varargs)
        MetaData.types.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MetaData.types.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MetaData.types', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'types', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'types(...)' code ##################

        str_129145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 461, 8), 'str', 'Return the list of attribute types.')
        
        # Assigning a ListComp to a Name (line 462):
        
        # Assigning a ListComp to a Name (line 462):
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'self' (line 462)
        self_129154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 60), 'self')
        # Obtaining the member '_attrnames' of a type (line 462)
        _attrnames_129155 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 462, 60), self_129154, '_attrnames')
        comprehension_129156 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 462, 22), _attrnames_129155)
        # Assigning a type to the variable 'name' (line 462)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 462, 22), 'name', comprehension_129156)
        
        # Obtaining the type of the subscript
        int_129146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 462, 45), 'int')
        
        # Obtaining the type of the subscript
        # Getting the type of 'name' (line 462)
        name_129147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 39), 'name')
        # Getting the type of 'self' (line 462)
        self_129148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 22), 'self')
        # Obtaining the member '_attributes' of a type (line 462)
        _attributes_129149 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 462, 22), self_129148, '_attributes')
        # Obtaining the member '__getitem__' of a type (line 462)
        getitem___129150 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 462, 22), _attributes_129149, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 462)
        subscript_call_result_129151 = invoke(stypy.reporting.localization.Localization(__file__, 462, 22), getitem___129150, name_129147)
        
        # Obtaining the member '__getitem__' of a type (line 462)
        getitem___129152 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 462, 22), subscript_call_result_129151, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 462)
        subscript_call_result_129153 = invoke(stypy.reporting.localization.Localization(__file__, 462, 22), getitem___129152, int_129146)
        
        list_129157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 462, 22), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 462, 22), list_129157, subscript_call_result_129153)
        # Assigning a type to the variable 'attr_types' (line 462)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 462, 8), 'attr_types', list_129157)
        # Getting the type of 'attr_types' (line 463)
        attr_types_129158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 15), 'attr_types')
        # Assigning a type to the variable 'stypy_return_type' (line 463)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 463, 8), 'stypy_return_type', attr_types_129158)
        
        # ################# End of 'types(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'types' in the type store
        # Getting the type of 'stypy_return_type' (line 460)
        stypy_return_type_129159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_129159)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'types'
        return stypy_return_type_129159


# Assigning a type to the variable 'MetaData' (line 400)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 400, 0), 'MetaData', MetaData)

@norecursion
def loadarff(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'loadarff'
    module_type_store = module_type_store.open_function_context('loadarff', 466, 0, False)
    
    # Passed parameters checking function
    loadarff.stypy_localization = localization
    loadarff.stypy_type_of_self = None
    loadarff.stypy_type_store = module_type_store
    loadarff.stypy_function_name = 'loadarff'
    loadarff.stypy_param_names_list = ['f']
    loadarff.stypy_varargs_param_name = None
    loadarff.stypy_kwargs_param_name = None
    loadarff.stypy_call_defaults = defaults
    loadarff.stypy_call_varargs = varargs
    loadarff.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'loadarff', ['f'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'loadarff', localization, ['f'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'loadarff(...)' code ##################

    str_129160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 535, (-1)), 'str', '\n    Read an arff file.\n\n    The data is returned as a record array, which can be accessed much like\n    a dictionary of numpy arrays.  For example, if one of the attributes is\n    called \'pressure\', then its first 10 data points can be accessed from the\n    ``data`` record array like so: ``data[\'pressure\'][0:10]``\n\n\n    Parameters\n    ----------\n    f : file-like or str\n       File-like object to read from, or filename to open.\n\n    Returns\n    -------\n    data : record array\n       The data of the arff file, accessible by attribute names.\n    meta : `MetaData`\n       Contains information about the arff file such as name and\n       type of attributes, the relation (name of the dataset), etc...\n\n    Raises\n    ------\n    ParseArffError\n        This is raised if the given file is not ARFF-formatted.\n    NotImplementedError\n        The ARFF file has an attribute which is not supported yet.\n\n    Notes\n    -----\n\n    This function should be able to read most arff files. Not\n    implemented functionality include:\n\n    * date type attributes\n    * string type attributes\n\n    It can read files with numeric and nominal attributes.  It cannot read\n    files with sparse data ({} in the file).  However, this function can\n    read files with missing data (? in the file), representing the data\n    points as NaNs.\n\n    Examples\n    --------\n    >>> from scipy.io import arff\n    >>> from io import StringIO\n    >>> content = """\n    ... @relation foo\n    ... @attribute width  numeric\n    ... @attribute height numeric\n    ... @attribute color  {red,green,blue,yellow,black}\n    ... @data\n    ... 5.0,3.25,blue\n    ... 4.5,3.75,green\n    ... 3.0,4.00,red\n    ... """\n    >>> f = StringIO(content)\n    >>> data, meta = arff.loadarff(f)\n    >>> data\n    array([(5.0, 3.25, \'blue\'), (4.5, 3.75, \'green\'), (3.0, 4.0, \'red\')],\n          dtype=[(\'width\', \'<f8\'), (\'height\', \'<f8\'), (\'color\', \'|S6\')])\n    >>> meta\n    Dataset: foo\n    \twidth\'s type is numeric\n    \theight\'s type is numeric\n    \tcolor\'s type is nominal, range is (\'red\', \'green\', \'blue\', \'yellow\', \'black\')\n\n    ')
    
    # Type idiom detected: calculating its left and rigth part (line 536)
    str_129161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 536, 18), 'str', 'read')
    # Getting the type of 'f' (line 536)
    f_129162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 15), 'f')
    
    (may_be_129163, more_types_in_union_129164) = may_provide_member(str_129161, f_129162)

    if may_be_129163:

        if more_types_in_union_129164:
            # Runtime conditional SSA (line 536)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'f' (line 536)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 536, 4), 'f', remove_not_member_provider_from_union(f_129162, 'read'))
        
        # Assigning a Name to a Name (line 537):
        
        # Assigning a Name to a Name (line 537):
        # Getting the type of 'f' (line 537)
        f_129165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 16), 'f')
        # Assigning a type to the variable 'ofile' (line 537)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 537, 8), 'ofile', f_129165)

        if more_types_in_union_129164:
            # Runtime conditional SSA for else branch (line 536)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_129163) or more_types_in_union_129164):
        # Assigning a type to the variable 'f' (line 536)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 536, 4), 'f', remove_member_provider_from_union(f_129162, 'read'))
        
        # Assigning a Call to a Name (line 539):
        
        # Assigning a Call to a Name (line 539):
        
        # Call to open(...): (line 539)
        # Processing the call arguments (line 539)
        # Getting the type of 'f' (line 539)
        f_129167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 21), 'f', False)
        str_129168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 539, 24), 'str', 'rt')
        # Processing the call keyword arguments (line 539)
        kwargs_129169 = {}
        # Getting the type of 'open' (line 539)
        open_129166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 16), 'open', False)
        # Calling open(args, kwargs) (line 539)
        open_call_result_129170 = invoke(stypy.reporting.localization.Localization(__file__, 539, 16), open_129166, *[f_129167, str_129168], **kwargs_129169)
        
        # Assigning a type to the variable 'ofile' (line 539)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 539, 8), 'ofile', open_call_result_129170)

        if (may_be_129163 and more_types_in_union_129164):
            # SSA join for if statement (line 536)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Try-finally block (line 540)
    
    # Call to _loadarff(...): (line 541)
    # Processing the call arguments (line 541)
    # Getting the type of 'ofile' (line 541)
    ofile_129172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 25), 'ofile', False)
    # Processing the call keyword arguments (line 541)
    kwargs_129173 = {}
    # Getting the type of '_loadarff' (line 541)
    _loadarff_129171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 15), '_loadarff', False)
    # Calling _loadarff(args, kwargs) (line 541)
    _loadarff_call_result_129174 = invoke(stypy.reporting.localization.Localization(__file__, 541, 15), _loadarff_129171, *[ofile_129172], **kwargs_129173)
    
    # Assigning a type to the variable 'stypy_return_type' (line 541)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 541, 8), 'stypy_return_type', _loadarff_call_result_129174)
    
    # finally branch of the try-finally block (line 540)
    
    
    # Getting the type of 'ofile' (line 543)
    ofile_129175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 11), 'ofile')
    # Getting the type of 'f' (line 543)
    f_129176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 24), 'f')
    # Applying the binary operator 'isnot' (line 543)
    result_is_not_129177 = python_operator(stypy.reporting.localization.Localization(__file__, 543, 11), 'isnot', ofile_129175, f_129176)
    
    # Testing the type of an if condition (line 543)
    if_condition_129178 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 543, 8), result_is_not_129177)
    # Assigning a type to the variable 'if_condition_129178' (line 543)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 543, 8), 'if_condition_129178', if_condition_129178)
    # SSA begins for if statement (line 543)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to close(...): (line 544)
    # Processing the call keyword arguments (line 544)
    kwargs_129181 = {}
    # Getting the type of 'ofile' (line 544)
    ofile_129179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 12), 'ofile', False)
    # Obtaining the member 'close' of a type (line 544)
    close_129180 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 544, 12), ofile_129179, 'close')
    # Calling close(args, kwargs) (line 544)
    close_call_result_129182 = invoke(stypy.reporting.localization.Localization(__file__, 544, 12), close_129180, *[], **kwargs_129181)
    
    # SSA join for if statement (line 543)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # ################# End of 'loadarff(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'loadarff' in the type store
    # Getting the type of 'stypy_return_type' (line 466)
    stypy_return_type_129183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_129183)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'loadarff'
    return stypy_return_type_129183

# Assigning a type to the variable 'loadarff' (line 466)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 466, 0), 'loadarff', loadarff)

@norecursion
def _loadarff(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_loadarff'
    module_type_store = module_type_store.open_function_context('_loadarff', 547, 0, False)
    
    # Passed parameters checking function
    _loadarff.stypy_localization = localization
    _loadarff.stypy_type_of_self = None
    _loadarff.stypy_type_store = module_type_store
    _loadarff.stypy_function_name = '_loadarff'
    _loadarff.stypy_param_names_list = ['ofile']
    _loadarff.stypy_varargs_param_name = None
    _loadarff.stypy_kwargs_param_name = None
    _loadarff.stypy_call_defaults = defaults
    _loadarff.stypy_call_varargs = varargs
    _loadarff.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_loadarff', ['ofile'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_loadarff', localization, ['ofile'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_loadarff(...)' code ##################

    
    
    # SSA begins for try-except statement (line 549)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Tuple (line 550):
    
    # Assigning a Subscript to a Name (line 550):
    
    # Obtaining the type of the subscript
    int_129184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 550, 8), 'int')
    
    # Call to read_header(...): (line 550)
    # Processing the call arguments (line 550)
    # Getting the type of 'ofile' (line 550)
    ofile_129186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 32), 'ofile', False)
    # Processing the call keyword arguments (line 550)
    kwargs_129187 = {}
    # Getting the type of 'read_header' (line 550)
    read_header_129185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 20), 'read_header', False)
    # Calling read_header(args, kwargs) (line 550)
    read_header_call_result_129188 = invoke(stypy.reporting.localization.Localization(__file__, 550, 20), read_header_129185, *[ofile_129186], **kwargs_129187)
    
    # Obtaining the member '__getitem__' of a type (line 550)
    getitem___129189 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 550, 8), read_header_call_result_129188, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 550)
    subscript_call_result_129190 = invoke(stypy.reporting.localization.Localization(__file__, 550, 8), getitem___129189, int_129184)
    
    # Assigning a type to the variable 'tuple_var_assignment_128250' (line 550)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 550, 8), 'tuple_var_assignment_128250', subscript_call_result_129190)
    
    # Assigning a Subscript to a Name (line 550):
    
    # Obtaining the type of the subscript
    int_129191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 550, 8), 'int')
    
    # Call to read_header(...): (line 550)
    # Processing the call arguments (line 550)
    # Getting the type of 'ofile' (line 550)
    ofile_129193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 32), 'ofile', False)
    # Processing the call keyword arguments (line 550)
    kwargs_129194 = {}
    # Getting the type of 'read_header' (line 550)
    read_header_129192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 20), 'read_header', False)
    # Calling read_header(args, kwargs) (line 550)
    read_header_call_result_129195 = invoke(stypy.reporting.localization.Localization(__file__, 550, 20), read_header_129192, *[ofile_129193], **kwargs_129194)
    
    # Obtaining the member '__getitem__' of a type (line 550)
    getitem___129196 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 550, 8), read_header_call_result_129195, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 550)
    subscript_call_result_129197 = invoke(stypy.reporting.localization.Localization(__file__, 550, 8), getitem___129196, int_129191)
    
    # Assigning a type to the variable 'tuple_var_assignment_128251' (line 550)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 550, 8), 'tuple_var_assignment_128251', subscript_call_result_129197)
    
    # Assigning a Name to a Name (line 550):
    # Getting the type of 'tuple_var_assignment_128250' (line 550)
    tuple_var_assignment_128250_129198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 8), 'tuple_var_assignment_128250')
    # Assigning a type to the variable 'rel' (line 550)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 550, 8), 'rel', tuple_var_assignment_128250_129198)
    
    # Assigning a Name to a Name (line 550):
    # Getting the type of 'tuple_var_assignment_128251' (line 550)
    tuple_var_assignment_128251_129199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 8), 'tuple_var_assignment_128251')
    # Assigning a type to the variable 'attr' (line 550)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 550, 13), 'attr', tuple_var_assignment_128251_129199)
    # SSA branch for the except part of a try statement (line 549)
    # SSA branch for the except 'ValueError' branch of a try statement (line 549)
    # Storing handler type
    module_type_store.open_ssa_branch('except')
    # Getting the type of 'ValueError' (line 551)
    ValueError_129200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 11), 'ValueError')
    # Assigning a type to the variable 'e' (line 551)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 551, 4), 'e', ValueError_129200)
    
    # Assigning a BinOp to a Name (line 552):
    
    # Assigning a BinOp to a Name (line 552):
    str_129201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 552, 14), 'str', 'Error while parsing header, error was: ')
    
    # Call to str(...): (line 552)
    # Processing the call arguments (line 552)
    # Getting the type of 'e' (line 552)
    e_129203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 62), 'e', False)
    # Processing the call keyword arguments (line 552)
    kwargs_129204 = {}
    # Getting the type of 'str' (line 552)
    str_129202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 58), 'str', False)
    # Calling str(args, kwargs) (line 552)
    str_call_result_129205 = invoke(stypy.reporting.localization.Localization(__file__, 552, 58), str_129202, *[e_129203], **kwargs_129204)
    
    # Applying the binary operator '+' (line 552)
    result_add_129206 = python_operator(stypy.reporting.localization.Localization(__file__, 552, 14), '+', str_129201, str_call_result_129205)
    
    # Assigning a type to the variable 'msg' (line 552)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 552, 8), 'msg', result_add_129206)
    
    # Call to ParseArffError(...): (line 553)
    # Processing the call arguments (line 553)
    # Getting the type of 'msg' (line 553)
    msg_129208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 29), 'msg', False)
    # Processing the call keyword arguments (line 553)
    kwargs_129209 = {}
    # Getting the type of 'ParseArffError' (line 553)
    ParseArffError_129207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 14), 'ParseArffError', False)
    # Calling ParseArffError(args, kwargs) (line 553)
    ParseArffError_call_result_129210 = invoke(stypy.reporting.localization.Localization(__file__, 553, 14), ParseArffError_129207, *[msg_129208], **kwargs_129209)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 553, 8), ParseArffError_call_result_129210, 'raise parameter', BaseException)
    # SSA join for try-except statement (line 549)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Name (line 556):
    
    # Assigning a Name to a Name (line 556):
    # Getting the type of 'False' (line 556)
    False_129211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 13), 'False')
    # Assigning a type to the variable 'hasstr' (line 556)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 556, 4), 'hasstr', False_129211)
    
    # Getting the type of 'attr' (line 557)
    attr_129212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 23), 'attr')
    # Testing the type of a for loop iterable (line 557)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 557, 4), attr_129212)
    # Getting the type of the for loop variable (line 557)
    for_loop_var_129213 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 557, 4), attr_129212)
    # Assigning a type to the variable 'name' (line 557)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 557, 4), 'name', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 557, 4), for_loop_var_129213))
    # Assigning a type to the variable 'value' (line 557)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 557, 4), 'value', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 557, 4), for_loop_var_129213))
    # SSA begins for a for statement (line 557)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 558):
    
    # Assigning a Call to a Name (line 558):
    
    # Call to parse_type(...): (line 558)
    # Processing the call arguments (line 558)
    # Getting the type of 'value' (line 558)
    value_129215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 26), 'value', False)
    # Processing the call keyword arguments (line 558)
    kwargs_129216 = {}
    # Getting the type of 'parse_type' (line 558)
    parse_type_129214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 15), 'parse_type', False)
    # Calling parse_type(args, kwargs) (line 558)
    parse_type_call_result_129217 = invoke(stypy.reporting.localization.Localization(__file__, 558, 15), parse_type_129214, *[value_129215], **kwargs_129216)
    
    # Assigning a type to the variable 'type' (line 558)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 558, 8), 'type', parse_type_call_result_129217)
    
    
    # Getting the type of 'type' (line 559)
    type_129218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 11), 'type')
    str_129219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 559, 19), 'str', 'string')
    # Applying the binary operator '==' (line 559)
    result_eq_129220 = python_operator(stypy.reporting.localization.Localization(__file__, 559, 11), '==', type_129218, str_129219)
    
    # Testing the type of an if condition (line 559)
    if_condition_129221 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 559, 8), result_eq_129220)
    # Assigning a type to the variable 'if_condition_129221' (line 559)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 559, 8), 'if_condition_129221', if_condition_129221)
    # SSA begins for if statement (line 559)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 560):
    
    # Assigning a Name to a Name (line 560):
    # Getting the type of 'True' (line 560)
    True_129222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 21), 'True')
    # Assigning a type to the variable 'hasstr' (line 560)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 560, 12), 'hasstr', True_129222)
    # SSA join for if statement (line 559)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 562):
    
    # Assigning a Call to a Name (line 562):
    
    # Call to MetaData(...): (line 562)
    # Processing the call arguments (line 562)
    # Getting the type of 'rel' (line 562)
    rel_129224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 20), 'rel', False)
    # Getting the type of 'attr' (line 562)
    attr_129225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 25), 'attr', False)
    # Processing the call keyword arguments (line 562)
    kwargs_129226 = {}
    # Getting the type of 'MetaData' (line 562)
    MetaData_129223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 11), 'MetaData', False)
    # Calling MetaData(args, kwargs) (line 562)
    MetaData_call_result_129227 = invoke(stypy.reporting.localization.Localization(__file__, 562, 11), MetaData_129223, *[rel_129224, attr_129225], **kwargs_129226)
    
    # Assigning a type to the variable 'meta' (line 562)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 562, 4), 'meta', MetaData_call_result_129227)
    
    # Assigning a Dict to a Name (line 571):
    
    # Assigning a Dict to a Name (line 571):
    
    # Obtaining an instance of the builtin type 'dict' (line 571)
    dict_129228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 571, 17), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 571)
    # Adding element type (key, value) (line 571)
    str_129229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 571, 18), 'str', 'real')
    # Getting the type of 'float' (line 571)
    float_129230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 26), 'float')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 571, 17), dict_129228, (str_129229, float_129230))
    # Adding element type (key, value) (line 571)
    str_129231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 571, 33), 'str', 'integer')
    # Getting the type of 'float' (line 571)
    float_129232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 44), 'float')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 571, 17), dict_129228, (str_129231, float_129232))
    # Adding element type (key, value) (line 571)
    str_129233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 571, 51), 'str', 'numeric')
    # Getting the type of 'float' (line 571)
    float_129234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 62), 'float')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 571, 17), dict_129228, (str_129233, float_129234))
    
    # Assigning a type to the variable 'acls2dtype' (line 571)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 571, 4), 'acls2dtype', dict_129228)
    
    # Assigning a Dict to a Name (line 572):
    
    # Assigning a Dict to a Name (line 572):
    
    # Obtaining an instance of the builtin type 'dict' (line 572)
    dict_129235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 572, 16), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 572)
    # Adding element type (key, value) (line 572)
    str_129236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 572, 17), 'str', 'real')
    # Getting the type of 'safe_float' (line 572)
    safe_float_129237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 25), 'safe_float')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 572, 16), dict_129235, (str_129236, safe_float_129237))
    # Adding element type (key, value) (line 572)
    str_129238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 573, 17), 'str', 'integer')
    # Getting the type of 'safe_float' (line 573)
    safe_float_129239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 28), 'safe_float')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 572, 16), dict_129235, (str_129238, safe_float_129239))
    # Adding element type (key, value) (line 572)
    str_129240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 574, 17), 'str', 'numeric')
    # Getting the type of 'safe_float' (line 574)
    safe_float_129241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 28), 'safe_float')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 572, 16), dict_129235, (str_129240, safe_float_129241))
    
    # Assigning a type to the variable 'acls2conv' (line 572)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 572, 4), 'acls2conv', dict_129235)
    
    # Assigning a List to a Name (line 575):
    
    # Assigning a List to a Name (line 575):
    
    # Obtaining an instance of the builtin type 'list' (line 575)
    list_129242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 575, 12), 'list')
    # Adding type elements to the builtin type 'list' instance (line 575)
    
    # Assigning a type to the variable 'descr' (line 575)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 575, 4), 'descr', list_129242)
    
    # Assigning a List to a Name (line 576):
    
    # Assigning a List to a Name (line 576):
    
    # Obtaining an instance of the builtin type 'list' (line 576)
    list_129243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 576, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 576)
    
    # Assigning a type to the variable 'convertors' (line 576)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 576, 4), 'convertors', list_129243)
    
    
    # Getting the type of 'hasstr' (line 577)
    hasstr_129244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 11), 'hasstr')
    # Applying the 'not' unary operator (line 577)
    result_not__129245 = python_operator(stypy.reporting.localization.Localization(__file__, 577, 7), 'not', hasstr_129244)
    
    # Testing the type of an if condition (line 577)
    if_condition_129246 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 577, 4), result_not__129245)
    # Assigning a type to the variable 'if_condition_129246' (line 577)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 577, 4), 'if_condition_129246', if_condition_129246)
    # SSA begins for if statement (line 577)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'attr' (line 578)
    attr_129247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 27), 'attr')
    # Testing the type of a for loop iterable (line 578)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 578, 8), attr_129247)
    # Getting the type of the for loop variable (line 578)
    for_loop_var_129248 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 578, 8), attr_129247)
    # Assigning a type to the variable 'name' (line 578)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 578, 8), 'name', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 578, 8), for_loop_var_129248))
    # Assigning a type to the variable 'value' (line 578)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 578, 8), 'value', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 578, 8), for_loop_var_129248))
    # SSA begins for a for statement (line 578)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 579):
    
    # Assigning a Call to a Name (line 579):
    
    # Call to parse_type(...): (line 579)
    # Processing the call arguments (line 579)
    # Getting the type of 'value' (line 579)
    value_129250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 30), 'value', False)
    # Processing the call keyword arguments (line 579)
    kwargs_129251 = {}
    # Getting the type of 'parse_type' (line 579)
    parse_type_129249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 19), 'parse_type', False)
    # Calling parse_type(args, kwargs) (line 579)
    parse_type_call_result_129252 = invoke(stypy.reporting.localization.Localization(__file__, 579, 19), parse_type_129249, *[value_129250], **kwargs_129251)
    
    # Assigning a type to the variable 'type' (line 579)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 579, 12), 'type', parse_type_call_result_129252)
    
    
    # Getting the type of 'type' (line 580)
    type_129253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 15), 'type')
    str_129254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 580, 23), 'str', 'date')
    # Applying the binary operator '==' (line 580)
    result_eq_129255 = python_operator(stypy.reporting.localization.Localization(__file__, 580, 15), '==', type_129253, str_129254)
    
    # Testing the type of an if condition (line 580)
    if_condition_129256 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 580, 12), result_eq_129255)
    # Assigning a type to the variable 'if_condition_129256' (line 580)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 580, 12), 'if_condition_129256', if_condition_129256)
    # SSA begins for if statement (line 580)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Tuple (line 581):
    
    # Assigning a Subscript to a Name (line 581):
    
    # Obtaining the type of the subscript
    int_129257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 581, 16), 'int')
    
    # Call to get_date_format(...): (line 581)
    # Processing the call arguments (line 581)
    # Getting the type of 'value' (line 581)
    value_129259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 61), 'value', False)
    # Processing the call keyword arguments (line 581)
    kwargs_129260 = {}
    # Getting the type of 'get_date_format' (line 581)
    get_date_format_129258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 45), 'get_date_format', False)
    # Calling get_date_format(args, kwargs) (line 581)
    get_date_format_call_result_129261 = invoke(stypy.reporting.localization.Localization(__file__, 581, 45), get_date_format_129258, *[value_129259], **kwargs_129260)
    
    # Obtaining the member '__getitem__' of a type (line 581)
    getitem___129262 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 581, 16), get_date_format_call_result_129261, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 581)
    subscript_call_result_129263 = invoke(stypy.reporting.localization.Localization(__file__, 581, 16), getitem___129262, int_129257)
    
    # Assigning a type to the variable 'tuple_var_assignment_128252' (line 581)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 581, 16), 'tuple_var_assignment_128252', subscript_call_result_129263)
    
    # Assigning a Subscript to a Name (line 581):
    
    # Obtaining the type of the subscript
    int_129264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 581, 16), 'int')
    
    # Call to get_date_format(...): (line 581)
    # Processing the call arguments (line 581)
    # Getting the type of 'value' (line 581)
    value_129266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 61), 'value', False)
    # Processing the call keyword arguments (line 581)
    kwargs_129267 = {}
    # Getting the type of 'get_date_format' (line 581)
    get_date_format_129265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 45), 'get_date_format', False)
    # Calling get_date_format(args, kwargs) (line 581)
    get_date_format_call_result_129268 = invoke(stypy.reporting.localization.Localization(__file__, 581, 45), get_date_format_129265, *[value_129266], **kwargs_129267)
    
    # Obtaining the member '__getitem__' of a type (line 581)
    getitem___129269 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 581, 16), get_date_format_call_result_129268, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 581)
    subscript_call_result_129270 = invoke(stypy.reporting.localization.Localization(__file__, 581, 16), getitem___129269, int_129264)
    
    # Assigning a type to the variable 'tuple_var_assignment_128253' (line 581)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 581, 16), 'tuple_var_assignment_128253', subscript_call_result_129270)
    
    # Assigning a Name to a Name (line 581):
    # Getting the type of 'tuple_var_assignment_128252' (line 581)
    tuple_var_assignment_128252_129271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 16), 'tuple_var_assignment_128252')
    # Assigning a type to the variable 'date_format' (line 581)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 581, 16), 'date_format', tuple_var_assignment_128252_129271)
    
    # Assigning a Name to a Name (line 581):
    # Getting the type of 'tuple_var_assignment_128253' (line 581)
    tuple_var_assignment_128253_129272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 16), 'tuple_var_assignment_128253')
    # Assigning a type to the variable 'datetime_unit' (line 581)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 581, 29), 'datetime_unit', tuple_var_assignment_128253_129272)
    
    # Call to append(...): (line 582)
    # Processing the call arguments (line 582)
    
    # Obtaining an instance of the builtin type 'tuple' (line 582)
    tuple_129275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 582, 30), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 582)
    # Adding element type (line 582)
    # Getting the type of 'name' (line 582)
    name_129276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 30), 'name', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 582, 30), tuple_129275, name_129276)
    # Adding element type (line 582)
    str_129277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 582, 36), 'str', 'datetime64[%s]')
    # Getting the type of 'datetime_unit' (line 582)
    datetime_unit_129278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 55), 'datetime_unit', False)
    # Applying the binary operator '%' (line 582)
    result_mod_129279 = python_operator(stypy.reporting.localization.Localization(__file__, 582, 36), '%', str_129277, datetime_unit_129278)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 582, 30), tuple_129275, result_mod_129279)
    
    # Processing the call keyword arguments (line 582)
    kwargs_129280 = {}
    # Getting the type of 'descr' (line 582)
    descr_129273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 16), 'descr', False)
    # Obtaining the member 'append' of a type (line 582)
    append_129274 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 582, 16), descr_129273, 'append')
    # Calling append(args, kwargs) (line 582)
    append_call_result_129281 = invoke(stypy.reporting.localization.Localization(__file__, 582, 16), append_129274, *[tuple_129275], **kwargs_129280)
    
    
    # Call to append(...): (line 583)
    # Processing the call arguments (line 583)
    
    # Call to partial(...): (line 583)
    # Processing the call arguments (line 583)
    # Getting the type of 'safe_date' (line 583)
    safe_date_129285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 42), 'safe_date', False)
    # Processing the call keyword arguments (line 583)
    # Getting the type of 'date_format' (line 583)
    date_format_129286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 65), 'date_format', False)
    keyword_129287 = date_format_129286
    # Getting the type of 'datetime_unit' (line 584)
    datetime_unit_129288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 56), 'datetime_unit', False)
    keyword_129289 = datetime_unit_129288
    kwargs_129290 = {'date_format': keyword_129287, 'datetime_unit': keyword_129289}
    # Getting the type of 'partial' (line 583)
    partial_129284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 34), 'partial', False)
    # Calling partial(args, kwargs) (line 583)
    partial_call_result_129291 = invoke(stypy.reporting.localization.Localization(__file__, 583, 34), partial_129284, *[safe_date_129285], **kwargs_129290)
    
    # Processing the call keyword arguments (line 583)
    kwargs_129292 = {}
    # Getting the type of 'convertors' (line 583)
    convertors_129282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 16), 'convertors', False)
    # Obtaining the member 'append' of a type (line 583)
    append_129283 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 583, 16), convertors_129282, 'append')
    # Calling append(args, kwargs) (line 583)
    append_call_result_129293 = invoke(stypy.reporting.localization.Localization(__file__, 583, 16), append_129283, *[partial_call_result_129291], **kwargs_129292)
    
    # SSA branch for the else part of an if statement (line 580)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'type' (line 585)
    type_129294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 17), 'type')
    str_129295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 585, 25), 'str', 'nominal')
    # Applying the binary operator '==' (line 585)
    result_eq_129296 = python_operator(stypy.reporting.localization.Localization(__file__, 585, 17), '==', type_129294, str_129295)
    
    # Testing the type of an if condition (line 585)
    if_condition_129297 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 585, 17), result_eq_129296)
    # Assigning a type to the variable 'if_condition_129297' (line 585)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 585, 17), 'if_condition_129297', if_condition_129297)
    # SSA begins for if statement (line 585)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 586):
    
    # Assigning a Call to a Name (line 586):
    
    # Call to maxnomlen(...): (line 586)
    # Processing the call arguments (line 586)
    # Getting the type of 'value' (line 586)
    value_129299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 30), 'value', False)
    # Processing the call keyword arguments (line 586)
    kwargs_129300 = {}
    # Getting the type of 'maxnomlen' (line 586)
    maxnomlen_129298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 20), 'maxnomlen', False)
    # Calling maxnomlen(args, kwargs) (line 586)
    maxnomlen_call_result_129301 = invoke(stypy.reporting.localization.Localization(__file__, 586, 20), maxnomlen_129298, *[value_129299], **kwargs_129300)
    
    # Assigning a type to the variable 'n' (line 586)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 586, 16), 'n', maxnomlen_call_result_129301)
    
    # Call to append(...): (line 587)
    # Processing the call arguments (line 587)
    
    # Obtaining an instance of the builtin type 'tuple' (line 587)
    tuple_129304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 587, 30), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 587)
    # Adding element type (line 587)
    # Getting the type of 'name' (line 587)
    name_129305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 30), 'name', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 587, 30), tuple_129304, name_129305)
    # Adding element type (line 587)
    str_129306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 587, 36), 'str', 'S%d')
    # Getting the type of 'n' (line 587)
    n_129307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 44), 'n', False)
    # Applying the binary operator '%' (line 587)
    result_mod_129308 = python_operator(stypy.reporting.localization.Localization(__file__, 587, 36), '%', str_129306, n_129307)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 587, 30), tuple_129304, result_mod_129308)
    
    # Processing the call keyword arguments (line 587)
    kwargs_129309 = {}
    # Getting the type of 'descr' (line 587)
    descr_129302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 16), 'descr', False)
    # Obtaining the member 'append' of a type (line 587)
    append_129303 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 587, 16), descr_129302, 'append')
    # Calling append(args, kwargs) (line 587)
    append_call_result_129310 = invoke(stypy.reporting.localization.Localization(__file__, 587, 16), append_129303, *[tuple_129304], **kwargs_129309)
    
    
    # Assigning a Call to a Name (line 588):
    
    # Assigning a Call to a Name (line 588):
    
    # Call to get_nom_val(...): (line 588)
    # Processing the call arguments (line 588)
    # Getting the type of 'value' (line 588)
    value_129312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 37), 'value', False)
    # Processing the call keyword arguments (line 588)
    kwargs_129313 = {}
    # Getting the type of 'get_nom_val' (line 588)
    get_nom_val_129311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 25), 'get_nom_val', False)
    # Calling get_nom_val(args, kwargs) (line 588)
    get_nom_val_call_result_129314 = invoke(stypy.reporting.localization.Localization(__file__, 588, 25), get_nom_val_129311, *[value_129312], **kwargs_129313)
    
    # Assigning a type to the variable 'pvalue' (line 588)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 588, 16), 'pvalue', get_nom_val_call_result_129314)
    
    # Call to append(...): (line 589)
    # Processing the call arguments (line 589)
    
    # Call to partial(...): (line 589)
    # Processing the call arguments (line 589)
    # Getting the type of 'safe_nominal' (line 589)
    safe_nominal_129318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 42), 'safe_nominal', False)
    # Processing the call keyword arguments (line 589)
    # Getting the type of 'pvalue' (line 589)
    pvalue_129319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 63), 'pvalue', False)
    keyword_129320 = pvalue_129319
    kwargs_129321 = {'pvalue': keyword_129320}
    # Getting the type of 'partial' (line 589)
    partial_129317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 34), 'partial', False)
    # Calling partial(args, kwargs) (line 589)
    partial_call_result_129322 = invoke(stypy.reporting.localization.Localization(__file__, 589, 34), partial_129317, *[safe_nominal_129318], **kwargs_129321)
    
    # Processing the call keyword arguments (line 589)
    kwargs_129323 = {}
    # Getting the type of 'convertors' (line 589)
    convertors_129315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 16), 'convertors', False)
    # Obtaining the member 'append' of a type (line 589)
    append_129316 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 589, 16), convertors_129315, 'append')
    # Calling append(args, kwargs) (line 589)
    append_call_result_129324 = invoke(stypy.reporting.localization.Localization(__file__, 589, 16), append_129316, *[partial_call_result_129322], **kwargs_129323)
    
    # SSA branch for the else part of an if statement (line 585)
    module_type_store.open_ssa_branch('else')
    
    # Call to append(...): (line 591)
    # Processing the call arguments (line 591)
    
    # Obtaining an instance of the builtin type 'tuple' (line 591)
    tuple_129327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 591, 30), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 591)
    # Adding element type (line 591)
    # Getting the type of 'name' (line 591)
    name_129328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 30), 'name', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 591, 30), tuple_129327, name_129328)
    # Adding element type (line 591)
    
    # Obtaining the type of the subscript
    # Getting the type of 'type' (line 591)
    type_129329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 47), 'type', False)
    # Getting the type of 'acls2dtype' (line 591)
    acls2dtype_129330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 36), 'acls2dtype', False)
    # Obtaining the member '__getitem__' of a type (line 591)
    getitem___129331 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 591, 36), acls2dtype_129330, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 591)
    subscript_call_result_129332 = invoke(stypy.reporting.localization.Localization(__file__, 591, 36), getitem___129331, type_129329)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 591, 30), tuple_129327, subscript_call_result_129332)
    
    # Processing the call keyword arguments (line 591)
    kwargs_129333 = {}
    # Getting the type of 'descr' (line 591)
    descr_129325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 16), 'descr', False)
    # Obtaining the member 'append' of a type (line 591)
    append_129326 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 591, 16), descr_129325, 'append')
    # Calling append(args, kwargs) (line 591)
    append_call_result_129334 = invoke(stypy.reporting.localization.Localization(__file__, 591, 16), append_129326, *[tuple_129327], **kwargs_129333)
    
    
    # Call to append(...): (line 592)
    # Processing the call arguments (line 592)
    # Getting the type of 'safe_float' (line 592)
    safe_float_129337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 34), 'safe_float', False)
    # Processing the call keyword arguments (line 592)
    kwargs_129338 = {}
    # Getting the type of 'convertors' (line 592)
    convertors_129335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 16), 'convertors', False)
    # Obtaining the member 'append' of a type (line 592)
    append_129336 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 592, 16), convertors_129335, 'append')
    # Calling append(args, kwargs) (line 592)
    append_call_result_129339 = invoke(stypy.reporting.localization.Localization(__file__, 592, 16), append_129336, *[safe_float_129337], **kwargs_129338)
    
    # SSA join for if statement (line 585)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 580)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 577)
    module_type_store.open_ssa_branch('else')
    
    # Call to NotImplementedError(...): (line 598)
    # Processing the call arguments (line 598)
    str_129341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 598, 34), 'str', 'String attributes not supported yet, sorry')
    # Processing the call keyword arguments (line 598)
    kwargs_129342 = {}
    # Getting the type of 'NotImplementedError' (line 598)
    NotImplementedError_129340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 14), 'NotImplementedError', False)
    # Calling NotImplementedError(args, kwargs) (line 598)
    NotImplementedError_call_result_129343 = invoke(stypy.reporting.localization.Localization(__file__, 598, 14), NotImplementedError_129340, *[str_129341], **kwargs_129342)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 598, 8), NotImplementedError_call_result_129343, 'raise parameter', BaseException)
    # SSA join for if statement (line 577)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 600):
    
    # Assigning a Call to a Name (line 600):
    
    # Call to len(...): (line 600)
    # Processing the call arguments (line 600)
    # Getting the type of 'convertors' (line 600)
    convertors_129345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 13), 'convertors', False)
    # Processing the call keyword arguments (line 600)
    kwargs_129346 = {}
    # Getting the type of 'len' (line 600)
    len_129344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 9), 'len', False)
    # Calling len(args, kwargs) (line 600)
    len_call_result_129347 = invoke(stypy.reporting.localization.Localization(__file__, 600, 9), len_129344, *[convertors_129345], **kwargs_129346)
    
    # Assigning a type to the variable 'ni' (line 600)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 600, 4), 'ni', len_call_result_129347)

    @norecursion
    def generator(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        str_129348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 602, 34), 'str', ',')
        defaults = [str_129348]
        # Create a new context for function 'generator'
        module_type_store = module_type_store.open_function_context('generator', 602, 4, False)
        
        # Passed parameters checking function
        generator.stypy_localization = localization
        generator.stypy_type_of_self = None
        generator.stypy_type_store = module_type_store
        generator.stypy_function_name = 'generator'
        generator.stypy_param_names_list = ['row_iter', 'delim']
        generator.stypy_varargs_param_name = None
        generator.stypy_kwargs_param_name = None
        generator.stypy_call_defaults = defaults
        generator.stypy_call_varargs = varargs
        generator.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'generator', ['row_iter', 'delim'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'generator', localization, ['row_iter', 'delim'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'generator(...)' code ##################

        
        # Assigning a Call to a Name (line 617):
        
        # Assigning a Call to a Name (line 617):
        
        # Call to list(...): (line 617)
        # Processing the call arguments (line 617)
        
        # Call to range(...): (line 617)
        # Processing the call arguments (line 617)
        # Getting the type of 'ni' (line 617)
        ni_129351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 27), 'ni', False)
        # Processing the call keyword arguments (line 617)
        kwargs_129352 = {}
        # Getting the type of 'range' (line 617)
        range_129350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 21), 'range', False)
        # Calling range(args, kwargs) (line 617)
        range_call_result_129353 = invoke(stypy.reporting.localization.Localization(__file__, 617, 21), range_129350, *[ni_129351], **kwargs_129352)
        
        # Processing the call keyword arguments (line 617)
        kwargs_129354 = {}
        # Getting the type of 'list' (line 617)
        list_129349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 16), 'list', False)
        # Calling list(args, kwargs) (line 617)
        list_call_result_129355 = invoke(stypy.reporting.localization.Localization(__file__, 617, 16), list_129349, *[range_call_result_129353], **kwargs_129354)
        
        # Assigning a type to the variable 'elems' (line 617)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 617, 8), 'elems', list_call_result_129355)
        
        # Getting the type of 'row_iter' (line 619)
        row_iter_129356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 19), 'row_iter')
        # Testing the type of a for loop iterable (line 619)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 619, 8), row_iter_129356)
        # Getting the type of the for loop variable (line 619)
        for_loop_var_129357 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 619, 8), row_iter_129356)
        # Assigning a type to the variable 'raw' (line 619)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 619, 8), 'raw', for_loop_var_129357)
        # SSA begins for a for statement (line 619)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Evaluating a boolean operation
        
        # Call to match(...): (line 622)
        # Processing the call arguments (line 622)
        # Getting the type of 'raw' (line 622)
        raw_129360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 31), 'raw', False)
        # Processing the call keyword arguments (line 622)
        kwargs_129361 = {}
        # Getting the type of 'r_comment' (line 622)
        r_comment_129358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 15), 'r_comment', False)
        # Obtaining the member 'match' of a type (line 622)
        match_129359 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 622, 15), r_comment_129358, 'match')
        # Calling match(args, kwargs) (line 622)
        match_call_result_129362 = invoke(stypy.reporting.localization.Localization(__file__, 622, 15), match_129359, *[raw_129360], **kwargs_129361)
        
        
        # Call to match(...): (line 622)
        # Processing the call arguments (line 622)
        # Getting the type of 'raw' (line 622)
        raw_129365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 53), 'raw', False)
        # Processing the call keyword arguments (line 622)
        kwargs_129366 = {}
        # Getting the type of 'r_empty' (line 622)
        r_empty_129363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 39), 'r_empty', False)
        # Obtaining the member 'match' of a type (line 622)
        match_129364 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 622, 39), r_empty_129363, 'match')
        # Calling match(args, kwargs) (line 622)
        match_call_result_129367 = invoke(stypy.reporting.localization.Localization(__file__, 622, 39), match_129364, *[raw_129365], **kwargs_129366)
        
        # Applying the binary operator 'or' (line 622)
        result_or_keyword_129368 = python_operator(stypy.reporting.localization.Localization(__file__, 622, 15), 'or', match_call_result_129362, match_call_result_129367)
        
        # Testing the type of an if condition (line 622)
        if_condition_129369 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 622, 12), result_or_keyword_129368)
        # Assigning a type to the variable 'if_condition_129369' (line 622)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 622, 12), 'if_condition_129369', if_condition_129369)
        # SSA begins for if statement (line 622)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # SSA join for if statement (line 622)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 624):
        
        # Assigning a Call to a Name (line 624):
        
        # Call to split(...): (line 624)
        # Processing the call arguments (line 624)
        # Getting the type of 'delim' (line 624)
        delim_129372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 624, 28), 'delim', False)
        # Processing the call keyword arguments (line 624)
        kwargs_129373 = {}
        # Getting the type of 'raw' (line 624)
        raw_129370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 624, 18), 'raw', False)
        # Obtaining the member 'split' of a type (line 624)
        split_129371 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 624, 18), raw_129370, 'split')
        # Calling split(args, kwargs) (line 624)
        split_call_result_129374 = invoke(stypy.reporting.localization.Localization(__file__, 624, 18), split_129371, *[delim_129372], **kwargs_129373)
        
        # Assigning a type to the variable 'row' (line 624)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 624, 12), 'row', split_call_result_129374)
        # Creating a generator
        
        # Call to tuple(...): (line 625)
        # Processing the call arguments (line 625)
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'elems' (line 625)
        elems_129386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 56), 'elems', False)
        comprehension_129387 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 625, 25), elems_129386)
        # Assigning a type to the variable 'i' (line 625)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 625, 25), 'i', comprehension_129387)
        
        # Call to (...): (line 625)
        # Processing the call arguments (line 625)
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 625)
        i_129380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 43), 'i', False)
        # Getting the type of 'row' (line 625)
        row_129381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 39), 'row', False)
        # Obtaining the member '__getitem__' of a type (line 625)
        getitem___129382 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 625, 39), row_129381, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 625)
        subscript_call_result_129383 = invoke(stypy.reporting.localization.Localization(__file__, 625, 39), getitem___129382, i_129380)
        
        # Processing the call keyword arguments (line 625)
        kwargs_129384 = {}
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 625)
        i_129376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 36), 'i', False)
        # Getting the type of 'convertors' (line 625)
        convertors_129377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 25), 'convertors', False)
        # Obtaining the member '__getitem__' of a type (line 625)
        getitem___129378 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 625, 25), convertors_129377, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 625)
        subscript_call_result_129379 = invoke(stypy.reporting.localization.Localization(__file__, 625, 25), getitem___129378, i_129376)
        
        # Calling (args, kwargs) (line 625)
        _call_result_129385 = invoke(stypy.reporting.localization.Localization(__file__, 625, 25), subscript_call_result_129379, *[subscript_call_result_129383], **kwargs_129384)
        
        list_129388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 625, 25), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 625, 25), list_129388, _call_result_129385)
        # Processing the call keyword arguments (line 625)
        kwargs_129389 = {}
        # Getting the type of 'tuple' (line 625)
        tuple_129375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 18), 'tuple', False)
        # Calling tuple(args, kwargs) (line 625)
        tuple_call_result_129390 = invoke(stypy.reporting.localization.Localization(__file__, 625, 18), tuple_129375, *[list_129388], **kwargs_129389)
        
        GeneratorType_129391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 625, 12), 'GeneratorType')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 625, 12), GeneratorType_129391, tuple_call_result_129390)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 625, 12), 'stypy_return_type', GeneratorType_129391)
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'generator(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'generator' in the type store
        # Getting the type of 'stypy_return_type' (line 602)
        stypy_return_type_129392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_129392)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'generator'
        return stypy_return_type_129392

    # Assigning a type to the variable 'generator' (line 602)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 602, 4), 'generator', generator)
    
    # Assigning a Call to a Name (line 627):
    
    # Assigning a Call to a Name (line 627):
    
    # Call to generator(...): (line 627)
    # Processing the call arguments (line 627)
    # Getting the type of 'ofile' (line 627)
    ofile_129394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 627, 18), 'ofile', False)
    # Processing the call keyword arguments (line 627)
    kwargs_129395 = {}
    # Getting the type of 'generator' (line 627)
    generator_129393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 627, 8), 'generator', False)
    # Calling generator(args, kwargs) (line 627)
    generator_call_result_129396 = invoke(stypy.reporting.localization.Localization(__file__, 627, 8), generator_129393, *[ofile_129394], **kwargs_129395)
    
    # Assigning a type to the variable 'a' (line 627)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 627, 4), 'a', generator_call_result_129396)
    
    # Assigning a Call to a Name (line 629):
    
    # Assigning a Call to a Name (line 629):
    
    # Call to fromiter(...): (line 629)
    # Processing the call arguments (line 629)
    # Getting the type of 'a' (line 629)
    a_129399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 629, 23), 'a', False)
    # Getting the type of 'descr' (line 629)
    descr_129400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 629, 26), 'descr', False)
    # Processing the call keyword arguments (line 629)
    kwargs_129401 = {}
    # Getting the type of 'np' (line 629)
    np_129397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 629, 11), 'np', False)
    # Obtaining the member 'fromiter' of a type (line 629)
    fromiter_129398 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 629, 11), np_129397, 'fromiter')
    # Calling fromiter(args, kwargs) (line 629)
    fromiter_call_result_129402 = invoke(stypy.reporting.localization.Localization(__file__, 629, 11), fromiter_129398, *[a_129399, descr_129400], **kwargs_129401)
    
    # Assigning a type to the variable 'data' (line 629)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 629, 4), 'data', fromiter_call_result_129402)
    
    # Obtaining an instance of the builtin type 'tuple' (line 630)
    tuple_129403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 630, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 630)
    # Adding element type (line 630)
    # Getting the type of 'data' (line 630)
    data_129404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 630, 11), 'data')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 630, 11), tuple_129403, data_129404)
    # Adding element type (line 630)
    # Getting the type of 'meta' (line 630)
    meta_129405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 630, 17), 'meta')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 630, 11), tuple_129403, meta_129405)
    
    # Assigning a type to the variable 'stypy_return_type' (line 630)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 630, 4), 'stypy_return_type', tuple_129403)
    
    # ################# End of '_loadarff(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_loadarff' in the type store
    # Getting the type of 'stypy_return_type' (line 547)
    stypy_return_type_129406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_129406)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_loadarff'
    return stypy_return_type_129406

# Assigning a type to the variable '_loadarff' (line 547)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 547, 0), '_loadarff', _loadarff)

@norecursion
def basic_stats(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'basic_stats'
    module_type_store = module_type_store.open_function_context('basic_stats', 636, 0, False)
    
    # Passed parameters checking function
    basic_stats.stypy_localization = localization
    basic_stats.stypy_type_of_self = None
    basic_stats.stypy_type_store = module_type_store
    basic_stats.stypy_function_name = 'basic_stats'
    basic_stats.stypy_param_names_list = ['data']
    basic_stats.stypy_varargs_param_name = None
    basic_stats.stypy_kwargs_param_name = None
    basic_stats.stypy_call_defaults = defaults
    basic_stats.stypy_call_varargs = varargs
    basic_stats.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'basic_stats', ['data'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'basic_stats', localization, ['data'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'basic_stats(...)' code ##################

    
    # Assigning a BinOp to a Name (line 637):
    
    # Assigning a BinOp to a Name (line 637):
    # Getting the type of 'data' (line 637)
    data_129407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 12), 'data')
    # Obtaining the member 'size' of a type (line 637)
    size_129408 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 637, 12), data_129407, 'size')
    float_129409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 637, 24), 'float')
    # Applying the binary operator '*' (line 637)
    result_mul_129410 = python_operator(stypy.reporting.localization.Localization(__file__, 637, 12), '*', size_129408, float_129409)
    
    # Getting the type of 'data' (line 637)
    data_129411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 30), 'data')
    # Obtaining the member 'size' of a type (line 637)
    size_129412 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 637, 30), data_129411, 'size')
    int_129413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 637, 42), 'int')
    # Applying the binary operator '-' (line 637)
    result_sub_129414 = python_operator(stypy.reporting.localization.Localization(__file__, 637, 30), '-', size_129412, int_129413)
    
    # Applying the binary operator 'div' (line 637)
    result_div_129415 = python_operator(stypy.reporting.localization.Localization(__file__, 637, 27), 'div', result_mul_129410, result_sub_129414)
    
    # Assigning a type to the variable 'nbfac' (line 637)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 637, 4), 'nbfac', result_div_129415)
    
    # Obtaining an instance of the builtin type 'tuple' (line 638)
    tuple_129416 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 638, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 638)
    # Adding element type (line 638)
    
    # Call to nanmin(...): (line 638)
    # Processing the call arguments (line 638)
    # Getting the type of 'data' (line 638)
    data_129419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 638, 21), 'data', False)
    # Processing the call keyword arguments (line 638)
    kwargs_129420 = {}
    # Getting the type of 'np' (line 638)
    np_129417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 638, 11), 'np', False)
    # Obtaining the member 'nanmin' of a type (line 638)
    nanmin_129418 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 638, 11), np_129417, 'nanmin')
    # Calling nanmin(args, kwargs) (line 638)
    nanmin_call_result_129421 = invoke(stypy.reporting.localization.Localization(__file__, 638, 11), nanmin_129418, *[data_129419], **kwargs_129420)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 638, 11), tuple_129416, nanmin_call_result_129421)
    # Adding element type (line 638)
    
    # Call to nanmax(...): (line 638)
    # Processing the call arguments (line 638)
    # Getting the type of 'data' (line 638)
    data_129424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 638, 38), 'data', False)
    # Processing the call keyword arguments (line 638)
    kwargs_129425 = {}
    # Getting the type of 'np' (line 638)
    np_129422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 638, 28), 'np', False)
    # Obtaining the member 'nanmax' of a type (line 638)
    nanmax_129423 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 638, 28), np_129422, 'nanmax')
    # Calling nanmax(args, kwargs) (line 638)
    nanmax_call_result_129426 = invoke(stypy.reporting.localization.Localization(__file__, 638, 28), nanmax_129423, *[data_129424], **kwargs_129425)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 638, 11), tuple_129416, nanmax_call_result_129426)
    # Adding element type (line 638)
    
    # Call to mean(...): (line 638)
    # Processing the call arguments (line 638)
    # Getting the type of 'data' (line 638)
    data_129429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 638, 53), 'data', False)
    # Processing the call keyword arguments (line 638)
    kwargs_129430 = {}
    # Getting the type of 'np' (line 638)
    np_129427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 638, 45), 'np', False)
    # Obtaining the member 'mean' of a type (line 638)
    mean_129428 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 638, 45), np_129427, 'mean')
    # Calling mean(args, kwargs) (line 638)
    mean_call_result_129431 = invoke(stypy.reporting.localization.Localization(__file__, 638, 45), mean_129428, *[data_129429], **kwargs_129430)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 638, 11), tuple_129416, mean_call_result_129431)
    # Adding element type (line 638)
    
    # Call to std(...): (line 638)
    # Processing the call arguments (line 638)
    # Getting the type of 'data' (line 638)
    data_129434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 638, 67), 'data', False)
    # Processing the call keyword arguments (line 638)
    kwargs_129435 = {}
    # Getting the type of 'np' (line 638)
    np_129432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 638, 60), 'np', False)
    # Obtaining the member 'std' of a type (line 638)
    std_129433 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 638, 60), np_129432, 'std')
    # Calling std(args, kwargs) (line 638)
    std_call_result_129436 = invoke(stypy.reporting.localization.Localization(__file__, 638, 60), std_129433, *[data_129434], **kwargs_129435)
    
    # Getting the type of 'nbfac' (line 638)
    nbfac_129437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 638, 75), 'nbfac')
    # Applying the binary operator '*' (line 638)
    result_mul_129438 = python_operator(stypy.reporting.localization.Localization(__file__, 638, 60), '*', std_call_result_129436, nbfac_129437)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 638, 11), tuple_129416, result_mul_129438)
    
    # Assigning a type to the variable 'stypy_return_type' (line 638)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 638, 4), 'stypy_return_type', tuple_129416)
    
    # ################# End of 'basic_stats(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'basic_stats' in the type store
    # Getting the type of 'stypy_return_type' (line 636)
    stypy_return_type_129439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_129439)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'basic_stats'
    return stypy_return_type_129439

# Assigning a type to the variable 'basic_stats' (line 636)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 636, 0), 'basic_stats', basic_stats)

@norecursion
def print_attribute(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'print_attribute'
    module_type_store = module_type_store.open_function_context('print_attribute', 641, 0, False)
    
    # Passed parameters checking function
    print_attribute.stypy_localization = localization
    print_attribute.stypy_type_of_self = None
    print_attribute.stypy_type_store = module_type_store
    print_attribute.stypy_function_name = 'print_attribute'
    print_attribute.stypy_param_names_list = ['name', 'tp', 'data']
    print_attribute.stypy_varargs_param_name = None
    print_attribute.stypy_kwargs_param_name = None
    print_attribute.stypy_call_defaults = defaults
    print_attribute.stypy_call_varargs = varargs
    print_attribute.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'print_attribute', ['name', 'tp', 'data'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'print_attribute', localization, ['name', 'tp', 'data'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'print_attribute(...)' code ##################

    
    # Assigning a Subscript to a Name (line 642):
    
    # Assigning a Subscript to a Name (line 642):
    
    # Obtaining the type of the subscript
    int_129440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 642, 14), 'int')
    # Getting the type of 'tp' (line 642)
    tp_129441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 642, 11), 'tp')
    # Obtaining the member '__getitem__' of a type (line 642)
    getitem___129442 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 642, 11), tp_129441, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 642)
    subscript_call_result_129443 = invoke(stypy.reporting.localization.Localization(__file__, 642, 11), getitem___129442, int_129440)
    
    # Assigning a type to the variable 'type' (line 642)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 642, 4), 'type', subscript_call_result_129443)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'type' (line 643)
    type_129444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 643, 7), 'type')
    str_129445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 643, 15), 'str', 'numeric')
    # Applying the binary operator '==' (line 643)
    result_eq_129446 = python_operator(stypy.reporting.localization.Localization(__file__, 643, 7), '==', type_129444, str_129445)
    
    
    # Getting the type of 'type' (line 643)
    type_129447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 643, 28), 'type')
    str_129448 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 643, 36), 'str', 'real')
    # Applying the binary operator '==' (line 643)
    result_eq_129449 = python_operator(stypy.reporting.localization.Localization(__file__, 643, 28), '==', type_129447, str_129448)
    
    # Applying the binary operator 'or' (line 643)
    result_or_keyword_129450 = python_operator(stypy.reporting.localization.Localization(__file__, 643, 7), 'or', result_eq_129446, result_eq_129449)
    
    # Getting the type of 'type' (line 643)
    type_129451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 643, 46), 'type')
    str_129452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 643, 54), 'str', 'integer')
    # Applying the binary operator '==' (line 643)
    result_eq_129453 = python_operator(stypy.reporting.localization.Localization(__file__, 643, 46), '==', type_129451, str_129452)
    
    # Applying the binary operator 'or' (line 643)
    result_or_keyword_129454 = python_operator(stypy.reporting.localization.Localization(__file__, 643, 7), 'or', result_or_keyword_129450, result_eq_129453)
    
    # Testing the type of an if condition (line 643)
    if_condition_129455 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 643, 4), result_or_keyword_129454)
    # Assigning a type to the variable 'if_condition_129455' (line 643)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 643, 4), 'if_condition_129455', if_condition_129455)
    # SSA begins for if statement (line 643)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Tuple (line 644):
    
    # Assigning a Subscript to a Name (line 644):
    
    # Obtaining the type of the subscript
    int_129456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 644, 8), 'int')
    
    # Call to basic_stats(...): (line 644)
    # Processing the call arguments (line 644)
    # Getting the type of 'data' (line 644)
    data_129458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 42), 'data', False)
    # Processing the call keyword arguments (line 644)
    kwargs_129459 = {}
    # Getting the type of 'basic_stats' (line 644)
    basic_stats_129457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 30), 'basic_stats', False)
    # Calling basic_stats(args, kwargs) (line 644)
    basic_stats_call_result_129460 = invoke(stypy.reporting.localization.Localization(__file__, 644, 30), basic_stats_129457, *[data_129458], **kwargs_129459)
    
    # Obtaining the member '__getitem__' of a type (line 644)
    getitem___129461 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 644, 8), basic_stats_call_result_129460, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 644)
    subscript_call_result_129462 = invoke(stypy.reporting.localization.Localization(__file__, 644, 8), getitem___129461, int_129456)
    
    # Assigning a type to the variable 'tuple_var_assignment_128254' (line 644)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 644, 8), 'tuple_var_assignment_128254', subscript_call_result_129462)
    
    # Assigning a Subscript to a Name (line 644):
    
    # Obtaining the type of the subscript
    int_129463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 644, 8), 'int')
    
    # Call to basic_stats(...): (line 644)
    # Processing the call arguments (line 644)
    # Getting the type of 'data' (line 644)
    data_129465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 42), 'data', False)
    # Processing the call keyword arguments (line 644)
    kwargs_129466 = {}
    # Getting the type of 'basic_stats' (line 644)
    basic_stats_129464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 30), 'basic_stats', False)
    # Calling basic_stats(args, kwargs) (line 644)
    basic_stats_call_result_129467 = invoke(stypy.reporting.localization.Localization(__file__, 644, 30), basic_stats_129464, *[data_129465], **kwargs_129466)
    
    # Obtaining the member '__getitem__' of a type (line 644)
    getitem___129468 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 644, 8), basic_stats_call_result_129467, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 644)
    subscript_call_result_129469 = invoke(stypy.reporting.localization.Localization(__file__, 644, 8), getitem___129468, int_129463)
    
    # Assigning a type to the variable 'tuple_var_assignment_128255' (line 644)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 644, 8), 'tuple_var_assignment_128255', subscript_call_result_129469)
    
    # Assigning a Subscript to a Name (line 644):
    
    # Obtaining the type of the subscript
    int_129470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 644, 8), 'int')
    
    # Call to basic_stats(...): (line 644)
    # Processing the call arguments (line 644)
    # Getting the type of 'data' (line 644)
    data_129472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 42), 'data', False)
    # Processing the call keyword arguments (line 644)
    kwargs_129473 = {}
    # Getting the type of 'basic_stats' (line 644)
    basic_stats_129471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 30), 'basic_stats', False)
    # Calling basic_stats(args, kwargs) (line 644)
    basic_stats_call_result_129474 = invoke(stypy.reporting.localization.Localization(__file__, 644, 30), basic_stats_129471, *[data_129472], **kwargs_129473)
    
    # Obtaining the member '__getitem__' of a type (line 644)
    getitem___129475 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 644, 8), basic_stats_call_result_129474, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 644)
    subscript_call_result_129476 = invoke(stypy.reporting.localization.Localization(__file__, 644, 8), getitem___129475, int_129470)
    
    # Assigning a type to the variable 'tuple_var_assignment_128256' (line 644)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 644, 8), 'tuple_var_assignment_128256', subscript_call_result_129476)
    
    # Assigning a Subscript to a Name (line 644):
    
    # Obtaining the type of the subscript
    int_129477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 644, 8), 'int')
    
    # Call to basic_stats(...): (line 644)
    # Processing the call arguments (line 644)
    # Getting the type of 'data' (line 644)
    data_129479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 42), 'data', False)
    # Processing the call keyword arguments (line 644)
    kwargs_129480 = {}
    # Getting the type of 'basic_stats' (line 644)
    basic_stats_129478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 30), 'basic_stats', False)
    # Calling basic_stats(args, kwargs) (line 644)
    basic_stats_call_result_129481 = invoke(stypy.reporting.localization.Localization(__file__, 644, 30), basic_stats_129478, *[data_129479], **kwargs_129480)
    
    # Obtaining the member '__getitem__' of a type (line 644)
    getitem___129482 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 644, 8), basic_stats_call_result_129481, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 644)
    subscript_call_result_129483 = invoke(stypy.reporting.localization.Localization(__file__, 644, 8), getitem___129482, int_129477)
    
    # Assigning a type to the variable 'tuple_var_assignment_128257' (line 644)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 644, 8), 'tuple_var_assignment_128257', subscript_call_result_129483)
    
    # Assigning a Name to a Name (line 644):
    # Getting the type of 'tuple_var_assignment_128254' (line 644)
    tuple_var_assignment_128254_129484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 8), 'tuple_var_assignment_128254')
    # Assigning a type to the variable 'min' (line 644)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 644, 8), 'min', tuple_var_assignment_128254_129484)
    
    # Assigning a Name to a Name (line 644):
    # Getting the type of 'tuple_var_assignment_128255' (line 644)
    tuple_var_assignment_128255_129485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 8), 'tuple_var_assignment_128255')
    # Assigning a type to the variable 'max' (line 644)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 644, 13), 'max', tuple_var_assignment_128255_129485)
    
    # Assigning a Name to a Name (line 644):
    # Getting the type of 'tuple_var_assignment_128256' (line 644)
    tuple_var_assignment_128256_129486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 8), 'tuple_var_assignment_128256')
    # Assigning a type to the variable 'mean' (line 644)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 644, 18), 'mean', tuple_var_assignment_128256_129486)
    
    # Assigning a Name to a Name (line 644):
    # Getting the type of 'tuple_var_assignment_128257' (line 644)
    tuple_var_assignment_128257_129487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 8), 'tuple_var_assignment_128257')
    # Assigning a type to the variable 'std' (line 644)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 644, 24), 'std', tuple_var_assignment_128257_129487)
    
    # Call to print(...): (line 645)
    # Processing the call arguments (line 645)
    str_129489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 645, 14), 'str', '%s,%s,%f,%f,%f,%f')
    
    # Obtaining an instance of the builtin type 'tuple' (line 645)
    tuple_129490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 645, 37), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 645)
    # Adding element type (line 645)
    # Getting the type of 'name' (line 645)
    name_129491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 645, 37), 'name', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 645, 37), tuple_129490, name_129491)
    # Adding element type (line 645)
    # Getting the type of 'type' (line 645)
    type_129492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 645, 43), 'type', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 645, 37), tuple_129490, type_129492)
    # Adding element type (line 645)
    # Getting the type of 'min' (line 645)
    min_129493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 645, 49), 'min', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 645, 37), tuple_129490, min_129493)
    # Adding element type (line 645)
    # Getting the type of 'max' (line 645)
    max_129494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 645, 54), 'max', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 645, 37), tuple_129490, max_129494)
    # Adding element type (line 645)
    # Getting the type of 'mean' (line 645)
    mean_129495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 645, 59), 'mean', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 645, 37), tuple_129490, mean_129495)
    # Adding element type (line 645)
    # Getting the type of 'std' (line 645)
    std_129496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 645, 65), 'std', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 645, 37), tuple_129490, std_129496)
    
    # Applying the binary operator '%' (line 645)
    result_mod_129497 = python_operator(stypy.reporting.localization.Localization(__file__, 645, 14), '%', str_129489, tuple_129490)
    
    # Processing the call keyword arguments (line 645)
    kwargs_129498 = {}
    # Getting the type of 'print' (line 645)
    print_129488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 645, 8), 'print', False)
    # Calling print(args, kwargs) (line 645)
    print_call_result_129499 = invoke(stypy.reporting.localization.Localization(__file__, 645, 8), print_129488, *[result_mod_129497], **kwargs_129498)
    
    # SSA branch for the else part of an if statement (line 643)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a BinOp to a Name (line 647):
    
    # Assigning a BinOp to a Name (line 647):
    # Getting the type of 'name' (line 647)
    name_129500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 647, 14), 'name')
    str_129501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 647, 21), 'str', ',{')
    # Applying the binary operator '+' (line 647)
    result_add_129502 = python_operator(stypy.reporting.localization.Localization(__file__, 647, 14), '+', name_129500, str_129501)
    
    # Assigning a type to the variable 'msg' (line 647)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 647, 8), 'msg', result_add_129502)
    
    
    # Call to range(...): (line 648)
    # Processing the call arguments (line 648)
    
    # Call to len(...): (line 648)
    # Processing the call arguments (line 648)
    
    # Obtaining the type of the subscript
    int_129505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 648, 30), 'int')
    # Getting the type of 'tp' (line 648)
    tp_129506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 648, 27), 'tp', False)
    # Obtaining the member '__getitem__' of a type (line 648)
    getitem___129507 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 648, 27), tp_129506, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 648)
    subscript_call_result_129508 = invoke(stypy.reporting.localization.Localization(__file__, 648, 27), getitem___129507, int_129505)
    
    # Processing the call keyword arguments (line 648)
    kwargs_129509 = {}
    # Getting the type of 'len' (line 648)
    len_129504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 648, 23), 'len', False)
    # Calling len(args, kwargs) (line 648)
    len_call_result_129510 = invoke(stypy.reporting.localization.Localization(__file__, 648, 23), len_129504, *[subscript_call_result_129508], **kwargs_129509)
    
    int_129511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 648, 34), 'int')
    # Applying the binary operator '-' (line 648)
    result_sub_129512 = python_operator(stypy.reporting.localization.Localization(__file__, 648, 23), '-', len_call_result_129510, int_129511)
    
    # Processing the call keyword arguments (line 648)
    kwargs_129513 = {}
    # Getting the type of 'range' (line 648)
    range_129503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 648, 17), 'range', False)
    # Calling range(args, kwargs) (line 648)
    range_call_result_129514 = invoke(stypy.reporting.localization.Localization(__file__, 648, 17), range_129503, *[result_sub_129512], **kwargs_129513)
    
    # Testing the type of a for loop iterable (line 648)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 648, 8), range_call_result_129514)
    # Getting the type of the for loop variable (line 648)
    for_loop_var_129515 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 648, 8), range_call_result_129514)
    # Assigning a type to the variable 'i' (line 648)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 648, 8), 'i', for_loop_var_129515)
    # SSA begins for a for statement (line 648)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Getting the type of 'msg' (line 649)
    msg_129516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 12), 'msg')
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 649)
    i_129517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 25), 'i')
    
    # Obtaining the type of the subscript
    int_129518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 649, 22), 'int')
    # Getting the type of 'tp' (line 649)
    tp_129519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 19), 'tp')
    # Obtaining the member '__getitem__' of a type (line 649)
    getitem___129520 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 649, 19), tp_129519, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 649)
    subscript_call_result_129521 = invoke(stypy.reporting.localization.Localization(__file__, 649, 19), getitem___129520, int_129518)
    
    # Obtaining the member '__getitem__' of a type (line 649)
    getitem___129522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 649, 19), subscript_call_result_129521, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 649)
    subscript_call_result_129523 = invoke(stypy.reporting.localization.Localization(__file__, 649, 19), getitem___129522, i_129517)
    
    str_129524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 649, 30), 'str', ',')
    # Applying the binary operator '+' (line 649)
    result_add_129525 = python_operator(stypy.reporting.localization.Localization(__file__, 649, 19), '+', subscript_call_result_129523, str_129524)
    
    # Applying the binary operator '+=' (line 649)
    result_iadd_129526 = python_operator(stypy.reporting.localization.Localization(__file__, 649, 12), '+=', msg_129516, result_add_129525)
    # Assigning a type to the variable 'msg' (line 649)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 649, 12), 'msg', result_iadd_129526)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'msg' (line 650)
    msg_129527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 8), 'msg')
    
    # Obtaining the type of the subscript
    int_129528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 650, 21), 'int')
    
    # Obtaining the type of the subscript
    int_129529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 650, 18), 'int')
    # Getting the type of 'tp' (line 650)
    tp_129530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 15), 'tp')
    # Obtaining the member '__getitem__' of a type (line 650)
    getitem___129531 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 650, 15), tp_129530, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 650)
    subscript_call_result_129532 = invoke(stypy.reporting.localization.Localization(__file__, 650, 15), getitem___129531, int_129529)
    
    # Obtaining the member '__getitem__' of a type (line 650)
    getitem___129533 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 650, 15), subscript_call_result_129532, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 650)
    subscript_call_result_129534 = invoke(stypy.reporting.localization.Localization(__file__, 650, 15), getitem___129533, int_129528)
    
    # Applying the binary operator '+=' (line 650)
    result_iadd_129535 = python_operator(stypy.reporting.localization.Localization(__file__, 650, 8), '+=', msg_129527, subscript_call_result_129534)
    # Assigning a type to the variable 'msg' (line 650)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 650, 8), 'msg', result_iadd_129535)
    
    
    # Getting the type of 'msg' (line 651)
    msg_129536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 8), 'msg')
    str_129537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 651, 15), 'str', '}')
    # Applying the binary operator '+=' (line 651)
    result_iadd_129538 = python_operator(stypy.reporting.localization.Localization(__file__, 651, 8), '+=', msg_129536, str_129537)
    # Assigning a type to the variable 'msg' (line 651)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 651, 8), 'msg', result_iadd_129538)
    
    
    # Call to print(...): (line 652)
    # Processing the call arguments (line 652)
    # Getting the type of 'msg' (line 652)
    msg_129540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 652, 14), 'msg', False)
    # Processing the call keyword arguments (line 652)
    kwargs_129541 = {}
    # Getting the type of 'print' (line 652)
    print_129539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 652, 8), 'print', False)
    # Calling print(args, kwargs) (line 652)
    print_call_result_129542 = invoke(stypy.reporting.localization.Localization(__file__, 652, 8), print_129539, *[msg_129540], **kwargs_129541)
    
    # SSA join for if statement (line 643)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'print_attribute(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'print_attribute' in the type store
    # Getting the type of 'stypy_return_type' (line 641)
    stypy_return_type_129543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 641, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_129543)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'print_attribute'
    return stypy_return_type_129543

# Assigning a type to the variable 'print_attribute' (line 641)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 641, 0), 'print_attribute', print_attribute)

@norecursion
def test_weka(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_weka'
    module_type_store = module_type_store.open_function_context('test_weka', 655, 0, False)
    
    # Passed parameters checking function
    test_weka.stypy_localization = localization
    test_weka.stypy_type_of_self = None
    test_weka.stypy_type_store = module_type_store
    test_weka.stypy_function_name = 'test_weka'
    test_weka.stypy_param_names_list = ['filename']
    test_weka.stypy_varargs_param_name = None
    test_weka.stypy_kwargs_param_name = None
    test_weka.stypy_call_defaults = defaults
    test_weka.stypy_call_varargs = varargs
    test_weka.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_weka', ['filename'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_weka', localization, ['filename'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_weka(...)' code ##################

    
    # Assigning a Call to a Tuple (line 656):
    
    # Assigning a Subscript to a Name (line 656):
    
    # Obtaining the type of the subscript
    int_129544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 656, 4), 'int')
    
    # Call to loadarff(...): (line 656)
    # Processing the call arguments (line 656)
    # Getting the type of 'filename' (line 656)
    filename_129546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 26), 'filename', False)
    # Processing the call keyword arguments (line 656)
    kwargs_129547 = {}
    # Getting the type of 'loadarff' (line 656)
    loadarff_129545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 17), 'loadarff', False)
    # Calling loadarff(args, kwargs) (line 656)
    loadarff_call_result_129548 = invoke(stypy.reporting.localization.Localization(__file__, 656, 17), loadarff_129545, *[filename_129546], **kwargs_129547)
    
    # Obtaining the member '__getitem__' of a type (line 656)
    getitem___129549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 656, 4), loadarff_call_result_129548, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 656)
    subscript_call_result_129550 = invoke(stypy.reporting.localization.Localization(__file__, 656, 4), getitem___129549, int_129544)
    
    # Assigning a type to the variable 'tuple_var_assignment_128258' (line 656)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 656, 4), 'tuple_var_assignment_128258', subscript_call_result_129550)
    
    # Assigning a Subscript to a Name (line 656):
    
    # Obtaining the type of the subscript
    int_129551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 656, 4), 'int')
    
    # Call to loadarff(...): (line 656)
    # Processing the call arguments (line 656)
    # Getting the type of 'filename' (line 656)
    filename_129553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 26), 'filename', False)
    # Processing the call keyword arguments (line 656)
    kwargs_129554 = {}
    # Getting the type of 'loadarff' (line 656)
    loadarff_129552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 17), 'loadarff', False)
    # Calling loadarff(args, kwargs) (line 656)
    loadarff_call_result_129555 = invoke(stypy.reporting.localization.Localization(__file__, 656, 17), loadarff_129552, *[filename_129553], **kwargs_129554)
    
    # Obtaining the member '__getitem__' of a type (line 656)
    getitem___129556 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 656, 4), loadarff_call_result_129555, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 656)
    subscript_call_result_129557 = invoke(stypy.reporting.localization.Localization(__file__, 656, 4), getitem___129556, int_129551)
    
    # Assigning a type to the variable 'tuple_var_assignment_128259' (line 656)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 656, 4), 'tuple_var_assignment_128259', subscript_call_result_129557)
    
    # Assigning a Name to a Name (line 656):
    # Getting the type of 'tuple_var_assignment_128258' (line 656)
    tuple_var_assignment_128258_129558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 4), 'tuple_var_assignment_128258')
    # Assigning a type to the variable 'data' (line 656)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 656, 4), 'data', tuple_var_assignment_128258_129558)
    
    # Assigning a Name to a Name (line 656):
    # Getting the type of 'tuple_var_assignment_128259' (line 656)
    tuple_var_assignment_128259_129559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 4), 'tuple_var_assignment_128259')
    # Assigning a type to the variable 'meta' (line 656)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 656, 10), 'meta', tuple_var_assignment_128259_129559)
    
    # Call to print(...): (line 657)
    # Processing the call arguments (line 657)
    
    # Call to len(...): (line 657)
    # Processing the call arguments (line 657)
    # Getting the type of 'data' (line 657)
    data_129562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 657, 14), 'data', False)
    # Obtaining the member 'dtype' of a type (line 657)
    dtype_129563 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 657, 14), data_129562, 'dtype')
    # Processing the call keyword arguments (line 657)
    kwargs_129564 = {}
    # Getting the type of 'len' (line 657)
    len_129561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 657, 10), 'len', False)
    # Calling len(args, kwargs) (line 657)
    len_call_result_129565 = invoke(stypy.reporting.localization.Localization(__file__, 657, 10), len_129561, *[dtype_129563], **kwargs_129564)
    
    # Processing the call keyword arguments (line 657)
    kwargs_129566 = {}
    # Getting the type of 'print' (line 657)
    print_129560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 657, 4), 'print', False)
    # Calling print(args, kwargs) (line 657)
    print_call_result_129567 = invoke(stypy.reporting.localization.Localization(__file__, 657, 4), print_129560, *[len_call_result_129565], **kwargs_129566)
    
    
    # Call to print(...): (line 658)
    # Processing the call arguments (line 658)
    # Getting the type of 'data' (line 658)
    data_129569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 658, 10), 'data', False)
    # Obtaining the member 'size' of a type (line 658)
    size_129570 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 658, 10), data_129569, 'size')
    # Processing the call keyword arguments (line 658)
    kwargs_129571 = {}
    # Getting the type of 'print' (line 658)
    print_129568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 658, 4), 'print', False)
    # Calling print(args, kwargs) (line 658)
    print_call_result_129572 = invoke(stypy.reporting.localization.Localization(__file__, 658, 4), print_129568, *[size_129570], **kwargs_129571)
    
    
    # Getting the type of 'meta' (line 659)
    meta_129573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 13), 'meta')
    # Testing the type of a for loop iterable (line 659)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 659, 4), meta_129573)
    # Getting the type of the for loop variable (line 659)
    for_loop_var_129574 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 659, 4), meta_129573)
    # Assigning a type to the variable 'i' (line 659)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 659, 4), 'i', for_loop_var_129574)
    # SSA begins for a for statement (line 659)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to print_attribute(...): (line 660)
    # Processing the call arguments (line 660)
    # Getting the type of 'i' (line 660)
    i_129576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 24), 'i', False)
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 660)
    i_129577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 32), 'i', False)
    # Getting the type of 'meta' (line 660)
    meta_129578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 27), 'meta', False)
    # Obtaining the member '__getitem__' of a type (line 660)
    getitem___129579 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 660, 27), meta_129578, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 660)
    subscript_call_result_129580 = invoke(stypy.reporting.localization.Localization(__file__, 660, 27), getitem___129579, i_129577)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 660)
    i_129581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 41), 'i', False)
    # Getting the type of 'data' (line 660)
    data_129582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 36), 'data', False)
    # Obtaining the member '__getitem__' of a type (line 660)
    getitem___129583 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 660, 36), data_129582, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 660)
    subscript_call_result_129584 = invoke(stypy.reporting.localization.Localization(__file__, 660, 36), getitem___129583, i_129581)
    
    # Processing the call keyword arguments (line 660)
    kwargs_129585 = {}
    # Getting the type of 'print_attribute' (line 660)
    print_attribute_129575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 8), 'print_attribute', False)
    # Calling print_attribute(args, kwargs) (line 660)
    print_attribute_call_result_129586 = invoke(stypy.reporting.localization.Localization(__file__, 660, 8), print_attribute_129575, *[i_129576, subscript_call_result_129580, subscript_call_result_129584], **kwargs_129585)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_weka(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_weka' in the type store
    # Getting the type of 'stypy_return_type' (line 655)
    stypy_return_type_129587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_129587)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_weka'
    return stypy_return_type_129587

# Assigning a type to the variable 'test_weka' (line 655)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 655, 0), 'test_weka', test_weka)

# Assigning a Name to a Attribute (line 663):

# Assigning a Name to a Attribute (line 663):
# Getting the type of 'False' (line 663)
False_129588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 21), 'False')
# Getting the type of 'test_weka' (line 663)
test_weka_129589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 0), 'test_weka')
# Setting the type of the member '__test__' of a type (line 663)
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 663, 0), test_weka_129589, '__test__', False_129588)

if (__name__ == '__main__'):
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 667, 4))
    
    # 'import sys' statement (line 667)
    import sys

    import_module(stypy.reporting.localization.Localization(__file__, 667, 4), 'sys', sys, module_type_store)
    
    
    # Assigning a Subscript to a Name (line 668):
    
    # Assigning a Subscript to a Name (line 668):
    
    # Obtaining the type of the subscript
    int_129590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 668, 24), 'int')
    # Getting the type of 'sys' (line 668)
    sys_129591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 15), 'sys')
    # Obtaining the member 'argv' of a type (line 668)
    argv_129592 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 668, 15), sys_129591, 'argv')
    # Obtaining the member '__getitem__' of a type (line 668)
    getitem___129593 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 668, 15), argv_129592, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 668)
    subscript_call_result_129594 = invoke(stypy.reporting.localization.Localization(__file__, 668, 15), getitem___129593, int_129590)
    
    # Assigning a type to the variable 'filename' (line 668)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 668, 4), 'filename', subscript_call_result_129594)
    
    # Call to test_weka(...): (line 669)
    # Processing the call arguments (line 669)
    # Getting the type of 'filename' (line 669)
    filename_129596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 669, 14), 'filename', False)
    # Processing the call keyword arguments (line 669)
    kwargs_129597 = {}
    # Getting the type of 'test_weka' (line 669)
    test_weka_129595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 669, 4), 'test_weka', False)
    # Calling test_weka(args, kwargs) (line 669)
    test_weka_call_result_129598 = invoke(stypy.reporting.localization.Localization(__file__, 669, 4), test_weka_129595, *[filename_129596], **kwargs_129597)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
