
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: This is a python interface to Adobe Font Metrics Files.  Although a
3: number of other python implementations exist, and may be more complete
4: than this, it was decided not to go with them because they were
5: either:
6: 
7:   1) copyrighted or used a non-BSD compatible license
8: 
9:   2) had too many dependencies and a free standing lib was needed
10: 
11:   3) Did more than needed and it was easier to write afresh rather than
12:      figure out how to get just what was needed.
13: 
14: It is pretty easy to use, and requires only built-in python libs:
15: 
16:     >>> from matplotlib import rcParams
17:     >>> import os.path
18:     >>> afm_fname = os.path.join(rcParams['datapath'],
19:     ...                         'fonts', 'afm', 'ptmr8a.afm')
20:     >>>
21:     >>> from matplotlib.afm import AFM
22:     >>> with open(afm_fname) as fh:
23:     ...     afm = AFM(fh)
24:     >>> afm.string_width_height('What the heck?')
25:     (6220.0, 694)
26:     >>> afm.get_fontname()
27:     'Times-Roman'
28:     >>> afm.get_kern_dist('A', 'f')
29:     0
30:     >>> afm.get_kern_dist('A', 'y')
31:     -92.0
32:     >>> afm.get_bbox_char('!')
33:     [130, -9, 238, 676]
34: 
35: '''
36: 
37: from __future__ import (absolute_import, division, print_function,
38:                         unicode_literals)
39: 
40: import six
41: from six.moves import map
42: 
43: import sys
44: import os
45: import re
46: from ._mathtext_data import uni2type1
47: 
48: # Convert string the a python type
49: 
50: # some afm files have floats where we are expecting ints -- there is
51: # probably a better way to handle this (support floats, round rather
52: # than truncate).  But I don't know what the best approach is now and
53: # this change to _to_int should at least prevent mpl from crashing on
54: # these JDH (2009-11-06)
55: 
56: 
57: def _to_int(x):
58:     return int(float(x))
59: 
60: 
61: _to_float = float
62: 
63: 
64: def _to_str(x):
65:     return x.decode('utf8')
66: 
67: 
68: def _to_list_of_ints(s):
69:     s = s.replace(b',', b' ')
70:     return [_to_int(val) for val in s.split()]
71: 
72: 
73: def _to_list_of_floats(s):
74:     return [_to_float(val) for val in s.split()]
75: 
76: 
77: def _to_bool(s):
78:     if s.lower().strip() in (b'false', b'0', b'no'):
79:         return False
80:     else:
81:         return True
82: 
83: 
84: def _sanity_check(fh):
85:     '''
86:     Check if the file at least looks like AFM.
87:     If not, raise :exc:`RuntimeError`.
88:     '''
89: 
90:     # Remember the file position in case the caller wants to
91:     # do something else with the file.
92:     pos = fh.tell()
93:     try:
94:         line = next(fh)
95:     finally:
96:         fh.seek(pos, 0)
97: 
98:     # AFM spec, Section 4: The StartFontMetrics keyword [followed by a
99:     # version number] must be the first line in the file, and the
100:     # EndFontMetrics keyword must be the last non-empty line in the
101:     # file. We just check the first line.
102:     if not line.startswith(b'StartFontMetrics'):
103:         raise RuntimeError('Not an AFM file')
104: 
105: 
106: def _parse_header(fh):
107:     '''
108:     Reads the font metrics header (up to the char metrics) and returns
109:     a dictionary mapping *key* to *val*.  *val* will be converted to the
110:     appropriate python type as necessary; e.g.:
111: 
112:         * 'False'->False
113:         * '0'->0
114:         * '-168 -218 1000 898'-> [-168, -218, 1000, 898]
115: 
116:     Dictionary keys are
117: 
118:       StartFontMetrics, FontName, FullName, FamilyName, Weight,
119:       ItalicAngle, IsFixedPitch, FontBBox, UnderlinePosition,
120:       UnderlineThickness, Version, Notice, EncodingScheme, CapHeight,
121:       XHeight, Ascender, Descender, StartCharMetrics
122: 
123:     '''
124:     headerConverters = {
125:         b'StartFontMetrics': _to_float,
126:         b'FontName': _to_str,
127:         b'FullName': _to_str,
128:         b'FamilyName': _to_str,
129:         b'Weight': _to_str,
130:         b'ItalicAngle': _to_float,
131:         b'IsFixedPitch': _to_bool,
132:         b'FontBBox': _to_list_of_ints,
133:         b'UnderlinePosition': _to_int,
134:         b'UnderlineThickness': _to_int,
135:         b'Version': _to_str,
136:         b'Notice': _to_str,
137:         b'EncodingScheme': _to_str,
138:         b'CapHeight': _to_float,  # Is the second version a mistake, or
139:         b'Capheight': _to_float,  # do some AFM files contain 'Capheight'? -JKS
140:         b'XHeight': _to_float,
141:         b'Ascender': _to_float,
142:         b'Descender': _to_float,
143:         b'StdHW': _to_float,
144:         b'StdVW': _to_float,
145:         b'StartCharMetrics': _to_int,
146:         b'CharacterSet': _to_str,
147:         b'Characters': _to_int,
148:         }
149: 
150:     d = {}
151:     for line in fh:
152:         line = line.rstrip()
153:         if line.startswith(b'Comment'):
154:             continue
155:         lst = line.split(b' ', 1)
156: 
157:         key = lst[0]
158:         if len(lst) == 2:
159:             val = lst[1]
160:         else:
161:             val = b''
162: 
163:         try:
164:             d[key] = headerConverters[key](val)
165:         except ValueError:
166:             print('Value error parsing header in AFM:',
167:                   key, val, file=sys.stderr)
168:             continue
169:         except KeyError:
170:             print('Found an unknown keyword in AFM header (was %r)' % key,
171:                   file=sys.stderr)
172:             continue
173:         if key == b'StartCharMetrics':
174:             return d
175:     raise RuntimeError('Bad parse')
176: 
177: 
178: def _parse_char_metrics(fh):
179:     '''
180:     Return a character metric dictionary.  Keys are the ASCII num of
181:     the character, values are a (*wx*, *name*, *bbox*) tuple, where
182:     *wx* is the character width, *name* is the postscript language
183:     name, and *bbox* is a (*llx*, *lly*, *urx*, *ury*) tuple.
184: 
185:     This function is incomplete per the standard, but thus far parses
186:     all the sample afm files tried.
187:     '''
188: 
189:     ascii_d = {}
190:     name_d = {}
191:     for line in fh:
192:         # We are defensively letting values be utf8. The spec requires
193:         # ascii, but there are non-compliant fonts in circulation
194:         line = _to_str(line.rstrip())  # Convert from byte-literal
195:         if line.startswith('EndCharMetrics'):
196:             return ascii_d, name_d
197:         # Split the metric line into a dictionary, keyed by metric identifiers
198:         vals = dict(s.strip().split(' ', 1) for s in line.split(';') if s)
199:         # There may be other metrics present, but only these are needed
200:         if not {'C', 'WX', 'N', 'B'}.issubset(vals):
201:             raise RuntimeError('Bad char metrics line: %s' % line)
202:         num = _to_int(vals['C'])
203:         wx = _to_float(vals['WX'])
204:         name = vals['N']
205:         bbox = _to_list_of_floats(vals['B'])
206:         bbox = list(map(int, bbox))
207:         # Workaround: If the character name is 'Euro', give it the
208:         # corresponding character code, according to WinAnsiEncoding (see PDF
209:         # Reference).
210:         if name == 'Euro':
211:             num = 128
212:         if num != -1:
213:             ascii_d[num] = (wx, name, bbox)
214:         name_d[name] = (wx, bbox)
215:     raise RuntimeError('Bad parse')
216: 
217: 
218: def _parse_kern_pairs(fh):
219:     '''
220:     Return a kern pairs dictionary; keys are (*char1*, *char2*) tuples and
221:     values are the kern pair value.  For example, a kern pairs line like
222:     ``KPX A y -50``
223: 
224:     will be represented as::
225: 
226:       d[ ('A', 'y') ] = -50
227: 
228:     '''
229: 
230:     line = next(fh)
231:     if not line.startswith(b'StartKernPairs'):
232:         raise RuntimeError('Bad start of kern pairs data: %s' % line)
233: 
234:     d = {}
235:     for line in fh:
236:         line = line.rstrip()
237:         if not line:
238:             continue
239:         if line.startswith(b'EndKernPairs'):
240:             next(fh)  # EndKernData
241:             return d
242:         vals = line.split()
243:         if len(vals) != 4 or vals[0] != b'KPX':
244:             raise RuntimeError('Bad kern pairs line: %s' % line)
245:         c1, c2, val = _to_str(vals[1]), _to_str(vals[2]), _to_float(vals[3])
246:         d[(c1, c2)] = val
247:     raise RuntimeError('Bad kern pairs parse')
248: 
249: 
250: def _parse_composites(fh):
251:     '''
252:     Return a composites dictionary.  Keys are the names of the
253:     composites.  Values are a num parts list of composite information,
254:     with each element being a (*name*, *dx*, *dy*) tuple.  Thus a
255:     composites line reading:
256: 
257:       CC Aacute 2 ; PCC A 0 0 ; PCC acute 160 170 ;
258: 
259:     will be represented as::
260: 
261:       d['Aacute'] = [ ('A', 0, 0), ('acute', 160, 170) ]
262: 
263:     '''
264:     d = {}
265:     for line in fh:
266:         line = line.rstrip()
267:         if not line:
268:             continue
269:         if line.startswith(b'EndComposites'):
270:             return d
271:         vals = line.split(b';')
272:         cc = vals[0].split()
273:         name, numParts = cc[1], _to_int(cc[2])
274:         pccParts = []
275:         for s in vals[1:-1]:
276:             pcc = s.split()
277:             name, dx, dy = pcc[1], _to_float(pcc[2]), _to_float(pcc[3])
278:             pccParts.append((name, dx, dy))
279:         d[name] = pccParts
280: 
281:     raise RuntimeError('Bad composites parse')
282: 
283: 
284: def _parse_optional(fh):
285:     '''
286:     Parse the optional fields for kern pair data and composites
287: 
288:     return value is a (*kernDict*, *compositeDict*) which are the
289:     return values from :func:`_parse_kern_pairs`, and
290:     :func:`_parse_composites` if the data exists, or empty dicts
291:     otherwise
292:     '''
293:     optional = {
294:         b'StartKernData': _parse_kern_pairs,
295:         b'StartComposites':  _parse_composites,
296:         }
297: 
298:     d = {b'StartKernData': {}, b'StartComposites': {}}
299:     for line in fh:
300:         line = line.rstrip()
301:         if not line:
302:             continue
303:         key = line.split()[0]
304: 
305:         if key in optional:
306:             d[key] = optional[key](fh)
307: 
308:     l = (d[b'StartKernData'], d[b'StartComposites'])
309:     return l
310: 
311: 
312: def parse_afm(fh):
313:     '''
314:     Parse the Adobe Font Metics file in file handle *fh*. Return value
315:     is a (*dhead*, *dcmetrics*, *dkernpairs*, *dcomposite*) tuple where
316:     *dhead* is a :func:`_parse_header` dict, *dcmetrics* is a
317:     :func:`_parse_composites` dict, *dkernpairs* is a
318:     :func:`_parse_kern_pairs` dict (possibly {}), and *dcomposite* is a
319:     :func:`_parse_composites` dict (possibly {})
320:     '''
321:     _sanity_check(fh)
322:     dhead = _parse_header(fh)
323:     dcmetrics_ascii, dcmetrics_name = _parse_char_metrics(fh)
324:     doptional = _parse_optional(fh)
325:     return dhead, dcmetrics_ascii, dcmetrics_name, doptional[0], doptional[1]
326: 
327: 
328: class AFM(object):
329: 
330:     def __init__(self, fh):
331:         '''
332:         Parse the AFM file in file object *fh*
333:         '''
334:         (dhead, dcmetrics_ascii, dcmetrics_name, dkernpairs, dcomposite) = \
335:             parse_afm(fh)
336:         self._header = dhead
337:         self._kern = dkernpairs
338:         self._metrics = dcmetrics_ascii
339:         self._metrics_by_name = dcmetrics_name
340:         self._composite = dcomposite
341: 
342:     def get_bbox_char(self, c, isord=False):
343:         if not isord:
344:             c = ord(c)
345:         wx, name, bbox = self._metrics[c]
346:         return bbox
347: 
348:     def string_width_height(self, s):
349:         '''
350:         Return the string width (including kerning) and string height
351:         as a (*w*, *h*) tuple.
352:         '''
353:         if not len(s):
354:             return 0, 0
355:         totalw = 0
356:         namelast = None
357:         miny = 1e9
358:         maxy = 0
359:         for c in s:
360:             if c == '\n':
361:                 continue
362:             wx, name, bbox = self._metrics[ord(c)]
363:             l, b, w, h = bbox
364: 
365:             # find the width with kerning
366:             try:
367:                 kp = self._kern[(namelast, name)]
368:             except KeyError:
369:                 kp = 0
370:             totalw += wx + kp
371: 
372:             # find the max y
373:             thismax = b + h
374:             if thismax > maxy:
375:                 maxy = thismax
376: 
377:             # find the min y
378:             thismin = b
379:             if thismin < miny:
380:                 miny = thismin
381:             namelast = name
382: 
383:         return totalw, maxy - miny
384: 
385:     def get_str_bbox_and_descent(self, s):
386:         '''
387:         Return the string bounding box
388:         '''
389:         if not len(s):
390:             return 0, 0, 0, 0
391:         totalw = 0
392:         namelast = None
393:         miny = 1e9
394:         maxy = 0
395:         left = 0
396:         if not isinstance(s, six.text_type):
397:             s = _to_str(s)
398:         for c in s:
399:             if c == '\n':
400:                 continue
401:             name = uni2type1.get(ord(c), 'question')
402:             try:
403:                 wx, bbox = self._metrics_by_name[name]
404:             except KeyError:
405:                 name = 'question'
406:                 wx, bbox = self._metrics_by_name[name]
407:             l, b, w, h = bbox
408:             if l < left:
409:                 left = l
410:             # find the width with kerning
411:             try:
412:                 kp = self._kern[(namelast, name)]
413:             except KeyError:
414:                 kp = 0
415:             totalw += wx + kp
416: 
417:             # find the max y
418:             thismax = b + h
419:             if thismax > maxy:
420:                 maxy = thismax
421: 
422:             # find the min y
423:             thismin = b
424:             if thismin < miny:
425:                 miny = thismin
426:             namelast = name
427: 
428:         return left, miny, totalw, maxy - miny, -miny
429: 
430:     def get_str_bbox(self, s):
431:         '''
432:         Return the string bounding box
433:         '''
434:         return self.get_str_bbox_and_descent(s)[:4]
435: 
436:     def get_name_char(self, c, isord=False):
437:         '''
438:         Get the name of the character, i.e., ';' is 'semicolon'
439:         '''
440:         if not isord:
441:             c = ord(c)
442:         wx, name, bbox = self._metrics[c]
443:         return name
444: 
445:     def get_width_char(self, c, isord=False):
446:         '''
447:         Get the width of the character from the character metric WX
448:         field
449:         '''
450:         if not isord:
451:             c = ord(c)
452:         wx, name, bbox = self._metrics[c]
453:         return wx
454: 
455:     def get_width_from_char_name(self, name):
456:         '''
457:         Get the width of the character from a type1 character name
458:         '''
459:         wx, bbox = self._metrics_by_name[name]
460:         return wx
461: 
462:     def get_height_char(self, c, isord=False):
463:         '''
464:         Get the height of character *c* from the bounding box.  This
465:         is the ink height (space is 0)
466:         '''
467:         if not isord:
468:             c = ord(c)
469:         wx, name, bbox = self._metrics[c]
470:         return bbox[-1]
471: 
472:     def get_kern_dist(self, c1, c2):
473:         '''
474:         Return the kerning pair distance (possibly 0) for chars *c1*
475:         and *c2*
476:         '''
477:         name1, name2 = self.get_name_char(c1), self.get_name_char(c2)
478:         return self.get_kern_dist_from_name(name1, name2)
479: 
480:     def get_kern_dist_from_name(self, name1, name2):
481:         '''
482:         Return the kerning pair distance (possibly 0) for chars
483:         *name1* and *name2*
484:         '''
485:         return self._kern.get((name1, name2), 0)
486: 
487:     def get_fontname(self):
488:         "Return the font name, e.g., 'Times-Roman'"
489:         return self._header[b'FontName']
490: 
491:     def get_fullname(self):
492:         "Return the font full name, e.g., 'Times-Roman'"
493:         name = self._header.get(b'FullName')
494:         if name is None:  # use FontName as a substitute
495:             name = self._header[b'FontName']
496:         return name
497: 
498:     def get_familyname(self):
499:         "Return the font family name, e.g., 'Times'"
500:         name = self._header.get(b'FamilyName')
501:         if name is not None:
502:             return name
503: 
504:         # FamilyName not specified so we'll make a guess
505:         name = self.get_fullname()
506:         extras = (br'(?i)([ -](regular|plain|italic|oblique|bold|semibold|'
507:                   br'light|ultralight|extra|condensed))+$')
508:         return re.sub(extras, '', name)
509: 
510:     @property
511:     def family_name(self):
512:         return self.get_familyname()
513: 
514:     def get_weight(self):
515:         "Return the font weight, e.g., 'Bold' or 'Roman'"
516:         return self._header[b'Weight']
517: 
518:     def get_angle(self):
519:         "Return the fontangle as float"
520:         return self._header[b'ItalicAngle']
521: 
522:     def get_capheight(self):
523:         "Return the cap height as float"
524:         return self._header[b'CapHeight']
525: 
526:     def get_xheight(self):
527:         "Return the xheight as float"
528:         return self._header[b'XHeight']
529: 
530:     def get_underline_thickness(self):
531:         "Return the underline thickness as float"
532:         return self._header[b'UnderlineThickness']
533: 
534:     def get_horizontal_stem_width(self):
535:         '''
536:         Return the standard horizontal stem width as float, or *None* if
537:         not specified in AFM file.
538:         '''
539:         return self._header.get(b'StdHW', None)
540: 
541:     def get_vertical_stem_width(self):
542:         '''
543:         Return the standard vertical stem width as float, or *None* if
544:         not specified in AFM file.
545:         '''
546:         return self._header.get(b'StdVW', None)
547: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

unicode_49 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, (-1)), 'unicode', u"\nThis is a python interface to Adobe Font Metrics Files.  Although a\nnumber of other python implementations exist, and may be more complete\nthan this, it was decided not to go with them because they were\neither:\n\n  1) copyrighted or used a non-BSD compatible license\n\n  2) had too many dependencies and a free standing lib was needed\n\n  3) Did more than needed and it was easier to write afresh rather than\n     figure out how to get just what was needed.\n\nIt is pretty easy to use, and requires only built-in python libs:\n\n    >>> from matplotlib import rcParams\n    >>> import os.path\n    >>> afm_fname = os.path.join(rcParams['datapath'],\n    ...                         'fonts', 'afm', 'ptmr8a.afm')\n    >>>\n    >>> from matplotlib.afm import AFM\n    >>> with open(afm_fname) as fh:\n    ...     afm = AFM(fh)\n    >>> afm.string_width_height('What the heck?')\n    (6220.0, 694)\n    >>> afm.get_fontname()\n    'Times-Roman'\n    >>> afm.get_kern_dist('A', 'f')\n    0\n    >>> afm.get_kern_dist('A', 'y')\n    -92.0\n    >>> afm.get_bbox_char('!')\n    [130, -9, 238, 676]\n\n")
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 40, 0))

# 'import six' statement (line 40)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_50 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 40, 0), 'six')

if (type(import_50) is not StypyTypeError):

    if (import_50 != 'pyd_module'):
        __import__(import_50)
        sys_modules_51 = sys.modules[import_50]
        import_module(stypy.reporting.localization.Localization(__file__, 40, 0), 'six', sys_modules_51.module_type_store, module_type_store)
    else:
        import six

        import_module(stypy.reporting.localization.Localization(__file__, 40, 0), 'six', six, module_type_store)

else:
    # Assigning a type to the variable 'six' (line 40)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 0), 'six', import_50)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 41, 0))

# 'from six.moves import map' statement (line 41)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_52 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 41, 0), 'six.moves')

if (type(import_52) is not StypyTypeError):

    if (import_52 != 'pyd_module'):
        __import__(import_52)
        sys_modules_53 = sys.modules[import_52]
        import_from_module(stypy.reporting.localization.Localization(__file__, 41, 0), 'six.moves', sys_modules_53.module_type_store, module_type_store, ['map'])
        nest_module(stypy.reporting.localization.Localization(__file__, 41, 0), __file__, sys_modules_53, sys_modules_53.module_type_store, module_type_store)
    else:
        from six.moves import map

        import_from_module(stypy.reporting.localization.Localization(__file__, 41, 0), 'six.moves', None, module_type_store, ['map'], [map])

else:
    # Assigning a type to the variable 'six.moves' (line 41)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 0), 'six.moves', import_52)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 43, 0))

# 'import sys' statement (line 43)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 43, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 44, 0))

# 'import os' statement (line 44)
import os

import_module(stypy.reporting.localization.Localization(__file__, 44, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 45, 0))

# 'import re' statement (line 45)
import re

import_module(stypy.reporting.localization.Localization(__file__, 45, 0), 're', re, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 46, 0))

# 'from matplotlib._mathtext_data import uni2type1' statement (line 46)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_54 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 46, 0), 'matplotlib._mathtext_data')

if (type(import_54) is not StypyTypeError):

    if (import_54 != 'pyd_module'):
        __import__(import_54)
        sys_modules_55 = sys.modules[import_54]
        import_from_module(stypy.reporting.localization.Localization(__file__, 46, 0), 'matplotlib._mathtext_data', sys_modules_55.module_type_store, module_type_store, ['uni2type1'])
        nest_module(stypy.reporting.localization.Localization(__file__, 46, 0), __file__, sys_modules_55, sys_modules_55.module_type_store, module_type_store)
    else:
        from matplotlib._mathtext_data import uni2type1

        import_from_module(stypy.reporting.localization.Localization(__file__, 46, 0), 'matplotlib._mathtext_data', None, module_type_store, ['uni2type1'], [uni2type1])

else:
    # Assigning a type to the variable 'matplotlib._mathtext_data' (line 46)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 0), 'matplotlib._mathtext_data', import_54)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')


@norecursion
def _to_int(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_to_int'
    module_type_store = module_type_store.open_function_context('_to_int', 57, 0, False)
    
    # Passed parameters checking function
    _to_int.stypy_localization = localization
    _to_int.stypy_type_of_self = None
    _to_int.stypy_type_store = module_type_store
    _to_int.stypy_function_name = '_to_int'
    _to_int.stypy_param_names_list = ['x']
    _to_int.stypy_varargs_param_name = None
    _to_int.stypy_kwargs_param_name = None
    _to_int.stypy_call_defaults = defaults
    _to_int.stypy_call_varargs = varargs
    _to_int.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_to_int', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_to_int', localization, ['x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_to_int(...)' code ##################

    
    # Call to int(...): (line 58)
    # Processing the call arguments (line 58)
    
    # Call to float(...): (line 58)
    # Processing the call arguments (line 58)
    # Getting the type of 'x' (line 58)
    x_58 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 21), 'x', False)
    # Processing the call keyword arguments (line 58)
    kwargs_59 = {}
    # Getting the type of 'float' (line 58)
    float_57 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 15), 'float', False)
    # Calling float(args, kwargs) (line 58)
    float_call_result_60 = invoke(stypy.reporting.localization.Localization(__file__, 58, 15), float_57, *[x_58], **kwargs_59)
    
    # Processing the call keyword arguments (line 58)
    kwargs_61 = {}
    # Getting the type of 'int' (line 58)
    int_56 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 11), 'int', False)
    # Calling int(args, kwargs) (line 58)
    int_call_result_62 = invoke(stypy.reporting.localization.Localization(__file__, 58, 11), int_56, *[float_call_result_60], **kwargs_61)
    
    # Assigning a type to the variable 'stypy_return_type' (line 58)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'stypy_return_type', int_call_result_62)
    
    # ################# End of '_to_int(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_to_int' in the type store
    # Getting the type of 'stypy_return_type' (line 57)
    stypy_return_type_63 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_63)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_to_int'
    return stypy_return_type_63

# Assigning a type to the variable '_to_int' (line 57)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 0), '_to_int', _to_int)

# Assigning a Name to a Name (line 61):

# Assigning a Name to a Name (line 61):
# Getting the type of 'float' (line 61)
float_64 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 12), 'float')
# Assigning a type to the variable '_to_float' (line 61)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 0), '_to_float', float_64)

@norecursion
def _to_str(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_to_str'
    module_type_store = module_type_store.open_function_context('_to_str', 64, 0, False)
    
    # Passed parameters checking function
    _to_str.stypy_localization = localization
    _to_str.stypy_type_of_self = None
    _to_str.stypy_type_store = module_type_store
    _to_str.stypy_function_name = '_to_str'
    _to_str.stypy_param_names_list = ['x']
    _to_str.stypy_varargs_param_name = None
    _to_str.stypy_kwargs_param_name = None
    _to_str.stypy_call_defaults = defaults
    _to_str.stypy_call_varargs = varargs
    _to_str.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_to_str', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_to_str', localization, ['x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_to_str(...)' code ##################

    
    # Call to decode(...): (line 65)
    # Processing the call arguments (line 65)
    unicode_67 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 20), 'unicode', u'utf8')
    # Processing the call keyword arguments (line 65)
    kwargs_68 = {}
    # Getting the type of 'x' (line 65)
    x_65 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 11), 'x', False)
    # Obtaining the member 'decode' of a type (line 65)
    decode_66 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 11), x_65, 'decode')
    # Calling decode(args, kwargs) (line 65)
    decode_call_result_69 = invoke(stypy.reporting.localization.Localization(__file__, 65, 11), decode_66, *[unicode_67], **kwargs_68)
    
    # Assigning a type to the variable 'stypy_return_type' (line 65)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'stypy_return_type', decode_call_result_69)
    
    # ################# End of '_to_str(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_to_str' in the type store
    # Getting the type of 'stypy_return_type' (line 64)
    stypy_return_type_70 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_70)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_to_str'
    return stypy_return_type_70

# Assigning a type to the variable '_to_str' (line 64)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 0), '_to_str', _to_str)

@norecursion
def _to_list_of_ints(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_to_list_of_ints'
    module_type_store = module_type_store.open_function_context('_to_list_of_ints', 68, 0, False)
    
    # Passed parameters checking function
    _to_list_of_ints.stypy_localization = localization
    _to_list_of_ints.stypy_type_of_self = None
    _to_list_of_ints.stypy_type_store = module_type_store
    _to_list_of_ints.stypy_function_name = '_to_list_of_ints'
    _to_list_of_ints.stypy_param_names_list = ['s']
    _to_list_of_ints.stypy_varargs_param_name = None
    _to_list_of_ints.stypy_kwargs_param_name = None
    _to_list_of_ints.stypy_call_defaults = defaults
    _to_list_of_ints.stypy_call_varargs = varargs
    _to_list_of_ints.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_to_list_of_ints', ['s'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_to_list_of_ints', localization, ['s'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_to_list_of_ints(...)' code ##################

    
    # Assigning a Call to a Name (line 69):
    
    # Assigning a Call to a Name (line 69):
    
    # Call to replace(...): (line 69)
    # Processing the call arguments (line 69)
    str_73 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 18), 'str', ',')
    str_74 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 24), 'str', ' ')
    # Processing the call keyword arguments (line 69)
    kwargs_75 = {}
    # Getting the type of 's' (line 69)
    s_71 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 's', False)
    # Obtaining the member 'replace' of a type (line 69)
    replace_72 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 8), s_71, 'replace')
    # Calling replace(args, kwargs) (line 69)
    replace_call_result_76 = invoke(stypy.reporting.localization.Localization(__file__, 69, 8), replace_72, *[str_73, str_74], **kwargs_75)
    
    # Assigning a type to the variable 's' (line 69)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 's', replace_call_result_76)
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to split(...): (line 70)
    # Processing the call keyword arguments (line 70)
    kwargs_83 = {}
    # Getting the type of 's' (line 70)
    s_81 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 36), 's', False)
    # Obtaining the member 'split' of a type (line 70)
    split_82 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 36), s_81, 'split')
    # Calling split(args, kwargs) (line 70)
    split_call_result_84 = invoke(stypy.reporting.localization.Localization(__file__, 70, 36), split_82, *[], **kwargs_83)
    
    comprehension_85 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 12), split_call_result_84)
    # Assigning a type to the variable 'val' (line 70)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 12), 'val', comprehension_85)
    
    # Call to _to_int(...): (line 70)
    # Processing the call arguments (line 70)
    # Getting the type of 'val' (line 70)
    val_78 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 20), 'val', False)
    # Processing the call keyword arguments (line 70)
    kwargs_79 = {}
    # Getting the type of '_to_int' (line 70)
    _to_int_77 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 12), '_to_int', False)
    # Calling _to_int(args, kwargs) (line 70)
    _to_int_call_result_80 = invoke(stypy.reporting.localization.Localization(__file__, 70, 12), _to_int_77, *[val_78], **kwargs_79)
    
    list_86 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 12), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 12), list_86, _to_int_call_result_80)
    # Assigning a type to the variable 'stypy_return_type' (line 70)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 4), 'stypy_return_type', list_86)
    
    # ################# End of '_to_list_of_ints(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_to_list_of_ints' in the type store
    # Getting the type of 'stypy_return_type' (line 68)
    stypy_return_type_87 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_87)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_to_list_of_ints'
    return stypy_return_type_87

# Assigning a type to the variable '_to_list_of_ints' (line 68)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 0), '_to_list_of_ints', _to_list_of_ints)

@norecursion
def _to_list_of_floats(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_to_list_of_floats'
    module_type_store = module_type_store.open_function_context('_to_list_of_floats', 73, 0, False)
    
    # Passed parameters checking function
    _to_list_of_floats.stypy_localization = localization
    _to_list_of_floats.stypy_type_of_self = None
    _to_list_of_floats.stypy_type_store = module_type_store
    _to_list_of_floats.stypy_function_name = '_to_list_of_floats'
    _to_list_of_floats.stypy_param_names_list = ['s']
    _to_list_of_floats.stypy_varargs_param_name = None
    _to_list_of_floats.stypy_kwargs_param_name = None
    _to_list_of_floats.stypy_call_defaults = defaults
    _to_list_of_floats.stypy_call_varargs = varargs
    _to_list_of_floats.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_to_list_of_floats', ['s'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_to_list_of_floats', localization, ['s'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_to_list_of_floats(...)' code ##################

    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to split(...): (line 74)
    # Processing the call keyword arguments (line 74)
    kwargs_94 = {}
    # Getting the type of 's' (line 74)
    s_92 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 38), 's', False)
    # Obtaining the member 'split' of a type (line 74)
    split_93 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 38), s_92, 'split')
    # Calling split(args, kwargs) (line 74)
    split_call_result_95 = invoke(stypy.reporting.localization.Localization(__file__, 74, 38), split_93, *[], **kwargs_94)
    
    comprehension_96 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 12), split_call_result_95)
    # Assigning a type to the variable 'val' (line 74)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 12), 'val', comprehension_96)
    
    # Call to _to_float(...): (line 74)
    # Processing the call arguments (line 74)
    # Getting the type of 'val' (line 74)
    val_89 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 22), 'val', False)
    # Processing the call keyword arguments (line 74)
    kwargs_90 = {}
    # Getting the type of '_to_float' (line 74)
    _to_float_88 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 12), '_to_float', False)
    # Calling _to_float(args, kwargs) (line 74)
    _to_float_call_result_91 = invoke(stypy.reporting.localization.Localization(__file__, 74, 12), _to_float_88, *[val_89], **kwargs_90)
    
    list_97 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 12), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 12), list_97, _to_float_call_result_91)
    # Assigning a type to the variable 'stypy_return_type' (line 74)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), 'stypy_return_type', list_97)
    
    # ################# End of '_to_list_of_floats(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_to_list_of_floats' in the type store
    # Getting the type of 'stypy_return_type' (line 73)
    stypy_return_type_98 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_98)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_to_list_of_floats'
    return stypy_return_type_98

# Assigning a type to the variable '_to_list_of_floats' (line 73)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 0), '_to_list_of_floats', _to_list_of_floats)

@norecursion
def _to_bool(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_to_bool'
    module_type_store = module_type_store.open_function_context('_to_bool', 77, 0, False)
    
    # Passed parameters checking function
    _to_bool.stypy_localization = localization
    _to_bool.stypy_type_of_self = None
    _to_bool.stypy_type_store = module_type_store
    _to_bool.stypy_function_name = '_to_bool'
    _to_bool.stypy_param_names_list = ['s']
    _to_bool.stypy_varargs_param_name = None
    _to_bool.stypy_kwargs_param_name = None
    _to_bool.stypy_call_defaults = defaults
    _to_bool.stypy_call_varargs = varargs
    _to_bool.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_to_bool', ['s'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_to_bool', localization, ['s'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_to_bool(...)' code ##################

    
    
    
    # Call to strip(...): (line 78)
    # Processing the call keyword arguments (line 78)
    kwargs_104 = {}
    
    # Call to lower(...): (line 78)
    # Processing the call keyword arguments (line 78)
    kwargs_101 = {}
    # Getting the type of 's' (line 78)
    s_99 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 7), 's', False)
    # Obtaining the member 'lower' of a type (line 78)
    lower_100 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 7), s_99, 'lower')
    # Calling lower(args, kwargs) (line 78)
    lower_call_result_102 = invoke(stypy.reporting.localization.Localization(__file__, 78, 7), lower_100, *[], **kwargs_101)
    
    # Obtaining the member 'strip' of a type (line 78)
    strip_103 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 7), lower_call_result_102, 'strip')
    # Calling strip(args, kwargs) (line 78)
    strip_call_result_105 = invoke(stypy.reporting.localization.Localization(__file__, 78, 7), strip_103, *[], **kwargs_104)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 78)
    tuple_106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 29), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 78)
    # Adding element type (line 78)
    str_107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 29), 'str', 'false')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 29), tuple_106, str_107)
    # Adding element type (line 78)
    str_108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 39), 'str', '0')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 29), tuple_106, str_108)
    # Adding element type (line 78)
    str_109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 45), 'str', 'no')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 29), tuple_106, str_109)
    
    # Applying the binary operator 'in' (line 78)
    result_contains_110 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 7), 'in', strip_call_result_105, tuple_106)
    
    # Testing the type of an if condition (line 78)
    if_condition_111 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 78, 4), result_contains_110)
    # Assigning a type to the variable 'if_condition_111' (line 78)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'if_condition_111', if_condition_111)
    # SSA begins for if statement (line 78)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'False' (line 79)
    False_112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 15), 'False')
    # Assigning a type to the variable 'stypy_return_type' (line 79)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'stypy_return_type', False_112)
    # SSA branch for the else part of an if statement (line 78)
    module_type_store.open_ssa_branch('else')
    # Getting the type of 'True' (line 81)
    True_113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 15), 'True')
    # Assigning a type to the variable 'stypy_return_type' (line 81)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'stypy_return_type', True_113)
    # SSA join for if statement (line 78)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of '_to_bool(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_to_bool' in the type store
    # Getting the type of 'stypy_return_type' (line 77)
    stypy_return_type_114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_114)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_to_bool'
    return stypy_return_type_114

# Assigning a type to the variable '_to_bool' (line 77)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 0), '_to_bool', _to_bool)

@norecursion
def _sanity_check(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_sanity_check'
    module_type_store = module_type_store.open_function_context('_sanity_check', 84, 0, False)
    
    # Passed parameters checking function
    _sanity_check.stypy_localization = localization
    _sanity_check.stypy_type_of_self = None
    _sanity_check.stypy_type_store = module_type_store
    _sanity_check.stypy_function_name = '_sanity_check'
    _sanity_check.stypy_param_names_list = ['fh']
    _sanity_check.stypy_varargs_param_name = None
    _sanity_check.stypy_kwargs_param_name = None
    _sanity_check.stypy_call_defaults = defaults
    _sanity_check.stypy_call_varargs = varargs
    _sanity_check.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_sanity_check', ['fh'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_sanity_check', localization, ['fh'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_sanity_check(...)' code ##################

    unicode_115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, (-1)), 'unicode', u'\n    Check if the file at least looks like AFM.\n    If not, raise :exc:`RuntimeError`.\n    ')
    
    # Assigning a Call to a Name (line 92):
    
    # Assigning a Call to a Name (line 92):
    
    # Call to tell(...): (line 92)
    # Processing the call keyword arguments (line 92)
    kwargs_118 = {}
    # Getting the type of 'fh' (line 92)
    fh_116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 10), 'fh', False)
    # Obtaining the member 'tell' of a type (line 92)
    tell_117 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 10), fh_116, 'tell')
    # Calling tell(args, kwargs) (line 92)
    tell_call_result_119 = invoke(stypy.reporting.localization.Localization(__file__, 92, 10), tell_117, *[], **kwargs_118)
    
    # Assigning a type to the variable 'pos' (line 92)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 4), 'pos', tell_call_result_119)
    
    # Try-finally block (line 93)
    
    # Assigning a Call to a Name (line 94):
    
    # Assigning a Call to a Name (line 94):
    
    # Call to next(...): (line 94)
    # Processing the call arguments (line 94)
    # Getting the type of 'fh' (line 94)
    fh_121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 20), 'fh', False)
    # Processing the call keyword arguments (line 94)
    kwargs_122 = {}
    # Getting the type of 'next' (line 94)
    next_120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 15), 'next', False)
    # Calling next(args, kwargs) (line 94)
    next_call_result_123 = invoke(stypy.reporting.localization.Localization(__file__, 94, 15), next_120, *[fh_121], **kwargs_122)
    
    # Assigning a type to the variable 'line' (line 94)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'line', next_call_result_123)
    
    # finally branch of the try-finally block (line 93)
    
    # Call to seek(...): (line 96)
    # Processing the call arguments (line 96)
    # Getting the type of 'pos' (line 96)
    pos_126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 16), 'pos', False)
    int_127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 21), 'int')
    # Processing the call keyword arguments (line 96)
    kwargs_128 = {}
    # Getting the type of 'fh' (line 96)
    fh_124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'fh', False)
    # Obtaining the member 'seek' of a type (line 96)
    seek_125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 8), fh_124, 'seek')
    # Calling seek(args, kwargs) (line 96)
    seek_call_result_129 = invoke(stypy.reporting.localization.Localization(__file__, 96, 8), seek_125, *[pos_126, int_127], **kwargs_128)
    
    
    
    
    
    # Call to startswith(...): (line 102)
    # Processing the call arguments (line 102)
    str_132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 27), 'str', 'StartFontMetrics')
    # Processing the call keyword arguments (line 102)
    kwargs_133 = {}
    # Getting the type of 'line' (line 102)
    line_130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 11), 'line', False)
    # Obtaining the member 'startswith' of a type (line 102)
    startswith_131 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 11), line_130, 'startswith')
    # Calling startswith(args, kwargs) (line 102)
    startswith_call_result_134 = invoke(stypy.reporting.localization.Localization(__file__, 102, 11), startswith_131, *[str_132], **kwargs_133)
    
    # Applying the 'not' unary operator (line 102)
    result_not__135 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 7), 'not', startswith_call_result_134)
    
    # Testing the type of an if condition (line 102)
    if_condition_136 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 102, 4), result_not__135)
    # Assigning a type to the variable 'if_condition_136' (line 102)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 4), 'if_condition_136', if_condition_136)
    # SSA begins for if statement (line 102)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to RuntimeError(...): (line 103)
    # Processing the call arguments (line 103)
    unicode_138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 27), 'unicode', u'Not an AFM file')
    # Processing the call keyword arguments (line 103)
    kwargs_139 = {}
    # Getting the type of 'RuntimeError' (line 103)
    RuntimeError_137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 14), 'RuntimeError', False)
    # Calling RuntimeError(args, kwargs) (line 103)
    RuntimeError_call_result_140 = invoke(stypy.reporting.localization.Localization(__file__, 103, 14), RuntimeError_137, *[unicode_138], **kwargs_139)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 103, 8), RuntimeError_call_result_140, 'raise parameter', BaseException)
    # SSA join for if statement (line 102)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of '_sanity_check(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_sanity_check' in the type store
    # Getting the type of 'stypy_return_type' (line 84)
    stypy_return_type_141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_141)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_sanity_check'
    return stypy_return_type_141

# Assigning a type to the variable '_sanity_check' (line 84)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 0), '_sanity_check', _sanity_check)

@norecursion
def _parse_header(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_parse_header'
    module_type_store = module_type_store.open_function_context('_parse_header', 106, 0, False)
    
    # Passed parameters checking function
    _parse_header.stypy_localization = localization
    _parse_header.stypy_type_of_self = None
    _parse_header.stypy_type_store = module_type_store
    _parse_header.stypy_function_name = '_parse_header'
    _parse_header.stypy_param_names_list = ['fh']
    _parse_header.stypy_varargs_param_name = None
    _parse_header.stypy_kwargs_param_name = None
    _parse_header.stypy_call_defaults = defaults
    _parse_header.stypy_call_varargs = varargs
    _parse_header.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_parse_header', ['fh'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_parse_header', localization, ['fh'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_parse_header(...)' code ##################

    unicode_142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, (-1)), 'unicode', u"\n    Reads the font metrics header (up to the char metrics) and returns\n    a dictionary mapping *key* to *val*.  *val* will be converted to the\n    appropriate python type as necessary; e.g.:\n\n        * 'False'->False\n        * '0'->0\n        * '-168 -218 1000 898'-> [-168, -218, 1000, 898]\n\n    Dictionary keys are\n\n      StartFontMetrics, FontName, FullName, FamilyName, Weight,\n      ItalicAngle, IsFixedPitch, FontBBox, UnderlinePosition,\n      UnderlineThickness, Version, Notice, EncodingScheme, CapHeight,\n      XHeight, Ascender, Descender, StartCharMetrics\n\n    ")
    
    # Assigning a Dict to a Name (line 124):
    
    # Assigning a Dict to a Name (line 124):
    
    # Obtaining an instance of the builtin type 'dict' (line 124)
    dict_143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 23), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 124)
    # Adding element type (key, value) (line 124)
    str_144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 8), 'str', 'StartFontMetrics')
    # Getting the type of '_to_float' (line 125)
    _to_float_145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 29), '_to_float')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 23), dict_143, (str_144, _to_float_145))
    # Adding element type (key, value) (line 124)
    str_146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 8), 'str', 'FontName')
    # Getting the type of '_to_str' (line 126)
    _to_str_147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 21), '_to_str')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 23), dict_143, (str_146, _to_str_147))
    # Adding element type (key, value) (line 124)
    str_148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 8), 'str', 'FullName')
    # Getting the type of '_to_str' (line 127)
    _to_str_149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 21), '_to_str')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 23), dict_143, (str_148, _to_str_149))
    # Adding element type (key, value) (line 124)
    str_150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 8), 'str', 'FamilyName')
    # Getting the type of '_to_str' (line 128)
    _to_str_151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 23), '_to_str')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 23), dict_143, (str_150, _to_str_151))
    # Adding element type (key, value) (line 124)
    str_152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 8), 'str', 'Weight')
    # Getting the type of '_to_str' (line 129)
    _to_str_153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 19), '_to_str')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 23), dict_143, (str_152, _to_str_153))
    # Adding element type (key, value) (line 124)
    str_154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 8), 'str', 'ItalicAngle')
    # Getting the type of '_to_float' (line 130)
    _to_float_155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 24), '_to_float')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 23), dict_143, (str_154, _to_float_155))
    # Adding element type (key, value) (line 124)
    str_156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 8), 'str', 'IsFixedPitch')
    # Getting the type of '_to_bool' (line 131)
    _to_bool_157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 25), '_to_bool')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 23), dict_143, (str_156, _to_bool_157))
    # Adding element type (key, value) (line 124)
    str_158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 8), 'str', 'FontBBox')
    # Getting the type of '_to_list_of_ints' (line 132)
    _to_list_of_ints_159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 21), '_to_list_of_ints')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 23), dict_143, (str_158, _to_list_of_ints_159))
    # Adding element type (key, value) (line 124)
    str_160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 8), 'str', 'UnderlinePosition')
    # Getting the type of '_to_int' (line 133)
    _to_int_161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 30), '_to_int')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 23), dict_143, (str_160, _to_int_161))
    # Adding element type (key, value) (line 124)
    str_162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 8), 'str', 'UnderlineThickness')
    # Getting the type of '_to_int' (line 134)
    _to_int_163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 31), '_to_int')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 23), dict_143, (str_162, _to_int_163))
    # Adding element type (key, value) (line 124)
    str_164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 8), 'str', 'Version')
    # Getting the type of '_to_str' (line 135)
    _to_str_165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 20), '_to_str')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 23), dict_143, (str_164, _to_str_165))
    # Adding element type (key, value) (line 124)
    str_166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 8), 'str', 'Notice')
    # Getting the type of '_to_str' (line 136)
    _to_str_167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 19), '_to_str')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 23), dict_143, (str_166, _to_str_167))
    # Adding element type (key, value) (line 124)
    str_168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 8), 'str', 'EncodingScheme')
    # Getting the type of '_to_str' (line 137)
    _to_str_169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 27), '_to_str')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 23), dict_143, (str_168, _to_str_169))
    # Adding element type (key, value) (line 124)
    str_170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 8), 'str', 'CapHeight')
    # Getting the type of '_to_float' (line 138)
    _to_float_171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 22), '_to_float')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 23), dict_143, (str_170, _to_float_171))
    # Adding element type (key, value) (line 124)
    str_172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 8), 'str', 'Capheight')
    # Getting the type of '_to_float' (line 139)
    _to_float_173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 22), '_to_float')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 23), dict_143, (str_172, _to_float_173))
    # Adding element type (key, value) (line 124)
    str_174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 8), 'str', 'XHeight')
    # Getting the type of '_to_float' (line 140)
    _to_float_175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 20), '_to_float')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 23), dict_143, (str_174, _to_float_175))
    # Adding element type (key, value) (line 124)
    str_176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 8), 'str', 'Ascender')
    # Getting the type of '_to_float' (line 141)
    _to_float_177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 21), '_to_float')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 23), dict_143, (str_176, _to_float_177))
    # Adding element type (key, value) (line 124)
    str_178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 8), 'str', 'Descender')
    # Getting the type of '_to_float' (line 142)
    _to_float_179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 22), '_to_float')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 23), dict_143, (str_178, _to_float_179))
    # Adding element type (key, value) (line 124)
    str_180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 8), 'str', 'StdHW')
    # Getting the type of '_to_float' (line 143)
    _to_float_181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 18), '_to_float')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 23), dict_143, (str_180, _to_float_181))
    # Adding element type (key, value) (line 124)
    str_182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 8), 'str', 'StdVW')
    # Getting the type of '_to_float' (line 144)
    _to_float_183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 18), '_to_float')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 23), dict_143, (str_182, _to_float_183))
    # Adding element type (key, value) (line 124)
    str_184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 8), 'str', 'StartCharMetrics')
    # Getting the type of '_to_int' (line 145)
    _to_int_185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 29), '_to_int')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 23), dict_143, (str_184, _to_int_185))
    # Adding element type (key, value) (line 124)
    str_186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 8), 'str', 'CharacterSet')
    # Getting the type of '_to_str' (line 146)
    _to_str_187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 25), '_to_str')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 23), dict_143, (str_186, _to_str_187))
    # Adding element type (key, value) (line 124)
    str_188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 8), 'str', 'Characters')
    # Getting the type of '_to_int' (line 147)
    _to_int_189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 23), '_to_int')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 23), dict_143, (str_188, _to_int_189))
    
    # Assigning a type to the variable 'headerConverters' (line 124)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 4), 'headerConverters', dict_143)
    
    # Assigning a Dict to a Name (line 150):
    
    # Assigning a Dict to a Name (line 150):
    
    # Obtaining an instance of the builtin type 'dict' (line 150)
    dict_190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 8), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 150)
    
    # Assigning a type to the variable 'd' (line 150)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 4), 'd', dict_190)
    
    # Getting the type of 'fh' (line 151)
    fh_191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 16), 'fh')
    # Testing the type of a for loop iterable (line 151)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 151, 4), fh_191)
    # Getting the type of the for loop variable (line 151)
    for_loop_var_192 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 151, 4), fh_191)
    # Assigning a type to the variable 'line' (line 151)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 4), 'line', for_loop_var_192)
    # SSA begins for a for statement (line 151)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 152):
    
    # Assigning a Call to a Name (line 152):
    
    # Call to rstrip(...): (line 152)
    # Processing the call keyword arguments (line 152)
    kwargs_195 = {}
    # Getting the type of 'line' (line 152)
    line_193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 15), 'line', False)
    # Obtaining the member 'rstrip' of a type (line 152)
    rstrip_194 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 15), line_193, 'rstrip')
    # Calling rstrip(args, kwargs) (line 152)
    rstrip_call_result_196 = invoke(stypy.reporting.localization.Localization(__file__, 152, 15), rstrip_194, *[], **kwargs_195)
    
    # Assigning a type to the variable 'line' (line 152)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 8), 'line', rstrip_call_result_196)
    
    
    # Call to startswith(...): (line 153)
    # Processing the call arguments (line 153)
    str_199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 27), 'str', 'Comment')
    # Processing the call keyword arguments (line 153)
    kwargs_200 = {}
    # Getting the type of 'line' (line 153)
    line_197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 11), 'line', False)
    # Obtaining the member 'startswith' of a type (line 153)
    startswith_198 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 11), line_197, 'startswith')
    # Calling startswith(args, kwargs) (line 153)
    startswith_call_result_201 = invoke(stypy.reporting.localization.Localization(__file__, 153, 11), startswith_198, *[str_199], **kwargs_200)
    
    # Testing the type of an if condition (line 153)
    if_condition_202 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 153, 8), startswith_call_result_201)
    # Assigning a type to the variable 'if_condition_202' (line 153)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 8), 'if_condition_202', if_condition_202)
    # SSA begins for if statement (line 153)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 153)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 155):
    
    # Assigning a Call to a Name (line 155):
    
    # Call to split(...): (line 155)
    # Processing the call arguments (line 155)
    str_205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 25), 'str', ' ')
    int_206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 31), 'int')
    # Processing the call keyword arguments (line 155)
    kwargs_207 = {}
    # Getting the type of 'line' (line 155)
    line_203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 14), 'line', False)
    # Obtaining the member 'split' of a type (line 155)
    split_204 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 14), line_203, 'split')
    # Calling split(args, kwargs) (line 155)
    split_call_result_208 = invoke(stypy.reporting.localization.Localization(__file__, 155, 14), split_204, *[str_205, int_206], **kwargs_207)
    
    # Assigning a type to the variable 'lst' (line 155)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 8), 'lst', split_call_result_208)
    
    # Assigning a Subscript to a Name (line 157):
    
    # Assigning a Subscript to a Name (line 157):
    
    # Obtaining the type of the subscript
    int_209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 18), 'int')
    # Getting the type of 'lst' (line 157)
    lst_210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 14), 'lst')
    # Obtaining the member '__getitem__' of a type (line 157)
    getitem___211 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 14), lst_210, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 157)
    subscript_call_result_212 = invoke(stypy.reporting.localization.Localization(__file__, 157, 14), getitem___211, int_209)
    
    # Assigning a type to the variable 'key' (line 157)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 8), 'key', subscript_call_result_212)
    
    
    
    # Call to len(...): (line 158)
    # Processing the call arguments (line 158)
    # Getting the type of 'lst' (line 158)
    lst_214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 15), 'lst', False)
    # Processing the call keyword arguments (line 158)
    kwargs_215 = {}
    # Getting the type of 'len' (line 158)
    len_213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 11), 'len', False)
    # Calling len(args, kwargs) (line 158)
    len_call_result_216 = invoke(stypy.reporting.localization.Localization(__file__, 158, 11), len_213, *[lst_214], **kwargs_215)
    
    int_217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 23), 'int')
    # Applying the binary operator '==' (line 158)
    result_eq_218 = python_operator(stypy.reporting.localization.Localization(__file__, 158, 11), '==', len_call_result_216, int_217)
    
    # Testing the type of an if condition (line 158)
    if_condition_219 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 158, 8), result_eq_218)
    # Assigning a type to the variable 'if_condition_219' (line 158)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'if_condition_219', if_condition_219)
    # SSA begins for if statement (line 158)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 159):
    
    # Assigning a Subscript to a Name (line 159):
    
    # Obtaining the type of the subscript
    int_220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 22), 'int')
    # Getting the type of 'lst' (line 159)
    lst_221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 18), 'lst')
    # Obtaining the member '__getitem__' of a type (line 159)
    getitem___222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 18), lst_221, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 159)
    subscript_call_result_223 = invoke(stypy.reporting.localization.Localization(__file__, 159, 18), getitem___222, int_220)
    
    # Assigning a type to the variable 'val' (line 159)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 12), 'val', subscript_call_result_223)
    # SSA branch for the else part of an if statement (line 158)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Str to a Name (line 161):
    
    # Assigning a Str to a Name (line 161):
    str_224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 18), 'str', '')
    # Assigning a type to the variable 'val' (line 161)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 12), 'val', str_224)
    # SSA join for if statement (line 158)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # SSA begins for try-except statement (line 163)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Subscript (line 164):
    
    # Assigning a Call to a Subscript (line 164):
    
    # Call to (...): (line 164)
    # Processing the call arguments (line 164)
    # Getting the type of 'val' (line 164)
    val_229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 43), 'val', False)
    # Processing the call keyword arguments (line 164)
    kwargs_230 = {}
    
    # Obtaining the type of the subscript
    # Getting the type of 'key' (line 164)
    key_225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 38), 'key', False)
    # Getting the type of 'headerConverters' (line 164)
    headerConverters_226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 21), 'headerConverters', False)
    # Obtaining the member '__getitem__' of a type (line 164)
    getitem___227 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 21), headerConverters_226, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 164)
    subscript_call_result_228 = invoke(stypy.reporting.localization.Localization(__file__, 164, 21), getitem___227, key_225)
    
    # Calling (args, kwargs) (line 164)
    _call_result_231 = invoke(stypy.reporting.localization.Localization(__file__, 164, 21), subscript_call_result_228, *[val_229], **kwargs_230)
    
    # Getting the type of 'd' (line 164)
    d_232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 12), 'd')
    # Getting the type of 'key' (line 164)
    key_233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 14), 'key')
    # Storing an element on a container (line 164)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 12), d_232, (key_233, _call_result_231))
    # SSA branch for the except part of a try statement (line 163)
    # SSA branch for the except 'ValueError' branch of a try statement (line 163)
    module_type_store.open_ssa_branch('except')
    
    # Call to print(...): (line 166)
    # Processing the call arguments (line 166)
    unicode_235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 18), 'unicode', u'Value error parsing header in AFM:')
    # Getting the type of 'key' (line 167)
    key_236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 18), 'key', False)
    # Getting the type of 'val' (line 167)
    val_237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 23), 'val', False)
    # Processing the call keyword arguments (line 166)
    # Getting the type of 'sys' (line 167)
    sys_238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 33), 'sys', False)
    # Obtaining the member 'stderr' of a type (line 167)
    stderr_239 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 33), sys_238, 'stderr')
    keyword_240 = stderr_239
    kwargs_241 = {'file': keyword_240}
    # Getting the type of 'print' (line 166)
    print_234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 12), 'print', False)
    # Calling print(args, kwargs) (line 166)
    print_call_result_242 = invoke(stypy.reporting.localization.Localization(__file__, 166, 12), print_234, *[unicode_235, key_236, val_237], **kwargs_241)
    
    # SSA branch for the except 'KeyError' branch of a try statement (line 163)
    module_type_store.open_ssa_branch('except')
    
    # Call to print(...): (line 170)
    # Processing the call arguments (line 170)
    unicode_244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 18), 'unicode', u'Found an unknown keyword in AFM header (was %r)')
    # Getting the type of 'key' (line 170)
    key_245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 70), 'key', False)
    # Applying the binary operator '%' (line 170)
    result_mod_246 = python_operator(stypy.reporting.localization.Localization(__file__, 170, 18), '%', unicode_244, key_245)
    
    # Processing the call keyword arguments (line 170)
    # Getting the type of 'sys' (line 171)
    sys_247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 23), 'sys', False)
    # Obtaining the member 'stderr' of a type (line 171)
    stderr_248 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 23), sys_247, 'stderr')
    keyword_249 = stderr_248
    kwargs_250 = {'file': keyword_249}
    # Getting the type of 'print' (line 170)
    print_243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 12), 'print', False)
    # Calling print(args, kwargs) (line 170)
    print_call_result_251 = invoke(stypy.reporting.localization.Localization(__file__, 170, 12), print_243, *[result_mod_246], **kwargs_250)
    
    # SSA join for try-except statement (line 163)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'key' (line 173)
    key_252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 11), 'key')
    str_253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 18), 'str', 'StartCharMetrics')
    # Applying the binary operator '==' (line 173)
    result_eq_254 = python_operator(stypy.reporting.localization.Localization(__file__, 173, 11), '==', key_252, str_253)
    
    # Testing the type of an if condition (line 173)
    if_condition_255 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 173, 8), result_eq_254)
    # Assigning a type to the variable 'if_condition_255' (line 173)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 8), 'if_condition_255', if_condition_255)
    # SSA begins for if statement (line 173)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'd' (line 174)
    d_256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 19), 'd')
    # Assigning a type to the variable 'stypy_return_type' (line 174)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 12), 'stypy_return_type', d_256)
    # SSA join for if statement (line 173)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to RuntimeError(...): (line 175)
    # Processing the call arguments (line 175)
    unicode_258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 23), 'unicode', u'Bad parse')
    # Processing the call keyword arguments (line 175)
    kwargs_259 = {}
    # Getting the type of 'RuntimeError' (line 175)
    RuntimeError_257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 10), 'RuntimeError', False)
    # Calling RuntimeError(args, kwargs) (line 175)
    RuntimeError_call_result_260 = invoke(stypy.reporting.localization.Localization(__file__, 175, 10), RuntimeError_257, *[unicode_258], **kwargs_259)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 175, 4), RuntimeError_call_result_260, 'raise parameter', BaseException)
    
    # ################# End of '_parse_header(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_parse_header' in the type store
    # Getting the type of 'stypy_return_type' (line 106)
    stypy_return_type_261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_261)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_parse_header'
    return stypy_return_type_261

# Assigning a type to the variable '_parse_header' (line 106)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 0), '_parse_header', _parse_header)

@norecursion
def _parse_char_metrics(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_parse_char_metrics'
    module_type_store = module_type_store.open_function_context('_parse_char_metrics', 178, 0, False)
    
    # Passed parameters checking function
    _parse_char_metrics.stypy_localization = localization
    _parse_char_metrics.stypy_type_of_self = None
    _parse_char_metrics.stypy_type_store = module_type_store
    _parse_char_metrics.stypy_function_name = '_parse_char_metrics'
    _parse_char_metrics.stypy_param_names_list = ['fh']
    _parse_char_metrics.stypy_varargs_param_name = None
    _parse_char_metrics.stypy_kwargs_param_name = None
    _parse_char_metrics.stypy_call_defaults = defaults
    _parse_char_metrics.stypy_call_varargs = varargs
    _parse_char_metrics.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_parse_char_metrics', ['fh'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_parse_char_metrics', localization, ['fh'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_parse_char_metrics(...)' code ##################

    unicode_262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, (-1)), 'unicode', u'\n    Return a character metric dictionary.  Keys are the ASCII num of\n    the character, values are a (*wx*, *name*, *bbox*) tuple, where\n    *wx* is the character width, *name* is the postscript language\n    name, and *bbox* is a (*llx*, *lly*, *urx*, *ury*) tuple.\n\n    This function is incomplete per the standard, but thus far parses\n    all the sample afm files tried.\n    ')
    
    # Assigning a Dict to a Name (line 189):
    
    # Assigning a Dict to a Name (line 189):
    
    # Obtaining an instance of the builtin type 'dict' (line 189)
    dict_263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 14), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 189)
    
    # Assigning a type to the variable 'ascii_d' (line 189)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 4), 'ascii_d', dict_263)
    
    # Assigning a Dict to a Name (line 190):
    
    # Assigning a Dict to a Name (line 190):
    
    # Obtaining an instance of the builtin type 'dict' (line 190)
    dict_264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 13), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 190)
    
    # Assigning a type to the variable 'name_d' (line 190)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 4), 'name_d', dict_264)
    
    # Getting the type of 'fh' (line 191)
    fh_265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 16), 'fh')
    # Testing the type of a for loop iterable (line 191)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 191, 4), fh_265)
    # Getting the type of the for loop variable (line 191)
    for_loop_var_266 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 191, 4), fh_265)
    # Assigning a type to the variable 'line' (line 191)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 4), 'line', for_loop_var_266)
    # SSA begins for a for statement (line 191)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 194):
    
    # Assigning a Call to a Name (line 194):
    
    # Call to _to_str(...): (line 194)
    # Processing the call arguments (line 194)
    
    # Call to rstrip(...): (line 194)
    # Processing the call keyword arguments (line 194)
    kwargs_270 = {}
    # Getting the type of 'line' (line 194)
    line_268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 23), 'line', False)
    # Obtaining the member 'rstrip' of a type (line 194)
    rstrip_269 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 23), line_268, 'rstrip')
    # Calling rstrip(args, kwargs) (line 194)
    rstrip_call_result_271 = invoke(stypy.reporting.localization.Localization(__file__, 194, 23), rstrip_269, *[], **kwargs_270)
    
    # Processing the call keyword arguments (line 194)
    kwargs_272 = {}
    # Getting the type of '_to_str' (line 194)
    _to_str_267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 15), '_to_str', False)
    # Calling _to_str(args, kwargs) (line 194)
    _to_str_call_result_273 = invoke(stypy.reporting.localization.Localization(__file__, 194, 15), _to_str_267, *[rstrip_call_result_271], **kwargs_272)
    
    # Assigning a type to the variable 'line' (line 194)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 8), 'line', _to_str_call_result_273)
    
    
    # Call to startswith(...): (line 195)
    # Processing the call arguments (line 195)
    unicode_276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 27), 'unicode', u'EndCharMetrics')
    # Processing the call keyword arguments (line 195)
    kwargs_277 = {}
    # Getting the type of 'line' (line 195)
    line_274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 11), 'line', False)
    # Obtaining the member 'startswith' of a type (line 195)
    startswith_275 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 11), line_274, 'startswith')
    # Calling startswith(args, kwargs) (line 195)
    startswith_call_result_278 = invoke(stypy.reporting.localization.Localization(__file__, 195, 11), startswith_275, *[unicode_276], **kwargs_277)
    
    # Testing the type of an if condition (line 195)
    if_condition_279 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 195, 8), startswith_call_result_278)
    # Assigning a type to the variable 'if_condition_279' (line 195)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 8), 'if_condition_279', if_condition_279)
    # SSA begins for if statement (line 195)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'tuple' (line 196)
    tuple_280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 19), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 196)
    # Adding element type (line 196)
    # Getting the type of 'ascii_d' (line 196)
    ascii_d_281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 19), 'ascii_d')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 196, 19), tuple_280, ascii_d_281)
    # Adding element type (line 196)
    # Getting the type of 'name_d' (line 196)
    name_d_282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 28), 'name_d')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 196, 19), tuple_280, name_d_282)
    
    # Assigning a type to the variable 'stypy_return_type' (line 196)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 12), 'stypy_return_type', tuple_280)
    # SSA join for if statement (line 195)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 198):
    
    # Assigning a Call to a Name (line 198):
    
    # Call to dict(...): (line 198)
    # Processing the call arguments (line 198)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 198, 20, True)
    # Calculating comprehension expression
    
    # Call to split(...): (line 198)
    # Processing the call arguments (line 198)
    unicode_296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 64), 'unicode', u';')
    # Processing the call keyword arguments (line 198)
    kwargs_297 = {}
    # Getting the type of 'line' (line 198)
    line_294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 53), 'line', False)
    # Obtaining the member 'split' of a type (line 198)
    split_295 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 53), line_294, 'split')
    # Calling split(args, kwargs) (line 198)
    split_call_result_298 = invoke(stypy.reporting.localization.Localization(__file__, 198, 53), split_295, *[unicode_296], **kwargs_297)
    
    comprehension_299 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 198, 20), split_call_result_298)
    # Assigning a type to the variable 's' (line 198)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 20), 's', comprehension_299)
    # Getting the type of 's' (line 198)
    s_293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 72), 's', False)
    
    # Call to split(...): (line 198)
    # Processing the call arguments (line 198)
    unicode_289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 36), 'unicode', u' ')
    int_290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 41), 'int')
    # Processing the call keyword arguments (line 198)
    kwargs_291 = {}
    
    # Call to strip(...): (line 198)
    # Processing the call keyword arguments (line 198)
    kwargs_286 = {}
    # Getting the type of 's' (line 198)
    s_284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 20), 's', False)
    # Obtaining the member 'strip' of a type (line 198)
    strip_285 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 20), s_284, 'strip')
    # Calling strip(args, kwargs) (line 198)
    strip_call_result_287 = invoke(stypy.reporting.localization.Localization(__file__, 198, 20), strip_285, *[], **kwargs_286)
    
    # Obtaining the member 'split' of a type (line 198)
    split_288 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 20), strip_call_result_287, 'split')
    # Calling split(args, kwargs) (line 198)
    split_call_result_292 = invoke(stypy.reporting.localization.Localization(__file__, 198, 20), split_288, *[unicode_289, int_290], **kwargs_291)
    
    list_300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 20), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 198, 20), list_300, split_call_result_292)
    # Processing the call keyword arguments (line 198)
    kwargs_301 = {}
    # Getting the type of 'dict' (line 198)
    dict_283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 15), 'dict', False)
    # Calling dict(args, kwargs) (line 198)
    dict_call_result_302 = invoke(stypy.reporting.localization.Localization(__file__, 198, 15), dict_283, *[list_300], **kwargs_301)
    
    # Assigning a type to the variable 'vals' (line 198)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 8), 'vals', dict_call_result_302)
    
    
    
    # Call to issubset(...): (line 200)
    # Processing the call arguments (line 200)
    # Getting the type of 'vals' (line 200)
    vals_309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 46), 'vals', False)
    # Processing the call keyword arguments (line 200)
    kwargs_310 = {}
    
    # Obtaining an instance of the builtin type 'set' (line 200)
    set_303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 15), 'set')
    # Adding type elements to the builtin type 'set' instance (line 200)
    # Adding element type (line 200)
    unicode_304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 16), 'unicode', u'C')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 15), set_303, unicode_304)
    # Adding element type (line 200)
    unicode_305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 21), 'unicode', u'WX')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 15), set_303, unicode_305)
    # Adding element type (line 200)
    unicode_306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 27), 'unicode', u'N')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 15), set_303, unicode_306)
    # Adding element type (line 200)
    unicode_307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 32), 'unicode', u'B')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 15), set_303, unicode_307)
    
    # Obtaining the member 'issubset' of a type (line 200)
    issubset_308 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 15), set_303, 'issubset')
    # Calling issubset(args, kwargs) (line 200)
    issubset_call_result_311 = invoke(stypy.reporting.localization.Localization(__file__, 200, 15), issubset_308, *[vals_309], **kwargs_310)
    
    # Applying the 'not' unary operator (line 200)
    result_not__312 = python_operator(stypy.reporting.localization.Localization(__file__, 200, 11), 'not', issubset_call_result_311)
    
    # Testing the type of an if condition (line 200)
    if_condition_313 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 200, 8), result_not__312)
    # Assigning a type to the variable 'if_condition_313' (line 200)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 8), 'if_condition_313', if_condition_313)
    # SSA begins for if statement (line 200)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to RuntimeError(...): (line 201)
    # Processing the call arguments (line 201)
    unicode_315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 31), 'unicode', u'Bad char metrics line: %s')
    # Getting the type of 'line' (line 201)
    line_316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 61), 'line', False)
    # Applying the binary operator '%' (line 201)
    result_mod_317 = python_operator(stypy.reporting.localization.Localization(__file__, 201, 31), '%', unicode_315, line_316)
    
    # Processing the call keyword arguments (line 201)
    kwargs_318 = {}
    # Getting the type of 'RuntimeError' (line 201)
    RuntimeError_314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 18), 'RuntimeError', False)
    # Calling RuntimeError(args, kwargs) (line 201)
    RuntimeError_call_result_319 = invoke(stypy.reporting.localization.Localization(__file__, 201, 18), RuntimeError_314, *[result_mod_317], **kwargs_318)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 201, 12), RuntimeError_call_result_319, 'raise parameter', BaseException)
    # SSA join for if statement (line 200)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 202):
    
    # Assigning a Call to a Name (line 202):
    
    # Call to _to_int(...): (line 202)
    # Processing the call arguments (line 202)
    
    # Obtaining the type of the subscript
    unicode_321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 27), 'unicode', u'C')
    # Getting the type of 'vals' (line 202)
    vals_322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 22), 'vals', False)
    # Obtaining the member '__getitem__' of a type (line 202)
    getitem___323 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 22), vals_322, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 202)
    subscript_call_result_324 = invoke(stypy.reporting.localization.Localization(__file__, 202, 22), getitem___323, unicode_321)
    
    # Processing the call keyword arguments (line 202)
    kwargs_325 = {}
    # Getting the type of '_to_int' (line 202)
    _to_int_320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 14), '_to_int', False)
    # Calling _to_int(args, kwargs) (line 202)
    _to_int_call_result_326 = invoke(stypy.reporting.localization.Localization(__file__, 202, 14), _to_int_320, *[subscript_call_result_324], **kwargs_325)
    
    # Assigning a type to the variable 'num' (line 202)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 8), 'num', _to_int_call_result_326)
    
    # Assigning a Call to a Name (line 203):
    
    # Assigning a Call to a Name (line 203):
    
    # Call to _to_float(...): (line 203)
    # Processing the call arguments (line 203)
    
    # Obtaining the type of the subscript
    unicode_328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 28), 'unicode', u'WX')
    # Getting the type of 'vals' (line 203)
    vals_329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 23), 'vals', False)
    # Obtaining the member '__getitem__' of a type (line 203)
    getitem___330 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 23), vals_329, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 203)
    subscript_call_result_331 = invoke(stypy.reporting.localization.Localization(__file__, 203, 23), getitem___330, unicode_328)
    
    # Processing the call keyword arguments (line 203)
    kwargs_332 = {}
    # Getting the type of '_to_float' (line 203)
    _to_float_327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 13), '_to_float', False)
    # Calling _to_float(args, kwargs) (line 203)
    _to_float_call_result_333 = invoke(stypy.reporting.localization.Localization(__file__, 203, 13), _to_float_327, *[subscript_call_result_331], **kwargs_332)
    
    # Assigning a type to the variable 'wx' (line 203)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 8), 'wx', _to_float_call_result_333)
    
    # Assigning a Subscript to a Name (line 204):
    
    # Assigning a Subscript to a Name (line 204):
    
    # Obtaining the type of the subscript
    unicode_334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 20), 'unicode', u'N')
    # Getting the type of 'vals' (line 204)
    vals_335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 15), 'vals')
    # Obtaining the member '__getitem__' of a type (line 204)
    getitem___336 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 15), vals_335, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 204)
    subscript_call_result_337 = invoke(stypy.reporting.localization.Localization(__file__, 204, 15), getitem___336, unicode_334)
    
    # Assigning a type to the variable 'name' (line 204)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 8), 'name', subscript_call_result_337)
    
    # Assigning a Call to a Name (line 205):
    
    # Assigning a Call to a Name (line 205):
    
    # Call to _to_list_of_floats(...): (line 205)
    # Processing the call arguments (line 205)
    
    # Obtaining the type of the subscript
    unicode_339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 39), 'unicode', u'B')
    # Getting the type of 'vals' (line 205)
    vals_340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 34), 'vals', False)
    # Obtaining the member '__getitem__' of a type (line 205)
    getitem___341 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 34), vals_340, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 205)
    subscript_call_result_342 = invoke(stypy.reporting.localization.Localization(__file__, 205, 34), getitem___341, unicode_339)
    
    # Processing the call keyword arguments (line 205)
    kwargs_343 = {}
    # Getting the type of '_to_list_of_floats' (line 205)
    _to_list_of_floats_338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 15), '_to_list_of_floats', False)
    # Calling _to_list_of_floats(args, kwargs) (line 205)
    _to_list_of_floats_call_result_344 = invoke(stypy.reporting.localization.Localization(__file__, 205, 15), _to_list_of_floats_338, *[subscript_call_result_342], **kwargs_343)
    
    # Assigning a type to the variable 'bbox' (line 205)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 8), 'bbox', _to_list_of_floats_call_result_344)
    
    # Assigning a Call to a Name (line 206):
    
    # Assigning a Call to a Name (line 206):
    
    # Call to list(...): (line 206)
    # Processing the call arguments (line 206)
    
    # Call to map(...): (line 206)
    # Processing the call arguments (line 206)
    # Getting the type of 'int' (line 206)
    int_347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 24), 'int', False)
    # Getting the type of 'bbox' (line 206)
    bbox_348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 29), 'bbox', False)
    # Processing the call keyword arguments (line 206)
    kwargs_349 = {}
    # Getting the type of 'map' (line 206)
    map_346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 20), 'map', False)
    # Calling map(args, kwargs) (line 206)
    map_call_result_350 = invoke(stypy.reporting.localization.Localization(__file__, 206, 20), map_346, *[int_347, bbox_348], **kwargs_349)
    
    # Processing the call keyword arguments (line 206)
    kwargs_351 = {}
    # Getting the type of 'list' (line 206)
    list_345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 15), 'list', False)
    # Calling list(args, kwargs) (line 206)
    list_call_result_352 = invoke(stypy.reporting.localization.Localization(__file__, 206, 15), list_345, *[map_call_result_350], **kwargs_351)
    
    # Assigning a type to the variable 'bbox' (line 206)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 8), 'bbox', list_call_result_352)
    
    
    # Getting the type of 'name' (line 210)
    name_353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 11), 'name')
    unicode_354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 19), 'unicode', u'Euro')
    # Applying the binary operator '==' (line 210)
    result_eq_355 = python_operator(stypy.reporting.localization.Localization(__file__, 210, 11), '==', name_353, unicode_354)
    
    # Testing the type of an if condition (line 210)
    if_condition_356 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 210, 8), result_eq_355)
    # Assigning a type to the variable 'if_condition_356' (line 210)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 8), 'if_condition_356', if_condition_356)
    # SSA begins for if statement (line 210)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 211):
    
    # Assigning a Num to a Name (line 211):
    int_357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 18), 'int')
    # Assigning a type to the variable 'num' (line 211)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 12), 'num', int_357)
    # SSA join for if statement (line 210)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'num' (line 212)
    num_358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 11), 'num')
    int_359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 18), 'int')
    # Applying the binary operator '!=' (line 212)
    result_ne_360 = python_operator(stypy.reporting.localization.Localization(__file__, 212, 11), '!=', num_358, int_359)
    
    # Testing the type of an if condition (line 212)
    if_condition_361 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 212, 8), result_ne_360)
    # Assigning a type to the variable 'if_condition_361' (line 212)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 8), 'if_condition_361', if_condition_361)
    # SSA begins for if statement (line 212)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Tuple to a Subscript (line 213):
    
    # Assigning a Tuple to a Subscript (line 213):
    
    # Obtaining an instance of the builtin type 'tuple' (line 213)
    tuple_362 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 28), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 213)
    # Adding element type (line 213)
    # Getting the type of 'wx' (line 213)
    wx_363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 28), 'wx')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 213, 28), tuple_362, wx_363)
    # Adding element type (line 213)
    # Getting the type of 'name' (line 213)
    name_364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 32), 'name')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 213, 28), tuple_362, name_364)
    # Adding element type (line 213)
    # Getting the type of 'bbox' (line 213)
    bbox_365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 38), 'bbox')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 213, 28), tuple_362, bbox_365)
    
    # Getting the type of 'ascii_d' (line 213)
    ascii_d_366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 12), 'ascii_d')
    # Getting the type of 'num' (line 213)
    num_367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 20), 'num')
    # Storing an element on a container (line 213)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 213, 12), ascii_d_366, (num_367, tuple_362))
    # SSA join for if statement (line 212)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Tuple to a Subscript (line 214):
    
    # Assigning a Tuple to a Subscript (line 214):
    
    # Obtaining an instance of the builtin type 'tuple' (line 214)
    tuple_368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 24), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 214)
    # Adding element type (line 214)
    # Getting the type of 'wx' (line 214)
    wx_369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 24), 'wx')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 214, 24), tuple_368, wx_369)
    # Adding element type (line 214)
    # Getting the type of 'bbox' (line 214)
    bbox_370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 28), 'bbox')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 214, 24), tuple_368, bbox_370)
    
    # Getting the type of 'name_d' (line 214)
    name_d_371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 8), 'name_d')
    # Getting the type of 'name' (line 214)
    name_372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 15), 'name')
    # Storing an element on a container (line 214)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 214, 8), name_d_371, (name_372, tuple_368))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to RuntimeError(...): (line 215)
    # Processing the call arguments (line 215)
    unicode_374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 23), 'unicode', u'Bad parse')
    # Processing the call keyword arguments (line 215)
    kwargs_375 = {}
    # Getting the type of 'RuntimeError' (line 215)
    RuntimeError_373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 10), 'RuntimeError', False)
    # Calling RuntimeError(args, kwargs) (line 215)
    RuntimeError_call_result_376 = invoke(stypy.reporting.localization.Localization(__file__, 215, 10), RuntimeError_373, *[unicode_374], **kwargs_375)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 215, 4), RuntimeError_call_result_376, 'raise parameter', BaseException)
    
    # ################# End of '_parse_char_metrics(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_parse_char_metrics' in the type store
    # Getting the type of 'stypy_return_type' (line 178)
    stypy_return_type_377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_377)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_parse_char_metrics'
    return stypy_return_type_377

# Assigning a type to the variable '_parse_char_metrics' (line 178)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 0), '_parse_char_metrics', _parse_char_metrics)

@norecursion
def _parse_kern_pairs(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_parse_kern_pairs'
    module_type_store = module_type_store.open_function_context('_parse_kern_pairs', 218, 0, False)
    
    # Passed parameters checking function
    _parse_kern_pairs.stypy_localization = localization
    _parse_kern_pairs.stypy_type_of_self = None
    _parse_kern_pairs.stypy_type_store = module_type_store
    _parse_kern_pairs.stypy_function_name = '_parse_kern_pairs'
    _parse_kern_pairs.stypy_param_names_list = ['fh']
    _parse_kern_pairs.stypy_varargs_param_name = None
    _parse_kern_pairs.stypy_kwargs_param_name = None
    _parse_kern_pairs.stypy_call_defaults = defaults
    _parse_kern_pairs.stypy_call_varargs = varargs
    _parse_kern_pairs.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_parse_kern_pairs', ['fh'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_parse_kern_pairs', localization, ['fh'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_parse_kern_pairs(...)' code ##################

    unicode_378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, (-1)), 'unicode', u"\n    Return a kern pairs dictionary; keys are (*char1*, *char2*) tuples and\n    values are the kern pair value.  For example, a kern pairs line like\n    ``KPX A y -50``\n\n    will be represented as::\n\n      d[ ('A', 'y') ] = -50\n\n    ")
    
    # Assigning a Call to a Name (line 230):
    
    # Assigning a Call to a Name (line 230):
    
    # Call to next(...): (line 230)
    # Processing the call arguments (line 230)
    # Getting the type of 'fh' (line 230)
    fh_380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 16), 'fh', False)
    # Processing the call keyword arguments (line 230)
    kwargs_381 = {}
    # Getting the type of 'next' (line 230)
    next_379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 11), 'next', False)
    # Calling next(args, kwargs) (line 230)
    next_call_result_382 = invoke(stypy.reporting.localization.Localization(__file__, 230, 11), next_379, *[fh_380], **kwargs_381)
    
    # Assigning a type to the variable 'line' (line 230)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 4), 'line', next_call_result_382)
    
    
    
    # Call to startswith(...): (line 231)
    # Processing the call arguments (line 231)
    str_385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 27), 'str', 'StartKernPairs')
    # Processing the call keyword arguments (line 231)
    kwargs_386 = {}
    # Getting the type of 'line' (line 231)
    line_383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 11), 'line', False)
    # Obtaining the member 'startswith' of a type (line 231)
    startswith_384 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 11), line_383, 'startswith')
    # Calling startswith(args, kwargs) (line 231)
    startswith_call_result_387 = invoke(stypy.reporting.localization.Localization(__file__, 231, 11), startswith_384, *[str_385], **kwargs_386)
    
    # Applying the 'not' unary operator (line 231)
    result_not__388 = python_operator(stypy.reporting.localization.Localization(__file__, 231, 7), 'not', startswith_call_result_387)
    
    # Testing the type of an if condition (line 231)
    if_condition_389 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 231, 4), result_not__388)
    # Assigning a type to the variable 'if_condition_389' (line 231)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 4), 'if_condition_389', if_condition_389)
    # SSA begins for if statement (line 231)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to RuntimeError(...): (line 232)
    # Processing the call arguments (line 232)
    unicode_391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 27), 'unicode', u'Bad start of kern pairs data: %s')
    # Getting the type of 'line' (line 232)
    line_392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 64), 'line', False)
    # Applying the binary operator '%' (line 232)
    result_mod_393 = python_operator(stypy.reporting.localization.Localization(__file__, 232, 27), '%', unicode_391, line_392)
    
    # Processing the call keyword arguments (line 232)
    kwargs_394 = {}
    # Getting the type of 'RuntimeError' (line 232)
    RuntimeError_390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 14), 'RuntimeError', False)
    # Calling RuntimeError(args, kwargs) (line 232)
    RuntimeError_call_result_395 = invoke(stypy.reporting.localization.Localization(__file__, 232, 14), RuntimeError_390, *[result_mod_393], **kwargs_394)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 232, 8), RuntimeError_call_result_395, 'raise parameter', BaseException)
    # SSA join for if statement (line 231)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Dict to a Name (line 234):
    
    # Assigning a Dict to a Name (line 234):
    
    # Obtaining an instance of the builtin type 'dict' (line 234)
    dict_396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 8), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 234)
    
    # Assigning a type to the variable 'd' (line 234)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 4), 'd', dict_396)
    
    # Getting the type of 'fh' (line 235)
    fh_397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 16), 'fh')
    # Testing the type of a for loop iterable (line 235)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 235, 4), fh_397)
    # Getting the type of the for loop variable (line 235)
    for_loop_var_398 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 235, 4), fh_397)
    # Assigning a type to the variable 'line' (line 235)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 4), 'line', for_loop_var_398)
    # SSA begins for a for statement (line 235)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 236):
    
    # Assigning a Call to a Name (line 236):
    
    # Call to rstrip(...): (line 236)
    # Processing the call keyword arguments (line 236)
    kwargs_401 = {}
    # Getting the type of 'line' (line 236)
    line_399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 15), 'line', False)
    # Obtaining the member 'rstrip' of a type (line 236)
    rstrip_400 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 15), line_399, 'rstrip')
    # Calling rstrip(args, kwargs) (line 236)
    rstrip_call_result_402 = invoke(stypy.reporting.localization.Localization(__file__, 236, 15), rstrip_400, *[], **kwargs_401)
    
    # Assigning a type to the variable 'line' (line 236)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 8), 'line', rstrip_call_result_402)
    
    
    # Getting the type of 'line' (line 237)
    line_403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 15), 'line')
    # Applying the 'not' unary operator (line 237)
    result_not__404 = python_operator(stypy.reporting.localization.Localization(__file__, 237, 11), 'not', line_403)
    
    # Testing the type of an if condition (line 237)
    if_condition_405 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 237, 8), result_not__404)
    # Assigning a type to the variable 'if_condition_405' (line 237)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 8), 'if_condition_405', if_condition_405)
    # SSA begins for if statement (line 237)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 237)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to startswith(...): (line 239)
    # Processing the call arguments (line 239)
    str_408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 27), 'str', 'EndKernPairs')
    # Processing the call keyword arguments (line 239)
    kwargs_409 = {}
    # Getting the type of 'line' (line 239)
    line_406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 11), 'line', False)
    # Obtaining the member 'startswith' of a type (line 239)
    startswith_407 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 11), line_406, 'startswith')
    # Calling startswith(args, kwargs) (line 239)
    startswith_call_result_410 = invoke(stypy.reporting.localization.Localization(__file__, 239, 11), startswith_407, *[str_408], **kwargs_409)
    
    # Testing the type of an if condition (line 239)
    if_condition_411 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 239, 8), startswith_call_result_410)
    # Assigning a type to the variable 'if_condition_411' (line 239)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 8), 'if_condition_411', if_condition_411)
    # SSA begins for if statement (line 239)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to next(...): (line 240)
    # Processing the call arguments (line 240)
    # Getting the type of 'fh' (line 240)
    fh_413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 17), 'fh', False)
    # Processing the call keyword arguments (line 240)
    kwargs_414 = {}
    # Getting the type of 'next' (line 240)
    next_412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 12), 'next', False)
    # Calling next(args, kwargs) (line 240)
    next_call_result_415 = invoke(stypy.reporting.localization.Localization(__file__, 240, 12), next_412, *[fh_413], **kwargs_414)
    
    # Getting the type of 'd' (line 241)
    d_416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 19), 'd')
    # Assigning a type to the variable 'stypy_return_type' (line 241)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 12), 'stypy_return_type', d_416)
    # SSA join for if statement (line 239)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 242):
    
    # Assigning a Call to a Name (line 242):
    
    # Call to split(...): (line 242)
    # Processing the call keyword arguments (line 242)
    kwargs_419 = {}
    # Getting the type of 'line' (line 242)
    line_417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 15), 'line', False)
    # Obtaining the member 'split' of a type (line 242)
    split_418 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 15), line_417, 'split')
    # Calling split(args, kwargs) (line 242)
    split_call_result_420 = invoke(stypy.reporting.localization.Localization(__file__, 242, 15), split_418, *[], **kwargs_419)
    
    # Assigning a type to the variable 'vals' (line 242)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 8), 'vals', split_call_result_420)
    
    
    # Evaluating a boolean operation
    
    
    # Call to len(...): (line 243)
    # Processing the call arguments (line 243)
    # Getting the type of 'vals' (line 243)
    vals_422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 15), 'vals', False)
    # Processing the call keyword arguments (line 243)
    kwargs_423 = {}
    # Getting the type of 'len' (line 243)
    len_421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 11), 'len', False)
    # Calling len(args, kwargs) (line 243)
    len_call_result_424 = invoke(stypy.reporting.localization.Localization(__file__, 243, 11), len_421, *[vals_422], **kwargs_423)
    
    int_425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 24), 'int')
    # Applying the binary operator '!=' (line 243)
    result_ne_426 = python_operator(stypy.reporting.localization.Localization(__file__, 243, 11), '!=', len_call_result_424, int_425)
    
    
    
    # Obtaining the type of the subscript
    int_427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 34), 'int')
    # Getting the type of 'vals' (line 243)
    vals_428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 29), 'vals')
    # Obtaining the member '__getitem__' of a type (line 243)
    getitem___429 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 29), vals_428, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 243)
    subscript_call_result_430 = invoke(stypy.reporting.localization.Localization(__file__, 243, 29), getitem___429, int_427)
    
    str_431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 40), 'str', 'KPX')
    # Applying the binary operator '!=' (line 243)
    result_ne_432 = python_operator(stypy.reporting.localization.Localization(__file__, 243, 29), '!=', subscript_call_result_430, str_431)
    
    # Applying the binary operator 'or' (line 243)
    result_or_keyword_433 = python_operator(stypy.reporting.localization.Localization(__file__, 243, 11), 'or', result_ne_426, result_ne_432)
    
    # Testing the type of an if condition (line 243)
    if_condition_434 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 243, 8), result_or_keyword_433)
    # Assigning a type to the variable 'if_condition_434' (line 243)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 8), 'if_condition_434', if_condition_434)
    # SSA begins for if statement (line 243)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to RuntimeError(...): (line 244)
    # Processing the call arguments (line 244)
    unicode_436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 31), 'unicode', u'Bad kern pairs line: %s')
    # Getting the type of 'line' (line 244)
    line_437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 59), 'line', False)
    # Applying the binary operator '%' (line 244)
    result_mod_438 = python_operator(stypy.reporting.localization.Localization(__file__, 244, 31), '%', unicode_436, line_437)
    
    # Processing the call keyword arguments (line 244)
    kwargs_439 = {}
    # Getting the type of 'RuntimeError' (line 244)
    RuntimeError_435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 18), 'RuntimeError', False)
    # Calling RuntimeError(args, kwargs) (line 244)
    RuntimeError_call_result_440 = invoke(stypy.reporting.localization.Localization(__file__, 244, 18), RuntimeError_435, *[result_mod_438], **kwargs_439)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 244, 12), RuntimeError_call_result_440, 'raise parameter', BaseException)
    # SSA join for if statement (line 243)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Tuple to a Tuple (line 245):
    
    # Assigning a Call to a Name (line 245):
    
    # Call to _to_str(...): (line 245)
    # Processing the call arguments (line 245)
    
    # Obtaining the type of the subscript
    int_442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 35), 'int')
    # Getting the type of 'vals' (line 245)
    vals_443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 30), 'vals', False)
    # Obtaining the member '__getitem__' of a type (line 245)
    getitem___444 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 30), vals_443, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 245)
    subscript_call_result_445 = invoke(stypy.reporting.localization.Localization(__file__, 245, 30), getitem___444, int_442)
    
    # Processing the call keyword arguments (line 245)
    kwargs_446 = {}
    # Getting the type of '_to_str' (line 245)
    _to_str_441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 22), '_to_str', False)
    # Calling _to_str(args, kwargs) (line 245)
    _to_str_call_result_447 = invoke(stypy.reporting.localization.Localization(__file__, 245, 22), _to_str_441, *[subscript_call_result_445], **kwargs_446)
    
    # Assigning a type to the variable 'tuple_assignment_1' (line 245)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 8), 'tuple_assignment_1', _to_str_call_result_447)
    
    # Assigning a Call to a Name (line 245):
    
    # Call to _to_str(...): (line 245)
    # Processing the call arguments (line 245)
    
    # Obtaining the type of the subscript
    int_449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 53), 'int')
    # Getting the type of 'vals' (line 245)
    vals_450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 48), 'vals', False)
    # Obtaining the member '__getitem__' of a type (line 245)
    getitem___451 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 48), vals_450, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 245)
    subscript_call_result_452 = invoke(stypy.reporting.localization.Localization(__file__, 245, 48), getitem___451, int_449)
    
    # Processing the call keyword arguments (line 245)
    kwargs_453 = {}
    # Getting the type of '_to_str' (line 245)
    _to_str_448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 40), '_to_str', False)
    # Calling _to_str(args, kwargs) (line 245)
    _to_str_call_result_454 = invoke(stypy.reporting.localization.Localization(__file__, 245, 40), _to_str_448, *[subscript_call_result_452], **kwargs_453)
    
    # Assigning a type to the variable 'tuple_assignment_2' (line 245)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 8), 'tuple_assignment_2', _to_str_call_result_454)
    
    # Assigning a Call to a Name (line 245):
    
    # Call to _to_float(...): (line 245)
    # Processing the call arguments (line 245)
    
    # Obtaining the type of the subscript
    int_456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 73), 'int')
    # Getting the type of 'vals' (line 245)
    vals_457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 68), 'vals', False)
    # Obtaining the member '__getitem__' of a type (line 245)
    getitem___458 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 68), vals_457, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 245)
    subscript_call_result_459 = invoke(stypy.reporting.localization.Localization(__file__, 245, 68), getitem___458, int_456)
    
    # Processing the call keyword arguments (line 245)
    kwargs_460 = {}
    # Getting the type of '_to_float' (line 245)
    _to_float_455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 58), '_to_float', False)
    # Calling _to_float(args, kwargs) (line 245)
    _to_float_call_result_461 = invoke(stypy.reporting.localization.Localization(__file__, 245, 58), _to_float_455, *[subscript_call_result_459], **kwargs_460)
    
    # Assigning a type to the variable 'tuple_assignment_3' (line 245)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 8), 'tuple_assignment_3', _to_float_call_result_461)
    
    # Assigning a Name to a Name (line 245):
    # Getting the type of 'tuple_assignment_1' (line 245)
    tuple_assignment_1_462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 8), 'tuple_assignment_1')
    # Assigning a type to the variable 'c1' (line 245)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 8), 'c1', tuple_assignment_1_462)
    
    # Assigning a Name to a Name (line 245):
    # Getting the type of 'tuple_assignment_2' (line 245)
    tuple_assignment_2_463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 8), 'tuple_assignment_2')
    # Assigning a type to the variable 'c2' (line 245)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 12), 'c2', tuple_assignment_2_463)
    
    # Assigning a Name to a Name (line 245):
    # Getting the type of 'tuple_assignment_3' (line 245)
    tuple_assignment_3_464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 8), 'tuple_assignment_3')
    # Assigning a type to the variable 'val' (line 245)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 16), 'val', tuple_assignment_3_464)
    
    # Assigning a Name to a Subscript (line 246):
    
    # Assigning a Name to a Subscript (line 246):
    # Getting the type of 'val' (line 246)
    val_465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 22), 'val')
    # Getting the type of 'd' (line 246)
    d_466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 8), 'd')
    
    # Obtaining an instance of the builtin type 'tuple' (line 246)
    tuple_467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 246)
    # Adding element type (line 246)
    # Getting the type of 'c1' (line 246)
    c1_468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 11), 'c1')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 246, 11), tuple_467, c1_468)
    # Adding element type (line 246)
    # Getting the type of 'c2' (line 246)
    c2_469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 15), 'c2')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 246, 11), tuple_467, c2_469)
    
    # Storing an element on a container (line 246)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 246, 8), d_466, (tuple_467, val_465))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to RuntimeError(...): (line 247)
    # Processing the call arguments (line 247)
    unicode_471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 23), 'unicode', u'Bad kern pairs parse')
    # Processing the call keyword arguments (line 247)
    kwargs_472 = {}
    # Getting the type of 'RuntimeError' (line 247)
    RuntimeError_470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 10), 'RuntimeError', False)
    # Calling RuntimeError(args, kwargs) (line 247)
    RuntimeError_call_result_473 = invoke(stypy.reporting.localization.Localization(__file__, 247, 10), RuntimeError_470, *[unicode_471], **kwargs_472)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 247, 4), RuntimeError_call_result_473, 'raise parameter', BaseException)
    
    # ################# End of '_parse_kern_pairs(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_parse_kern_pairs' in the type store
    # Getting the type of 'stypy_return_type' (line 218)
    stypy_return_type_474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_474)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_parse_kern_pairs'
    return stypy_return_type_474

# Assigning a type to the variable '_parse_kern_pairs' (line 218)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 0), '_parse_kern_pairs', _parse_kern_pairs)

@norecursion
def _parse_composites(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_parse_composites'
    module_type_store = module_type_store.open_function_context('_parse_composites', 250, 0, False)
    
    # Passed parameters checking function
    _parse_composites.stypy_localization = localization
    _parse_composites.stypy_type_of_self = None
    _parse_composites.stypy_type_store = module_type_store
    _parse_composites.stypy_function_name = '_parse_composites'
    _parse_composites.stypy_param_names_list = ['fh']
    _parse_composites.stypy_varargs_param_name = None
    _parse_composites.stypy_kwargs_param_name = None
    _parse_composites.stypy_call_defaults = defaults
    _parse_composites.stypy_call_varargs = varargs
    _parse_composites.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_parse_composites', ['fh'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_parse_composites', localization, ['fh'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_parse_composites(...)' code ##################

    unicode_475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, (-1)), 'unicode', u"\n    Return a composites dictionary.  Keys are the names of the\n    composites.  Values are a num parts list of composite information,\n    with each element being a (*name*, *dx*, *dy*) tuple.  Thus a\n    composites line reading:\n\n      CC Aacute 2 ; PCC A 0 0 ; PCC acute 160 170 ;\n\n    will be represented as::\n\n      d['Aacute'] = [ ('A', 0, 0), ('acute', 160, 170) ]\n\n    ")
    
    # Assigning a Dict to a Name (line 264):
    
    # Assigning a Dict to a Name (line 264):
    
    # Obtaining an instance of the builtin type 'dict' (line 264)
    dict_476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 8), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 264)
    
    # Assigning a type to the variable 'd' (line 264)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 4), 'd', dict_476)
    
    # Getting the type of 'fh' (line 265)
    fh_477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 16), 'fh')
    # Testing the type of a for loop iterable (line 265)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 265, 4), fh_477)
    # Getting the type of the for loop variable (line 265)
    for_loop_var_478 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 265, 4), fh_477)
    # Assigning a type to the variable 'line' (line 265)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 4), 'line', for_loop_var_478)
    # SSA begins for a for statement (line 265)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 266):
    
    # Assigning a Call to a Name (line 266):
    
    # Call to rstrip(...): (line 266)
    # Processing the call keyword arguments (line 266)
    kwargs_481 = {}
    # Getting the type of 'line' (line 266)
    line_479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 15), 'line', False)
    # Obtaining the member 'rstrip' of a type (line 266)
    rstrip_480 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 15), line_479, 'rstrip')
    # Calling rstrip(args, kwargs) (line 266)
    rstrip_call_result_482 = invoke(stypy.reporting.localization.Localization(__file__, 266, 15), rstrip_480, *[], **kwargs_481)
    
    # Assigning a type to the variable 'line' (line 266)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 266, 8), 'line', rstrip_call_result_482)
    
    
    # Getting the type of 'line' (line 267)
    line_483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 15), 'line')
    # Applying the 'not' unary operator (line 267)
    result_not__484 = python_operator(stypy.reporting.localization.Localization(__file__, 267, 11), 'not', line_483)
    
    # Testing the type of an if condition (line 267)
    if_condition_485 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 267, 8), result_not__484)
    # Assigning a type to the variable 'if_condition_485' (line 267)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 8), 'if_condition_485', if_condition_485)
    # SSA begins for if statement (line 267)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 267)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to startswith(...): (line 269)
    # Processing the call arguments (line 269)
    str_488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 27), 'str', 'EndComposites')
    # Processing the call keyword arguments (line 269)
    kwargs_489 = {}
    # Getting the type of 'line' (line 269)
    line_486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 11), 'line', False)
    # Obtaining the member 'startswith' of a type (line 269)
    startswith_487 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 11), line_486, 'startswith')
    # Calling startswith(args, kwargs) (line 269)
    startswith_call_result_490 = invoke(stypy.reporting.localization.Localization(__file__, 269, 11), startswith_487, *[str_488], **kwargs_489)
    
    # Testing the type of an if condition (line 269)
    if_condition_491 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 269, 8), startswith_call_result_490)
    # Assigning a type to the variable 'if_condition_491' (line 269)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 8), 'if_condition_491', if_condition_491)
    # SSA begins for if statement (line 269)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'd' (line 270)
    d_492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 19), 'd')
    # Assigning a type to the variable 'stypy_return_type' (line 270)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 270, 12), 'stypy_return_type', d_492)
    # SSA join for if statement (line 269)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 271):
    
    # Assigning a Call to a Name (line 271):
    
    # Call to split(...): (line 271)
    # Processing the call arguments (line 271)
    str_495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 26), 'str', ';')
    # Processing the call keyword arguments (line 271)
    kwargs_496 = {}
    # Getting the type of 'line' (line 271)
    line_493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 15), 'line', False)
    # Obtaining the member 'split' of a type (line 271)
    split_494 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 15), line_493, 'split')
    # Calling split(args, kwargs) (line 271)
    split_call_result_497 = invoke(stypy.reporting.localization.Localization(__file__, 271, 15), split_494, *[str_495], **kwargs_496)
    
    # Assigning a type to the variable 'vals' (line 271)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 8), 'vals', split_call_result_497)
    
    # Assigning a Call to a Name (line 272):
    
    # Assigning a Call to a Name (line 272):
    
    # Call to split(...): (line 272)
    # Processing the call keyword arguments (line 272)
    kwargs_503 = {}
    
    # Obtaining the type of the subscript
    int_498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 18), 'int')
    # Getting the type of 'vals' (line 272)
    vals_499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 13), 'vals', False)
    # Obtaining the member '__getitem__' of a type (line 272)
    getitem___500 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 13), vals_499, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 272)
    subscript_call_result_501 = invoke(stypy.reporting.localization.Localization(__file__, 272, 13), getitem___500, int_498)
    
    # Obtaining the member 'split' of a type (line 272)
    split_502 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 13), subscript_call_result_501, 'split')
    # Calling split(args, kwargs) (line 272)
    split_call_result_504 = invoke(stypy.reporting.localization.Localization(__file__, 272, 13), split_502, *[], **kwargs_503)
    
    # Assigning a type to the variable 'cc' (line 272)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 8), 'cc', split_call_result_504)
    
    # Assigning a Tuple to a Tuple (line 273):
    
    # Assigning a Subscript to a Name (line 273):
    
    # Obtaining the type of the subscript
    int_505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 28), 'int')
    # Getting the type of 'cc' (line 273)
    cc_506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 25), 'cc')
    # Obtaining the member '__getitem__' of a type (line 273)
    getitem___507 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 25), cc_506, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 273)
    subscript_call_result_508 = invoke(stypy.reporting.localization.Localization(__file__, 273, 25), getitem___507, int_505)
    
    # Assigning a type to the variable 'tuple_assignment_4' (line 273)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 8), 'tuple_assignment_4', subscript_call_result_508)
    
    # Assigning a Call to a Name (line 273):
    
    # Call to _to_int(...): (line 273)
    # Processing the call arguments (line 273)
    
    # Obtaining the type of the subscript
    int_510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 43), 'int')
    # Getting the type of 'cc' (line 273)
    cc_511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 40), 'cc', False)
    # Obtaining the member '__getitem__' of a type (line 273)
    getitem___512 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 40), cc_511, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 273)
    subscript_call_result_513 = invoke(stypy.reporting.localization.Localization(__file__, 273, 40), getitem___512, int_510)
    
    # Processing the call keyword arguments (line 273)
    kwargs_514 = {}
    # Getting the type of '_to_int' (line 273)
    _to_int_509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 32), '_to_int', False)
    # Calling _to_int(args, kwargs) (line 273)
    _to_int_call_result_515 = invoke(stypy.reporting.localization.Localization(__file__, 273, 32), _to_int_509, *[subscript_call_result_513], **kwargs_514)
    
    # Assigning a type to the variable 'tuple_assignment_5' (line 273)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 8), 'tuple_assignment_5', _to_int_call_result_515)
    
    # Assigning a Name to a Name (line 273):
    # Getting the type of 'tuple_assignment_4' (line 273)
    tuple_assignment_4_516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 8), 'tuple_assignment_4')
    # Assigning a type to the variable 'name' (line 273)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 8), 'name', tuple_assignment_4_516)
    
    # Assigning a Name to a Name (line 273):
    # Getting the type of 'tuple_assignment_5' (line 273)
    tuple_assignment_5_517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 8), 'tuple_assignment_5')
    # Assigning a type to the variable 'numParts' (line 273)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 14), 'numParts', tuple_assignment_5_517)
    
    # Assigning a List to a Name (line 274):
    
    # Assigning a List to a Name (line 274):
    
    # Obtaining an instance of the builtin type 'list' (line 274)
    list_518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 274)
    
    # Assigning a type to the variable 'pccParts' (line 274)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 8), 'pccParts', list_518)
    
    
    # Obtaining the type of the subscript
    int_519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 22), 'int')
    int_520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 24), 'int')
    slice_521 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 275, 17), int_519, int_520, None)
    # Getting the type of 'vals' (line 275)
    vals_522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 17), 'vals')
    # Obtaining the member '__getitem__' of a type (line 275)
    getitem___523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 17), vals_522, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 275)
    subscript_call_result_524 = invoke(stypy.reporting.localization.Localization(__file__, 275, 17), getitem___523, slice_521)
    
    # Testing the type of a for loop iterable (line 275)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 275, 8), subscript_call_result_524)
    # Getting the type of the for loop variable (line 275)
    for_loop_var_525 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 275, 8), subscript_call_result_524)
    # Assigning a type to the variable 's' (line 275)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 8), 's', for_loop_var_525)
    # SSA begins for a for statement (line 275)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 276):
    
    # Assigning a Call to a Name (line 276):
    
    # Call to split(...): (line 276)
    # Processing the call keyword arguments (line 276)
    kwargs_528 = {}
    # Getting the type of 's' (line 276)
    s_526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 18), 's', False)
    # Obtaining the member 'split' of a type (line 276)
    split_527 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 18), s_526, 'split')
    # Calling split(args, kwargs) (line 276)
    split_call_result_529 = invoke(stypy.reporting.localization.Localization(__file__, 276, 18), split_527, *[], **kwargs_528)
    
    # Assigning a type to the variable 'pcc' (line 276)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 12), 'pcc', split_call_result_529)
    
    # Assigning a Tuple to a Tuple (line 277):
    
    # Assigning a Subscript to a Name (line 277):
    
    # Obtaining the type of the subscript
    int_530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 31), 'int')
    # Getting the type of 'pcc' (line 277)
    pcc_531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 27), 'pcc')
    # Obtaining the member '__getitem__' of a type (line 277)
    getitem___532 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 27), pcc_531, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 277)
    subscript_call_result_533 = invoke(stypy.reporting.localization.Localization(__file__, 277, 27), getitem___532, int_530)
    
    # Assigning a type to the variable 'tuple_assignment_6' (line 277)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 12), 'tuple_assignment_6', subscript_call_result_533)
    
    # Assigning a Call to a Name (line 277):
    
    # Call to _to_float(...): (line 277)
    # Processing the call arguments (line 277)
    
    # Obtaining the type of the subscript
    int_535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 49), 'int')
    # Getting the type of 'pcc' (line 277)
    pcc_536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 45), 'pcc', False)
    # Obtaining the member '__getitem__' of a type (line 277)
    getitem___537 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 45), pcc_536, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 277)
    subscript_call_result_538 = invoke(stypy.reporting.localization.Localization(__file__, 277, 45), getitem___537, int_535)
    
    # Processing the call keyword arguments (line 277)
    kwargs_539 = {}
    # Getting the type of '_to_float' (line 277)
    _to_float_534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 35), '_to_float', False)
    # Calling _to_float(args, kwargs) (line 277)
    _to_float_call_result_540 = invoke(stypy.reporting.localization.Localization(__file__, 277, 35), _to_float_534, *[subscript_call_result_538], **kwargs_539)
    
    # Assigning a type to the variable 'tuple_assignment_7' (line 277)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 12), 'tuple_assignment_7', _to_float_call_result_540)
    
    # Assigning a Call to a Name (line 277):
    
    # Call to _to_float(...): (line 277)
    # Processing the call arguments (line 277)
    
    # Obtaining the type of the subscript
    int_542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 68), 'int')
    # Getting the type of 'pcc' (line 277)
    pcc_543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 64), 'pcc', False)
    # Obtaining the member '__getitem__' of a type (line 277)
    getitem___544 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 64), pcc_543, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 277)
    subscript_call_result_545 = invoke(stypy.reporting.localization.Localization(__file__, 277, 64), getitem___544, int_542)
    
    # Processing the call keyword arguments (line 277)
    kwargs_546 = {}
    # Getting the type of '_to_float' (line 277)
    _to_float_541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 54), '_to_float', False)
    # Calling _to_float(args, kwargs) (line 277)
    _to_float_call_result_547 = invoke(stypy.reporting.localization.Localization(__file__, 277, 54), _to_float_541, *[subscript_call_result_545], **kwargs_546)
    
    # Assigning a type to the variable 'tuple_assignment_8' (line 277)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 12), 'tuple_assignment_8', _to_float_call_result_547)
    
    # Assigning a Name to a Name (line 277):
    # Getting the type of 'tuple_assignment_6' (line 277)
    tuple_assignment_6_548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 12), 'tuple_assignment_6')
    # Assigning a type to the variable 'name' (line 277)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 12), 'name', tuple_assignment_6_548)
    
    # Assigning a Name to a Name (line 277):
    # Getting the type of 'tuple_assignment_7' (line 277)
    tuple_assignment_7_549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 12), 'tuple_assignment_7')
    # Assigning a type to the variable 'dx' (line 277)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 18), 'dx', tuple_assignment_7_549)
    
    # Assigning a Name to a Name (line 277):
    # Getting the type of 'tuple_assignment_8' (line 277)
    tuple_assignment_8_550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 12), 'tuple_assignment_8')
    # Assigning a type to the variable 'dy' (line 277)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 22), 'dy', tuple_assignment_8_550)
    
    # Call to append(...): (line 278)
    # Processing the call arguments (line 278)
    
    # Obtaining an instance of the builtin type 'tuple' (line 278)
    tuple_553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 29), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 278)
    # Adding element type (line 278)
    # Getting the type of 'name' (line 278)
    name_554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 29), 'name', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 278, 29), tuple_553, name_554)
    # Adding element type (line 278)
    # Getting the type of 'dx' (line 278)
    dx_555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 35), 'dx', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 278, 29), tuple_553, dx_555)
    # Adding element type (line 278)
    # Getting the type of 'dy' (line 278)
    dy_556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 39), 'dy', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 278, 29), tuple_553, dy_556)
    
    # Processing the call keyword arguments (line 278)
    kwargs_557 = {}
    # Getting the type of 'pccParts' (line 278)
    pccParts_551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 12), 'pccParts', False)
    # Obtaining the member 'append' of a type (line 278)
    append_552 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 12), pccParts_551, 'append')
    # Calling append(args, kwargs) (line 278)
    append_call_result_558 = invoke(stypy.reporting.localization.Localization(__file__, 278, 12), append_552, *[tuple_553], **kwargs_557)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Subscript (line 279):
    
    # Assigning a Name to a Subscript (line 279):
    # Getting the type of 'pccParts' (line 279)
    pccParts_559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 18), 'pccParts')
    # Getting the type of 'd' (line 279)
    d_560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 8), 'd')
    # Getting the type of 'name' (line 279)
    name_561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 10), 'name')
    # Storing an element on a container (line 279)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 279, 8), d_560, (name_561, pccParts_559))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to RuntimeError(...): (line 281)
    # Processing the call arguments (line 281)
    unicode_563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 23), 'unicode', u'Bad composites parse')
    # Processing the call keyword arguments (line 281)
    kwargs_564 = {}
    # Getting the type of 'RuntimeError' (line 281)
    RuntimeError_562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 10), 'RuntimeError', False)
    # Calling RuntimeError(args, kwargs) (line 281)
    RuntimeError_call_result_565 = invoke(stypy.reporting.localization.Localization(__file__, 281, 10), RuntimeError_562, *[unicode_563], **kwargs_564)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 281, 4), RuntimeError_call_result_565, 'raise parameter', BaseException)
    
    # ################# End of '_parse_composites(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_parse_composites' in the type store
    # Getting the type of 'stypy_return_type' (line 250)
    stypy_return_type_566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_566)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_parse_composites'
    return stypy_return_type_566

# Assigning a type to the variable '_parse_composites' (line 250)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 0), '_parse_composites', _parse_composites)

@norecursion
def _parse_optional(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_parse_optional'
    module_type_store = module_type_store.open_function_context('_parse_optional', 284, 0, False)
    
    # Passed parameters checking function
    _parse_optional.stypy_localization = localization
    _parse_optional.stypy_type_of_self = None
    _parse_optional.stypy_type_store = module_type_store
    _parse_optional.stypy_function_name = '_parse_optional'
    _parse_optional.stypy_param_names_list = ['fh']
    _parse_optional.stypy_varargs_param_name = None
    _parse_optional.stypy_kwargs_param_name = None
    _parse_optional.stypy_call_defaults = defaults
    _parse_optional.stypy_call_varargs = varargs
    _parse_optional.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_parse_optional', ['fh'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_parse_optional', localization, ['fh'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_parse_optional(...)' code ##################

    unicode_567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, (-1)), 'unicode', u'\n    Parse the optional fields for kern pair data and composites\n\n    return value is a (*kernDict*, *compositeDict*) which are the\n    return values from :func:`_parse_kern_pairs`, and\n    :func:`_parse_composites` if the data exists, or empty dicts\n    otherwise\n    ')
    
    # Assigning a Dict to a Name (line 293):
    
    # Assigning a Dict to a Name (line 293):
    
    # Obtaining an instance of the builtin type 'dict' (line 293)
    dict_568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 15), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 293)
    # Adding element type (key, value) (line 293)
    str_569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 8), 'str', 'StartKernData')
    # Getting the type of '_parse_kern_pairs' (line 294)
    _parse_kern_pairs_570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 26), '_parse_kern_pairs')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 293, 15), dict_568, (str_569, _parse_kern_pairs_570))
    # Adding element type (key, value) (line 293)
    str_571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 8), 'str', 'StartComposites')
    # Getting the type of '_parse_composites' (line 295)
    _parse_composites_572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 29), '_parse_composites')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 293, 15), dict_568, (str_571, _parse_composites_572))
    
    # Assigning a type to the variable 'optional' (line 293)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 4), 'optional', dict_568)
    
    # Assigning a Dict to a Name (line 298):
    
    # Assigning a Dict to a Name (line 298):
    
    # Obtaining an instance of the builtin type 'dict' (line 298)
    dict_573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 8), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 298)
    # Adding element type (key, value) (line 298)
    str_574 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 9), 'str', 'StartKernData')
    
    # Obtaining an instance of the builtin type 'dict' (line 298)
    dict_575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 27), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 298)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 298, 8), dict_573, (str_574, dict_575))
    # Adding element type (key, value) (line 298)
    str_576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 31), 'str', 'StartComposites')
    
    # Obtaining an instance of the builtin type 'dict' (line 298)
    dict_577 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 51), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 298)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 298, 8), dict_573, (str_576, dict_577))
    
    # Assigning a type to the variable 'd' (line 298)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 4), 'd', dict_573)
    
    # Getting the type of 'fh' (line 299)
    fh_578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 16), 'fh')
    # Testing the type of a for loop iterable (line 299)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 299, 4), fh_578)
    # Getting the type of the for loop variable (line 299)
    for_loop_var_579 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 299, 4), fh_578)
    # Assigning a type to the variable 'line' (line 299)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 4), 'line', for_loop_var_579)
    # SSA begins for a for statement (line 299)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 300):
    
    # Assigning a Call to a Name (line 300):
    
    # Call to rstrip(...): (line 300)
    # Processing the call keyword arguments (line 300)
    kwargs_582 = {}
    # Getting the type of 'line' (line 300)
    line_580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 15), 'line', False)
    # Obtaining the member 'rstrip' of a type (line 300)
    rstrip_581 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 15), line_580, 'rstrip')
    # Calling rstrip(args, kwargs) (line 300)
    rstrip_call_result_583 = invoke(stypy.reporting.localization.Localization(__file__, 300, 15), rstrip_581, *[], **kwargs_582)
    
    # Assigning a type to the variable 'line' (line 300)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 8), 'line', rstrip_call_result_583)
    
    
    # Getting the type of 'line' (line 301)
    line_584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 15), 'line')
    # Applying the 'not' unary operator (line 301)
    result_not__585 = python_operator(stypy.reporting.localization.Localization(__file__, 301, 11), 'not', line_584)
    
    # Testing the type of an if condition (line 301)
    if_condition_586 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 301, 8), result_not__585)
    # Assigning a type to the variable 'if_condition_586' (line 301)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 8), 'if_condition_586', if_condition_586)
    # SSA begins for if statement (line 301)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 301)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Subscript to a Name (line 303):
    
    # Assigning a Subscript to a Name (line 303):
    
    # Obtaining the type of the subscript
    int_587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 27), 'int')
    
    # Call to split(...): (line 303)
    # Processing the call keyword arguments (line 303)
    kwargs_590 = {}
    # Getting the type of 'line' (line 303)
    line_588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 14), 'line', False)
    # Obtaining the member 'split' of a type (line 303)
    split_589 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 14), line_588, 'split')
    # Calling split(args, kwargs) (line 303)
    split_call_result_591 = invoke(stypy.reporting.localization.Localization(__file__, 303, 14), split_589, *[], **kwargs_590)
    
    # Obtaining the member '__getitem__' of a type (line 303)
    getitem___592 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 14), split_call_result_591, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 303)
    subscript_call_result_593 = invoke(stypy.reporting.localization.Localization(__file__, 303, 14), getitem___592, int_587)
    
    # Assigning a type to the variable 'key' (line 303)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 8), 'key', subscript_call_result_593)
    
    
    # Getting the type of 'key' (line 305)
    key_594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 11), 'key')
    # Getting the type of 'optional' (line 305)
    optional_595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 18), 'optional')
    # Applying the binary operator 'in' (line 305)
    result_contains_596 = python_operator(stypy.reporting.localization.Localization(__file__, 305, 11), 'in', key_594, optional_595)
    
    # Testing the type of an if condition (line 305)
    if_condition_597 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 305, 8), result_contains_596)
    # Assigning a type to the variable 'if_condition_597' (line 305)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 305, 8), 'if_condition_597', if_condition_597)
    # SSA begins for if statement (line 305)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Subscript (line 306):
    
    # Assigning a Call to a Subscript (line 306):
    
    # Call to (...): (line 306)
    # Processing the call arguments (line 306)
    # Getting the type of 'fh' (line 306)
    fh_602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 35), 'fh', False)
    # Processing the call keyword arguments (line 306)
    kwargs_603 = {}
    
    # Obtaining the type of the subscript
    # Getting the type of 'key' (line 306)
    key_598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 30), 'key', False)
    # Getting the type of 'optional' (line 306)
    optional_599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 21), 'optional', False)
    # Obtaining the member '__getitem__' of a type (line 306)
    getitem___600 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 21), optional_599, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 306)
    subscript_call_result_601 = invoke(stypy.reporting.localization.Localization(__file__, 306, 21), getitem___600, key_598)
    
    # Calling (args, kwargs) (line 306)
    _call_result_604 = invoke(stypy.reporting.localization.Localization(__file__, 306, 21), subscript_call_result_601, *[fh_602], **kwargs_603)
    
    # Getting the type of 'd' (line 306)
    d_605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 12), 'd')
    # Getting the type of 'key' (line 306)
    key_606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 14), 'key')
    # Storing an element on a container (line 306)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 306, 12), d_605, (key_606, _call_result_604))
    # SSA join for if statement (line 305)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Tuple to a Name (line 308):
    
    # Assigning a Tuple to a Name (line 308):
    
    # Obtaining an instance of the builtin type 'tuple' (line 308)
    tuple_607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 308)
    # Adding element type (line 308)
    
    # Obtaining the type of the subscript
    str_608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, 11), 'str', 'StartKernData')
    # Getting the type of 'd' (line 308)
    d_609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 9), 'd')
    # Obtaining the member '__getitem__' of a type (line 308)
    getitem___610 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 9), d_609, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 308)
    subscript_call_result_611 = invoke(stypy.reporting.localization.Localization(__file__, 308, 9), getitem___610, str_608)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 308, 9), tuple_607, subscript_call_result_611)
    # Adding element type (line 308)
    
    # Obtaining the type of the subscript
    str_612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, 32), 'str', 'StartComposites')
    # Getting the type of 'd' (line 308)
    d_613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 30), 'd')
    # Obtaining the member '__getitem__' of a type (line 308)
    getitem___614 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 30), d_613, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 308)
    subscript_call_result_615 = invoke(stypy.reporting.localization.Localization(__file__, 308, 30), getitem___614, str_612)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 308, 9), tuple_607, subscript_call_result_615)
    
    # Assigning a type to the variable 'l' (line 308)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 308, 4), 'l', tuple_607)
    # Getting the type of 'l' (line 309)
    l_616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 11), 'l')
    # Assigning a type to the variable 'stypy_return_type' (line 309)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 4), 'stypy_return_type', l_616)
    
    # ################# End of '_parse_optional(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_parse_optional' in the type store
    # Getting the type of 'stypy_return_type' (line 284)
    stypy_return_type_617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_617)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_parse_optional'
    return stypy_return_type_617

# Assigning a type to the variable '_parse_optional' (line 284)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 0), '_parse_optional', _parse_optional)

@norecursion
def parse_afm(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'parse_afm'
    module_type_store = module_type_store.open_function_context('parse_afm', 312, 0, False)
    
    # Passed parameters checking function
    parse_afm.stypy_localization = localization
    parse_afm.stypy_type_of_self = None
    parse_afm.stypy_type_store = module_type_store
    parse_afm.stypy_function_name = 'parse_afm'
    parse_afm.stypy_param_names_list = ['fh']
    parse_afm.stypy_varargs_param_name = None
    parse_afm.stypy_kwargs_param_name = None
    parse_afm.stypy_call_defaults = defaults
    parse_afm.stypy_call_varargs = varargs
    parse_afm.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'parse_afm', ['fh'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'parse_afm', localization, ['fh'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'parse_afm(...)' code ##################

    unicode_618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, (-1)), 'unicode', u'\n    Parse the Adobe Font Metics file in file handle *fh*. Return value\n    is a (*dhead*, *dcmetrics*, *dkernpairs*, *dcomposite*) tuple where\n    *dhead* is a :func:`_parse_header` dict, *dcmetrics* is a\n    :func:`_parse_composites` dict, *dkernpairs* is a\n    :func:`_parse_kern_pairs` dict (possibly {}), and *dcomposite* is a\n    :func:`_parse_composites` dict (possibly {})\n    ')
    
    # Call to _sanity_check(...): (line 321)
    # Processing the call arguments (line 321)
    # Getting the type of 'fh' (line 321)
    fh_620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 18), 'fh', False)
    # Processing the call keyword arguments (line 321)
    kwargs_621 = {}
    # Getting the type of '_sanity_check' (line 321)
    _sanity_check_619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 4), '_sanity_check', False)
    # Calling _sanity_check(args, kwargs) (line 321)
    _sanity_check_call_result_622 = invoke(stypy.reporting.localization.Localization(__file__, 321, 4), _sanity_check_619, *[fh_620], **kwargs_621)
    
    
    # Assigning a Call to a Name (line 322):
    
    # Assigning a Call to a Name (line 322):
    
    # Call to _parse_header(...): (line 322)
    # Processing the call arguments (line 322)
    # Getting the type of 'fh' (line 322)
    fh_624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 26), 'fh', False)
    # Processing the call keyword arguments (line 322)
    kwargs_625 = {}
    # Getting the type of '_parse_header' (line 322)
    _parse_header_623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 12), '_parse_header', False)
    # Calling _parse_header(args, kwargs) (line 322)
    _parse_header_call_result_626 = invoke(stypy.reporting.localization.Localization(__file__, 322, 12), _parse_header_623, *[fh_624], **kwargs_625)
    
    # Assigning a type to the variable 'dhead' (line 322)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 322, 4), 'dhead', _parse_header_call_result_626)
    
    # Assigning a Call to a Tuple (line 323):
    
    # Assigning a Call to a Name:
    
    # Call to _parse_char_metrics(...): (line 323)
    # Processing the call arguments (line 323)
    # Getting the type of 'fh' (line 323)
    fh_628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 58), 'fh', False)
    # Processing the call keyword arguments (line 323)
    kwargs_629 = {}
    # Getting the type of '_parse_char_metrics' (line 323)
    _parse_char_metrics_627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 38), '_parse_char_metrics', False)
    # Calling _parse_char_metrics(args, kwargs) (line 323)
    _parse_char_metrics_call_result_630 = invoke(stypy.reporting.localization.Localization(__file__, 323, 38), _parse_char_metrics_627, *[fh_628], **kwargs_629)
    
    # Assigning a type to the variable 'call_assignment_9' (line 323)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 4), 'call_assignment_9', _parse_char_metrics_call_result_630)
    
    # Assigning a Call to a Name (line 323):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 4), 'int')
    # Processing the call keyword arguments
    kwargs_634 = {}
    # Getting the type of 'call_assignment_9' (line 323)
    call_assignment_9_631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 4), 'call_assignment_9', False)
    # Obtaining the member '__getitem__' of a type (line 323)
    getitem___632 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 323, 4), call_assignment_9_631, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_635 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___632, *[int_633], **kwargs_634)
    
    # Assigning a type to the variable 'call_assignment_10' (line 323)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 4), 'call_assignment_10', getitem___call_result_635)
    
    # Assigning a Name to a Name (line 323):
    # Getting the type of 'call_assignment_10' (line 323)
    call_assignment_10_636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 4), 'call_assignment_10')
    # Assigning a type to the variable 'dcmetrics_ascii' (line 323)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 4), 'dcmetrics_ascii', call_assignment_10_636)
    
    # Assigning a Call to a Name (line 323):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 4), 'int')
    # Processing the call keyword arguments
    kwargs_640 = {}
    # Getting the type of 'call_assignment_9' (line 323)
    call_assignment_9_637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 4), 'call_assignment_9', False)
    # Obtaining the member '__getitem__' of a type (line 323)
    getitem___638 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 323, 4), call_assignment_9_637, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_641 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___638, *[int_639], **kwargs_640)
    
    # Assigning a type to the variable 'call_assignment_11' (line 323)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 4), 'call_assignment_11', getitem___call_result_641)
    
    # Assigning a Name to a Name (line 323):
    # Getting the type of 'call_assignment_11' (line 323)
    call_assignment_11_642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 4), 'call_assignment_11')
    # Assigning a type to the variable 'dcmetrics_name' (line 323)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 21), 'dcmetrics_name', call_assignment_11_642)
    
    # Assigning a Call to a Name (line 324):
    
    # Assigning a Call to a Name (line 324):
    
    # Call to _parse_optional(...): (line 324)
    # Processing the call arguments (line 324)
    # Getting the type of 'fh' (line 324)
    fh_644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 32), 'fh', False)
    # Processing the call keyword arguments (line 324)
    kwargs_645 = {}
    # Getting the type of '_parse_optional' (line 324)
    _parse_optional_643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 16), '_parse_optional', False)
    # Calling _parse_optional(args, kwargs) (line 324)
    _parse_optional_call_result_646 = invoke(stypy.reporting.localization.Localization(__file__, 324, 16), _parse_optional_643, *[fh_644], **kwargs_645)
    
    # Assigning a type to the variable 'doptional' (line 324)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 324, 4), 'doptional', _parse_optional_call_result_646)
    
    # Obtaining an instance of the builtin type 'tuple' (line 325)
    tuple_647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 325)
    # Adding element type (line 325)
    # Getting the type of 'dhead' (line 325)
    dhead_648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 11), 'dhead')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 325, 11), tuple_647, dhead_648)
    # Adding element type (line 325)
    # Getting the type of 'dcmetrics_ascii' (line 325)
    dcmetrics_ascii_649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 18), 'dcmetrics_ascii')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 325, 11), tuple_647, dcmetrics_ascii_649)
    # Adding element type (line 325)
    # Getting the type of 'dcmetrics_name' (line 325)
    dcmetrics_name_650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 35), 'dcmetrics_name')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 325, 11), tuple_647, dcmetrics_name_650)
    # Adding element type (line 325)
    
    # Obtaining the type of the subscript
    int_651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 61), 'int')
    # Getting the type of 'doptional' (line 325)
    doptional_652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 51), 'doptional')
    # Obtaining the member '__getitem__' of a type (line 325)
    getitem___653 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 325, 51), doptional_652, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 325)
    subscript_call_result_654 = invoke(stypy.reporting.localization.Localization(__file__, 325, 51), getitem___653, int_651)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 325, 11), tuple_647, subscript_call_result_654)
    # Adding element type (line 325)
    
    # Obtaining the type of the subscript
    int_655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 75), 'int')
    # Getting the type of 'doptional' (line 325)
    doptional_656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 65), 'doptional')
    # Obtaining the member '__getitem__' of a type (line 325)
    getitem___657 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 325, 65), doptional_656, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 325)
    subscript_call_result_658 = invoke(stypy.reporting.localization.Localization(__file__, 325, 65), getitem___657, int_655)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 325, 11), tuple_647, subscript_call_result_658)
    
    # Assigning a type to the variable 'stypy_return_type' (line 325)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 4), 'stypy_return_type', tuple_647)
    
    # ################# End of 'parse_afm(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'parse_afm' in the type store
    # Getting the type of 'stypy_return_type' (line 312)
    stypy_return_type_659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_659)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'parse_afm'
    return stypy_return_type_659

# Assigning a type to the variable 'parse_afm' (line 312)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 0), 'parse_afm', parse_afm)
# Declaration of the 'AFM' class

class AFM(object, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 330, 4, False)
        # Assigning a type to the variable 'self' (line 331)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'AFM.__init__', ['fh'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['fh'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        unicode_660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, (-1)), 'unicode', u'\n        Parse the AFM file in file object *fh*\n        ')
        
        # Assigning a Call to a Tuple (line 334):
        
        # Assigning a Call to a Name:
        
        # Call to parse_afm(...): (line 335)
        # Processing the call arguments (line 335)
        # Getting the type of 'fh' (line 335)
        fh_662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 22), 'fh', False)
        # Processing the call keyword arguments (line 335)
        kwargs_663 = {}
        # Getting the type of 'parse_afm' (line 335)
        parse_afm_661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 12), 'parse_afm', False)
        # Calling parse_afm(args, kwargs) (line 335)
        parse_afm_call_result_664 = invoke(stypy.reporting.localization.Localization(__file__, 335, 12), parse_afm_661, *[fh_662], **kwargs_663)
        
        # Assigning a type to the variable 'call_assignment_12' (line 334)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 8), 'call_assignment_12', parse_afm_call_result_664)
        
        # Assigning a Call to a Name (line 334):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 8), 'int')
        # Processing the call keyword arguments
        kwargs_668 = {}
        # Getting the type of 'call_assignment_12' (line 334)
        call_assignment_12_665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 8), 'call_assignment_12', False)
        # Obtaining the member '__getitem__' of a type (line 334)
        getitem___666 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 8), call_assignment_12_665, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_669 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___666, *[int_667], **kwargs_668)
        
        # Assigning a type to the variable 'call_assignment_13' (line 334)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 8), 'call_assignment_13', getitem___call_result_669)
        
        # Assigning a Name to a Name (line 334):
        # Getting the type of 'call_assignment_13' (line 334)
        call_assignment_13_670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 8), 'call_assignment_13')
        # Assigning a type to the variable 'dhead' (line 334)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 9), 'dhead', call_assignment_13_670)
        
        # Assigning a Call to a Name (line 334):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 8), 'int')
        # Processing the call keyword arguments
        kwargs_674 = {}
        # Getting the type of 'call_assignment_12' (line 334)
        call_assignment_12_671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 8), 'call_assignment_12', False)
        # Obtaining the member '__getitem__' of a type (line 334)
        getitem___672 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 8), call_assignment_12_671, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_675 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___672, *[int_673], **kwargs_674)
        
        # Assigning a type to the variable 'call_assignment_14' (line 334)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 8), 'call_assignment_14', getitem___call_result_675)
        
        # Assigning a Name to a Name (line 334):
        # Getting the type of 'call_assignment_14' (line 334)
        call_assignment_14_676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 8), 'call_assignment_14')
        # Assigning a type to the variable 'dcmetrics_ascii' (line 334)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 16), 'dcmetrics_ascii', call_assignment_14_676)
        
        # Assigning a Call to a Name (line 334):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 8), 'int')
        # Processing the call keyword arguments
        kwargs_680 = {}
        # Getting the type of 'call_assignment_12' (line 334)
        call_assignment_12_677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 8), 'call_assignment_12', False)
        # Obtaining the member '__getitem__' of a type (line 334)
        getitem___678 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 8), call_assignment_12_677, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_681 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___678, *[int_679], **kwargs_680)
        
        # Assigning a type to the variable 'call_assignment_15' (line 334)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 8), 'call_assignment_15', getitem___call_result_681)
        
        # Assigning a Name to a Name (line 334):
        # Getting the type of 'call_assignment_15' (line 334)
        call_assignment_15_682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 8), 'call_assignment_15')
        # Assigning a type to the variable 'dcmetrics_name' (line 334)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 33), 'dcmetrics_name', call_assignment_15_682)
        
        # Assigning a Call to a Name (line 334):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 8), 'int')
        # Processing the call keyword arguments
        kwargs_686 = {}
        # Getting the type of 'call_assignment_12' (line 334)
        call_assignment_12_683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 8), 'call_assignment_12', False)
        # Obtaining the member '__getitem__' of a type (line 334)
        getitem___684 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 8), call_assignment_12_683, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_687 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___684, *[int_685], **kwargs_686)
        
        # Assigning a type to the variable 'call_assignment_16' (line 334)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 8), 'call_assignment_16', getitem___call_result_687)
        
        # Assigning a Name to a Name (line 334):
        # Getting the type of 'call_assignment_16' (line 334)
        call_assignment_16_688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 8), 'call_assignment_16')
        # Assigning a type to the variable 'dkernpairs' (line 334)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 49), 'dkernpairs', call_assignment_16_688)
        
        # Assigning a Call to a Name (line 334):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 8), 'int')
        # Processing the call keyword arguments
        kwargs_692 = {}
        # Getting the type of 'call_assignment_12' (line 334)
        call_assignment_12_689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 8), 'call_assignment_12', False)
        # Obtaining the member '__getitem__' of a type (line 334)
        getitem___690 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 8), call_assignment_12_689, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_693 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___690, *[int_691], **kwargs_692)
        
        # Assigning a type to the variable 'call_assignment_17' (line 334)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 8), 'call_assignment_17', getitem___call_result_693)
        
        # Assigning a Name to a Name (line 334):
        # Getting the type of 'call_assignment_17' (line 334)
        call_assignment_17_694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 8), 'call_assignment_17')
        # Assigning a type to the variable 'dcomposite' (line 334)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 61), 'dcomposite', call_assignment_17_694)
        
        # Assigning a Name to a Attribute (line 336):
        
        # Assigning a Name to a Attribute (line 336):
        # Getting the type of 'dhead' (line 336)
        dhead_695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 23), 'dhead')
        # Getting the type of 'self' (line 336)
        self_696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 8), 'self')
        # Setting the type of the member '_header' of a type (line 336)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 8), self_696, '_header', dhead_695)
        
        # Assigning a Name to a Attribute (line 337):
        
        # Assigning a Name to a Attribute (line 337):
        # Getting the type of 'dkernpairs' (line 337)
        dkernpairs_697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 21), 'dkernpairs')
        # Getting the type of 'self' (line 337)
        self_698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 8), 'self')
        # Setting the type of the member '_kern' of a type (line 337)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 8), self_698, '_kern', dkernpairs_697)
        
        # Assigning a Name to a Attribute (line 338):
        
        # Assigning a Name to a Attribute (line 338):
        # Getting the type of 'dcmetrics_ascii' (line 338)
        dcmetrics_ascii_699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 24), 'dcmetrics_ascii')
        # Getting the type of 'self' (line 338)
        self_700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 8), 'self')
        # Setting the type of the member '_metrics' of a type (line 338)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 8), self_700, '_metrics', dcmetrics_ascii_699)
        
        # Assigning a Name to a Attribute (line 339):
        
        # Assigning a Name to a Attribute (line 339):
        # Getting the type of 'dcmetrics_name' (line 339)
        dcmetrics_name_701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 32), 'dcmetrics_name')
        # Getting the type of 'self' (line 339)
        self_702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 8), 'self')
        # Setting the type of the member '_metrics_by_name' of a type (line 339)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 339, 8), self_702, '_metrics_by_name', dcmetrics_name_701)
        
        # Assigning a Name to a Attribute (line 340):
        
        # Assigning a Name to a Attribute (line 340):
        # Getting the type of 'dcomposite' (line 340)
        dcomposite_703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 26), 'dcomposite')
        # Getting the type of 'self' (line 340)
        self_704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 8), 'self')
        # Setting the type of the member '_composite' of a type (line 340)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 340, 8), self_704, '_composite', dcomposite_703)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def get_bbox_char(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'False' (line 342)
        False_705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 37), 'False')
        defaults = [False_705]
        # Create a new context for function 'get_bbox_char'
        module_type_store = module_type_store.open_function_context('get_bbox_char', 342, 4, False)
        # Assigning a type to the variable 'self' (line 343)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        AFM.get_bbox_char.__dict__.__setitem__('stypy_localization', localization)
        AFM.get_bbox_char.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        AFM.get_bbox_char.__dict__.__setitem__('stypy_type_store', module_type_store)
        AFM.get_bbox_char.__dict__.__setitem__('stypy_function_name', 'AFM.get_bbox_char')
        AFM.get_bbox_char.__dict__.__setitem__('stypy_param_names_list', ['c', 'isord'])
        AFM.get_bbox_char.__dict__.__setitem__('stypy_varargs_param_name', None)
        AFM.get_bbox_char.__dict__.__setitem__('stypy_kwargs_param_name', None)
        AFM.get_bbox_char.__dict__.__setitem__('stypy_call_defaults', defaults)
        AFM.get_bbox_char.__dict__.__setitem__('stypy_call_varargs', varargs)
        AFM.get_bbox_char.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        AFM.get_bbox_char.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'AFM.get_bbox_char', ['c', 'isord'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_bbox_char', localization, ['c', 'isord'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_bbox_char(...)' code ##################

        
        
        # Getting the type of 'isord' (line 343)
        isord_706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 15), 'isord')
        # Applying the 'not' unary operator (line 343)
        result_not__707 = python_operator(stypy.reporting.localization.Localization(__file__, 343, 11), 'not', isord_706)
        
        # Testing the type of an if condition (line 343)
        if_condition_708 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 343, 8), result_not__707)
        # Assigning a type to the variable 'if_condition_708' (line 343)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 8), 'if_condition_708', if_condition_708)
        # SSA begins for if statement (line 343)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 344):
        
        # Assigning a Call to a Name (line 344):
        
        # Call to ord(...): (line 344)
        # Processing the call arguments (line 344)
        # Getting the type of 'c' (line 344)
        c_710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 20), 'c', False)
        # Processing the call keyword arguments (line 344)
        kwargs_711 = {}
        # Getting the type of 'ord' (line 344)
        ord_709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 16), 'ord', False)
        # Calling ord(args, kwargs) (line 344)
        ord_call_result_712 = invoke(stypy.reporting.localization.Localization(__file__, 344, 16), ord_709, *[c_710], **kwargs_711)
        
        # Assigning a type to the variable 'c' (line 344)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 344, 12), 'c', ord_call_result_712)
        # SSA join for if statement (line 343)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Subscript to a Tuple (line 345):
        
        # Assigning a Subscript to a Name (line 345):
        
        # Obtaining the type of the subscript
        int_713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 8), 'int')
        
        # Obtaining the type of the subscript
        # Getting the type of 'c' (line 345)
        c_714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 39), 'c')
        # Getting the type of 'self' (line 345)
        self_715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 25), 'self')
        # Obtaining the member '_metrics' of a type (line 345)
        _metrics_716 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 345, 25), self_715, '_metrics')
        # Obtaining the member '__getitem__' of a type (line 345)
        getitem___717 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 345, 25), _metrics_716, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 345)
        subscript_call_result_718 = invoke(stypy.reporting.localization.Localization(__file__, 345, 25), getitem___717, c_714)
        
        # Obtaining the member '__getitem__' of a type (line 345)
        getitem___719 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 345, 8), subscript_call_result_718, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 345)
        subscript_call_result_720 = invoke(stypy.reporting.localization.Localization(__file__, 345, 8), getitem___719, int_713)
        
        # Assigning a type to the variable 'tuple_var_assignment_18' (line 345)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 345, 8), 'tuple_var_assignment_18', subscript_call_result_720)
        
        # Assigning a Subscript to a Name (line 345):
        
        # Obtaining the type of the subscript
        int_721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 8), 'int')
        
        # Obtaining the type of the subscript
        # Getting the type of 'c' (line 345)
        c_722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 39), 'c')
        # Getting the type of 'self' (line 345)
        self_723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 25), 'self')
        # Obtaining the member '_metrics' of a type (line 345)
        _metrics_724 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 345, 25), self_723, '_metrics')
        # Obtaining the member '__getitem__' of a type (line 345)
        getitem___725 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 345, 25), _metrics_724, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 345)
        subscript_call_result_726 = invoke(stypy.reporting.localization.Localization(__file__, 345, 25), getitem___725, c_722)
        
        # Obtaining the member '__getitem__' of a type (line 345)
        getitem___727 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 345, 8), subscript_call_result_726, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 345)
        subscript_call_result_728 = invoke(stypy.reporting.localization.Localization(__file__, 345, 8), getitem___727, int_721)
        
        # Assigning a type to the variable 'tuple_var_assignment_19' (line 345)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 345, 8), 'tuple_var_assignment_19', subscript_call_result_728)
        
        # Assigning a Subscript to a Name (line 345):
        
        # Obtaining the type of the subscript
        int_729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 8), 'int')
        
        # Obtaining the type of the subscript
        # Getting the type of 'c' (line 345)
        c_730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 39), 'c')
        # Getting the type of 'self' (line 345)
        self_731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 25), 'self')
        # Obtaining the member '_metrics' of a type (line 345)
        _metrics_732 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 345, 25), self_731, '_metrics')
        # Obtaining the member '__getitem__' of a type (line 345)
        getitem___733 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 345, 25), _metrics_732, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 345)
        subscript_call_result_734 = invoke(stypy.reporting.localization.Localization(__file__, 345, 25), getitem___733, c_730)
        
        # Obtaining the member '__getitem__' of a type (line 345)
        getitem___735 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 345, 8), subscript_call_result_734, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 345)
        subscript_call_result_736 = invoke(stypy.reporting.localization.Localization(__file__, 345, 8), getitem___735, int_729)
        
        # Assigning a type to the variable 'tuple_var_assignment_20' (line 345)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 345, 8), 'tuple_var_assignment_20', subscript_call_result_736)
        
        # Assigning a Name to a Name (line 345):
        # Getting the type of 'tuple_var_assignment_18' (line 345)
        tuple_var_assignment_18_737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 8), 'tuple_var_assignment_18')
        # Assigning a type to the variable 'wx' (line 345)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 345, 8), 'wx', tuple_var_assignment_18_737)
        
        # Assigning a Name to a Name (line 345):
        # Getting the type of 'tuple_var_assignment_19' (line 345)
        tuple_var_assignment_19_738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 8), 'tuple_var_assignment_19')
        # Assigning a type to the variable 'name' (line 345)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 345, 12), 'name', tuple_var_assignment_19_738)
        
        # Assigning a Name to a Name (line 345):
        # Getting the type of 'tuple_var_assignment_20' (line 345)
        tuple_var_assignment_20_739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 8), 'tuple_var_assignment_20')
        # Assigning a type to the variable 'bbox' (line 345)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 345, 18), 'bbox', tuple_var_assignment_20_739)
        # Getting the type of 'bbox' (line 346)
        bbox_740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 15), 'bbox')
        # Assigning a type to the variable 'stypy_return_type' (line 346)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 346, 8), 'stypy_return_type', bbox_740)
        
        # ################# End of 'get_bbox_char(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_bbox_char' in the type store
        # Getting the type of 'stypy_return_type' (line 342)
        stypy_return_type_741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_741)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_bbox_char'
        return stypy_return_type_741


    @norecursion
    def string_width_height(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'string_width_height'
        module_type_store = module_type_store.open_function_context('string_width_height', 348, 4, False)
        # Assigning a type to the variable 'self' (line 349)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        AFM.string_width_height.__dict__.__setitem__('stypy_localization', localization)
        AFM.string_width_height.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        AFM.string_width_height.__dict__.__setitem__('stypy_type_store', module_type_store)
        AFM.string_width_height.__dict__.__setitem__('stypy_function_name', 'AFM.string_width_height')
        AFM.string_width_height.__dict__.__setitem__('stypy_param_names_list', ['s'])
        AFM.string_width_height.__dict__.__setitem__('stypy_varargs_param_name', None)
        AFM.string_width_height.__dict__.__setitem__('stypy_kwargs_param_name', None)
        AFM.string_width_height.__dict__.__setitem__('stypy_call_defaults', defaults)
        AFM.string_width_height.__dict__.__setitem__('stypy_call_varargs', varargs)
        AFM.string_width_height.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        AFM.string_width_height.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'AFM.string_width_height', ['s'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'string_width_height', localization, ['s'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'string_width_height(...)' code ##################

        unicode_742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 352, (-1)), 'unicode', u'\n        Return the string width (including kerning) and string height\n        as a (*w*, *h*) tuple.\n        ')
        
        
        
        # Call to len(...): (line 353)
        # Processing the call arguments (line 353)
        # Getting the type of 's' (line 353)
        s_744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 19), 's', False)
        # Processing the call keyword arguments (line 353)
        kwargs_745 = {}
        # Getting the type of 'len' (line 353)
        len_743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 15), 'len', False)
        # Calling len(args, kwargs) (line 353)
        len_call_result_746 = invoke(stypy.reporting.localization.Localization(__file__, 353, 15), len_743, *[s_744], **kwargs_745)
        
        # Applying the 'not' unary operator (line 353)
        result_not__747 = python_operator(stypy.reporting.localization.Localization(__file__, 353, 11), 'not', len_call_result_746)
        
        # Testing the type of an if condition (line 353)
        if_condition_748 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 353, 8), result_not__747)
        # Assigning a type to the variable 'if_condition_748' (line 353)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 353, 8), 'if_condition_748', if_condition_748)
        # SSA begins for if statement (line 353)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Obtaining an instance of the builtin type 'tuple' (line 354)
        tuple_749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 19), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 354)
        # Adding element type (line 354)
        int_750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 354, 19), tuple_749, int_750)
        # Adding element type (line 354)
        int_751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 354, 19), tuple_749, int_751)
        
        # Assigning a type to the variable 'stypy_return_type' (line 354)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 354, 12), 'stypy_return_type', tuple_749)
        # SSA join for if statement (line 353)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Num to a Name (line 355):
        
        # Assigning a Num to a Name (line 355):
        int_752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 17), 'int')
        # Assigning a type to the variable 'totalw' (line 355)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 355, 8), 'totalw', int_752)
        
        # Assigning a Name to a Name (line 356):
        
        # Assigning a Name to a Name (line 356):
        # Getting the type of 'None' (line 356)
        None_753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 19), 'None')
        # Assigning a type to the variable 'namelast' (line 356)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 356, 8), 'namelast', None_753)
        
        # Assigning a Num to a Name (line 357):
        
        # Assigning a Num to a Name (line 357):
        float_754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 357, 15), 'float')
        # Assigning a type to the variable 'miny' (line 357)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 357, 8), 'miny', float_754)
        
        # Assigning a Num to a Name (line 358):
        
        # Assigning a Num to a Name (line 358):
        int_755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 358, 15), 'int')
        # Assigning a type to the variable 'maxy' (line 358)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 358, 8), 'maxy', int_755)
        
        # Getting the type of 's' (line 359)
        s_756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 17), 's')
        # Testing the type of a for loop iterable (line 359)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 359, 8), s_756)
        # Getting the type of the for loop variable (line 359)
        for_loop_var_757 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 359, 8), s_756)
        # Assigning a type to the variable 'c' (line 359)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 359, 8), 'c', for_loop_var_757)
        # SSA begins for a for statement (line 359)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Getting the type of 'c' (line 360)
        c_758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 15), 'c')
        unicode_759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 360, 20), 'unicode', u'\n')
        # Applying the binary operator '==' (line 360)
        result_eq_760 = python_operator(stypy.reporting.localization.Localization(__file__, 360, 15), '==', c_758, unicode_759)
        
        # Testing the type of an if condition (line 360)
        if_condition_761 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 360, 12), result_eq_760)
        # Assigning a type to the variable 'if_condition_761' (line 360)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 360, 12), 'if_condition_761', if_condition_761)
        # SSA begins for if statement (line 360)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # SSA join for if statement (line 360)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Subscript to a Tuple (line 362):
        
        # Assigning a Subscript to a Name (line 362):
        
        # Obtaining the type of the subscript
        int_762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 12), 'int')
        
        # Obtaining the type of the subscript
        
        # Call to ord(...): (line 362)
        # Processing the call arguments (line 362)
        # Getting the type of 'c' (line 362)
        c_764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 47), 'c', False)
        # Processing the call keyword arguments (line 362)
        kwargs_765 = {}
        # Getting the type of 'ord' (line 362)
        ord_763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 43), 'ord', False)
        # Calling ord(args, kwargs) (line 362)
        ord_call_result_766 = invoke(stypy.reporting.localization.Localization(__file__, 362, 43), ord_763, *[c_764], **kwargs_765)
        
        # Getting the type of 'self' (line 362)
        self_767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 29), 'self')
        # Obtaining the member '_metrics' of a type (line 362)
        _metrics_768 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 29), self_767, '_metrics')
        # Obtaining the member '__getitem__' of a type (line 362)
        getitem___769 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 29), _metrics_768, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 362)
        subscript_call_result_770 = invoke(stypy.reporting.localization.Localization(__file__, 362, 29), getitem___769, ord_call_result_766)
        
        # Obtaining the member '__getitem__' of a type (line 362)
        getitem___771 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 12), subscript_call_result_770, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 362)
        subscript_call_result_772 = invoke(stypy.reporting.localization.Localization(__file__, 362, 12), getitem___771, int_762)
        
        # Assigning a type to the variable 'tuple_var_assignment_21' (line 362)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 12), 'tuple_var_assignment_21', subscript_call_result_772)
        
        # Assigning a Subscript to a Name (line 362):
        
        # Obtaining the type of the subscript
        int_773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 12), 'int')
        
        # Obtaining the type of the subscript
        
        # Call to ord(...): (line 362)
        # Processing the call arguments (line 362)
        # Getting the type of 'c' (line 362)
        c_775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 47), 'c', False)
        # Processing the call keyword arguments (line 362)
        kwargs_776 = {}
        # Getting the type of 'ord' (line 362)
        ord_774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 43), 'ord', False)
        # Calling ord(args, kwargs) (line 362)
        ord_call_result_777 = invoke(stypy.reporting.localization.Localization(__file__, 362, 43), ord_774, *[c_775], **kwargs_776)
        
        # Getting the type of 'self' (line 362)
        self_778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 29), 'self')
        # Obtaining the member '_metrics' of a type (line 362)
        _metrics_779 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 29), self_778, '_metrics')
        # Obtaining the member '__getitem__' of a type (line 362)
        getitem___780 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 29), _metrics_779, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 362)
        subscript_call_result_781 = invoke(stypy.reporting.localization.Localization(__file__, 362, 29), getitem___780, ord_call_result_777)
        
        # Obtaining the member '__getitem__' of a type (line 362)
        getitem___782 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 12), subscript_call_result_781, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 362)
        subscript_call_result_783 = invoke(stypy.reporting.localization.Localization(__file__, 362, 12), getitem___782, int_773)
        
        # Assigning a type to the variable 'tuple_var_assignment_22' (line 362)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 12), 'tuple_var_assignment_22', subscript_call_result_783)
        
        # Assigning a Subscript to a Name (line 362):
        
        # Obtaining the type of the subscript
        int_784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 12), 'int')
        
        # Obtaining the type of the subscript
        
        # Call to ord(...): (line 362)
        # Processing the call arguments (line 362)
        # Getting the type of 'c' (line 362)
        c_786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 47), 'c', False)
        # Processing the call keyword arguments (line 362)
        kwargs_787 = {}
        # Getting the type of 'ord' (line 362)
        ord_785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 43), 'ord', False)
        # Calling ord(args, kwargs) (line 362)
        ord_call_result_788 = invoke(stypy.reporting.localization.Localization(__file__, 362, 43), ord_785, *[c_786], **kwargs_787)
        
        # Getting the type of 'self' (line 362)
        self_789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 29), 'self')
        # Obtaining the member '_metrics' of a type (line 362)
        _metrics_790 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 29), self_789, '_metrics')
        # Obtaining the member '__getitem__' of a type (line 362)
        getitem___791 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 29), _metrics_790, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 362)
        subscript_call_result_792 = invoke(stypy.reporting.localization.Localization(__file__, 362, 29), getitem___791, ord_call_result_788)
        
        # Obtaining the member '__getitem__' of a type (line 362)
        getitem___793 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 12), subscript_call_result_792, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 362)
        subscript_call_result_794 = invoke(stypy.reporting.localization.Localization(__file__, 362, 12), getitem___793, int_784)
        
        # Assigning a type to the variable 'tuple_var_assignment_23' (line 362)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 12), 'tuple_var_assignment_23', subscript_call_result_794)
        
        # Assigning a Name to a Name (line 362):
        # Getting the type of 'tuple_var_assignment_21' (line 362)
        tuple_var_assignment_21_795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 12), 'tuple_var_assignment_21')
        # Assigning a type to the variable 'wx' (line 362)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 12), 'wx', tuple_var_assignment_21_795)
        
        # Assigning a Name to a Name (line 362):
        # Getting the type of 'tuple_var_assignment_22' (line 362)
        tuple_var_assignment_22_796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 12), 'tuple_var_assignment_22')
        # Assigning a type to the variable 'name' (line 362)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 16), 'name', tuple_var_assignment_22_796)
        
        # Assigning a Name to a Name (line 362):
        # Getting the type of 'tuple_var_assignment_23' (line 362)
        tuple_var_assignment_23_797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 12), 'tuple_var_assignment_23')
        # Assigning a type to the variable 'bbox' (line 362)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 22), 'bbox', tuple_var_assignment_23_797)
        
        # Assigning a Name to a Tuple (line 363):
        
        # Assigning a Subscript to a Name (line 363):
        
        # Obtaining the type of the subscript
        int_798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 363, 12), 'int')
        # Getting the type of 'bbox' (line 363)
        bbox_799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 25), 'bbox')
        # Obtaining the member '__getitem__' of a type (line 363)
        getitem___800 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 363, 12), bbox_799, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 363)
        subscript_call_result_801 = invoke(stypy.reporting.localization.Localization(__file__, 363, 12), getitem___800, int_798)
        
        # Assigning a type to the variable 'tuple_var_assignment_24' (line 363)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 12), 'tuple_var_assignment_24', subscript_call_result_801)
        
        # Assigning a Subscript to a Name (line 363):
        
        # Obtaining the type of the subscript
        int_802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 363, 12), 'int')
        # Getting the type of 'bbox' (line 363)
        bbox_803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 25), 'bbox')
        # Obtaining the member '__getitem__' of a type (line 363)
        getitem___804 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 363, 12), bbox_803, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 363)
        subscript_call_result_805 = invoke(stypy.reporting.localization.Localization(__file__, 363, 12), getitem___804, int_802)
        
        # Assigning a type to the variable 'tuple_var_assignment_25' (line 363)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 12), 'tuple_var_assignment_25', subscript_call_result_805)
        
        # Assigning a Subscript to a Name (line 363):
        
        # Obtaining the type of the subscript
        int_806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 363, 12), 'int')
        # Getting the type of 'bbox' (line 363)
        bbox_807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 25), 'bbox')
        # Obtaining the member '__getitem__' of a type (line 363)
        getitem___808 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 363, 12), bbox_807, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 363)
        subscript_call_result_809 = invoke(stypy.reporting.localization.Localization(__file__, 363, 12), getitem___808, int_806)
        
        # Assigning a type to the variable 'tuple_var_assignment_26' (line 363)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 12), 'tuple_var_assignment_26', subscript_call_result_809)
        
        # Assigning a Subscript to a Name (line 363):
        
        # Obtaining the type of the subscript
        int_810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 363, 12), 'int')
        # Getting the type of 'bbox' (line 363)
        bbox_811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 25), 'bbox')
        # Obtaining the member '__getitem__' of a type (line 363)
        getitem___812 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 363, 12), bbox_811, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 363)
        subscript_call_result_813 = invoke(stypy.reporting.localization.Localization(__file__, 363, 12), getitem___812, int_810)
        
        # Assigning a type to the variable 'tuple_var_assignment_27' (line 363)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 12), 'tuple_var_assignment_27', subscript_call_result_813)
        
        # Assigning a Name to a Name (line 363):
        # Getting the type of 'tuple_var_assignment_24' (line 363)
        tuple_var_assignment_24_814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 12), 'tuple_var_assignment_24')
        # Assigning a type to the variable 'l' (line 363)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 12), 'l', tuple_var_assignment_24_814)
        
        # Assigning a Name to a Name (line 363):
        # Getting the type of 'tuple_var_assignment_25' (line 363)
        tuple_var_assignment_25_815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 12), 'tuple_var_assignment_25')
        # Assigning a type to the variable 'b' (line 363)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 15), 'b', tuple_var_assignment_25_815)
        
        # Assigning a Name to a Name (line 363):
        # Getting the type of 'tuple_var_assignment_26' (line 363)
        tuple_var_assignment_26_816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 12), 'tuple_var_assignment_26')
        # Assigning a type to the variable 'w' (line 363)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 18), 'w', tuple_var_assignment_26_816)
        
        # Assigning a Name to a Name (line 363):
        # Getting the type of 'tuple_var_assignment_27' (line 363)
        tuple_var_assignment_27_817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 12), 'tuple_var_assignment_27')
        # Assigning a type to the variable 'h' (line 363)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 21), 'h', tuple_var_assignment_27_817)
        
        
        # SSA begins for try-except statement (line 366)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Subscript to a Name (line 367):
        
        # Assigning a Subscript to a Name (line 367):
        
        # Obtaining the type of the subscript
        
        # Obtaining an instance of the builtin type 'tuple' (line 367)
        tuple_818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 367, 33), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 367)
        # Adding element type (line 367)
        # Getting the type of 'namelast' (line 367)
        namelast_819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 33), 'namelast')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 367, 33), tuple_818, namelast_819)
        # Adding element type (line 367)
        # Getting the type of 'name' (line 367)
        name_820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 43), 'name')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 367, 33), tuple_818, name_820)
        
        # Getting the type of 'self' (line 367)
        self_821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 21), 'self')
        # Obtaining the member '_kern' of a type (line 367)
        _kern_822 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 367, 21), self_821, '_kern')
        # Obtaining the member '__getitem__' of a type (line 367)
        getitem___823 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 367, 21), _kern_822, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 367)
        subscript_call_result_824 = invoke(stypy.reporting.localization.Localization(__file__, 367, 21), getitem___823, tuple_818)
        
        # Assigning a type to the variable 'kp' (line 367)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 367, 16), 'kp', subscript_call_result_824)
        # SSA branch for the except part of a try statement (line 366)
        # SSA branch for the except 'KeyError' branch of a try statement (line 366)
        module_type_store.open_ssa_branch('except')
        
        # Assigning a Num to a Name (line 369):
        
        # Assigning a Num to a Name (line 369):
        int_825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 369, 21), 'int')
        # Assigning a type to the variable 'kp' (line 369)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 369, 16), 'kp', int_825)
        # SSA join for try-except statement (line 366)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'totalw' (line 370)
        totalw_826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 12), 'totalw')
        # Getting the type of 'wx' (line 370)
        wx_827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 22), 'wx')
        # Getting the type of 'kp' (line 370)
        kp_828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 27), 'kp')
        # Applying the binary operator '+' (line 370)
        result_add_829 = python_operator(stypy.reporting.localization.Localization(__file__, 370, 22), '+', wx_827, kp_828)
        
        # Applying the binary operator '+=' (line 370)
        result_iadd_830 = python_operator(stypy.reporting.localization.Localization(__file__, 370, 12), '+=', totalw_826, result_add_829)
        # Assigning a type to the variable 'totalw' (line 370)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 12), 'totalw', result_iadd_830)
        
        
        # Assigning a BinOp to a Name (line 373):
        
        # Assigning a BinOp to a Name (line 373):
        # Getting the type of 'b' (line 373)
        b_831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 22), 'b')
        # Getting the type of 'h' (line 373)
        h_832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 26), 'h')
        # Applying the binary operator '+' (line 373)
        result_add_833 = python_operator(stypy.reporting.localization.Localization(__file__, 373, 22), '+', b_831, h_832)
        
        # Assigning a type to the variable 'thismax' (line 373)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 373, 12), 'thismax', result_add_833)
        
        
        # Getting the type of 'thismax' (line 374)
        thismax_834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 15), 'thismax')
        # Getting the type of 'maxy' (line 374)
        maxy_835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 25), 'maxy')
        # Applying the binary operator '>' (line 374)
        result_gt_836 = python_operator(stypy.reporting.localization.Localization(__file__, 374, 15), '>', thismax_834, maxy_835)
        
        # Testing the type of an if condition (line 374)
        if_condition_837 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 374, 12), result_gt_836)
        # Assigning a type to the variable 'if_condition_837' (line 374)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 374, 12), 'if_condition_837', if_condition_837)
        # SSA begins for if statement (line 374)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 375):
        
        # Assigning a Name to a Name (line 375):
        # Getting the type of 'thismax' (line 375)
        thismax_838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 23), 'thismax')
        # Assigning a type to the variable 'maxy' (line 375)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 375, 16), 'maxy', thismax_838)
        # SSA join for if statement (line 374)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Name (line 378):
        
        # Assigning a Name to a Name (line 378):
        # Getting the type of 'b' (line 378)
        b_839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 22), 'b')
        # Assigning a type to the variable 'thismin' (line 378)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 378, 12), 'thismin', b_839)
        
        
        # Getting the type of 'thismin' (line 379)
        thismin_840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 15), 'thismin')
        # Getting the type of 'miny' (line 379)
        miny_841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 25), 'miny')
        # Applying the binary operator '<' (line 379)
        result_lt_842 = python_operator(stypy.reporting.localization.Localization(__file__, 379, 15), '<', thismin_840, miny_841)
        
        # Testing the type of an if condition (line 379)
        if_condition_843 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 379, 12), result_lt_842)
        # Assigning a type to the variable 'if_condition_843' (line 379)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 379, 12), 'if_condition_843', if_condition_843)
        # SSA begins for if statement (line 379)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 380):
        
        # Assigning a Name to a Name (line 380):
        # Getting the type of 'thismin' (line 380)
        thismin_844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 23), 'thismin')
        # Assigning a type to the variable 'miny' (line 380)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 380, 16), 'miny', thismin_844)
        # SSA join for if statement (line 379)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Name (line 381):
        
        # Assigning a Name to a Name (line 381):
        # Getting the type of 'name' (line 381)
        name_845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 23), 'name')
        # Assigning a type to the variable 'namelast' (line 381)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 381, 12), 'namelast', name_845)
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 383)
        tuple_846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 383)
        # Adding element type (line 383)
        # Getting the type of 'totalw' (line 383)
        totalw_847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 15), 'totalw')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 383, 15), tuple_846, totalw_847)
        # Adding element type (line 383)
        # Getting the type of 'maxy' (line 383)
        maxy_848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 23), 'maxy')
        # Getting the type of 'miny' (line 383)
        miny_849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 30), 'miny')
        # Applying the binary operator '-' (line 383)
        result_sub_850 = python_operator(stypy.reporting.localization.Localization(__file__, 383, 23), '-', maxy_848, miny_849)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 383, 15), tuple_846, result_sub_850)
        
        # Assigning a type to the variable 'stypy_return_type' (line 383)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 8), 'stypy_return_type', tuple_846)
        
        # ################# End of 'string_width_height(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'string_width_height' in the type store
        # Getting the type of 'stypy_return_type' (line 348)
        stypy_return_type_851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_851)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'string_width_height'
        return stypy_return_type_851


    @norecursion
    def get_str_bbox_and_descent(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_str_bbox_and_descent'
        module_type_store = module_type_store.open_function_context('get_str_bbox_and_descent', 385, 4, False)
        # Assigning a type to the variable 'self' (line 386)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 386, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        AFM.get_str_bbox_and_descent.__dict__.__setitem__('stypy_localization', localization)
        AFM.get_str_bbox_and_descent.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        AFM.get_str_bbox_and_descent.__dict__.__setitem__('stypy_type_store', module_type_store)
        AFM.get_str_bbox_and_descent.__dict__.__setitem__('stypy_function_name', 'AFM.get_str_bbox_and_descent')
        AFM.get_str_bbox_and_descent.__dict__.__setitem__('stypy_param_names_list', ['s'])
        AFM.get_str_bbox_and_descent.__dict__.__setitem__('stypy_varargs_param_name', None)
        AFM.get_str_bbox_and_descent.__dict__.__setitem__('stypy_kwargs_param_name', None)
        AFM.get_str_bbox_and_descent.__dict__.__setitem__('stypy_call_defaults', defaults)
        AFM.get_str_bbox_and_descent.__dict__.__setitem__('stypy_call_varargs', varargs)
        AFM.get_str_bbox_and_descent.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        AFM.get_str_bbox_and_descent.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'AFM.get_str_bbox_and_descent', ['s'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_str_bbox_and_descent', localization, ['s'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_str_bbox_and_descent(...)' code ##################

        unicode_852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 388, (-1)), 'unicode', u'\n        Return the string bounding box\n        ')
        
        
        
        # Call to len(...): (line 389)
        # Processing the call arguments (line 389)
        # Getting the type of 's' (line 389)
        s_854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 19), 's', False)
        # Processing the call keyword arguments (line 389)
        kwargs_855 = {}
        # Getting the type of 'len' (line 389)
        len_853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 15), 'len', False)
        # Calling len(args, kwargs) (line 389)
        len_call_result_856 = invoke(stypy.reporting.localization.Localization(__file__, 389, 15), len_853, *[s_854], **kwargs_855)
        
        # Applying the 'not' unary operator (line 389)
        result_not__857 = python_operator(stypy.reporting.localization.Localization(__file__, 389, 11), 'not', len_call_result_856)
        
        # Testing the type of an if condition (line 389)
        if_condition_858 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 389, 8), result_not__857)
        # Assigning a type to the variable 'if_condition_858' (line 389)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 389, 8), 'if_condition_858', if_condition_858)
        # SSA begins for if statement (line 389)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Obtaining an instance of the builtin type 'tuple' (line 390)
        tuple_859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 19), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 390)
        # Adding element type (line 390)
        int_860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 390, 19), tuple_859, int_860)
        # Adding element type (line 390)
        int_861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 390, 19), tuple_859, int_861)
        # Adding element type (line 390)
        int_862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 390, 19), tuple_859, int_862)
        # Adding element type (line 390)
        int_863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 390, 19), tuple_859, int_863)
        
        # Assigning a type to the variable 'stypy_return_type' (line 390)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 390, 12), 'stypy_return_type', tuple_859)
        # SSA join for if statement (line 389)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Num to a Name (line 391):
        
        # Assigning a Num to a Name (line 391):
        int_864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 391, 17), 'int')
        # Assigning a type to the variable 'totalw' (line 391)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 391, 8), 'totalw', int_864)
        
        # Assigning a Name to a Name (line 392):
        
        # Assigning a Name to a Name (line 392):
        # Getting the type of 'None' (line 392)
        None_865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 19), 'None')
        # Assigning a type to the variable 'namelast' (line 392)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 392, 8), 'namelast', None_865)
        
        # Assigning a Num to a Name (line 393):
        
        # Assigning a Num to a Name (line 393):
        float_866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 15), 'float')
        # Assigning a type to the variable 'miny' (line 393)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 393, 8), 'miny', float_866)
        
        # Assigning a Num to a Name (line 394):
        
        # Assigning a Num to a Name (line 394):
        int_867 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 15), 'int')
        # Assigning a type to the variable 'maxy' (line 394)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 394, 8), 'maxy', int_867)
        
        # Assigning a Num to a Name (line 395):
        
        # Assigning a Num to a Name (line 395):
        int_868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 395, 15), 'int')
        # Assigning a type to the variable 'left' (line 395)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 395, 8), 'left', int_868)
        
        
        
        # Call to isinstance(...): (line 396)
        # Processing the call arguments (line 396)
        # Getting the type of 's' (line 396)
        s_870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 26), 's', False)
        # Getting the type of 'six' (line 396)
        six_871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 29), 'six', False)
        # Obtaining the member 'text_type' of a type (line 396)
        text_type_872 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 396, 29), six_871, 'text_type')
        # Processing the call keyword arguments (line 396)
        kwargs_873 = {}
        # Getting the type of 'isinstance' (line 396)
        isinstance_869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 15), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 396)
        isinstance_call_result_874 = invoke(stypy.reporting.localization.Localization(__file__, 396, 15), isinstance_869, *[s_870, text_type_872], **kwargs_873)
        
        # Applying the 'not' unary operator (line 396)
        result_not__875 = python_operator(stypy.reporting.localization.Localization(__file__, 396, 11), 'not', isinstance_call_result_874)
        
        # Testing the type of an if condition (line 396)
        if_condition_876 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 396, 8), result_not__875)
        # Assigning a type to the variable 'if_condition_876' (line 396)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 396, 8), 'if_condition_876', if_condition_876)
        # SSA begins for if statement (line 396)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 397):
        
        # Assigning a Call to a Name (line 397):
        
        # Call to _to_str(...): (line 397)
        # Processing the call arguments (line 397)
        # Getting the type of 's' (line 397)
        s_878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 24), 's', False)
        # Processing the call keyword arguments (line 397)
        kwargs_879 = {}
        # Getting the type of '_to_str' (line 397)
        _to_str_877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 16), '_to_str', False)
        # Calling _to_str(args, kwargs) (line 397)
        _to_str_call_result_880 = invoke(stypy.reporting.localization.Localization(__file__, 397, 16), _to_str_877, *[s_878], **kwargs_879)
        
        # Assigning a type to the variable 's' (line 397)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 397, 12), 's', _to_str_call_result_880)
        # SSA join for if statement (line 396)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 's' (line 398)
        s_881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 17), 's')
        # Testing the type of a for loop iterable (line 398)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 398, 8), s_881)
        # Getting the type of the for loop variable (line 398)
        for_loop_var_882 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 398, 8), s_881)
        # Assigning a type to the variable 'c' (line 398)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 398, 8), 'c', for_loop_var_882)
        # SSA begins for a for statement (line 398)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Getting the type of 'c' (line 399)
        c_883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 15), 'c')
        unicode_884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 20), 'unicode', u'\n')
        # Applying the binary operator '==' (line 399)
        result_eq_885 = python_operator(stypy.reporting.localization.Localization(__file__, 399, 15), '==', c_883, unicode_884)
        
        # Testing the type of an if condition (line 399)
        if_condition_886 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 399, 12), result_eq_885)
        # Assigning a type to the variable 'if_condition_886' (line 399)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 399, 12), 'if_condition_886', if_condition_886)
        # SSA begins for if statement (line 399)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # SSA join for if statement (line 399)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 401):
        
        # Assigning a Call to a Name (line 401):
        
        # Call to get(...): (line 401)
        # Processing the call arguments (line 401)
        
        # Call to ord(...): (line 401)
        # Processing the call arguments (line 401)
        # Getting the type of 'c' (line 401)
        c_890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 37), 'c', False)
        # Processing the call keyword arguments (line 401)
        kwargs_891 = {}
        # Getting the type of 'ord' (line 401)
        ord_889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 33), 'ord', False)
        # Calling ord(args, kwargs) (line 401)
        ord_call_result_892 = invoke(stypy.reporting.localization.Localization(__file__, 401, 33), ord_889, *[c_890], **kwargs_891)
        
        unicode_893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 401, 41), 'unicode', u'question')
        # Processing the call keyword arguments (line 401)
        kwargs_894 = {}
        # Getting the type of 'uni2type1' (line 401)
        uni2type1_887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 19), 'uni2type1', False)
        # Obtaining the member 'get' of a type (line 401)
        get_888 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 401, 19), uni2type1_887, 'get')
        # Calling get(args, kwargs) (line 401)
        get_call_result_895 = invoke(stypy.reporting.localization.Localization(__file__, 401, 19), get_888, *[ord_call_result_892, unicode_893], **kwargs_894)
        
        # Assigning a type to the variable 'name' (line 401)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 401, 12), 'name', get_call_result_895)
        
        
        # SSA begins for try-except statement (line 402)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Subscript to a Tuple (line 403):
        
        # Assigning a Subscript to a Name (line 403):
        
        # Obtaining the type of the subscript
        int_896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 403, 16), 'int')
        
        # Obtaining the type of the subscript
        # Getting the type of 'name' (line 403)
        name_897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 49), 'name')
        # Getting the type of 'self' (line 403)
        self_898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 27), 'self')
        # Obtaining the member '_metrics_by_name' of a type (line 403)
        _metrics_by_name_899 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 403, 27), self_898, '_metrics_by_name')
        # Obtaining the member '__getitem__' of a type (line 403)
        getitem___900 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 403, 27), _metrics_by_name_899, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 403)
        subscript_call_result_901 = invoke(stypy.reporting.localization.Localization(__file__, 403, 27), getitem___900, name_897)
        
        # Obtaining the member '__getitem__' of a type (line 403)
        getitem___902 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 403, 16), subscript_call_result_901, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 403)
        subscript_call_result_903 = invoke(stypy.reporting.localization.Localization(__file__, 403, 16), getitem___902, int_896)
        
        # Assigning a type to the variable 'tuple_var_assignment_28' (line 403)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 403, 16), 'tuple_var_assignment_28', subscript_call_result_903)
        
        # Assigning a Subscript to a Name (line 403):
        
        # Obtaining the type of the subscript
        int_904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 403, 16), 'int')
        
        # Obtaining the type of the subscript
        # Getting the type of 'name' (line 403)
        name_905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 49), 'name')
        # Getting the type of 'self' (line 403)
        self_906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 27), 'self')
        # Obtaining the member '_metrics_by_name' of a type (line 403)
        _metrics_by_name_907 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 403, 27), self_906, '_metrics_by_name')
        # Obtaining the member '__getitem__' of a type (line 403)
        getitem___908 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 403, 27), _metrics_by_name_907, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 403)
        subscript_call_result_909 = invoke(stypy.reporting.localization.Localization(__file__, 403, 27), getitem___908, name_905)
        
        # Obtaining the member '__getitem__' of a type (line 403)
        getitem___910 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 403, 16), subscript_call_result_909, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 403)
        subscript_call_result_911 = invoke(stypy.reporting.localization.Localization(__file__, 403, 16), getitem___910, int_904)
        
        # Assigning a type to the variable 'tuple_var_assignment_29' (line 403)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 403, 16), 'tuple_var_assignment_29', subscript_call_result_911)
        
        # Assigning a Name to a Name (line 403):
        # Getting the type of 'tuple_var_assignment_28' (line 403)
        tuple_var_assignment_28_912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 16), 'tuple_var_assignment_28')
        # Assigning a type to the variable 'wx' (line 403)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 403, 16), 'wx', tuple_var_assignment_28_912)
        
        # Assigning a Name to a Name (line 403):
        # Getting the type of 'tuple_var_assignment_29' (line 403)
        tuple_var_assignment_29_913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 16), 'tuple_var_assignment_29')
        # Assigning a type to the variable 'bbox' (line 403)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 403, 20), 'bbox', tuple_var_assignment_29_913)
        # SSA branch for the except part of a try statement (line 402)
        # SSA branch for the except 'KeyError' branch of a try statement (line 402)
        module_type_store.open_ssa_branch('except')
        
        # Assigning a Str to a Name (line 405):
        
        # Assigning a Str to a Name (line 405):
        unicode_914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 23), 'unicode', u'question')
        # Assigning a type to the variable 'name' (line 405)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 405, 16), 'name', unicode_914)
        
        # Assigning a Subscript to a Tuple (line 406):
        
        # Assigning a Subscript to a Name (line 406):
        
        # Obtaining the type of the subscript
        int_915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 16), 'int')
        
        # Obtaining the type of the subscript
        # Getting the type of 'name' (line 406)
        name_916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 49), 'name')
        # Getting the type of 'self' (line 406)
        self_917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 27), 'self')
        # Obtaining the member '_metrics_by_name' of a type (line 406)
        _metrics_by_name_918 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 27), self_917, '_metrics_by_name')
        # Obtaining the member '__getitem__' of a type (line 406)
        getitem___919 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 27), _metrics_by_name_918, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 406)
        subscript_call_result_920 = invoke(stypy.reporting.localization.Localization(__file__, 406, 27), getitem___919, name_916)
        
        # Obtaining the member '__getitem__' of a type (line 406)
        getitem___921 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 16), subscript_call_result_920, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 406)
        subscript_call_result_922 = invoke(stypy.reporting.localization.Localization(__file__, 406, 16), getitem___921, int_915)
        
        # Assigning a type to the variable 'tuple_var_assignment_30' (line 406)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 406, 16), 'tuple_var_assignment_30', subscript_call_result_922)
        
        # Assigning a Subscript to a Name (line 406):
        
        # Obtaining the type of the subscript
        int_923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 16), 'int')
        
        # Obtaining the type of the subscript
        # Getting the type of 'name' (line 406)
        name_924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 49), 'name')
        # Getting the type of 'self' (line 406)
        self_925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 27), 'self')
        # Obtaining the member '_metrics_by_name' of a type (line 406)
        _metrics_by_name_926 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 27), self_925, '_metrics_by_name')
        # Obtaining the member '__getitem__' of a type (line 406)
        getitem___927 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 27), _metrics_by_name_926, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 406)
        subscript_call_result_928 = invoke(stypy.reporting.localization.Localization(__file__, 406, 27), getitem___927, name_924)
        
        # Obtaining the member '__getitem__' of a type (line 406)
        getitem___929 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 16), subscript_call_result_928, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 406)
        subscript_call_result_930 = invoke(stypy.reporting.localization.Localization(__file__, 406, 16), getitem___929, int_923)
        
        # Assigning a type to the variable 'tuple_var_assignment_31' (line 406)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 406, 16), 'tuple_var_assignment_31', subscript_call_result_930)
        
        # Assigning a Name to a Name (line 406):
        # Getting the type of 'tuple_var_assignment_30' (line 406)
        tuple_var_assignment_30_931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 16), 'tuple_var_assignment_30')
        # Assigning a type to the variable 'wx' (line 406)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 406, 16), 'wx', tuple_var_assignment_30_931)
        
        # Assigning a Name to a Name (line 406):
        # Getting the type of 'tuple_var_assignment_31' (line 406)
        tuple_var_assignment_31_932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 16), 'tuple_var_assignment_31')
        # Assigning a type to the variable 'bbox' (line 406)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 406, 20), 'bbox', tuple_var_assignment_31_932)
        # SSA join for try-except statement (line 402)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Tuple (line 407):
        
        # Assigning a Subscript to a Name (line 407):
        
        # Obtaining the type of the subscript
        int_933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 407, 12), 'int')
        # Getting the type of 'bbox' (line 407)
        bbox_934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 25), 'bbox')
        # Obtaining the member '__getitem__' of a type (line 407)
        getitem___935 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 407, 12), bbox_934, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 407)
        subscript_call_result_936 = invoke(stypy.reporting.localization.Localization(__file__, 407, 12), getitem___935, int_933)
        
        # Assigning a type to the variable 'tuple_var_assignment_32' (line 407)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 407, 12), 'tuple_var_assignment_32', subscript_call_result_936)
        
        # Assigning a Subscript to a Name (line 407):
        
        # Obtaining the type of the subscript
        int_937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 407, 12), 'int')
        # Getting the type of 'bbox' (line 407)
        bbox_938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 25), 'bbox')
        # Obtaining the member '__getitem__' of a type (line 407)
        getitem___939 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 407, 12), bbox_938, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 407)
        subscript_call_result_940 = invoke(stypy.reporting.localization.Localization(__file__, 407, 12), getitem___939, int_937)
        
        # Assigning a type to the variable 'tuple_var_assignment_33' (line 407)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 407, 12), 'tuple_var_assignment_33', subscript_call_result_940)
        
        # Assigning a Subscript to a Name (line 407):
        
        # Obtaining the type of the subscript
        int_941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 407, 12), 'int')
        # Getting the type of 'bbox' (line 407)
        bbox_942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 25), 'bbox')
        # Obtaining the member '__getitem__' of a type (line 407)
        getitem___943 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 407, 12), bbox_942, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 407)
        subscript_call_result_944 = invoke(stypy.reporting.localization.Localization(__file__, 407, 12), getitem___943, int_941)
        
        # Assigning a type to the variable 'tuple_var_assignment_34' (line 407)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 407, 12), 'tuple_var_assignment_34', subscript_call_result_944)
        
        # Assigning a Subscript to a Name (line 407):
        
        # Obtaining the type of the subscript
        int_945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 407, 12), 'int')
        # Getting the type of 'bbox' (line 407)
        bbox_946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 25), 'bbox')
        # Obtaining the member '__getitem__' of a type (line 407)
        getitem___947 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 407, 12), bbox_946, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 407)
        subscript_call_result_948 = invoke(stypy.reporting.localization.Localization(__file__, 407, 12), getitem___947, int_945)
        
        # Assigning a type to the variable 'tuple_var_assignment_35' (line 407)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 407, 12), 'tuple_var_assignment_35', subscript_call_result_948)
        
        # Assigning a Name to a Name (line 407):
        # Getting the type of 'tuple_var_assignment_32' (line 407)
        tuple_var_assignment_32_949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 12), 'tuple_var_assignment_32')
        # Assigning a type to the variable 'l' (line 407)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 407, 12), 'l', tuple_var_assignment_32_949)
        
        # Assigning a Name to a Name (line 407):
        # Getting the type of 'tuple_var_assignment_33' (line 407)
        tuple_var_assignment_33_950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 12), 'tuple_var_assignment_33')
        # Assigning a type to the variable 'b' (line 407)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 407, 15), 'b', tuple_var_assignment_33_950)
        
        # Assigning a Name to a Name (line 407):
        # Getting the type of 'tuple_var_assignment_34' (line 407)
        tuple_var_assignment_34_951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 12), 'tuple_var_assignment_34')
        # Assigning a type to the variable 'w' (line 407)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 407, 18), 'w', tuple_var_assignment_34_951)
        
        # Assigning a Name to a Name (line 407):
        # Getting the type of 'tuple_var_assignment_35' (line 407)
        tuple_var_assignment_35_952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 12), 'tuple_var_assignment_35')
        # Assigning a type to the variable 'h' (line 407)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 407, 21), 'h', tuple_var_assignment_35_952)
        
        
        # Getting the type of 'l' (line 408)
        l_953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 15), 'l')
        # Getting the type of 'left' (line 408)
        left_954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 19), 'left')
        # Applying the binary operator '<' (line 408)
        result_lt_955 = python_operator(stypy.reporting.localization.Localization(__file__, 408, 15), '<', l_953, left_954)
        
        # Testing the type of an if condition (line 408)
        if_condition_956 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 408, 12), result_lt_955)
        # Assigning a type to the variable 'if_condition_956' (line 408)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 408, 12), 'if_condition_956', if_condition_956)
        # SSA begins for if statement (line 408)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 409):
        
        # Assigning a Name to a Name (line 409):
        # Getting the type of 'l' (line 409)
        l_957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 23), 'l')
        # Assigning a type to the variable 'left' (line 409)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 409, 16), 'left', l_957)
        # SSA join for if statement (line 408)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # SSA begins for try-except statement (line 411)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Subscript to a Name (line 412):
        
        # Assigning a Subscript to a Name (line 412):
        
        # Obtaining the type of the subscript
        
        # Obtaining an instance of the builtin type 'tuple' (line 412)
        tuple_958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 412, 33), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 412)
        # Adding element type (line 412)
        # Getting the type of 'namelast' (line 412)
        namelast_959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 33), 'namelast')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 412, 33), tuple_958, namelast_959)
        # Adding element type (line 412)
        # Getting the type of 'name' (line 412)
        name_960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 43), 'name')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 412, 33), tuple_958, name_960)
        
        # Getting the type of 'self' (line 412)
        self_961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 21), 'self')
        # Obtaining the member '_kern' of a type (line 412)
        _kern_962 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 412, 21), self_961, '_kern')
        # Obtaining the member '__getitem__' of a type (line 412)
        getitem___963 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 412, 21), _kern_962, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 412)
        subscript_call_result_964 = invoke(stypy.reporting.localization.Localization(__file__, 412, 21), getitem___963, tuple_958)
        
        # Assigning a type to the variable 'kp' (line 412)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 412, 16), 'kp', subscript_call_result_964)
        # SSA branch for the except part of a try statement (line 411)
        # SSA branch for the except 'KeyError' branch of a try statement (line 411)
        module_type_store.open_ssa_branch('except')
        
        # Assigning a Num to a Name (line 414):
        
        # Assigning a Num to a Name (line 414):
        int_965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 414, 21), 'int')
        # Assigning a type to the variable 'kp' (line 414)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 414, 16), 'kp', int_965)
        # SSA join for try-except statement (line 411)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'totalw' (line 415)
        totalw_966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 12), 'totalw')
        # Getting the type of 'wx' (line 415)
        wx_967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 22), 'wx')
        # Getting the type of 'kp' (line 415)
        kp_968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 27), 'kp')
        # Applying the binary operator '+' (line 415)
        result_add_969 = python_operator(stypy.reporting.localization.Localization(__file__, 415, 22), '+', wx_967, kp_968)
        
        # Applying the binary operator '+=' (line 415)
        result_iadd_970 = python_operator(stypy.reporting.localization.Localization(__file__, 415, 12), '+=', totalw_966, result_add_969)
        # Assigning a type to the variable 'totalw' (line 415)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 415, 12), 'totalw', result_iadd_970)
        
        
        # Assigning a BinOp to a Name (line 418):
        
        # Assigning a BinOp to a Name (line 418):
        # Getting the type of 'b' (line 418)
        b_971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 22), 'b')
        # Getting the type of 'h' (line 418)
        h_972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 26), 'h')
        # Applying the binary operator '+' (line 418)
        result_add_973 = python_operator(stypy.reporting.localization.Localization(__file__, 418, 22), '+', b_971, h_972)
        
        # Assigning a type to the variable 'thismax' (line 418)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 418, 12), 'thismax', result_add_973)
        
        
        # Getting the type of 'thismax' (line 419)
        thismax_974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 15), 'thismax')
        # Getting the type of 'maxy' (line 419)
        maxy_975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 25), 'maxy')
        # Applying the binary operator '>' (line 419)
        result_gt_976 = python_operator(stypy.reporting.localization.Localization(__file__, 419, 15), '>', thismax_974, maxy_975)
        
        # Testing the type of an if condition (line 419)
        if_condition_977 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 419, 12), result_gt_976)
        # Assigning a type to the variable 'if_condition_977' (line 419)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 419, 12), 'if_condition_977', if_condition_977)
        # SSA begins for if statement (line 419)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 420):
        
        # Assigning a Name to a Name (line 420):
        # Getting the type of 'thismax' (line 420)
        thismax_978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 23), 'thismax')
        # Assigning a type to the variable 'maxy' (line 420)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 420, 16), 'maxy', thismax_978)
        # SSA join for if statement (line 419)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Name (line 423):
        
        # Assigning a Name to a Name (line 423):
        # Getting the type of 'b' (line 423)
        b_979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 22), 'b')
        # Assigning a type to the variable 'thismin' (line 423)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 423, 12), 'thismin', b_979)
        
        
        # Getting the type of 'thismin' (line 424)
        thismin_980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 15), 'thismin')
        # Getting the type of 'miny' (line 424)
        miny_981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 25), 'miny')
        # Applying the binary operator '<' (line 424)
        result_lt_982 = python_operator(stypy.reporting.localization.Localization(__file__, 424, 15), '<', thismin_980, miny_981)
        
        # Testing the type of an if condition (line 424)
        if_condition_983 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 424, 12), result_lt_982)
        # Assigning a type to the variable 'if_condition_983' (line 424)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 424, 12), 'if_condition_983', if_condition_983)
        # SSA begins for if statement (line 424)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 425):
        
        # Assigning a Name to a Name (line 425):
        # Getting the type of 'thismin' (line 425)
        thismin_984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 23), 'thismin')
        # Assigning a type to the variable 'miny' (line 425)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 425, 16), 'miny', thismin_984)
        # SSA join for if statement (line 424)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Name (line 426):
        
        # Assigning a Name to a Name (line 426):
        # Getting the type of 'name' (line 426)
        name_985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 23), 'name')
        # Assigning a type to the variable 'namelast' (line 426)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 426, 12), 'namelast', name_985)
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 428)
        tuple_986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 428, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 428)
        # Adding element type (line 428)
        # Getting the type of 'left' (line 428)
        left_987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 15), 'left')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 428, 15), tuple_986, left_987)
        # Adding element type (line 428)
        # Getting the type of 'miny' (line 428)
        miny_988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 21), 'miny')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 428, 15), tuple_986, miny_988)
        # Adding element type (line 428)
        # Getting the type of 'totalw' (line 428)
        totalw_989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 27), 'totalw')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 428, 15), tuple_986, totalw_989)
        # Adding element type (line 428)
        # Getting the type of 'maxy' (line 428)
        maxy_990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 35), 'maxy')
        # Getting the type of 'miny' (line 428)
        miny_991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 42), 'miny')
        # Applying the binary operator '-' (line 428)
        result_sub_992 = python_operator(stypy.reporting.localization.Localization(__file__, 428, 35), '-', maxy_990, miny_991)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 428, 15), tuple_986, result_sub_992)
        # Adding element type (line 428)
        
        # Getting the type of 'miny' (line 428)
        miny_993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 49), 'miny')
        # Applying the 'usub' unary operator (line 428)
        result___neg___994 = python_operator(stypy.reporting.localization.Localization(__file__, 428, 48), 'usub', miny_993)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 428, 15), tuple_986, result___neg___994)
        
        # Assigning a type to the variable 'stypy_return_type' (line 428)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 428, 8), 'stypy_return_type', tuple_986)
        
        # ################# End of 'get_str_bbox_and_descent(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_str_bbox_and_descent' in the type store
        # Getting the type of 'stypy_return_type' (line 385)
        stypy_return_type_995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_995)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_str_bbox_and_descent'
        return stypy_return_type_995


    @norecursion
    def get_str_bbox(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_str_bbox'
        module_type_store = module_type_store.open_function_context('get_str_bbox', 430, 4, False)
        # Assigning a type to the variable 'self' (line 431)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 431, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        AFM.get_str_bbox.__dict__.__setitem__('stypy_localization', localization)
        AFM.get_str_bbox.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        AFM.get_str_bbox.__dict__.__setitem__('stypy_type_store', module_type_store)
        AFM.get_str_bbox.__dict__.__setitem__('stypy_function_name', 'AFM.get_str_bbox')
        AFM.get_str_bbox.__dict__.__setitem__('stypy_param_names_list', ['s'])
        AFM.get_str_bbox.__dict__.__setitem__('stypy_varargs_param_name', None)
        AFM.get_str_bbox.__dict__.__setitem__('stypy_kwargs_param_name', None)
        AFM.get_str_bbox.__dict__.__setitem__('stypy_call_defaults', defaults)
        AFM.get_str_bbox.__dict__.__setitem__('stypy_call_varargs', varargs)
        AFM.get_str_bbox.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        AFM.get_str_bbox.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'AFM.get_str_bbox', ['s'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_str_bbox', localization, ['s'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_str_bbox(...)' code ##################

        unicode_996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, (-1)), 'unicode', u'\n        Return the string bounding box\n        ')
        
        # Obtaining the type of the subscript
        int_997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, 49), 'int')
        slice_998 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 434, 15), None, int_997, None)
        
        # Call to get_str_bbox_and_descent(...): (line 434)
        # Processing the call arguments (line 434)
        # Getting the type of 's' (line 434)
        s_1001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 45), 's', False)
        # Processing the call keyword arguments (line 434)
        kwargs_1002 = {}
        # Getting the type of 'self' (line 434)
        self_999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 15), 'self', False)
        # Obtaining the member 'get_str_bbox_and_descent' of a type (line 434)
        get_str_bbox_and_descent_1000 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 434, 15), self_999, 'get_str_bbox_and_descent')
        # Calling get_str_bbox_and_descent(args, kwargs) (line 434)
        get_str_bbox_and_descent_call_result_1003 = invoke(stypy.reporting.localization.Localization(__file__, 434, 15), get_str_bbox_and_descent_1000, *[s_1001], **kwargs_1002)
        
        # Obtaining the member '__getitem__' of a type (line 434)
        getitem___1004 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 434, 15), get_str_bbox_and_descent_call_result_1003, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 434)
        subscript_call_result_1005 = invoke(stypy.reporting.localization.Localization(__file__, 434, 15), getitem___1004, slice_998)
        
        # Assigning a type to the variable 'stypy_return_type' (line 434)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 434, 8), 'stypy_return_type', subscript_call_result_1005)
        
        # ################# End of 'get_str_bbox(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_str_bbox' in the type store
        # Getting the type of 'stypy_return_type' (line 430)
        stypy_return_type_1006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1006)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_str_bbox'
        return stypy_return_type_1006


    @norecursion
    def get_name_char(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'False' (line 436)
        False_1007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 37), 'False')
        defaults = [False_1007]
        # Create a new context for function 'get_name_char'
        module_type_store = module_type_store.open_function_context('get_name_char', 436, 4, False)
        # Assigning a type to the variable 'self' (line 437)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 437, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        AFM.get_name_char.__dict__.__setitem__('stypy_localization', localization)
        AFM.get_name_char.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        AFM.get_name_char.__dict__.__setitem__('stypy_type_store', module_type_store)
        AFM.get_name_char.__dict__.__setitem__('stypy_function_name', 'AFM.get_name_char')
        AFM.get_name_char.__dict__.__setitem__('stypy_param_names_list', ['c', 'isord'])
        AFM.get_name_char.__dict__.__setitem__('stypy_varargs_param_name', None)
        AFM.get_name_char.__dict__.__setitem__('stypy_kwargs_param_name', None)
        AFM.get_name_char.__dict__.__setitem__('stypy_call_defaults', defaults)
        AFM.get_name_char.__dict__.__setitem__('stypy_call_varargs', varargs)
        AFM.get_name_char.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        AFM.get_name_char.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'AFM.get_name_char', ['c', 'isord'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_name_char', localization, ['c', 'isord'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_name_char(...)' code ##################

        unicode_1008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 439, (-1)), 'unicode', u"\n        Get the name of the character, i.e., ';' is 'semicolon'\n        ")
        
        
        # Getting the type of 'isord' (line 440)
        isord_1009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 15), 'isord')
        # Applying the 'not' unary operator (line 440)
        result_not__1010 = python_operator(stypy.reporting.localization.Localization(__file__, 440, 11), 'not', isord_1009)
        
        # Testing the type of an if condition (line 440)
        if_condition_1011 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 440, 8), result_not__1010)
        # Assigning a type to the variable 'if_condition_1011' (line 440)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 440, 8), 'if_condition_1011', if_condition_1011)
        # SSA begins for if statement (line 440)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 441):
        
        # Assigning a Call to a Name (line 441):
        
        # Call to ord(...): (line 441)
        # Processing the call arguments (line 441)
        # Getting the type of 'c' (line 441)
        c_1013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 20), 'c', False)
        # Processing the call keyword arguments (line 441)
        kwargs_1014 = {}
        # Getting the type of 'ord' (line 441)
        ord_1012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 16), 'ord', False)
        # Calling ord(args, kwargs) (line 441)
        ord_call_result_1015 = invoke(stypy.reporting.localization.Localization(__file__, 441, 16), ord_1012, *[c_1013], **kwargs_1014)
        
        # Assigning a type to the variable 'c' (line 441)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 441, 12), 'c', ord_call_result_1015)
        # SSA join for if statement (line 440)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Subscript to a Tuple (line 442):
        
        # Assigning a Subscript to a Name (line 442):
        
        # Obtaining the type of the subscript
        int_1016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 442, 8), 'int')
        
        # Obtaining the type of the subscript
        # Getting the type of 'c' (line 442)
        c_1017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 39), 'c')
        # Getting the type of 'self' (line 442)
        self_1018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 25), 'self')
        # Obtaining the member '_metrics' of a type (line 442)
        _metrics_1019 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 442, 25), self_1018, '_metrics')
        # Obtaining the member '__getitem__' of a type (line 442)
        getitem___1020 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 442, 25), _metrics_1019, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 442)
        subscript_call_result_1021 = invoke(stypy.reporting.localization.Localization(__file__, 442, 25), getitem___1020, c_1017)
        
        # Obtaining the member '__getitem__' of a type (line 442)
        getitem___1022 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 442, 8), subscript_call_result_1021, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 442)
        subscript_call_result_1023 = invoke(stypy.reporting.localization.Localization(__file__, 442, 8), getitem___1022, int_1016)
        
        # Assigning a type to the variable 'tuple_var_assignment_36' (line 442)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 442, 8), 'tuple_var_assignment_36', subscript_call_result_1023)
        
        # Assigning a Subscript to a Name (line 442):
        
        # Obtaining the type of the subscript
        int_1024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 442, 8), 'int')
        
        # Obtaining the type of the subscript
        # Getting the type of 'c' (line 442)
        c_1025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 39), 'c')
        # Getting the type of 'self' (line 442)
        self_1026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 25), 'self')
        # Obtaining the member '_metrics' of a type (line 442)
        _metrics_1027 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 442, 25), self_1026, '_metrics')
        # Obtaining the member '__getitem__' of a type (line 442)
        getitem___1028 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 442, 25), _metrics_1027, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 442)
        subscript_call_result_1029 = invoke(stypy.reporting.localization.Localization(__file__, 442, 25), getitem___1028, c_1025)
        
        # Obtaining the member '__getitem__' of a type (line 442)
        getitem___1030 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 442, 8), subscript_call_result_1029, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 442)
        subscript_call_result_1031 = invoke(stypy.reporting.localization.Localization(__file__, 442, 8), getitem___1030, int_1024)
        
        # Assigning a type to the variable 'tuple_var_assignment_37' (line 442)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 442, 8), 'tuple_var_assignment_37', subscript_call_result_1031)
        
        # Assigning a Subscript to a Name (line 442):
        
        # Obtaining the type of the subscript
        int_1032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 442, 8), 'int')
        
        # Obtaining the type of the subscript
        # Getting the type of 'c' (line 442)
        c_1033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 39), 'c')
        # Getting the type of 'self' (line 442)
        self_1034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 25), 'self')
        # Obtaining the member '_metrics' of a type (line 442)
        _metrics_1035 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 442, 25), self_1034, '_metrics')
        # Obtaining the member '__getitem__' of a type (line 442)
        getitem___1036 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 442, 25), _metrics_1035, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 442)
        subscript_call_result_1037 = invoke(stypy.reporting.localization.Localization(__file__, 442, 25), getitem___1036, c_1033)
        
        # Obtaining the member '__getitem__' of a type (line 442)
        getitem___1038 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 442, 8), subscript_call_result_1037, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 442)
        subscript_call_result_1039 = invoke(stypy.reporting.localization.Localization(__file__, 442, 8), getitem___1038, int_1032)
        
        # Assigning a type to the variable 'tuple_var_assignment_38' (line 442)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 442, 8), 'tuple_var_assignment_38', subscript_call_result_1039)
        
        # Assigning a Name to a Name (line 442):
        # Getting the type of 'tuple_var_assignment_36' (line 442)
        tuple_var_assignment_36_1040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 8), 'tuple_var_assignment_36')
        # Assigning a type to the variable 'wx' (line 442)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 442, 8), 'wx', tuple_var_assignment_36_1040)
        
        # Assigning a Name to a Name (line 442):
        # Getting the type of 'tuple_var_assignment_37' (line 442)
        tuple_var_assignment_37_1041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 8), 'tuple_var_assignment_37')
        # Assigning a type to the variable 'name' (line 442)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 442, 12), 'name', tuple_var_assignment_37_1041)
        
        # Assigning a Name to a Name (line 442):
        # Getting the type of 'tuple_var_assignment_38' (line 442)
        tuple_var_assignment_38_1042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 8), 'tuple_var_assignment_38')
        # Assigning a type to the variable 'bbox' (line 442)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 442, 18), 'bbox', tuple_var_assignment_38_1042)
        # Getting the type of 'name' (line 443)
        name_1043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 15), 'name')
        # Assigning a type to the variable 'stypy_return_type' (line 443)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 443, 8), 'stypy_return_type', name_1043)
        
        # ################# End of 'get_name_char(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_name_char' in the type store
        # Getting the type of 'stypy_return_type' (line 436)
        stypy_return_type_1044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1044)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_name_char'
        return stypy_return_type_1044


    @norecursion
    def get_width_char(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'False' (line 445)
        False_1045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 38), 'False')
        defaults = [False_1045]
        # Create a new context for function 'get_width_char'
        module_type_store = module_type_store.open_function_context('get_width_char', 445, 4, False)
        # Assigning a type to the variable 'self' (line 446)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 446, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        AFM.get_width_char.__dict__.__setitem__('stypy_localization', localization)
        AFM.get_width_char.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        AFM.get_width_char.__dict__.__setitem__('stypy_type_store', module_type_store)
        AFM.get_width_char.__dict__.__setitem__('stypy_function_name', 'AFM.get_width_char')
        AFM.get_width_char.__dict__.__setitem__('stypy_param_names_list', ['c', 'isord'])
        AFM.get_width_char.__dict__.__setitem__('stypy_varargs_param_name', None)
        AFM.get_width_char.__dict__.__setitem__('stypy_kwargs_param_name', None)
        AFM.get_width_char.__dict__.__setitem__('stypy_call_defaults', defaults)
        AFM.get_width_char.__dict__.__setitem__('stypy_call_varargs', varargs)
        AFM.get_width_char.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        AFM.get_width_char.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'AFM.get_width_char', ['c', 'isord'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_width_char', localization, ['c', 'isord'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_width_char(...)' code ##################

        unicode_1046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 449, (-1)), 'unicode', u'\n        Get the width of the character from the character metric WX\n        field\n        ')
        
        
        # Getting the type of 'isord' (line 450)
        isord_1047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 15), 'isord')
        # Applying the 'not' unary operator (line 450)
        result_not__1048 = python_operator(stypy.reporting.localization.Localization(__file__, 450, 11), 'not', isord_1047)
        
        # Testing the type of an if condition (line 450)
        if_condition_1049 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 450, 8), result_not__1048)
        # Assigning a type to the variable 'if_condition_1049' (line 450)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 450, 8), 'if_condition_1049', if_condition_1049)
        # SSA begins for if statement (line 450)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 451):
        
        # Assigning a Call to a Name (line 451):
        
        # Call to ord(...): (line 451)
        # Processing the call arguments (line 451)
        # Getting the type of 'c' (line 451)
        c_1051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 20), 'c', False)
        # Processing the call keyword arguments (line 451)
        kwargs_1052 = {}
        # Getting the type of 'ord' (line 451)
        ord_1050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 16), 'ord', False)
        # Calling ord(args, kwargs) (line 451)
        ord_call_result_1053 = invoke(stypy.reporting.localization.Localization(__file__, 451, 16), ord_1050, *[c_1051], **kwargs_1052)
        
        # Assigning a type to the variable 'c' (line 451)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 451, 12), 'c', ord_call_result_1053)
        # SSA join for if statement (line 450)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Subscript to a Tuple (line 452):
        
        # Assigning a Subscript to a Name (line 452):
        
        # Obtaining the type of the subscript
        int_1054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 452, 8), 'int')
        
        # Obtaining the type of the subscript
        # Getting the type of 'c' (line 452)
        c_1055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 39), 'c')
        # Getting the type of 'self' (line 452)
        self_1056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 25), 'self')
        # Obtaining the member '_metrics' of a type (line 452)
        _metrics_1057 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 452, 25), self_1056, '_metrics')
        # Obtaining the member '__getitem__' of a type (line 452)
        getitem___1058 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 452, 25), _metrics_1057, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 452)
        subscript_call_result_1059 = invoke(stypy.reporting.localization.Localization(__file__, 452, 25), getitem___1058, c_1055)
        
        # Obtaining the member '__getitem__' of a type (line 452)
        getitem___1060 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 452, 8), subscript_call_result_1059, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 452)
        subscript_call_result_1061 = invoke(stypy.reporting.localization.Localization(__file__, 452, 8), getitem___1060, int_1054)
        
        # Assigning a type to the variable 'tuple_var_assignment_39' (line 452)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 452, 8), 'tuple_var_assignment_39', subscript_call_result_1061)
        
        # Assigning a Subscript to a Name (line 452):
        
        # Obtaining the type of the subscript
        int_1062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 452, 8), 'int')
        
        # Obtaining the type of the subscript
        # Getting the type of 'c' (line 452)
        c_1063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 39), 'c')
        # Getting the type of 'self' (line 452)
        self_1064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 25), 'self')
        # Obtaining the member '_metrics' of a type (line 452)
        _metrics_1065 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 452, 25), self_1064, '_metrics')
        # Obtaining the member '__getitem__' of a type (line 452)
        getitem___1066 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 452, 25), _metrics_1065, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 452)
        subscript_call_result_1067 = invoke(stypy.reporting.localization.Localization(__file__, 452, 25), getitem___1066, c_1063)
        
        # Obtaining the member '__getitem__' of a type (line 452)
        getitem___1068 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 452, 8), subscript_call_result_1067, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 452)
        subscript_call_result_1069 = invoke(stypy.reporting.localization.Localization(__file__, 452, 8), getitem___1068, int_1062)
        
        # Assigning a type to the variable 'tuple_var_assignment_40' (line 452)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 452, 8), 'tuple_var_assignment_40', subscript_call_result_1069)
        
        # Assigning a Subscript to a Name (line 452):
        
        # Obtaining the type of the subscript
        int_1070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 452, 8), 'int')
        
        # Obtaining the type of the subscript
        # Getting the type of 'c' (line 452)
        c_1071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 39), 'c')
        # Getting the type of 'self' (line 452)
        self_1072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 25), 'self')
        # Obtaining the member '_metrics' of a type (line 452)
        _metrics_1073 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 452, 25), self_1072, '_metrics')
        # Obtaining the member '__getitem__' of a type (line 452)
        getitem___1074 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 452, 25), _metrics_1073, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 452)
        subscript_call_result_1075 = invoke(stypy.reporting.localization.Localization(__file__, 452, 25), getitem___1074, c_1071)
        
        # Obtaining the member '__getitem__' of a type (line 452)
        getitem___1076 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 452, 8), subscript_call_result_1075, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 452)
        subscript_call_result_1077 = invoke(stypy.reporting.localization.Localization(__file__, 452, 8), getitem___1076, int_1070)
        
        # Assigning a type to the variable 'tuple_var_assignment_41' (line 452)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 452, 8), 'tuple_var_assignment_41', subscript_call_result_1077)
        
        # Assigning a Name to a Name (line 452):
        # Getting the type of 'tuple_var_assignment_39' (line 452)
        tuple_var_assignment_39_1078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 8), 'tuple_var_assignment_39')
        # Assigning a type to the variable 'wx' (line 452)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 452, 8), 'wx', tuple_var_assignment_39_1078)
        
        # Assigning a Name to a Name (line 452):
        # Getting the type of 'tuple_var_assignment_40' (line 452)
        tuple_var_assignment_40_1079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 8), 'tuple_var_assignment_40')
        # Assigning a type to the variable 'name' (line 452)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 452, 12), 'name', tuple_var_assignment_40_1079)
        
        # Assigning a Name to a Name (line 452):
        # Getting the type of 'tuple_var_assignment_41' (line 452)
        tuple_var_assignment_41_1080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 8), 'tuple_var_assignment_41')
        # Assigning a type to the variable 'bbox' (line 452)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 452, 18), 'bbox', tuple_var_assignment_41_1080)
        # Getting the type of 'wx' (line 453)
        wx_1081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 15), 'wx')
        # Assigning a type to the variable 'stypy_return_type' (line 453)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 453, 8), 'stypy_return_type', wx_1081)
        
        # ################# End of 'get_width_char(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_width_char' in the type store
        # Getting the type of 'stypy_return_type' (line 445)
        stypy_return_type_1082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1082)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_width_char'
        return stypy_return_type_1082


    @norecursion
    def get_width_from_char_name(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_width_from_char_name'
        module_type_store = module_type_store.open_function_context('get_width_from_char_name', 455, 4, False)
        # Assigning a type to the variable 'self' (line 456)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 456, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        AFM.get_width_from_char_name.__dict__.__setitem__('stypy_localization', localization)
        AFM.get_width_from_char_name.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        AFM.get_width_from_char_name.__dict__.__setitem__('stypy_type_store', module_type_store)
        AFM.get_width_from_char_name.__dict__.__setitem__('stypy_function_name', 'AFM.get_width_from_char_name')
        AFM.get_width_from_char_name.__dict__.__setitem__('stypy_param_names_list', ['name'])
        AFM.get_width_from_char_name.__dict__.__setitem__('stypy_varargs_param_name', None)
        AFM.get_width_from_char_name.__dict__.__setitem__('stypy_kwargs_param_name', None)
        AFM.get_width_from_char_name.__dict__.__setitem__('stypy_call_defaults', defaults)
        AFM.get_width_from_char_name.__dict__.__setitem__('stypy_call_varargs', varargs)
        AFM.get_width_from_char_name.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        AFM.get_width_from_char_name.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'AFM.get_width_from_char_name', ['name'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_width_from_char_name', localization, ['name'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_width_from_char_name(...)' code ##################

        unicode_1083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 458, (-1)), 'unicode', u'\n        Get the width of the character from a type1 character name\n        ')
        
        # Assigning a Subscript to a Tuple (line 459):
        
        # Assigning a Subscript to a Name (line 459):
        
        # Obtaining the type of the subscript
        int_1084 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 459, 8), 'int')
        
        # Obtaining the type of the subscript
        # Getting the type of 'name' (line 459)
        name_1085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 41), 'name')
        # Getting the type of 'self' (line 459)
        self_1086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 19), 'self')
        # Obtaining the member '_metrics_by_name' of a type (line 459)
        _metrics_by_name_1087 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 459, 19), self_1086, '_metrics_by_name')
        # Obtaining the member '__getitem__' of a type (line 459)
        getitem___1088 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 459, 19), _metrics_by_name_1087, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 459)
        subscript_call_result_1089 = invoke(stypy.reporting.localization.Localization(__file__, 459, 19), getitem___1088, name_1085)
        
        # Obtaining the member '__getitem__' of a type (line 459)
        getitem___1090 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 459, 8), subscript_call_result_1089, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 459)
        subscript_call_result_1091 = invoke(stypy.reporting.localization.Localization(__file__, 459, 8), getitem___1090, int_1084)
        
        # Assigning a type to the variable 'tuple_var_assignment_42' (line 459)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 459, 8), 'tuple_var_assignment_42', subscript_call_result_1091)
        
        # Assigning a Subscript to a Name (line 459):
        
        # Obtaining the type of the subscript
        int_1092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 459, 8), 'int')
        
        # Obtaining the type of the subscript
        # Getting the type of 'name' (line 459)
        name_1093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 41), 'name')
        # Getting the type of 'self' (line 459)
        self_1094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 19), 'self')
        # Obtaining the member '_metrics_by_name' of a type (line 459)
        _metrics_by_name_1095 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 459, 19), self_1094, '_metrics_by_name')
        # Obtaining the member '__getitem__' of a type (line 459)
        getitem___1096 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 459, 19), _metrics_by_name_1095, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 459)
        subscript_call_result_1097 = invoke(stypy.reporting.localization.Localization(__file__, 459, 19), getitem___1096, name_1093)
        
        # Obtaining the member '__getitem__' of a type (line 459)
        getitem___1098 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 459, 8), subscript_call_result_1097, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 459)
        subscript_call_result_1099 = invoke(stypy.reporting.localization.Localization(__file__, 459, 8), getitem___1098, int_1092)
        
        # Assigning a type to the variable 'tuple_var_assignment_43' (line 459)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 459, 8), 'tuple_var_assignment_43', subscript_call_result_1099)
        
        # Assigning a Name to a Name (line 459):
        # Getting the type of 'tuple_var_assignment_42' (line 459)
        tuple_var_assignment_42_1100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 8), 'tuple_var_assignment_42')
        # Assigning a type to the variable 'wx' (line 459)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 459, 8), 'wx', tuple_var_assignment_42_1100)
        
        # Assigning a Name to a Name (line 459):
        # Getting the type of 'tuple_var_assignment_43' (line 459)
        tuple_var_assignment_43_1101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 8), 'tuple_var_assignment_43')
        # Assigning a type to the variable 'bbox' (line 459)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 459, 12), 'bbox', tuple_var_assignment_43_1101)
        # Getting the type of 'wx' (line 460)
        wx_1102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 15), 'wx')
        # Assigning a type to the variable 'stypy_return_type' (line 460)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 460, 8), 'stypy_return_type', wx_1102)
        
        # ################# End of 'get_width_from_char_name(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_width_from_char_name' in the type store
        # Getting the type of 'stypy_return_type' (line 455)
        stypy_return_type_1103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1103)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_width_from_char_name'
        return stypy_return_type_1103


    @norecursion
    def get_height_char(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'False' (line 462)
        False_1104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 39), 'False')
        defaults = [False_1104]
        # Create a new context for function 'get_height_char'
        module_type_store = module_type_store.open_function_context('get_height_char', 462, 4, False)
        # Assigning a type to the variable 'self' (line 463)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 463, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        AFM.get_height_char.__dict__.__setitem__('stypy_localization', localization)
        AFM.get_height_char.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        AFM.get_height_char.__dict__.__setitem__('stypy_type_store', module_type_store)
        AFM.get_height_char.__dict__.__setitem__('stypy_function_name', 'AFM.get_height_char')
        AFM.get_height_char.__dict__.__setitem__('stypy_param_names_list', ['c', 'isord'])
        AFM.get_height_char.__dict__.__setitem__('stypy_varargs_param_name', None)
        AFM.get_height_char.__dict__.__setitem__('stypy_kwargs_param_name', None)
        AFM.get_height_char.__dict__.__setitem__('stypy_call_defaults', defaults)
        AFM.get_height_char.__dict__.__setitem__('stypy_call_varargs', varargs)
        AFM.get_height_char.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        AFM.get_height_char.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'AFM.get_height_char', ['c', 'isord'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_height_char', localization, ['c', 'isord'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_height_char(...)' code ##################

        unicode_1105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 466, (-1)), 'unicode', u'\n        Get the height of character *c* from the bounding box.  This\n        is the ink height (space is 0)\n        ')
        
        
        # Getting the type of 'isord' (line 467)
        isord_1106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 15), 'isord')
        # Applying the 'not' unary operator (line 467)
        result_not__1107 = python_operator(stypy.reporting.localization.Localization(__file__, 467, 11), 'not', isord_1106)
        
        # Testing the type of an if condition (line 467)
        if_condition_1108 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 467, 8), result_not__1107)
        # Assigning a type to the variable 'if_condition_1108' (line 467)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 467, 8), 'if_condition_1108', if_condition_1108)
        # SSA begins for if statement (line 467)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 468):
        
        # Assigning a Call to a Name (line 468):
        
        # Call to ord(...): (line 468)
        # Processing the call arguments (line 468)
        # Getting the type of 'c' (line 468)
        c_1110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 20), 'c', False)
        # Processing the call keyword arguments (line 468)
        kwargs_1111 = {}
        # Getting the type of 'ord' (line 468)
        ord_1109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 16), 'ord', False)
        # Calling ord(args, kwargs) (line 468)
        ord_call_result_1112 = invoke(stypy.reporting.localization.Localization(__file__, 468, 16), ord_1109, *[c_1110], **kwargs_1111)
        
        # Assigning a type to the variable 'c' (line 468)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 468, 12), 'c', ord_call_result_1112)
        # SSA join for if statement (line 467)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Subscript to a Tuple (line 469):
        
        # Assigning a Subscript to a Name (line 469):
        
        # Obtaining the type of the subscript
        int_1113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 469, 8), 'int')
        
        # Obtaining the type of the subscript
        # Getting the type of 'c' (line 469)
        c_1114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 39), 'c')
        # Getting the type of 'self' (line 469)
        self_1115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 25), 'self')
        # Obtaining the member '_metrics' of a type (line 469)
        _metrics_1116 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 469, 25), self_1115, '_metrics')
        # Obtaining the member '__getitem__' of a type (line 469)
        getitem___1117 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 469, 25), _metrics_1116, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 469)
        subscript_call_result_1118 = invoke(stypy.reporting.localization.Localization(__file__, 469, 25), getitem___1117, c_1114)
        
        # Obtaining the member '__getitem__' of a type (line 469)
        getitem___1119 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 469, 8), subscript_call_result_1118, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 469)
        subscript_call_result_1120 = invoke(stypy.reporting.localization.Localization(__file__, 469, 8), getitem___1119, int_1113)
        
        # Assigning a type to the variable 'tuple_var_assignment_44' (line 469)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 469, 8), 'tuple_var_assignment_44', subscript_call_result_1120)
        
        # Assigning a Subscript to a Name (line 469):
        
        # Obtaining the type of the subscript
        int_1121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 469, 8), 'int')
        
        # Obtaining the type of the subscript
        # Getting the type of 'c' (line 469)
        c_1122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 39), 'c')
        # Getting the type of 'self' (line 469)
        self_1123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 25), 'self')
        # Obtaining the member '_metrics' of a type (line 469)
        _metrics_1124 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 469, 25), self_1123, '_metrics')
        # Obtaining the member '__getitem__' of a type (line 469)
        getitem___1125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 469, 25), _metrics_1124, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 469)
        subscript_call_result_1126 = invoke(stypy.reporting.localization.Localization(__file__, 469, 25), getitem___1125, c_1122)
        
        # Obtaining the member '__getitem__' of a type (line 469)
        getitem___1127 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 469, 8), subscript_call_result_1126, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 469)
        subscript_call_result_1128 = invoke(stypy.reporting.localization.Localization(__file__, 469, 8), getitem___1127, int_1121)
        
        # Assigning a type to the variable 'tuple_var_assignment_45' (line 469)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 469, 8), 'tuple_var_assignment_45', subscript_call_result_1128)
        
        # Assigning a Subscript to a Name (line 469):
        
        # Obtaining the type of the subscript
        int_1129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 469, 8), 'int')
        
        # Obtaining the type of the subscript
        # Getting the type of 'c' (line 469)
        c_1130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 39), 'c')
        # Getting the type of 'self' (line 469)
        self_1131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 25), 'self')
        # Obtaining the member '_metrics' of a type (line 469)
        _metrics_1132 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 469, 25), self_1131, '_metrics')
        # Obtaining the member '__getitem__' of a type (line 469)
        getitem___1133 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 469, 25), _metrics_1132, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 469)
        subscript_call_result_1134 = invoke(stypy.reporting.localization.Localization(__file__, 469, 25), getitem___1133, c_1130)
        
        # Obtaining the member '__getitem__' of a type (line 469)
        getitem___1135 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 469, 8), subscript_call_result_1134, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 469)
        subscript_call_result_1136 = invoke(stypy.reporting.localization.Localization(__file__, 469, 8), getitem___1135, int_1129)
        
        # Assigning a type to the variable 'tuple_var_assignment_46' (line 469)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 469, 8), 'tuple_var_assignment_46', subscript_call_result_1136)
        
        # Assigning a Name to a Name (line 469):
        # Getting the type of 'tuple_var_assignment_44' (line 469)
        tuple_var_assignment_44_1137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 8), 'tuple_var_assignment_44')
        # Assigning a type to the variable 'wx' (line 469)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 469, 8), 'wx', tuple_var_assignment_44_1137)
        
        # Assigning a Name to a Name (line 469):
        # Getting the type of 'tuple_var_assignment_45' (line 469)
        tuple_var_assignment_45_1138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 8), 'tuple_var_assignment_45')
        # Assigning a type to the variable 'name' (line 469)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 469, 12), 'name', tuple_var_assignment_45_1138)
        
        # Assigning a Name to a Name (line 469):
        # Getting the type of 'tuple_var_assignment_46' (line 469)
        tuple_var_assignment_46_1139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 8), 'tuple_var_assignment_46')
        # Assigning a type to the variable 'bbox' (line 469)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 469, 18), 'bbox', tuple_var_assignment_46_1139)
        
        # Obtaining the type of the subscript
        int_1140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 470, 20), 'int')
        # Getting the type of 'bbox' (line 470)
        bbox_1141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 15), 'bbox')
        # Obtaining the member '__getitem__' of a type (line 470)
        getitem___1142 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 470, 15), bbox_1141, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 470)
        subscript_call_result_1143 = invoke(stypy.reporting.localization.Localization(__file__, 470, 15), getitem___1142, int_1140)
        
        # Assigning a type to the variable 'stypy_return_type' (line 470)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 470, 8), 'stypy_return_type', subscript_call_result_1143)
        
        # ################# End of 'get_height_char(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_height_char' in the type store
        # Getting the type of 'stypy_return_type' (line 462)
        stypy_return_type_1144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1144)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_height_char'
        return stypy_return_type_1144


    @norecursion
    def get_kern_dist(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_kern_dist'
        module_type_store = module_type_store.open_function_context('get_kern_dist', 472, 4, False)
        # Assigning a type to the variable 'self' (line 473)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 473, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        AFM.get_kern_dist.__dict__.__setitem__('stypy_localization', localization)
        AFM.get_kern_dist.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        AFM.get_kern_dist.__dict__.__setitem__('stypy_type_store', module_type_store)
        AFM.get_kern_dist.__dict__.__setitem__('stypy_function_name', 'AFM.get_kern_dist')
        AFM.get_kern_dist.__dict__.__setitem__('stypy_param_names_list', ['c1', 'c2'])
        AFM.get_kern_dist.__dict__.__setitem__('stypy_varargs_param_name', None)
        AFM.get_kern_dist.__dict__.__setitem__('stypy_kwargs_param_name', None)
        AFM.get_kern_dist.__dict__.__setitem__('stypy_call_defaults', defaults)
        AFM.get_kern_dist.__dict__.__setitem__('stypy_call_varargs', varargs)
        AFM.get_kern_dist.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        AFM.get_kern_dist.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'AFM.get_kern_dist', ['c1', 'c2'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_kern_dist', localization, ['c1', 'c2'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_kern_dist(...)' code ##################

        unicode_1145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 476, (-1)), 'unicode', u'\n        Return the kerning pair distance (possibly 0) for chars *c1*\n        and *c2*\n        ')
        
        # Assigning a Tuple to a Tuple (line 477):
        
        # Assigning a Call to a Name (line 477):
        
        # Call to get_name_char(...): (line 477)
        # Processing the call arguments (line 477)
        # Getting the type of 'c1' (line 477)
        c1_1148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 42), 'c1', False)
        # Processing the call keyword arguments (line 477)
        kwargs_1149 = {}
        # Getting the type of 'self' (line 477)
        self_1146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 23), 'self', False)
        # Obtaining the member 'get_name_char' of a type (line 477)
        get_name_char_1147 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 477, 23), self_1146, 'get_name_char')
        # Calling get_name_char(args, kwargs) (line 477)
        get_name_char_call_result_1150 = invoke(stypy.reporting.localization.Localization(__file__, 477, 23), get_name_char_1147, *[c1_1148], **kwargs_1149)
        
        # Assigning a type to the variable 'tuple_assignment_47' (line 477)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 477, 8), 'tuple_assignment_47', get_name_char_call_result_1150)
        
        # Assigning a Call to a Name (line 477):
        
        # Call to get_name_char(...): (line 477)
        # Processing the call arguments (line 477)
        # Getting the type of 'c2' (line 477)
        c2_1153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 66), 'c2', False)
        # Processing the call keyword arguments (line 477)
        kwargs_1154 = {}
        # Getting the type of 'self' (line 477)
        self_1151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 47), 'self', False)
        # Obtaining the member 'get_name_char' of a type (line 477)
        get_name_char_1152 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 477, 47), self_1151, 'get_name_char')
        # Calling get_name_char(args, kwargs) (line 477)
        get_name_char_call_result_1155 = invoke(stypy.reporting.localization.Localization(__file__, 477, 47), get_name_char_1152, *[c2_1153], **kwargs_1154)
        
        # Assigning a type to the variable 'tuple_assignment_48' (line 477)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 477, 8), 'tuple_assignment_48', get_name_char_call_result_1155)
        
        # Assigning a Name to a Name (line 477):
        # Getting the type of 'tuple_assignment_47' (line 477)
        tuple_assignment_47_1156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 8), 'tuple_assignment_47')
        # Assigning a type to the variable 'name1' (line 477)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 477, 8), 'name1', tuple_assignment_47_1156)
        
        # Assigning a Name to a Name (line 477):
        # Getting the type of 'tuple_assignment_48' (line 477)
        tuple_assignment_48_1157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 8), 'tuple_assignment_48')
        # Assigning a type to the variable 'name2' (line 477)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 477, 15), 'name2', tuple_assignment_48_1157)
        
        # Call to get_kern_dist_from_name(...): (line 478)
        # Processing the call arguments (line 478)
        # Getting the type of 'name1' (line 478)
        name1_1160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 44), 'name1', False)
        # Getting the type of 'name2' (line 478)
        name2_1161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 51), 'name2', False)
        # Processing the call keyword arguments (line 478)
        kwargs_1162 = {}
        # Getting the type of 'self' (line 478)
        self_1158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 15), 'self', False)
        # Obtaining the member 'get_kern_dist_from_name' of a type (line 478)
        get_kern_dist_from_name_1159 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 478, 15), self_1158, 'get_kern_dist_from_name')
        # Calling get_kern_dist_from_name(args, kwargs) (line 478)
        get_kern_dist_from_name_call_result_1163 = invoke(stypy.reporting.localization.Localization(__file__, 478, 15), get_kern_dist_from_name_1159, *[name1_1160, name2_1161], **kwargs_1162)
        
        # Assigning a type to the variable 'stypy_return_type' (line 478)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 478, 8), 'stypy_return_type', get_kern_dist_from_name_call_result_1163)
        
        # ################# End of 'get_kern_dist(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_kern_dist' in the type store
        # Getting the type of 'stypy_return_type' (line 472)
        stypy_return_type_1164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1164)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_kern_dist'
        return stypy_return_type_1164


    @norecursion
    def get_kern_dist_from_name(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_kern_dist_from_name'
        module_type_store = module_type_store.open_function_context('get_kern_dist_from_name', 480, 4, False)
        # Assigning a type to the variable 'self' (line 481)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 481, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        AFM.get_kern_dist_from_name.__dict__.__setitem__('stypy_localization', localization)
        AFM.get_kern_dist_from_name.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        AFM.get_kern_dist_from_name.__dict__.__setitem__('stypy_type_store', module_type_store)
        AFM.get_kern_dist_from_name.__dict__.__setitem__('stypy_function_name', 'AFM.get_kern_dist_from_name')
        AFM.get_kern_dist_from_name.__dict__.__setitem__('stypy_param_names_list', ['name1', 'name2'])
        AFM.get_kern_dist_from_name.__dict__.__setitem__('stypy_varargs_param_name', None)
        AFM.get_kern_dist_from_name.__dict__.__setitem__('stypy_kwargs_param_name', None)
        AFM.get_kern_dist_from_name.__dict__.__setitem__('stypy_call_defaults', defaults)
        AFM.get_kern_dist_from_name.__dict__.__setitem__('stypy_call_varargs', varargs)
        AFM.get_kern_dist_from_name.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        AFM.get_kern_dist_from_name.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'AFM.get_kern_dist_from_name', ['name1', 'name2'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_kern_dist_from_name', localization, ['name1', 'name2'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_kern_dist_from_name(...)' code ##################

        unicode_1165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 484, (-1)), 'unicode', u'\n        Return the kerning pair distance (possibly 0) for chars\n        *name1* and *name2*\n        ')
        
        # Call to get(...): (line 485)
        # Processing the call arguments (line 485)
        
        # Obtaining an instance of the builtin type 'tuple' (line 485)
        tuple_1169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 485, 31), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 485)
        # Adding element type (line 485)
        # Getting the type of 'name1' (line 485)
        name1_1170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 31), 'name1', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 485, 31), tuple_1169, name1_1170)
        # Adding element type (line 485)
        # Getting the type of 'name2' (line 485)
        name2_1171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 38), 'name2', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 485, 31), tuple_1169, name2_1171)
        
        int_1172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 485, 46), 'int')
        # Processing the call keyword arguments (line 485)
        kwargs_1173 = {}
        # Getting the type of 'self' (line 485)
        self_1166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 15), 'self', False)
        # Obtaining the member '_kern' of a type (line 485)
        _kern_1167 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 485, 15), self_1166, '_kern')
        # Obtaining the member 'get' of a type (line 485)
        get_1168 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 485, 15), _kern_1167, 'get')
        # Calling get(args, kwargs) (line 485)
        get_call_result_1174 = invoke(stypy.reporting.localization.Localization(__file__, 485, 15), get_1168, *[tuple_1169, int_1172], **kwargs_1173)
        
        # Assigning a type to the variable 'stypy_return_type' (line 485)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 485, 8), 'stypy_return_type', get_call_result_1174)
        
        # ################# End of 'get_kern_dist_from_name(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_kern_dist_from_name' in the type store
        # Getting the type of 'stypy_return_type' (line 480)
        stypy_return_type_1175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1175)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_kern_dist_from_name'
        return stypy_return_type_1175


    @norecursion
    def get_fontname(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_fontname'
        module_type_store = module_type_store.open_function_context('get_fontname', 487, 4, False)
        # Assigning a type to the variable 'self' (line 488)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 488, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        AFM.get_fontname.__dict__.__setitem__('stypy_localization', localization)
        AFM.get_fontname.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        AFM.get_fontname.__dict__.__setitem__('stypy_type_store', module_type_store)
        AFM.get_fontname.__dict__.__setitem__('stypy_function_name', 'AFM.get_fontname')
        AFM.get_fontname.__dict__.__setitem__('stypy_param_names_list', [])
        AFM.get_fontname.__dict__.__setitem__('stypy_varargs_param_name', None)
        AFM.get_fontname.__dict__.__setitem__('stypy_kwargs_param_name', None)
        AFM.get_fontname.__dict__.__setitem__('stypy_call_defaults', defaults)
        AFM.get_fontname.__dict__.__setitem__('stypy_call_varargs', varargs)
        AFM.get_fontname.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        AFM.get_fontname.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'AFM.get_fontname', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_fontname', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_fontname(...)' code ##################

        unicode_1176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 488, 8), 'unicode', u"Return the font name, e.g., 'Times-Roman'")
        
        # Obtaining the type of the subscript
        str_1177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 489, 28), 'str', 'FontName')
        # Getting the type of 'self' (line 489)
        self_1178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 15), 'self')
        # Obtaining the member '_header' of a type (line 489)
        _header_1179 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 489, 15), self_1178, '_header')
        # Obtaining the member '__getitem__' of a type (line 489)
        getitem___1180 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 489, 15), _header_1179, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 489)
        subscript_call_result_1181 = invoke(stypy.reporting.localization.Localization(__file__, 489, 15), getitem___1180, str_1177)
        
        # Assigning a type to the variable 'stypy_return_type' (line 489)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 489, 8), 'stypy_return_type', subscript_call_result_1181)
        
        # ################# End of 'get_fontname(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_fontname' in the type store
        # Getting the type of 'stypy_return_type' (line 487)
        stypy_return_type_1182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1182)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_fontname'
        return stypy_return_type_1182


    @norecursion
    def get_fullname(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_fullname'
        module_type_store = module_type_store.open_function_context('get_fullname', 491, 4, False)
        # Assigning a type to the variable 'self' (line 492)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 492, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        AFM.get_fullname.__dict__.__setitem__('stypy_localization', localization)
        AFM.get_fullname.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        AFM.get_fullname.__dict__.__setitem__('stypy_type_store', module_type_store)
        AFM.get_fullname.__dict__.__setitem__('stypy_function_name', 'AFM.get_fullname')
        AFM.get_fullname.__dict__.__setitem__('stypy_param_names_list', [])
        AFM.get_fullname.__dict__.__setitem__('stypy_varargs_param_name', None)
        AFM.get_fullname.__dict__.__setitem__('stypy_kwargs_param_name', None)
        AFM.get_fullname.__dict__.__setitem__('stypy_call_defaults', defaults)
        AFM.get_fullname.__dict__.__setitem__('stypy_call_varargs', varargs)
        AFM.get_fullname.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        AFM.get_fullname.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'AFM.get_fullname', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_fullname', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_fullname(...)' code ##################

        unicode_1183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 492, 8), 'unicode', u"Return the font full name, e.g., 'Times-Roman'")
        
        # Assigning a Call to a Name (line 493):
        
        # Assigning a Call to a Name (line 493):
        
        # Call to get(...): (line 493)
        # Processing the call arguments (line 493)
        str_1187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 493, 32), 'str', 'FullName')
        # Processing the call keyword arguments (line 493)
        kwargs_1188 = {}
        # Getting the type of 'self' (line 493)
        self_1184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 15), 'self', False)
        # Obtaining the member '_header' of a type (line 493)
        _header_1185 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 493, 15), self_1184, '_header')
        # Obtaining the member 'get' of a type (line 493)
        get_1186 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 493, 15), _header_1185, 'get')
        # Calling get(args, kwargs) (line 493)
        get_call_result_1189 = invoke(stypy.reporting.localization.Localization(__file__, 493, 15), get_1186, *[str_1187], **kwargs_1188)
        
        # Assigning a type to the variable 'name' (line 493)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 493, 8), 'name', get_call_result_1189)
        
        # Type idiom detected: calculating its left and rigth part (line 494)
        # Getting the type of 'name' (line 494)
        name_1190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 11), 'name')
        # Getting the type of 'None' (line 494)
        None_1191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 19), 'None')
        
        (may_be_1192, more_types_in_union_1193) = may_be_none(name_1190, None_1191)

        if may_be_1192:

            if more_types_in_union_1193:
                # Runtime conditional SSA (line 494)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Subscript to a Name (line 495):
            
            # Assigning a Subscript to a Name (line 495):
            
            # Obtaining the type of the subscript
            str_1194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 495, 32), 'str', 'FontName')
            # Getting the type of 'self' (line 495)
            self_1195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 495, 19), 'self')
            # Obtaining the member '_header' of a type (line 495)
            _header_1196 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 495, 19), self_1195, '_header')
            # Obtaining the member '__getitem__' of a type (line 495)
            getitem___1197 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 495, 19), _header_1196, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 495)
            subscript_call_result_1198 = invoke(stypy.reporting.localization.Localization(__file__, 495, 19), getitem___1197, str_1194)
            
            # Assigning a type to the variable 'name' (line 495)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 495, 12), 'name', subscript_call_result_1198)

            if more_types_in_union_1193:
                # SSA join for if statement (line 494)
                module_type_store = module_type_store.join_ssa_context()


        
        # Getting the type of 'name' (line 496)
        name_1199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 15), 'name')
        # Assigning a type to the variable 'stypy_return_type' (line 496)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 496, 8), 'stypy_return_type', name_1199)
        
        # ################# End of 'get_fullname(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_fullname' in the type store
        # Getting the type of 'stypy_return_type' (line 491)
        stypy_return_type_1200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1200)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_fullname'
        return stypy_return_type_1200


    @norecursion
    def get_familyname(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_familyname'
        module_type_store = module_type_store.open_function_context('get_familyname', 498, 4, False)
        # Assigning a type to the variable 'self' (line 499)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 499, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        AFM.get_familyname.__dict__.__setitem__('stypy_localization', localization)
        AFM.get_familyname.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        AFM.get_familyname.__dict__.__setitem__('stypy_type_store', module_type_store)
        AFM.get_familyname.__dict__.__setitem__('stypy_function_name', 'AFM.get_familyname')
        AFM.get_familyname.__dict__.__setitem__('stypy_param_names_list', [])
        AFM.get_familyname.__dict__.__setitem__('stypy_varargs_param_name', None)
        AFM.get_familyname.__dict__.__setitem__('stypy_kwargs_param_name', None)
        AFM.get_familyname.__dict__.__setitem__('stypy_call_defaults', defaults)
        AFM.get_familyname.__dict__.__setitem__('stypy_call_varargs', varargs)
        AFM.get_familyname.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        AFM.get_familyname.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'AFM.get_familyname', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_familyname', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_familyname(...)' code ##################

        unicode_1201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 499, 8), 'unicode', u"Return the font family name, e.g., 'Times'")
        
        # Assigning a Call to a Name (line 500):
        
        # Assigning a Call to a Name (line 500):
        
        # Call to get(...): (line 500)
        # Processing the call arguments (line 500)
        str_1205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 500, 32), 'str', 'FamilyName')
        # Processing the call keyword arguments (line 500)
        kwargs_1206 = {}
        # Getting the type of 'self' (line 500)
        self_1202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 15), 'self', False)
        # Obtaining the member '_header' of a type (line 500)
        _header_1203 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 500, 15), self_1202, '_header')
        # Obtaining the member 'get' of a type (line 500)
        get_1204 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 500, 15), _header_1203, 'get')
        # Calling get(args, kwargs) (line 500)
        get_call_result_1207 = invoke(stypy.reporting.localization.Localization(__file__, 500, 15), get_1204, *[str_1205], **kwargs_1206)
        
        # Assigning a type to the variable 'name' (line 500)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 500, 8), 'name', get_call_result_1207)
        
        # Type idiom detected: calculating its left and rigth part (line 501)
        # Getting the type of 'name' (line 501)
        name_1208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 8), 'name')
        # Getting the type of 'None' (line 501)
        None_1209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 23), 'None')
        
        (may_be_1210, more_types_in_union_1211) = may_not_be_none(name_1208, None_1209)

        if may_be_1210:

            if more_types_in_union_1211:
                # Runtime conditional SSA (line 501)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Getting the type of 'name' (line 502)
            name_1212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 19), 'name')
            # Assigning a type to the variable 'stypy_return_type' (line 502)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 502, 12), 'stypy_return_type', name_1212)

            if more_types_in_union_1211:
                # SSA join for if statement (line 501)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Name (line 505):
        
        # Assigning a Call to a Name (line 505):
        
        # Call to get_fullname(...): (line 505)
        # Processing the call keyword arguments (line 505)
        kwargs_1215 = {}
        # Getting the type of 'self' (line 505)
        self_1213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 15), 'self', False)
        # Obtaining the member 'get_fullname' of a type (line 505)
        get_fullname_1214 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 505, 15), self_1213, 'get_fullname')
        # Calling get_fullname(args, kwargs) (line 505)
        get_fullname_call_result_1216 = invoke(stypy.reporting.localization.Localization(__file__, 505, 15), get_fullname_1214, *[], **kwargs_1215)
        
        # Assigning a type to the variable 'name' (line 505)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 505, 8), 'name', get_fullname_call_result_1216)
        
        # Assigning a Str to a Name (line 506):
        
        # Assigning a Str to a Name (line 506):
        str_1217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 506, 18), 'str', '(?i)([ -](regular|plain|italic|oblique|bold|semibold|light|ultralight|extra|condensed))+$')
        # Assigning a type to the variable 'extras' (line 506)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 506, 8), 'extras', str_1217)
        
        # Call to sub(...): (line 508)
        # Processing the call arguments (line 508)
        # Getting the type of 'extras' (line 508)
        extras_1220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 22), 'extras', False)
        unicode_1221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 508, 30), 'unicode', u'')
        # Getting the type of 'name' (line 508)
        name_1222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 34), 'name', False)
        # Processing the call keyword arguments (line 508)
        kwargs_1223 = {}
        # Getting the type of 're' (line 508)
        re_1218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 15), 're', False)
        # Obtaining the member 'sub' of a type (line 508)
        sub_1219 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 508, 15), re_1218, 'sub')
        # Calling sub(args, kwargs) (line 508)
        sub_call_result_1224 = invoke(stypy.reporting.localization.Localization(__file__, 508, 15), sub_1219, *[extras_1220, unicode_1221, name_1222], **kwargs_1223)
        
        # Assigning a type to the variable 'stypy_return_type' (line 508)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 508, 8), 'stypy_return_type', sub_call_result_1224)
        
        # ################# End of 'get_familyname(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_familyname' in the type store
        # Getting the type of 'stypy_return_type' (line 498)
        stypy_return_type_1225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1225)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_familyname'
        return stypy_return_type_1225


    @norecursion
    def family_name(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'family_name'
        module_type_store = module_type_store.open_function_context('family_name', 510, 4, False)
        # Assigning a type to the variable 'self' (line 511)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 511, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        AFM.family_name.__dict__.__setitem__('stypy_localization', localization)
        AFM.family_name.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        AFM.family_name.__dict__.__setitem__('stypy_type_store', module_type_store)
        AFM.family_name.__dict__.__setitem__('stypy_function_name', 'AFM.family_name')
        AFM.family_name.__dict__.__setitem__('stypy_param_names_list', [])
        AFM.family_name.__dict__.__setitem__('stypy_varargs_param_name', None)
        AFM.family_name.__dict__.__setitem__('stypy_kwargs_param_name', None)
        AFM.family_name.__dict__.__setitem__('stypy_call_defaults', defaults)
        AFM.family_name.__dict__.__setitem__('stypy_call_varargs', varargs)
        AFM.family_name.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        AFM.family_name.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'AFM.family_name', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'family_name', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'family_name(...)' code ##################

        
        # Call to get_familyname(...): (line 512)
        # Processing the call keyword arguments (line 512)
        kwargs_1228 = {}
        # Getting the type of 'self' (line 512)
        self_1226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 15), 'self', False)
        # Obtaining the member 'get_familyname' of a type (line 512)
        get_familyname_1227 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 512, 15), self_1226, 'get_familyname')
        # Calling get_familyname(args, kwargs) (line 512)
        get_familyname_call_result_1229 = invoke(stypy.reporting.localization.Localization(__file__, 512, 15), get_familyname_1227, *[], **kwargs_1228)
        
        # Assigning a type to the variable 'stypy_return_type' (line 512)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 512, 8), 'stypy_return_type', get_familyname_call_result_1229)
        
        # ################# End of 'family_name(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'family_name' in the type store
        # Getting the type of 'stypy_return_type' (line 510)
        stypy_return_type_1230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1230)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'family_name'
        return stypy_return_type_1230


    @norecursion
    def get_weight(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_weight'
        module_type_store = module_type_store.open_function_context('get_weight', 514, 4, False)
        # Assigning a type to the variable 'self' (line 515)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 515, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        AFM.get_weight.__dict__.__setitem__('stypy_localization', localization)
        AFM.get_weight.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        AFM.get_weight.__dict__.__setitem__('stypy_type_store', module_type_store)
        AFM.get_weight.__dict__.__setitem__('stypy_function_name', 'AFM.get_weight')
        AFM.get_weight.__dict__.__setitem__('stypy_param_names_list', [])
        AFM.get_weight.__dict__.__setitem__('stypy_varargs_param_name', None)
        AFM.get_weight.__dict__.__setitem__('stypy_kwargs_param_name', None)
        AFM.get_weight.__dict__.__setitem__('stypy_call_defaults', defaults)
        AFM.get_weight.__dict__.__setitem__('stypy_call_varargs', varargs)
        AFM.get_weight.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        AFM.get_weight.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'AFM.get_weight', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_weight', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_weight(...)' code ##################

        unicode_1231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 515, 8), 'unicode', u"Return the font weight, e.g., 'Bold' or 'Roman'")
        
        # Obtaining the type of the subscript
        str_1232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 516, 28), 'str', 'Weight')
        # Getting the type of 'self' (line 516)
        self_1233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 15), 'self')
        # Obtaining the member '_header' of a type (line 516)
        _header_1234 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 516, 15), self_1233, '_header')
        # Obtaining the member '__getitem__' of a type (line 516)
        getitem___1235 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 516, 15), _header_1234, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 516)
        subscript_call_result_1236 = invoke(stypy.reporting.localization.Localization(__file__, 516, 15), getitem___1235, str_1232)
        
        # Assigning a type to the variable 'stypy_return_type' (line 516)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 516, 8), 'stypy_return_type', subscript_call_result_1236)
        
        # ################# End of 'get_weight(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_weight' in the type store
        # Getting the type of 'stypy_return_type' (line 514)
        stypy_return_type_1237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1237)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_weight'
        return stypy_return_type_1237


    @norecursion
    def get_angle(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_angle'
        module_type_store = module_type_store.open_function_context('get_angle', 518, 4, False)
        # Assigning a type to the variable 'self' (line 519)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 519, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        AFM.get_angle.__dict__.__setitem__('stypy_localization', localization)
        AFM.get_angle.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        AFM.get_angle.__dict__.__setitem__('stypy_type_store', module_type_store)
        AFM.get_angle.__dict__.__setitem__('stypy_function_name', 'AFM.get_angle')
        AFM.get_angle.__dict__.__setitem__('stypy_param_names_list', [])
        AFM.get_angle.__dict__.__setitem__('stypy_varargs_param_name', None)
        AFM.get_angle.__dict__.__setitem__('stypy_kwargs_param_name', None)
        AFM.get_angle.__dict__.__setitem__('stypy_call_defaults', defaults)
        AFM.get_angle.__dict__.__setitem__('stypy_call_varargs', varargs)
        AFM.get_angle.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        AFM.get_angle.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'AFM.get_angle', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_angle', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_angle(...)' code ##################

        unicode_1238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 519, 8), 'unicode', u'Return the fontangle as float')
        
        # Obtaining the type of the subscript
        str_1239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 520, 28), 'str', 'ItalicAngle')
        # Getting the type of 'self' (line 520)
        self_1240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 15), 'self')
        # Obtaining the member '_header' of a type (line 520)
        _header_1241 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 520, 15), self_1240, '_header')
        # Obtaining the member '__getitem__' of a type (line 520)
        getitem___1242 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 520, 15), _header_1241, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 520)
        subscript_call_result_1243 = invoke(stypy.reporting.localization.Localization(__file__, 520, 15), getitem___1242, str_1239)
        
        # Assigning a type to the variable 'stypy_return_type' (line 520)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 520, 8), 'stypy_return_type', subscript_call_result_1243)
        
        # ################# End of 'get_angle(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_angle' in the type store
        # Getting the type of 'stypy_return_type' (line 518)
        stypy_return_type_1244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1244)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_angle'
        return stypy_return_type_1244


    @norecursion
    def get_capheight(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_capheight'
        module_type_store = module_type_store.open_function_context('get_capheight', 522, 4, False)
        # Assigning a type to the variable 'self' (line 523)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 523, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        AFM.get_capheight.__dict__.__setitem__('stypy_localization', localization)
        AFM.get_capheight.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        AFM.get_capheight.__dict__.__setitem__('stypy_type_store', module_type_store)
        AFM.get_capheight.__dict__.__setitem__('stypy_function_name', 'AFM.get_capheight')
        AFM.get_capheight.__dict__.__setitem__('stypy_param_names_list', [])
        AFM.get_capheight.__dict__.__setitem__('stypy_varargs_param_name', None)
        AFM.get_capheight.__dict__.__setitem__('stypy_kwargs_param_name', None)
        AFM.get_capheight.__dict__.__setitem__('stypy_call_defaults', defaults)
        AFM.get_capheight.__dict__.__setitem__('stypy_call_varargs', varargs)
        AFM.get_capheight.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        AFM.get_capheight.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'AFM.get_capheight', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_capheight', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_capheight(...)' code ##################

        unicode_1245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 523, 8), 'unicode', u'Return the cap height as float')
        
        # Obtaining the type of the subscript
        str_1246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 524, 28), 'str', 'CapHeight')
        # Getting the type of 'self' (line 524)
        self_1247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 15), 'self')
        # Obtaining the member '_header' of a type (line 524)
        _header_1248 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 524, 15), self_1247, '_header')
        # Obtaining the member '__getitem__' of a type (line 524)
        getitem___1249 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 524, 15), _header_1248, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 524)
        subscript_call_result_1250 = invoke(stypy.reporting.localization.Localization(__file__, 524, 15), getitem___1249, str_1246)
        
        # Assigning a type to the variable 'stypy_return_type' (line 524)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 524, 8), 'stypy_return_type', subscript_call_result_1250)
        
        # ################# End of 'get_capheight(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_capheight' in the type store
        # Getting the type of 'stypy_return_type' (line 522)
        stypy_return_type_1251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1251)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_capheight'
        return stypy_return_type_1251


    @norecursion
    def get_xheight(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_xheight'
        module_type_store = module_type_store.open_function_context('get_xheight', 526, 4, False)
        # Assigning a type to the variable 'self' (line 527)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 527, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        AFM.get_xheight.__dict__.__setitem__('stypy_localization', localization)
        AFM.get_xheight.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        AFM.get_xheight.__dict__.__setitem__('stypy_type_store', module_type_store)
        AFM.get_xheight.__dict__.__setitem__('stypy_function_name', 'AFM.get_xheight')
        AFM.get_xheight.__dict__.__setitem__('stypy_param_names_list', [])
        AFM.get_xheight.__dict__.__setitem__('stypy_varargs_param_name', None)
        AFM.get_xheight.__dict__.__setitem__('stypy_kwargs_param_name', None)
        AFM.get_xheight.__dict__.__setitem__('stypy_call_defaults', defaults)
        AFM.get_xheight.__dict__.__setitem__('stypy_call_varargs', varargs)
        AFM.get_xheight.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        AFM.get_xheight.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'AFM.get_xheight', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_xheight', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_xheight(...)' code ##################

        unicode_1252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 527, 8), 'unicode', u'Return the xheight as float')
        
        # Obtaining the type of the subscript
        str_1253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 528, 28), 'str', 'XHeight')
        # Getting the type of 'self' (line 528)
        self_1254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 15), 'self')
        # Obtaining the member '_header' of a type (line 528)
        _header_1255 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 528, 15), self_1254, '_header')
        # Obtaining the member '__getitem__' of a type (line 528)
        getitem___1256 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 528, 15), _header_1255, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 528)
        subscript_call_result_1257 = invoke(stypy.reporting.localization.Localization(__file__, 528, 15), getitem___1256, str_1253)
        
        # Assigning a type to the variable 'stypy_return_type' (line 528)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 528, 8), 'stypy_return_type', subscript_call_result_1257)
        
        # ################# End of 'get_xheight(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_xheight' in the type store
        # Getting the type of 'stypy_return_type' (line 526)
        stypy_return_type_1258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1258)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_xheight'
        return stypy_return_type_1258


    @norecursion
    def get_underline_thickness(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_underline_thickness'
        module_type_store = module_type_store.open_function_context('get_underline_thickness', 530, 4, False)
        # Assigning a type to the variable 'self' (line 531)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 531, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        AFM.get_underline_thickness.__dict__.__setitem__('stypy_localization', localization)
        AFM.get_underline_thickness.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        AFM.get_underline_thickness.__dict__.__setitem__('stypy_type_store', module_type_store)
        AFM.get_underline_thickness.__dict__.__setitem__('stypy_function_name', 'AFM.get_underline_thickness')
        AFM.get_underline_thickness.__dict__.__setitem__('stypy_param_names_list', [])
        AFM.get_underline_thickness.__dict__.__setitem__('stypy_varargs_param_name', None)
        AFM.get_underline_thickness.__dict__.__setitem__('stypy_kwargs_param_name', None)
        AFM.get_underline_thickness.__dict__.__setitem__('stypy_call_defaults', defaults)
        AFM.get_underline_thickness.__dict__.__setitem__('stypy_call_varargs', varargs)
        AFM.get_underline_thickness.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        AFM.get_underline_thickness.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'AFM.get_underline_thickness', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_underline_thickness', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_underline_thickness(...)' code ##################

        unicode_1259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 531, 8), 'unicode', u'Return the underline thickness as float')
        
        # Obtaining the type of the subscript
        str_1260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 532, 28), 'str', 'UnderlineThickness')
        # Getting the type of 'self' (line 532)
        self_1261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 15), 'self')
        # Obtaining the member '_header' of a type (line 532)
        _header_1262 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 532, 15), self_1261, '_header')
        # Obtaining the member '__getitem__' of a type (line 532)
        getitem___1263 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 532, 15), _header_1262, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 532)
        subscript_call_result_1264 = invoke(stypy.reporting.localization.Localization(__file__, 532, 15), getitem___1263, str_1260)
        
        # Assigning a type to the variable 'stypy_return_type' (line 532)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 532, 8), 'stypy_return_type', subscript_call_result_1264)
        
        # ################# End of 'get_underline_thickness(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_underline_thickness' in the type store
        # Getting the type of 'stypy_return_type' (line 530)
        stypy_return_type_1265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1265)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_underline_thickness'
        return stypy_return_type_1265


    @norecursion
    def get_horizontal_stem_width(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_horizontal_stem_width'
        module_type_store = module_type_store.open_function_context('get_horizontal_stem_width', 534, 4, False)
        # Assigning a type to the variable 'self' (line 535)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 535, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        AFM.get_horizontal_stem_width.__dict__.__setitem__('stypy_localization', localization)
        AFM.get_horizontal_stem_width.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        AFM.get_horizontal_stem_width.__dict__.__setitem__('stypy_type_store', module_type_store)
        AFM.get_horizontal_stem_width.__dict__.__setitem__('stypy_function_name', 'AFM.get_horizontal_stem_width')
        AFM.get_horizontal_stem_width.__dict__.__setitem__('stypy_param_names_list', [])
        AFM.get_horizontal_stem_width.__dict__.__setitem__('stypy_varargs_param_name', None)
        AFM.get_horizontal_stem_width.__dict__.__setitem__('stypy_kwargs_param_name', None)
        AFM.get_horizontal_stem_width.__dict__.__setitem__('stypy_call_defaults', defaults)
        AFM.get_horizontal_stem_width.__dict__.__setitem__('stypy_call_varargs', varargs)
        AFM.get_horizontal_stem_width.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        AFM.get_horizontal_stem_width.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'AFM.get_horizontal_stem_width', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_horizontal_stem_width', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_horizontal_stem_width(...)' code ##################

        unicode_1266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 538, (-1)), 'unicode', u'\n        Return the standard horizontal stem width as float, or *None* if\n        not specified in AFM file.\n        ')
        
        # Call to get(...): (line 539)
        # Processing the call arguments (line 539)
        str_1270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 539, 32), 'str', 'StdHW')
        # Getting the type of 'None' (line 539)
        None_1271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 42), 'None', False)
        # Processing the call keyword arguments (line 539)
        kwargs_1272 = {}
        # Getting the type of 'self' (line 539)
        self_1267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 15), 'self', False)
        # Obtaining the member '_header' of a type (line 539)
        _header_1268 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 539, 15), self_1267, '_header')
        # Obtaining the member 'get' of a type (line 539)
        get_1269 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 539, 15), _header_1268, 'get')
        # Calling get(args, kwargs) (line 539)
        get_call_result_1273 = invoke(stypy.reporting.localization.Localization(__file__, 539, 15), get_1269, *[str_1270, None_1271], **kwargs_1272)
        
        # Assigning a type to the variable 'stypy_return_type' (line 539)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 539, 8), 'stypy_return_type', get_call_result_1273)
        
        # ################# End of 'get_horizontal_stem_width(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_horizontal_stem_width' in the type store
        # Getting the type of 'stypy_return_type' (line 534)
        stypy_return_type_1274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1274)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_horizontal_stem_width'
        return stypy_return_type_1274


    @norecursion
    def get_vertical_stem_width(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_vertical_stem_width'
        module_type_store = module_type_store.open_function_context('get_vertical_stem_width', 541, 4, False)
        # Assigning a type to the variable 'self' (line 542)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 542, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        AFM.get_vertical_stem_width.__dict__.__setitem__('stypy_localization', localization)
        AFM.get_vertical_stem_width.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        AFM.get_vertical_stem_width.__dict__.__setitem__('stypy_type_store', module_type_store)
        AFM.get_vertical_stem_width.__dict__.__setitem__('stypy_function_name', 'AFM.get_vertical_stem_width')
        AFM.get_vertical_stem_width.__dict__.__setitem__('stypy_param_names_list', [])
        AFM.get_vertical_stem_width.__dict__.__setitem__('stypy_varargs_param_name', None)
        AFM.get_vertical_stem_width.__dict__.__setitem__('stypy_kwargs_param_name', None)
        AFM.get_vertical_stem_width.__dict__.__setitem__('stypy_call_defaults', defaults)
        AFM.get_vertical_stem_width.__dict__.__setitem__('stypy_call_varargs', varargs)
        AFM.get_vertical_stem_width.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        AFM.get_vertical_stem_width.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'AFM.get_vertical_stem_width', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_vertical_stem_width', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_vertical_stem_width(...)' code ##################

        unicode_1275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 545, (-1)), 'unicode', u'\n        Return the standard vertical stem width as float, or *None* if\n        not specified in AFM file.\n        ')
        
        # Call to get(...): (line 546)
        # Processing the call arguments (line 546)
        str_1279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 546, 32), 'str', 'StdVW')
        # Getting the type of 'None' (line 546)
        None_1280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 42), 'None', False)
        # Processing the call keyword arguments (line 546)
        kwargs_1281 = {}
        # Getting the type of 'self' (line 546)
        self_1276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 15), 'self', False)
        # Obtaining the member '_header' of a type (line 546)
        _header_1277 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 546, 15), self_1276, '_header')
        # Obtaining the member 'get' of a type (line 546)
        get_1278 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 546, 15), _header_1277, 'get')
        # Calling get(args, kwargs) (line 546)
        get_call_result_1282 = invoke(stypy.reporting.localization.Localization(__file__, 546, 15), get_1278, *[str_1279, None_1280], **kwargs_1281)
        
        # Assigning a type to the variable 'stypy_return_type' (line 546)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 546, 8), 'stypy_return_type', get_call_result_1282)
        
        # ################# End of 'get_vertical_stem_width(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_vertical_stem_width' in the type store
        # Getting the type of 'stypy_return_type' (line 541)
        stypy_return_type_1283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1283)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_vertical_stem_width'
        return stypy_return_type_1283


# Assigning a type to the variable 'AFM' (line 328)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 328, 0), 'AFM', AFM)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
