
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: A module for finding, managing, and using fonts across platforms.
3: 
4: This module provides a single :class:`FontManager` instance that can
5: be shared across backends and platforms.  The :func:`findfont`
6: function returns the best TrueType (TTF) font file in the local or
7: system font path that matches the specified :class:`FontProperties`
8: instance.  The :class:`FontManager` also handles Adobe Font Metrics
9: (AFM) font files for use by the PostScript backend.
10: 
11: The design is based on the `W3C Cascading Style Sheet, Level 1 (CSS1)
12: font specification <http://www.w3.org/TR/1998/REC-CSS2-19980512/>`_.
13: Future versions may implement the Level 2 or 2.1 specifications.
14: 
15: Experimental support is included for using `fontconfig` on Unix
16: variant platforms (Linux, OS X, Solaris).  To enable it, set the
17: constant ``USE_FONTCONFIG`` in this file to ``True``.  Fontconfig has
18: the advantage that it is the standard way to look up fonts on X11
19: platforms, so if a font is installed, it is much more likely to be
20: found.
21: '''
22: from __future__ import (absolute_import, division, print_function,
23:                         unicode_literals)
24: 
25: import six
26: from six.moves import cPickle as pickle
27: 
28: '''
29: KNOWN ISSUES
30: 
31:   - documentation
32:   - font variant is untested
33:   - font stretch is incomplete
34:   - font size is incomplete
35:   - font size_adjust is incomplete
36:   - default font algorithm needs improvement and testing
37:   - setWeights function needs improvement
38:   - 'light' is an invalid weight value, remove it.
39:   - update_fonts not implemented
40: 
41: Authors   : John Hunter <jdhunter@ace.bsd.uchicago.edu>
42:             Paul Barrett <Barrett@STScI.Edu>
43:             Michael Droettboom <mdroe@STScI.edu>
44: Copyright : John Hunter (2004,2005), Paul Barrett (2004,2005)
45: License   : matplotlib license (PSF compatible)
46:             The font directory code is from ttfquery,
47:             see license/LICENSE_TTFQUERY.
48: '''
49: 
50: from collections import Iterable
51: import json
52: import os
53: import sys
54: from threading import Timer
55: import warnings
56: 
57: import matplotlib
58: from matplotlib import afm, cbook, ft2font, rcParams, get_cachedir
59: from matplotlib.compat import subprocess
60: from matplotlib.fontconfig_pattern import (
61:     parse_fontconfig_pattern, generate_fontconfig_pattern)
62: 
63: try:
64:     from functools import lru_cache
65: except ImportError:
66:     from backports.functools_lru_cache import lru_cache
67: 
68: 
69: USE_FONTCONFIG = False
70: verbose = matplotlib.verbose
71: 
72: font_scalings = {
73:     'xx-small' : 0.579,
74:     'x-small'  : 0.694,
75:     'small'    : 0.833,
76:     'medium'   : 1.0,
77:     'large'    : 1.200,
78:     'x-large'  : 1.440,
79:     'xx-large' : 1.728,
80:     'larger'   : 1.2,
81:     'smaller'  : 0.833,
82:     None       : 1.0}
83: 
84: stretch_dict = {
85:     'ultra-condensed' : 100,
86:     'extra-condensed' : 200,
87:     'condensed'       : 300,
88:     'semi-condensed'  : 400,
89:     'normal'          : 500,
90:     'semi-expanded'   : 600,
91:     'expanded'        : 700,
92:     'extra-expanded'  : 800,
93:     'ultra-expanded'  : 900}
94: 
95: weight_dict = {
96:     'ultralight' : 100,
97:     'light'      : 200,
98:     'normal'     : 400,
99:     'regular'    : 400,
100:     'book'       : 400,
101:     'medium'     : 500,
102:     'roman'      : 500,
103:     'semibold'   : 600,
104:     'demibold'   : 600,
105:     'demi'       : 600,
106:     'bold'       : 700,
107:     'heavy'      : 800,
108:     'extra bold' : 800,
109:     'black'      : 900}
110: 
111: font_family_aliases = {
112:     'serif',
113:     'sans-serif',
114:     'sans serif',
115:     'cursive',
116:     'fantasy',
117:     'monospace',
118:     'sans'}
119: 
120: #  OS Font paths
121: MSFolders = \
122:     r'Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders'
123: 
124: 
125: MSFontDirectories = [
126:     r'SOFTWARE\Microsoft\Windows NT\CurrentVersion\Fonts',
127:     r'SOFTWARE\Microsoft\Windows\CurrentVersion\Fonts']
128: 
129: 
130: X11FontDirectories = [
131:     # an old standard installation point
132:     "/usr/X11R6/lib/X11/fonts/TTF/",
133:     "/usr/X11/lib/X11/fonts",
134:     # here is the new standard location for fonts
135:     "/usr/share/fonts/",
136:     # documented as a good place to install new fonts
137:     "/usr/local/share/fonts/",
138:     # common application, not really useful
139:     "/usr/lib/openoffice/share/fonts/truetype/",
140:     ]
141: 
142: OSXFontDirectories = [
143:     "/Library/Fonts/",
144:     "/Network/Library/Fonts/",
145:     "/System/Library/Fonts/",
146:     # fonts installed via MacPorts
147:     "/opt/local/share/fonts"
148:     ""
149: ]
150: 
151: if not USE_FONTCONFIG and sys.platform != 'win32':
152:     home = os.environ.get('HOME')
153:     if home is not None:
154:         # user fonts on OSX
155:         path = os.path.join(home, 'Library', 'Fonts')
156:         OSXFontDirectories.append(path)
157:         path = os.path.join(home, '.fonts')
158:         X11FontDirectories.append(path)
159: 
160: 
161: def get_fontext_synonyms(fontext):
162:     '''
163:     Return a list of file extensions extensions that are synonyms for
164:     the given file extension *fileext*.
165:     '''
166:     return {'ttf': ('ttf', 'otf'),
167:             'otf': ('ttf', 'otf'),
168:             'afm': ('afm',)}[fontext]
169: 
170: 
171: def list_fonts(directory, extensions):
172:     '''
173:     Return a list of all fonts matching any of the extensions,
174:     possibly upper-cased, found recursively under the directory.
175:     '''
176:     pattern = ';'.join(['*.%s;*.%s' % (ext, ext.upper())
177:                         for ext in extensions])
178:     return cbook.listFiles(directory, pattern)
179: 
180: 
181: def win32FontDirectory():
182:     '''
183:     Return the user-specified font directory for Win32.  This is
184:     looked up from the registry key::
185: 
186:       \\\\HKEY_CURRENT_USER\\Software\\Microsoft\\Windows\\CurrentVersion\\Explorer\\Shell Folders\\Fonts
187: 
188:     If the key is not found, $WINDIR/Fonts will be returned.
189:     '''
190:     try:
191:         from six.moves import winreg
192:     except ImportError:
193:         pass  # Fall through to default
194:     else:
195:         try:
196:             user = winreg.OpenKey(winreg.HKEY_CURRENT_USER, MSFolders)
197:             try:
198:                 try:
199:                     return winreg.QueryValueEx(user, 'Fonts')[0]
200:                 except OSError:
201:                     pass  # Fall through to default
202:             finally:
203:                 winreg.CloseKey(user)
204:         except OSError:
205:             pass  # Fall through to default
206:     return os.path.join(os.environ['WINDIR'], 'Fonts')
207: 
208: 
209: def win32InstalledFonts(directory=None, fontext='ttf'):
210:     '''
211:     Search for fonts in the specified font directory, or use the
212:     system directories if none given.  A list of TrueType font
213:     filenames are returned by default, or AFM fonts if *fontext* ==
214:     'afm'.
215:     '''
216: 
217:     from six.moves import winreg
218:     if directory is None:
219:         directory = win32FontDirectory()
220: 
221:     fontext = get_fontext_synonyms(fontext)
222: 
223:     key, items = None, {}
224:     for fontdir in MSFontDirectories:
225:         try:
226:             local = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, fontdir)
227:         except OSError:
228:             continue
229: 
230:         if not local:
231:             return list_fonts(directory, fontext)
232:         try:
233:             for j in range(winreg.QueryInfoKey(local)[1]):
234:                 try:
235:                     key, direc, any = winreg.EnumValue( local, j)
236:                     if not isinstance(direc, six.string_types):
237:                         continue
238:                     if not os.path.dirname(direc):
239:                         direc = os.path.join(directory, direc)
240:                     direc = os.path.abspath(direc).lower()
241:                     if os.path.splitext(direc)[1][1:] in fontext:
242:                         items[direc] = 1
243:                 except EnvironmentError:
244:                     continue
245:                 except WindowsError:
246:                     continue
247:                 except MemoryError:
248:                     continue
249:             return list(items)
250:         finally:
251:             winreg.CloseKey(local)
252:     return None
253: 
254: 
255: def OSXInstalledFonts(directories=None, fontext='ttf'):
256:     '''
257:     Get list of font files on OS X - ignores font suffix by default.
258:     '''
259:     if directories is None:
260:         directories = OSXFontDirectories
261: 
262:     fontext = get_fontext_synonyms(fontext)
263: 
264:     files = []
265:     for path in directories:
266:         if fontext is None:
267:             files.extend(cbook.listFiles(path, '*'))
268:         else:
269:             files.extend(list_fonts(path, fontext))
270:     return files
271: 
272: 
273: @lru_cache()
274: def _call_fc_list():
275:     '''Cache and list the font filenames known to `fc-list`.
276:     '''
277:     # Delay the warning by 5s.
278:     timer = Timer(5, lambda: warnings.warn(
279:         'Matplotlib is building the font cache using fc-list. '
280:         'This may take a moment.'))
281:     timer.start()
282:     try:
283:         out = subprocess.check_output([str('fc-list'), '--format=%{file}\\n'])
284:     except (OSError, subprocess.CalledProcessError):
285:         return []
286:     finally:
287:         timer.cancel()
288:     fnames = []
289:     for fname in out.split(b'\n'):
290:         try:
291:             fname = six.text_type(fname, sys.getfilesystemencoding())
292:         except UnicodeDecodeError:
293:             continue
294:         fnames.append(fname)
295:     return fnames
296: 
297: 
298: def get_fontconfig_fonts(fontext='ttf'):
299:     '''List the font filenames known to `fc-list` having the given extension.
300:     '''
301:     fontext = get_fontext_synonyms(fontext)
302:     return [fname for fname in _call_fc_list()
303:             if os.path.splitext(fname)[1][1:] in fontext]
304: 
305: 
306: def findSystemFonts(fontpaths=None, fontext='ttf'):
307:     '''
308:     Search for fonts in the specified font paths.  If no paths are
309:     given, will use a standard set of system paths, as well as the
310:     list of fonts tracked by fontconfig if fontconfig is installed and
311:     available.  A list of TrueType fonts are returned by default with
312:     AFM fonts as an option.
313:     '''
314:     fontfiles = set()
315:     fontexts = get_fontext_synonyms(fontext)
316: 
317:     if fontpaths is None:
318:         if sys.platform == 'win32':
319:             fontdir = win32FontDirectory()
320: 
321:             fontpaths = [fontdir]
322:             # now get all installed fonts directly...
323:             for f in win32InstalledFonts(fontdir):
324:                 base, ext = os.path.splitext(f)
325:                 if len(ext)>1 and ext[1:].lower() in fontexts:
326:                     fontfiles.add(f)
327:         else:
328:             fontpaths = X11FontDirectories
329:             # check for OS X & load its fonts if present
330:             if sys.platform == 'darwin':
331:                 for f in OSXInstalledFonts(fontext=fontext):
332:                     fontfiles.add(f)
333: 
334:             for f in get_fontconfig_fonts(fontext):
335:                 fontfiles.add(f)
336: 
337:     elif isinstance(fontpaths, six.string_types):
338:         fontpaths = [fontpaths]
339: 
340:     for path in fontpaths:
341:         files = list_fonts(path, fontexts)
342:         for fname in files:
343:             fontfiles.add(os.path.abspath(fname))
344: 
345:     return [fname for fname in fontfiles if os.path.exists(fname)]
346: 
347: 
348: @cbook.deprecated("2.1")
349: def weight_as_number(weight):
350:     '''
351:     Return the weight property as a numeric value.  String values
352:     are converted to their corresponding numeric value.
353:     '''
354:     if isinstance(weight, six.string_types):
355:         try:
356:             weight = weight_dict[weight.lower()]
357:         except KeyError:
358:             weight = 400
359:     elif weight in range(100, 1000, 100):
360:         pass
361:     else:
362:         raise ValueError('weight not a valid integer')
363:     return weight
364: 
365: 
366: class FontEntry(object):
367:     '''
368:     A class for storing Font properties.  It is used when populating
369:     the font lookup dictionary.
370:     '''
371:     def __init__(self,
372:                  fname  ='',
373:                  name   ='',
374:                  style  ='normal',
375:                  variant='normal',
376:                  weight ='normal',
377:                  stretch='normal',
378:                  size   ='medium',
379:                  ):
380:         self.fname   = fname
381:         self.name    = name
382:         self.style   = style
383:         self.variant = variant
384:         self.weight  = weight
385:         self.stretch = stretch
386:         try:
387:             self.size = str(float(size))
388:         except ValueError:
389:             self.size = size
390: 
391:     def __repr__(self):
392:         return "<Font '%s' (%s) %s %s %s %s>" % (
393:             self.name, os.path.basename(self.fname), self.style, self.variant,
394:             self.weight, self.stretch)
395: 
396: 
397: def ttfFontProperty(font):
398:     '''
399:     A function for populating the :class:`FontKey` by extracting
400:     information from the TrueType font file.
401: 
402:     *font* is a :class:`FT2Font` instance.
403:     '''
404:     name = font.family_name
405: 
406:     #  Styles are: italic, oblique, and normal (default)
407: 
408:     sfnt = font.get_sfnt()
409:     sfnt2 = sfnt.get((1,0,0,2))
410:     sfnt4 = sfnt.get((1,0,0,4))
411:     if sfnt2:
412:         sfnt2 = sfnt2.decode('macroman').lower()
413:     else:
414:         sfnt2 = ''
415:     if sfnt4:
416:         sfnt4 = sfnt4.decode('macroman').lower()
417:     else:
418:         sfnt4 = ''
419:     if sfnt4.find('oblique') >= 0:
420:         style = 'oblique'
421:     elif sfnt4.find('italic') >= 0:
422:         style = 'italic'
423:     elif sfnt2.find('regular') >= 0:
424:         style = 'normal'
425:     elif font.style_flags & ft2font.ITALIC:
426:         style = 'italic'
427:     else:
428:         style = 'normal'
429: 
430:     #  Variants are: small-caps and normal (default)
431: 
432:     #  !!!!  Untested
433:     if name.lower() in ['capitals', 'small-caps']:
434:         variant = 'small-caps'
435:     else:
436:         variant = 'normal'
437: 
438:     weight = next((w for w in weight_dict if sfnt4.find(w) >= 0), None)
439:     if not weight:
440:         if font.style_flags & ft2font.BOLD:
441:             weight = 700
442:         else:
443:             weight = 400
444: 
445:     #  Stretch can be absolute and relative
446:     #  Absolute stretches are: ultra-condensed, extra-condensed, condensed,
447:     #    semi-condensed, normal, semi-expanded, expanded, extra-expanded,
448:     #    and ultra-expanded.
449:     #  Relative stretches are: wider, narrower
450:     #  Child value is: inherit
451: 
452:     if (sfnt4.find('narrow') >= 0 or sfnt4.find('condensed') >= 0 or
453:             sfnt4.find('cond') >= 0):
454:         stretch = 'condensed'
455:     elif sfnt4.find('demi cond') >= 0:
456:         stretch = 'semi-condensed'
457:     elif sfnt4.find('wide') >= 0 or sfnt4.find('expanded') >= 0:
458:         stretch = 'expanded'
459:     else:
460:         stretch = 'normal'
461: 
462:     #  Sizes can be absolute and relative.
463:     #  Absolute sizes are: xx-small, x-small, small, medium, large, x-large,
464:     #    and xx-large.
465:     #  Relative sizes are: larger, smaller
466:     #  Length value is an absolute font size, e.g., 12pt
467:     #  Percentage values are in 'em's.  Most robust specification.
468: 
469:     #  !!!!  Incomplete
470:     if font.scalable:
471:         size = 'scalable'
472:     else:
473:         size = str(float(font.get_fontsize()))
474: 
475:     #  !!!!  Incomplete
476:     size_adjust = None
477: 
478:     return FontEntry(font.fname, name, style, variant, weight, stretch, size)
479: 
480: 
481: def afmFontProperty(fontpath, font):
482:     '''
483:     A function for populating a :class:`FontKey` instance by
484:     extracting information from the AFM font file.
485: 
486:     *font* is a class:`AFM` instance.
487:     '''
488: 
489:     name = font.get_familyname()
490:     fontname = font.get_fontname().lower()
491: 
492:     #  Styles are: italic, oblique, and normal (default)
493: 
494:     if font.get_angle() != 0 or name.lower().find('italic') >= 0:
495:         style = 'italic'
496:     elif name.lower().find('oblique') >= 0:
497:         style = 'oblique'
498:     else:
499:         style = 'normal'
500: 
501:     #  Variants are: small-caps and normal (default)
502: 
503:     # !!!!  Untested
504:     if name.lower() in ['capitals', 'small-caps']:
505:         variant = 'small-caps'
506:     else:
507:         variant = 'normal'
508: 
509:     weight = font.get_weight().lower()
510: 
511:     #  Stretch can be absolute and relative
512:     #  Absolute stretches are: ultra-condensed, extra-condensed, condensed,
513:     #    semi-condensed, normal, semi-expanded, expanded, extra-expanded,
514:     #    and ultra-expanded.
515:     #  Relative stretches are: wider, narrower
516:     #  Child value is: inherit
517:     if fontname.find('narrow') >= 0 or fontname.find('condensed') >= 0 or \
518:            fontname.find('cond') >= 0:
519:         stretch = 'condensed'
520:     elif fontname.find('demi cond') >= 0:
521:         stretch = 'semi-condensed'
522:     elif fontname.find('wide') >= 0 or fontname.find('expanded') >= 0:
523:         stretch = 'expanded'
524:     else:
525:         stretch = 'normal'
526: 
527:     #  Sizes can be absolute and relative.
528:     #  Absolute sizes are: xx-small, x-small, small, medium, large, x-large,
529:     #    and xx-large.
530:     #  Relative sizes are: larger, smaller
531:     #  Length value is an absolute font size, e.g., 12pt
532:     #  Percentage values are in 'em's.  Most robust specification.
533: 
534:     #  All AFM fonts are apparently scalable.
535: 
536:     size = 'scalable'
537: 
538:     # !!!!  Incomplete
539:     size_adjust = None
540: 
541:     return FontEntry(fontpath, name, style, variant, weight, stretch, size)
542: 
543: 
544: def createFontList(fontfiles, fontext='ttf'):
545:     '''
546:     A function to create a font lookup list.  The default is to create
547:     a list of TrueType fonts.  An AFM font list can optionally be
548:     created.
549:     '''
550: 
551:     fontlist = []
552:     #  Add fonts from list of known font files.
553:     seen = {}
554:     for fpath in fontfiles:
555:         verbose.report('createFontDict: %s' % (fpath), 'debug')
556:         fname = os.path.split(fpath)[1]
557:         if fname in seen:
558:             continue
559:         else:
560:             seen[fname] = 1
561:         if fontext == 'afm':
562:             try:
563:                 fh = open(fpath, 'rb')
564:             except EnvironmentError:
565:                 verbose.report("Could not open font file %s" % fpath)
566:                 continue
567:             try:
568:                 font = afm.AFM(fh)
569:             except RuntimeError:
570:                 verbose.report("Could not parse font file %s" % fpath)
571:                 continue
572:             finally:
573:                 fh.close()
574:             try:
575:                 prop = afmFontProperty(fpath, font)
576:             except KeyError:
577:                 continue
578:         else:
579:             try:
580:                 font = ft2font.FT2Font(fpath)
581:             except RuntimeError:
582:                 verbose.report("Could not open font file %s" % fpath)
583:                 continue
584:             except UnicodeError:
585:                 verbose.report("Cannot handle unicode filenames")
586:                 # print >> sys.stderr, 'Bad file is', fpath
587:                 continue
588:             except IOError:
589:                 verbose.report("IO error - cannot open font file %s" % fpath)
590:                 continue
591:             try:
592:                 prop = ttfFontProperty(font)
593:             except (KeyError, RuntimeError, ValueError):
594:                 continue
595: 
596:         fontlist.append(prop)
597:     return fontlist
598: 
599: 
600: class FontProperties(object):
601:     '''
602:     A class for storing and manipulating font properties.
603: 
604:     The font properties are those described in the `W3C Cascading
605:     Style Sheet, Level 1
606:     <http://www.w3.org/TR/1998/REC-CSS2-19980512/>`_ font
607:     specification.  The six properties are:
608: 
609:       - family: A list of font names in decreasing order of priority.
610:         The items may include a generic font family name, either
611:         'serif', 'sans-serif', 'cursive', 'fantasy', or 'monospace'.
612:         In that case, the actual font to be used will be looked up
613:         from the associated rcParam in :file:`matplotlibrc`.
614: 
615:       - style: Either 'normal', 'italic' or 'oblique'.
616: 
617:       - variant: Either 'normal' or 'small-caps'.
618: 
619:       - stretch: A numeric value in the range 0-1000 or one of
620:         'ultra-condensed', 'extra-condensed', 'condensed',
621:         'semi-condensed', 'normal', 'semi-expanded', 'expanded',
622:         'extra-expanded' or 'ultra-expanded'
623: 
624:       - weight: A numeric value in the range 0-1000 or one of
625:         'ultralight', 'light', 'normal', 'regular', 'book', 'medium',
626:         'roman', 'semibold', 'demibold', 'demi', 'bold', 'heavy',
627:         'extra bold', 'black'
628: 
629:       - size: Either an relative value of 'xx-small', 'x-small',
630:         'small', 'medium', 'large', 'x-large', 'xx-large' or an
631:         absolute font size, e.g., 12
632: 
633:     The default font property for TrueType fonts (as specified in the
634:     default :file:`matplotlibrc` file) is::
635: 
636:       sans-serif, normal, normal, normal, normal, scalable.
637: 
638:     Alternatively, a font may be specified using an absolute path to a
639:     .ttf file, by using the *fname* kwarg.
640: 
641:     The preferred usage of font sizes is to use the relative values,
642:     e.g.,  'large', instead of absolute font sizes, e.g., 12.  This
643:     approach allows all text sizes to be made larger or smaller based
644:     on the font manager's default font size.
645: 
646:     This class will also accept a `fontconfig
647:     <https://www.freedesktop.org/wiki/Software/fontconfig/>`_ pattern, if it is
648:     the only argument provided.  See the documentation on `fontconfig patterns
649:     <https://www.freedesktop.org/software/fontconfig/fontconfig-user.html>`_.
650:     This support does not require fontconfig to be installed.  We are merely
651:     borrowing its pattern syntax for use here.
652: 
653:     Note that matplotlib's internal font manager and fontconfig use a
654:     different algorithm to lookup fonts, so the results of the same pattern
655:     may be different in matplotlib than in other applications that use
656:     fontconfig.
657:     '''
658: 
659:     def __init__(self,
660:                  family = None,
661:                  style  = None,
662:                  variant= None,
663:                  weight = None,
664:                  stretch= None,
665:                  size   = None,
666:                  fname  = None, # if this is set, it's a hardcoded filename to use
667:                  _init   = None  # used only by copy()
668:                  ):
669:         self._family = _normalize_font_family(rcParams['font.family'])
670:         self._slant = rcParams['font.style']
671:         self._variant = rcParams['font.variant']
672:         self._weight = rcParams['font.weight']
673:         self._stretch = rcParams['font.stretch']
674:         self._size = rcParams['font.size']
675:         self._file = None
676: 
677:         # This is used only by copy()
678:         if _init is not None:
679:             self.__dict__.update(_init.__dict__)
680:             return
681: 
682:         if isinstance(family, six.string_types):
683:             # Treat family as a fontconfig pattern if it is the only
684:             # parameter provided.
685:             if (style is None and
686:                 variant is None and
687:                 weight is None and
688:                 stretch is None and
689:                 size is None and
690:                 fname is None):
691:                 self.set_fontconfig_pattern(family)
692:                 return
693: 
694:         self.set_family(family)
695:         self.set_style(style)
696:         self.set_variant(variant)
697:         self.set_weight(weight)
698:         self.set_stretch(stretch)
699:         self.set_file(fname)
700:         self.set_size(size)
701: 
702:     def _parse_fontconfig_pattern(self, pattern):
703:         return parse_fontconfig_pattern(pattern)
704: 
705:     def __hash__(self):
706:         l = (tuple(self.get_family()),
707:              self.get_slant(),
708:              self.get_variant(),
709:              self.get_weight(),
710:              self.get_stretch(),
711:              self.get_size_in_points(),
712:              self.get_file())
713:         return hash(l)
714: 
715:     def __eq__(self, other):
716:         return hash(self) == hash(other)
717: 
718:     def __ne__(self, other):
719:         return hash(self) != hash(other)
720: 
721:     def __str__(self):
722:         return self.get_fontconfig_pattern()
723: 
724:     def get_family(self):
725:         '''
726:         Return a list of font names that comprise the font family.
727:         '''
728:         return self._family
729: 
730:     def get_name(self):
731:         '''
732:         Return the name of the font that best matches the font
733:         properties.
734:         '''
735:         return get_font(findfont(self)).family_name
736: 
737:     def get_style(self):
738:         '''
739:         Return the font style.  Values are: 'normal', 'italic' or
740:         'oblique'.
741:         '''
742:         return self._slant
743:     get_slant = get_style
744: 
745:     def get_variant(self):
746:         '''
747:         Return the font variant.  Values are: 'normal' or
748:         'small-caps'.
749:         '''
750:         return self._variant
751: 
752:     def get_weight(self):
753:         '''
754:         Set the font weight.  Options are: A numeric value in the
755:         range 0-1000 or one of 'light', 'normal', 'regular', 'book',
756:         'medium', 'roman', 'semibold', 'demibold', 'demi', 'bold',
757:         'heavy', 'extra bold', 'black'
758:         '''
759:         return self._weight
760: 
761:     def get_stretch(self):
762:         '''
763:         Return the font stretch or width.  Options are: 'ultra-condensed',
764:         'extra-condensed', 'condensed', 'semi-condensed', 'normal',
765:         'semi-expanded', 'expanded', 'extra-expanded', 'ultra-expanded'.
766:         '''
767:         return self._stretch
768: 
769:     def get_size(self):
770:         '''
771:         Return the font size.
772:         '''
773:         return self._size
774: 
775:     def get_size_in_points(self):
776:         return self._size
777: 
778:     def get_file(self):
779:         '''
780:         Return the filename of the associated font.
781:         '''
782:         return self._file
783: 
784:     def get_fontconfig_pattern(self):
785:         '''
786:         Get a fontconfig pattern suitable for looking up the font as
787:         specified with fontconfig's ``fc-match`` utility.
788: 
789:         See the documentation on `fontconfig patterns
790:         <https://www.freedesktop.org/software/fontconfig/fontconfig-user.html>`_.
791: 
792:         This support does not require fontconfig to be installed or
793:         support for it to be enabled.  We are merely borrowing its
794:         pattern syntax for use here.
795:         '''
796:         return generate_fontconfig_pattern(self)
797: 
798:     def set_family(self, family):
799:         '''
800:         Change the font family.  May be either an alias (generic name
801:         is CSS parlance), such as: 'serif', 'sans-serif', 'cursive',
802:         'fantasy', or 'monospace', a real font name or a list of real
803:         font names.  Real font names are not supported when
804:         `text.usetex` is `True`.
805:         '''
806:         if family is None:
807:             family = rcParams['font.family']
808:         self._family = _normalize_font_family(family)
809:     set_name = set_family
810: 
811:     def set_style(self, style):
812:         '''
813:         Set the font style.  Values are: 'normal', 'italic' or
814:         'oblique'.
815:         '''
816:         if style is None:
817:             style = rcParams['font.style']
818:         if style not in ('normal', 'italic', 'oblique'):
819:             raise ValueError("style must be normal, italic or oblique")
820:         self._slant = style
821:     set_slant = set_style
822: 
823:     def set_variant(self, variant):
824:         '''
825:         Set the font variant.  Values are: 'normal' or 'small-caps'.
826:         '''
827:         if variant is None:
828:             variant = rcParams['font.variant']
829:         if variant not in ('normal', 'small-caps'):
830:             raise ValueError("variant must be normal or small-caps")
831:         self._variant = variant
832: 
833:     def set_weight(self, weight):
834:         '''
835:         Set the font weight.  May be either a numeric value in the
836:         range 0-1000 or one of 'ultralight', 'light', 'normal',
837:         'regular', 'book', 'medium', 'roman', 'semibold', 'demibold',
838:         'demi', 'bold', 'heavy', 'extra bold', 'black'
839:         '''
840:         if weight is None:
841:             weight = rcParams['font.weight']
842:         try:
843:             weight = int(weight)
844:             if weight < 0 or weight > 1000:
845:                 raise ValueError()
846:         except ValueError:
847:             if weight not in weight_dict:
848:                 raise ValueError("weight is invalid")
849:         self._weight = weight
850: 
851:     def set_stretch(self, stretch):
852:         '''
853:         Set the font stretch or width.  Options are: 'ultra-condensed',
854:         'extra-condensed', 'condensed', 'semi-condensed', 'normal',
855:         'semi-expanded', 'expanded', 'extra-expanded' or
856:         'ultra-expanded', or a numeric value in the range 0-1000.
857:         '''
858:         if stretch is None:
859:             stretch = rcParams['font.stretch']
860:         try:
861:             stretch = int(stretch)
862:             if stretch < 0 or stretch > 1000:
863:                 raise ValueError()
864:         except ValueError:
865:             if stretch not in stretch_dict:
866:                 raise ValueError("stretch is invalid")
867:         self._stretch = stretch
868: 
869:     def set_size(self, size):
870:         '''
871:         Set the font size.  Either an relative value of 'xx-small',
872:         'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'
873:         or an absolute font size, e.g., 12.
874:         '''
875:         if size is None:
876:             size = rcParams['font.size']
877:         try:
878:             size = float(size)
879:         except ValueError:
880:             try:
881:                 scale = font_scalings[size]
882:             except KeyError:
883:                 raise ValueError(
884:                     "Size is invalid. Valid font size are "
885:                     + ", ".join(map(str, font_scalings)))
886:             else:
887:                 size = scale * FontManager.get_default_size()
888:         self._size = size
889: 
890:     def set_file(self, file):
891:         '''
892:         Set the filename of the fontfile to use.  In this case, all
893:         other properties will be ignored.
894:         '''
895:         self._file = file
896: 
897:     def set_fontconfig_pattern(self, pattern):
898:         '''
899:         Set the properties by parsing a fontconfig *pattern*.
900: 
901:         See the documentation on `fontconfig patterns
902:         <https://www.freedesktop.org/software/fontconfig/fontconfig-user.html>`_.
903: 
904:         This support does not require fontconfig to be installed or
905:         support for it to be enabled.  We are merely borrowing its
906:         pattern syntax for use here.
907:         '''
908:         for key, val in six.iteritems(self._parse_fontconfig_pattern(pattern)):
909:             if type(val) == list:
910:                 getattr(self, "set_" + key)(val[0])
911:             else:
912:                 getattr(self, "set_" + key)(val)
913: 
914:     def copy(self):
915:         '''Return a deep copy of self'''
916:         return FontProperties(_init=self)
917: 
918: 
919: @cbook.deprecated("2.1")
920: def ttfdict_to_fnames(d):
921:     '''
922:     flatten a ttfdict to all the filenames it contains
923:     '''
924:     fnames = []
925:     for named in six.itervalues(d):
926:         for styled in six.itervalues(named):
927:             for variantd in six.itervalues(styled):
928:                 for weightd in six.itervalues(variantd):
929:                     for stretchd in six.itervalues(weightd):
930:                         for fname in six.itervalues(stretchd):
931:                             fnames.append(fname)
932:     return fnames
933: 
934: 
935: class JSONEncoder(json.JSONEncoder):
936:     def default(self, o):
937:         if isinstance(o, FontManager):
938:             return dict(o.__dict__, _class='FontManager')
939:         elif isinstance(o, FontEntry):
940:             return dict(o.__dict__, _class='FontEntry')
941:         else:
942:             return super(JSONEncoder, self).default(o)
943: 
944: 
945: def _json_decode(o):
946:     cls = o.pop('_class', None)
947:     if cls is None:
948:         return o
949:     elif cls == 'FontManager':
950:         r = FontManager.__new__(FontManager)
951:         r.__dict__.update(o)
952:         return r
953:     elif cls == 'FontEntry':
954:         r = FontEntry.__new__(FontEntry)
955:         r.__dict__.update(o)
956:         return r
957:     else:
958:         raise ValueError("don't know how to deserialize _class=%s" % cls)
959: 
960: 
961: def json_dump(data, filename):
962:     '''Dumps a data structure as JSON in the named file.
963:     Handles FontManager and its fields.'''
964: 
965:     with open(filename, 'w') as fh:
966:         json.dump(data, fh, cls=JSONEncoder, indent=2)
967: 
968: 
969: def json_load(filename):
970:     '''Loads a data structure as JSON from the named file.
971:     Handles FontManager and its fields.'''
972: 
973:     with open(filename, 'r') as fh:
974:         return json.load(fh, object_hook=_json_decode)
975: 
976: 
977: def _normalize_font_family(family):
978:     if isinstance(family, six.string_types):
979:         family = [six.text_type(family)]
980:     elif isinstance(family, Iterable):
981:         family = [six.text_type(f) for f in family]
982:     return family
983: 
984: 
985: class TempCache(object):
986:     '''
987:     A class to store temporary caches that are (a) not saved to disk
988:     and (b) invalidated whenever certain font-related
989:     rcParams---namely the family lookup lists---are changed or the
990:     font cache is reloaded.  This avoids the expensive linear search
991:     through all fonts every time a font is looked up.
992:     '''
993:     # A list of rcparam names that, when changed, invalidated this
994:     # cache.
995:     invalidating_rcparams = (
996:         'font.serif', 'font.sans-serif', 'font.cursive', 'font.fantasy',
997:         'font.monospace')
998: 
999:     def __init__(self):
1000:         self._lookup_cache = {}
1001:         self._last_rcParams = self.make_rcparams_key()
1002: 
1003:     def make_rcparams_key(self):
1004:         return [id(fontManager)] + [
1005:             rcParams[param] for param in self.invalidating_rcparams]
1006: 
1007:     def get(self, prop):
1008:         key = self.make_rcparams_key()
1009:         if key != self._last_rcParams:
1010:             self._lookup_cache = {}
1011:             self._last_rcParams = key
1012:         return self._lookup_cache.get(prop)
1013: 
1014:     def set(self, prop, value):
1015:         key = self.make_rcparams_key()
1016:         if key != self._last_rcParams:
1017:             self._lookup_cache = {}
1018:             self._last_rcParams = key
1019:         self._lookup_cache[prop] = value
1020: 
1021: 
1022: class FontManager(object):
1023:     '''
1024:     On import, the :class:`FontManager` singleton instance creates a
1025:     list of TrueType fonts based on the font properties: name, style,
1026:     variant, weight, stretch, and size.  The :meth:`findfont` method
1027:     does a nearest neighbor search to find the font that most closely
1028:     matches the specification.  If no good enough match is found, a
1029:     default font is returned.
1030:     '''
1031:     # Increment this version number whenever the font cache data
1032:     # format or behavior has changed and requires a existing font
1033:     # cache files to be rebuilt.
1034:     __version__ = 201
1035: 
1036:     def __init__(self, size=None, weight='normal'):
1037:         self._version = self.__version__
1038: 
1039:         self.__default_weight = weight
1040:         self.default_size = size
1041: 
1042:         paths = [os.path.join(rcParams['datapath'], 'fonts', 'ttf'),
1043:                  os.path.join(rcParams['datapath'], 'fonts', 'afm'),
1044:                  os.path.join(rcParams['datapath'], 'fonts', 'pdfcorefonts')]
1045: 
1046:         #  Create list of font paths
1047:         for pathname in ['TTFPATH', 'AFMPATH']:
1048:             if pathname in os.environ:
1049:                 ttfpath = os.environ[pathname]
1050:                 if ttfpath.find(';') >= 0: #win32 style
1051:                     paths.extend(ttfpath.split(';'))
1052:                 elif ttfpath.find(':') >= 0: # unix style
1053:                     paths.extend(ttfpath.split(':'))
1054:                 else:
1055:                     paths.append(ttfpath)
1056: 
1057:         verbose.report('font search path %s'%(str(paths)))
1058:         #  Load TrueType fonts and create font dictionary.
1059: 
1060:         self.ttffiles = findSystemFonts(paths) + findSystemFonts()
1061:         self.defaultFamily = {
1062:             'ttf': 'DejaVu Sans',
1063:             'afm': 'Helvetica'}
1064:         self.defaultFont = {}
1065: 
1066:         for fname in self.ttffiles:
1067:             verbose.report('trying fontname %s' % fname, 'debug')
1068:             if fname.lower().find('DejaVuSans.ttf')>=0:
1069:                 self.defaultFont['ttf'] = fname
1070:                 break
1071:         else:
1072:             # use anything
1073:             self.defaultFont['ttf'] = self.ttffiles[0]
1074: 
1075:         self.ttflist = createFontList(self.ttffiles)
1076: 
1077:         self.afmfiles = findSystemFonts(paths, fontext='afm') + \
1078:             findSystemFonts(fontext='afm')
1079:         self.afmlist = createFontList(self.afmfiles, fontext='afm')
1080:         if len(self.afmfiles):
1081:             self.defaultFont['afm'] = self.afmfiles[0]
1082:         else:
1083:             self.defaultFont['afm'] = None
1084: 
1085:     def get_default_weight(self):
1086:         '''
1087:         Return the default font weight.
1088:         '''
1089:         return self.__default_weight
1090: 
1091:     @staticmethod
1092:     def get_default_size():
1093:         '''
1094:         Return the default font size.
1095:         '''
1096:         return rcParams['font.size']
1097: 
1098:     def set_default_weight(self, weight):
1099:         '''
1100:         Set the default font weight.  The initial value is 'normal'.
1101:         '''
1102:         self.__default_weight = weight
1103: 
1104:     def update_fonts(self, filenames):
1105:         '''
1106:         Update the font dictionary with new font files.
1107:         Currently not implemented.
1108:         '''
1109:         #  !!!!  Needs implementing
1110:         raise NotImplementedError
1111: 
1112:     # Each of the scoring functions below should return a value between
1113:     # 0.0 (perfect match) and 1.0 (terrible match)
1114:     def score_family(self, families, family2):
1115:         '''
1116:         Returns a match score between the list of font families in
1117:         *families* and the font family name *family2*.
1118: 
1119:         An exact match at the head of the list returns 0.0.
1120: 
1121:         A match further down the list will return between 0 and 1.
1122: 
1123:         No match will return 1.0.
1124:         '''
1125:         if not isinstance(families, (list, tuple)):
1126:             families = [families]
1127:         elif len(families) == 0:
1128:             return 1.0
1129:         family2 = family2.lower()
1130:         step = 1 / len(families)
1131:         for i, family1 in enumerate(families):
1132:             family1 = family1.lower()
1133:             if family1 in font_family_aliases:
1134:                 if family1 in ('sans', 'sans serif'):
1135:                     family1 = 'sans-serif'
1136:                 options = rcParams['font.' + family1]
1137:                 options = [x.lower() for x in options]
1138:                 if family2 in options:
1139:                     idx = options.index(family2)
1140:                     return (i + (idx / len(options))) * step
1141:             elif family1 == family2:
1142:                 # The score should be weighted by where in the
1143:                 # list the font was found.
1144:                 return i * step
1145:         return 1.0
1146: 
1147:     def score_style(self, style1, style2):
1148:         '''
1149:         Returns a match score between *style1* and *style2*.
1150: 
1151:         An exact match returns 0.0.
1152: 
1153:         A match between 'italic' and 'oblique' returns 0.1.
1154: 
1155:         No match returns 1.0.
1156:         '''
1157:         if style1 == style2:
1158:             return 0.0
1159:         elif style1 in ('italic', 'oblique') and \
1160:                 style2 in ('italic', 'oblique'):
1161:             return 0.1
1162:         return 1.0
1163: 
1164:     def score_variant(self, variant1, variant2):
1165:         '''
1166:         Returns a match score between *variant1* and *variant2*.
1167: 
1168:         An exact match returns 0.0, otherwise 1.0.
1169:         '''
1170:         if variant1 == variant2:
1171:             return 0.0
1172:         else:
1173:             return 1.0
1174: 
1175:     def score_stretch(self, stretch1, stretch2):
1176:         '''
1177:         Returns a match score between *stretch1* and *stretch2*.
1178: 
1179:         The result is the absolute value of the difference between the
1180:         CSS numeric values of *stretch1* and *stretch2*, normalized
1181:         between 0.0 and 1.0.
1182:         '''
1183:         try:
1184:             stretchval1 = int(stretch1)
1185:         except ValueError:
1186:             stretchval1 = stretch_dict.get(stretch1, 500)
1187:         try:
1188:             stretchval2 = int(stretch2)
1189:         except ValueError:
1190:             stretchval2 = stretch_dict.get(stretch2, 500)
1191:         return abs(stretchval1 - stretchval2) / 1000.0
1192: 
1193:     def score_weight(self, weight1, weight2):
1194:         '''
1195:         Returns a match score between *weight1* and *weight2*.
1196: 
1197:         The result is 0.0 if both weight1 and weight 2 are given as strings
1198:         and have the same value.
1199: 
1200:         Otherwise, the result is the absolute value of the difference between the
1201:         CSS numeric values of *weight1* and *weight2*, normalized
1202:         between 0.05 and 1.0.
1203:         '''
1204: 
1205:         # exact match of the weight names (e.g. weight1 == weight2 == "regular")
1206:         if (isinstance(weight1, six.string_types) and
1207:                 isinstance(weight2, six.string_types) and
1208:                 weight1 == weight2):
1209:             return 0.0
1210:         try:
1211:             weightval1 = int(weight1)
1212:         except ValueError:
1213:             weightval1 = weight_dict.get(weight1, 500)
1214:         try:
1215:             weightval2 = int(weight2)
1216:         except ValueError:
1217:             weightval2 = weight_dict.get(weight2, 500)
1218:         return 0.95*(abs(weightval1 - weightval2) / 1000.0) + 0.05
1219: 
1220:     def score_size(self, size1, size2):
1221:         '''
1222:         Returns a match score between *size1* and *size2*.
1223: 
1224:         If *size2* (the size specified in the font file) is 'scalable', this
1225:         function always returns 0.0, since any font size can be generated.
1226: 
1227:         Otherwise, the result is the absolute distance between *size1* and
1228:         *size2*, normalized so that the usual range of font sizes (6pt -
1229:         72pt) will lie between 0.0 and 1.0.
1230:         '''
1231:         if size2 == 'scalable':
1232:             return 0.0
1233:         # Size value should have already been
1234:         try:
1235:             sizeval1 = float(size1)
1236:         except ValueError:
1237:             sizeval1 = self.default_size * font_scalings[size1]
1238:         try:
1239:             sizeval2 = float(size2)
1240:         except ValueError:
1241:             return 1.0
1242:         return abs(sizeval1 - sizeval2) / 72.0
1243: 
1244:     def findfont(self, prop, fontext='ttf', directory=None,
1245:                  fallback_to_default=True, rebuild_if_missing=True):
1246:         '''
1247:         Search the font list for the font that most closely matches
1248:         the :class:`FontProperties` *prop*.
1249: 
1250:         :meth:`findfont` performs a nearest neighbor search.  Each
1251:         font is given a similarity score to the target font
1252:         properties.  The first font with the highest score is
1253:         returned.  If no matches below a certain threshold are found,
1254:         the default font (usually DejaVu Sans) is returned.
1255: 
1256:         `directory`, is specified, will only return fonts from the
1257:         given directory (or subdirectory of that directory).
1258: 
1259:         The result is cached, so subsequent lookups don't have to
1260:         perform the O(n) nearest neighbor search.
1261: 
1262:         If `fallback_to_default` is True, will fallback to the default
1263:         font family (usually "DejaVu Sans" or "Helvetica") if
1264:         the first lookup hard-fails.
1265: 
1266:         See the `W3C Cascading Style Sheet, Level 1
1267:         <http://www.w3.org/TR/1998/REC-CSS2-19980512/>`_ documentation
1268:         for a description of the font finding algorithm.
1269:         '''
1270:         if not isinstance(prop, FontProperties):
1271:             prop = FontProperties(prop)
1272:         fname = prop.get_file()
1273:         if fname is not None:
1274:             verbose.report('findfont returning %s'%fname, 'debug')
1275:             return fname
1276: 
1277:         if fontext == 'afm':
1278:             fontlist = self.afmlist
1279:         else:
1280:             fontlist = self.ttflist
1281: 
1282:         if directory is None:
1283:             cached = _lookup_cache[fontext].get(prop)
1284:             if cached is not None:
1285:                 return cached
1286:         else:
1287:             directory = os.path.normcase(directory)
1288: 
1289:         best_score = 1e64
1290:         best_font = None
1291: 
1292:         for font in fontlist:
1293:             if (directory is not None and
1294:                     os.path.commonprefix([os.path.normcase(font.fname),
1295:                                           directory]) != directory):
1296:                 continue
1297:             # Matching family should have highest priority, so it is multiplied
1298:             # by 10.0
1299:             score = \
1300:                 self.score_family(prop.get_family(), font.name) * 10.0 + \
1301:                 self.score_style(prop.get_style(), font.style) + \
1302:                 self.score_variant(prop.get_variant(), font.variant) + \
1303:                 self.score_weight(prop.get_weight(), font.weight) + \
1304:                 self.score_stretch(prop.get_stretch(), font.stretch) + \
1305:                 self.score_size(prop.get_size(), font.size)
1306:             if score < best_score:
1307:                 best_score = score
1308:                 best_font = font
1309:             if score == 0:
1310:                 break
1311: 
1312:         if best_font is None or best_score >= 10.0:
1313:             if fallback_to_default:
1314:                 warnings.warn(
1315:                     'findfont: Font family %s not found. Falling back to %s' %
1316:                     (prop.get_family(), self.defaultFamily[fontext]))
1317:                 default_prop = prop.copy()
1318:                 default_prop.set_family(self.defaultFamily[fontext])
1319:                 return self.findfont(default_prop, fontext, directory, False)
1320:             else:
1321:                 # This is a hard fail -- we can't find anything reasonable,
1322:                 # so just return the DejuVuSans.ttf
1323:                 warnings.warn(
1324:                     'findfont: Could not match %s. Returning %s' %
1325:                     (prop, self.defaultFont[fontext]),
1326:                     UserWarning)
1327:                 result = self.defaultFont[fontext]
1328:         else:
1329:             verbose.report(
1330:                 'findfont: Matching %s to %s (%s) with score of %f' %
1331:                 (prop, best_font.name, repr(best_font.fname), best_score))
1332:             result = best_font.fname
1333: 
1334:         if not os.path.isfile(result):
1335:             if rebuild_if_missing:
1336:                 verbose.report(
1337:                     'findfont: Found a missing font file.  Rebuilding cache.')
1338:                 _rebuild()
1339:                 return fontManager.findfont(
1340:                     prop, fontext, directory, True, False)
1341:             else:
1342:                 raise ValueError("No valid font could be found")
1343: 
1344:         if directory is None:
1345:             _lookup_cache[fontext].set(prop, result)
1346:         return result
1347: 
1348: _is_opentype_cff_font_cache = {}
1349: def is_opentype_cff_font(filename):
1350:     '''
1351:     Returns True if the given font is a Postscript Compact Font Format
1352:     Font embedded in an OpenType wrapper.  Used by the PostScript and
1353:     PDF backends that can not subset these fonts.
1354:     '''
1355:     if os.path.splitext(filename)[1].lower() == '.otf':
1356:         result = _is_opentype_cff_font_cache.get(filename)
1357:         if result is None:
1358:             with open(filename, 'rb') as fd:
1359:                 tag = fd.read(4)
1360:             result = (tag == b'OTTO')
1361:             _is_opentype_cff_font_cache[filename] = result
1362:         return result
1363:     return False
1364: 
1365: fontManager = None
1366: _fmcache = None
1367: 
1368: 
1369: get_font = lru_cache(64)(ft2font.FT2Font)
1370: 
1371: 
1372: # The experimental fontconfig-based backend.
1373: if USE_FONTCONFIG and sys.platform != 'win32':
1374:     import re
1375: 
1376:     def fc_match(pattern, fontext):
1377:         fontexts = get_fontext_synonyms(fontext)
1378:         ext = "." + fontext
1379:         try:
1380:             pipe = subprocess.Popen(
1381:                 ['fc-match', '-s', '--format=%{file}\\n', pattern],
1382:                 stdout=subprocess.PIPE,
1383:                 stderr=subprocess.PIPE)
1384:             output = pipe.communicate()[0]
1385:         except (OSError, IOError):
1386:             return None
1387: 
1388:         # The bulk of the output from fc-list is ascii, so we keep the
1389:         # result in bytes and parse it as bytes, until we extract the
1390:         # filename, which is in sys.filesystemencoding().
1391:         if pipe.returncode == 0:
1392:             for fname in output.split(b'\n'):
1393:                 try:
1394:                     fname = six.text_type(fname, sys.getfilesystemencoding())
1395:                 except UnicodeDecodeError:
1396:                     continue
1397:                 if os.path.splitext(fname)[1][1:] in fontexts:
1398:                     return fname
1399:         return None
1400: 
1401:     _fc_match_cache = {}
1402: 
1403:     def findfont(prop, fontext='ttf'):
1404:         if not isinstance(prop, six.string_types):
1405:             prop = prop.get_fontconfig_pattern()
1406:         cached = _fc_match_cache.get(prop)
1407:         if cached is not None:
1408:             return cached
1409: 
1410:         result = fc_match(prop, fontext)
1411:         if result is None:
1412:             result = fc_match(':', fontext)
1413: 
1414:         _fc_match_cache[prop] = result
1415:         return result
1416: 
1417: else:
1418:     _fmcache = None
1419: 
1420:     cachedir = get_cachedir()
1421:     if cachedir is not None:
1422:         _fmcache = os.path.join(cachedir, 'fontList.json')
1423: 
1424:     fontManager = None
1425: 
1426:     _lookup_cache = {
1427:         'ttf': TempCache(),
1428:         'afm': TempCache()
1429:     }
1430: 
1431:     def _rebuild():
1432:         global fontManager
1433: 
1434:         fontManager = FontManager()
1435: 
1436:         if _fmcache:
1437:             with cbook.Locked(cachedir):
1438:                 json_dump(fontManager, _fmcache)
1439: 
1440:         verbose.report("generated new fontManager")
1441: 
1442:     if _fmcache:
1443:         try:
1444:             fontManager = json_load(_fmcache)
1445:             if (not hasattr(fontManager, '_version') or
1446:                 fontManager._version != FontManager.__version__):
1447:                 _rebuild()
1448:             else:
1449:                 fontManager.default_size = None
1450:                 verbose.report("Using fontManager instance from %s" % _fmcache)
1451:         except cbook.Locked.TimeoutError:
1452:             raise
1453:         except:
1454:             _rebuild()
1455:     else:
1456:         _rebuild()
1457: 
1458:     def findfont(prop, **kw):
1459:         global fontManager
1460:         font = fontManager.findfont(prop, **kw)
1461:         return font
1462: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

unicode_57317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, (-1)), 'unicode', u'\nA module for finding, managing, and using fonts across platforms.\n\nThis module provides a single :class:`FontManager` instance that can\nbe shared across backends and platforms.  The :func:`findfont`\nfunction returns the best TrueType (TTF) font file in the local or\nsystem font path that matches the specified :class:`FontProperties`\ninstance.  The :class:`FontManager` also handles Adobe Font Metrics\n(AFM) font files for use by the PostScript backend.\n\nThe design is based on the `W3C Cascading Style Sheet, Level 1 (CSS1)\nfont specification <http://www.w3.org/TR/1998/REC-CSS2-19980512/>`_.\nFuture versions may implement the Level 2 or 2.1 specifications.\n\nExperimental support is included for using `fontconfig` on Unix\nvariant platforms (Linux, OS X, Solaris).  To enable it, set the\nconstant ``USE_FONTCONFIG`` in this file to ``True``.  Fontconfig has\nthe advantage that it is the standard way to look up fonts on X11\nplatforms, so if a font is installed, it is much more likely to be\nfound.\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 25, 0))

# 'import six' statement (line 25)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_57318 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 25, 0), 'six')

if (type(import_57318) is not StypyTypeError):

    if (import_57318 != 'pyd_module'):
        __import__(import_57318)
        sys_modules_57319 = sys.modules[import_57318]
        import_module(stypy.reporting.localization.Localization(__file__, 25, 0), 'six', sys_modules_57319.module_type_store, module_type_store)
    else:
        import six

        import_module(stypy.reporting.localization.Localization(__file__, 25, 0), 'six', six, module_type_store)

else:
    # Assigning a type to the variable 'six' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'six', import_57318)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 26, 0))

# 'from six.moves import pickle' statement (line 26)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_57320 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'six.moves')

if (type(import_57320) is not StypyTypeError):

    if (import_57320 != 'pyd_module'):
        __import__(import_57320)
        sys_modules_57321 = sys.modules[import_57320]
        import_from_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'six.moves', sys_modules_57321.module_type_store, module_type_store, ['cPickle'])
        nest_module(stypy.reporting.localization.Localization(__file__, 26, 0), __file__, sys_modules_57321, sys_modules_57321.module_type_store, module_type_store)
    else:
        from six.moves import cPickle as pickle

        import_from_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'six.moves', None, module_type_store, ['cPickle'], [pickle])

else:
    # Assigning a type to the variable 'six.moves' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'six.moves', import_57320)

# Adding an alias
module_type_store.add_alias('pickle', 'cPickle')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

unicode_57322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, (-1)), 'unicode', u"\nKNOWN ISSUES\n\n  - documentation\n  - font variant is untested\n  - font stretch is incomplete\n  - font size is incomplete\n  - font size_adjust is incomplete\n  - default font algorithm needs improvement and testing\n  - setWeights function needs improvement\n  - 'light' is an invalid weight value, remove it.\n  - update_fonts not implemented\n\nAuthors   : John Hunter <jdhunter@ace.bsd.uchicago.edu>\n            Paul Barrett <Barrett@STScI.Edu>\n            Michael Droettboom <mdroe@STScI.edu>\nCopyright : John Hunter (2004,2005), Paul Barrett (2004,2005)\nLicense   : matplotlib license (PSF compatible)\n            The font directory code is from ttfquery,\n            see license/LICENSE_TTFQUERY.\n")
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 50, 0))

# 'from collections import Iterable' statement (line 50)
try:
    from collections import Iterable

except:
    Iterable = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 50, 0), 'collections', None, module_type_store, ['Iterable'], [Iterable])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 51, 0))

# 'import json' statement (line 51)
import json

import_module(stypy.reporting.localization.Localization(__file__, 51, 0), 'json', json, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 52, 0))

# 'import os' statement (line 52)
import os

import_module(stypy.reporting.localization.Localization(__file__, 52, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 53, 0))

# 'import sys' statement (line 53)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 53, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 54, 0))

# 'from threading import Timer' statement (line 54)
try:
    from threading import Timer

except:
    Timer = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 54, 0), 'threading', None, module_type_store, ['Timer'], [Timer])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 55, 0))

# 'import warnings' statement (line 55)
import warnings

import_module(stypy.reporting.localization.Localization(__file__, 55, 0), 'warnings', warnings, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 57, 0))

# 'import matplotlib' statement (line 57)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_57323 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 57, 0), 'matplotlib')

if (type(import_57323) is not StypyTypeError):

    if (import_57323 != 'pyd_module'):
        __import__(import_57323)
        sys_modules_57324 = sys.modules[import_57323]
        import_module(stypy.reporting.localization.Localization(__file__, 57, 0), 'matplotlib', sys_modules_57324.module_type_store, module_type_store)
    else:
        import matplotlib

        import_module(stypy.reporting.localization.Localization(__file__, 57, 0), 'matplotlib', matplotlib, module_type_store)

else:
    # Assigning a type to the variable 'matplotlib' (line 57)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 0), 'matplotlib', import_57323)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 58, 0))

# 'from matplotlib import afm, cbook, ft2font, rcParams, get_cachedir' statement (line 58)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_57325 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 58, 0), 'matplotlib')

if (type(import_57325) is not StypyTypeError):

    if (import_57325 != 'pyd_module'):
        __import__(import_57325)
        sys_modules_57326 = sys.modules[import_57325]
        import_from_module(stypy.reporting.localization.Localization(__file__, 58, 0), 'matplotlib', sys_modules_57326.module_type_store, module_type_store, ['afm', 'cbook', 'ft2font', 'rcParams', 'get_cachedir'])
        nest_module(stypy.reporting.localization.Localization(__file__, 58, 0), __file__, sys_modules_57326, sys_modules_57326.module_type_store, module_type_store)
    else:
        from matplotlib import afm, cbook, ft2font, rcParams, get_cachedir

        import_from_module(stypy.reporting.localization.Localization(__file__, 58, 0), 'matplotlib', None, module_type_store, ['afm', 'cbook', 'ft2font', 'rcParams', 'get_cachedir'], [afm, cbook, ft2font, rcParams, get_cachedir])

else:
    # Assigning a type to the variable 'matplotlib' (line 58)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 0), 'matplotlib', import_57325)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 59, 0))

# 'from matplotlib.compat import subprocess' statement (line 59)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_57327 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 59, 0), 'matplotlib.compat')

if (type(import_57327) is not StypyTypeError):

    if (import_57327 != 'pyd_module'):
        __import__(import_57327)
        sys_modules_57328 = sys.modules[import_57327]
        import_from_module(stypy.reporting.localization.Localization(__file__, 59, 0), 'matplotlib.compat', sys_modules_57328.module_type_store, module_type_store, ['subprocess'])
        nest_module(stypy.reporting.localization.Localization(__file__, 59, 0), __file__, sys_modules_57328, sys_modules_57328.module_type_store, module_type_store)
    else:
        from matplotlib.compat import subprocess

        import_from_module(stypy.reporting.localization.Localization(__file__, 59, 0), 'matplotlib.compat', None, module_type_store, ['subprocess'], [subprocess])

else:
    # Assigning a type to the variable 'matplotlib.compat' (line 59)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 0), 'matplotlib.compat', import_57327)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 60, 0))

# 'from matplotlib.fontconfig_pattern import parse_fontconfig_pattern, generate_fontconfig_pattern' statement (line 60)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_57329 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 60, 0), 'matplotlib.fontconfig_pattern')

if (type(import_57329) is not StypyTypeError):

    if (import_57329 != 'pyd_module'):
        __import__(import_57329)
        sys_modules_57330 = sys.modules[import_57329]
        import_from_module(stypy.reporting.localization.Localization(__file__, 60, 0), 'matplotlib.fontconfig_pattern', sys_modules_57330.module_type_store, module_type_store, ['parse_fontconfig_pattern', 'generate_fontconfig_pattern'])
        nest_module(stypy.reporting.localization.Localization(__file__, 60, 0), __file__, sys_modules_57330, sys_modules_57330.module_type_store, module_type_store)
    else:
        from matplotlib.fontconfig_pattern import parse_fontconfig_pattern, generate_fontconfig_pattern

        import_from_module(stypy.reporting.localization.Localization(__file__, 60, 0), 'matplotlib.fontconfig_pattern', None, module_type_store, ['parse_fontconfig_pattern', 'generate_fontconfig_pattern'], [parse_fontconfig_pattern, generate_fontconfig_pattern])

else:
    # Assigning a type to the variable 'matplotlib.fontconfig_pattern' (line 60)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 0), 'matplotlib.fontconfig_pattern', import_57329)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')



# SSA begins for try-except statement (line 63)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 64, 4))

# 'from functools import lru_cache' statement (line 64)
try:
    from functools import lru_cache

except:
    lru_cache = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 64, 4), 'functools', None, module_type_store, ['lru_cache'], [lru_cache])

# SSA branch for the except part of a try statement (line 63)
# SSA branch for the except 'ImportError' branch of a try statement (line 63)
module_type_store.open_ssa_branch('except')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 66, 4))

# 'from backports.functools_lru_cache import lru_cache' statement (line 66)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_57331 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 66, 4), 'backports.functools_lru_cache')

if (type(import_57331) is not StypyTypeError):

    if (import_57331 != 'pyd_module'):
        __import__(import_57331)
        sys_modules_57332 = sys.modules[import_57331]
        import_from_module(stypy.reporting.localization.Localization(__file__, 66, 4), 'backports.functools_lru_cache', sys_modules_57332.module_type_store, module_type_store, ['lru_cache'])
        nest_module(stypy.reporting.localization.Localization(__file__, 66, 4), __file__, sys_modules_57332, sys_modules_57332.module_type_store, module_type_store)
    else:
        from backports.functools_lru_cache import lru_cache

        import_from_module(stypy.reporting.localization.Localization(__file__, 66, 4), 'backports.functools_lru_cache', None, module_type_store, ['lru_cache'], [lru_cache])

else:
    # Assigning a type to the variable 'backports.functools_lru_cache' (line 66)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'backports.functools_lru_cache', import_57331)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

# SSA join for try-except statement (line 63)
module_type_store = module_type_store.join_ssa_context()


# Assigning a Name to a Name (line 69):

# Assigning a Name to a Name (line 69):
# Getting the type of 'False' (line 69)
False_57333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 17), 'False')
# Assigning a type to the variable 'USE_FONTCONFIG' (line 69)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 0), 'USE_FONTCONFIG', False_57333)

# Assigning a Attribute to a Name (line 70):

# Assigning a Attribute to a Name (line 70):
# Getting the type of 'matplotlib' (line 70)
matplotlib_57334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 10), 'matplotlib')
# Obtaining the member 'verbose' of a type (line 70)
verbose_57335 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 10), matplotlib_57334, 'verbose')
# Assigning a type to the variable 'verbose' (line 70)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 0), 'verbose', verbose_57335)

# Assigning a Dict to a Name (line 72):

# Assigning a Dict to a Name (line 72):

# Obtaining an instance of the builtin type 'dict' (line 72)
dict_57336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 16), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 72)
# Adding element type (key, value) (line 72)
unicode_57337 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 4), 'unicode', u'xx-small')
float_57338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 17), 'float')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 16), dict_57336, (unicode_57337, float_57338))
# Adding element type (key, value) (line 72)
unicode_57339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 4), 'unicode', u'x-small')
float_57340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 17), 'float')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 16), dict_57336, (unicode_57339, float_57340))
# Adding element type (key, value) (line 72)
unicode_57341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 4), 'unicode', u'small')
float_57342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 17), 'float')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 16), dict_57336, (unicode_57341, float_57342))
# Adding element type (key, value) (line 72)
unicode_57343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 4), 'unicode', u'medium')
float_57344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 17), 'float')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 16), dict_57336, (unicode_57343, float_57344))
# Adding element type (key, value) (line 72)
unicode_57345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 4), 'unicode', u'large')
float_57346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 17), 'float')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 16), dict_57336, (unicode_57345, float_57346))
# Adding element type (key, value) (line 72)
unicode_57347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 4), 'unicode', u'x-large')
float_57348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 17), 'float')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 16), dict_57336, (unicode_57347, float_57348))
# Adding element type (key, value) (line 72)
unicode_57349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 4), 'unicode', u'xx-large')
float_57350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 17), 'float')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 16), dict_57336, (unicode_57349, float_57350))
# Adding element type (key, value) (line 72)
unicode_57351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 4), 'unicode', u'larger')
float_57352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 17), 'float')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 16), dict_57336, (unicode_57351, float_57352))
# Adding element type (key, value) (line 72)
unicode_57353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 4), 'unicode', u'smaller')
float_57354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 17), 'float')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 16), dict_57336, (unicode_57353, float_57354))
# Adding element type (key, value) (line 72)
# Getting the type of 'None' (line 82)
None_57355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'None')
float_57356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 17), 'float')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 16), dict_57336, (None_57355, float_57356))

# Assigning a type to the variable 'font_scalings' (line 72)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 0), 'font_scalings', dict_57336)

# Assigning a Dict to a Name (line 84):

# Assigning a Dict to a Name (line 84):

# Obtaining an instance of the builtin type 'dict' (line 84)
dict_57357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 15), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 84)
# Adding element type (key, value) (line 84)
unicode_57358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 4), 'unicode', u'ultra-condensed')
int_57359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 24), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 15), dict_57357, (unicode_57358, int_57359))
# Adding element type (key, value) (line 84)
unicode_57360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 4), 'unicode', u'extra-condensed')
int_57361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 24), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 15), dict_57357, (unicode_57360, int_57361))
# Adding element type (key, value) (line 84)
unicode_57362 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 4), 'unicode', u'condensed')
int_57363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 24), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 15), dict_57357, (unicode_57362, int_57363))
# Adding element type (key, value) (line 84)
unicode_57364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 4), 'unicode', u'semi-condensed')
int_57365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 24), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 15), dict_57357, (unicode_57364, int_57365))
# Adding element type (key, value) (line 84)
unicode_57366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 4), 'unicode', u'normal')
int_57367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 24), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 15), dict_57357, (unicode_57366, int_57367))
# Adding element type (key, value) (line 84)
unicode_57368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 4), 'unicode', u'semi-expanded')
int_57369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 24), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 15), dict_57357, (unicode_57368, int_57369))
# Adding element type (key, value) (line 84)
unicode_57370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 4), 'unicode', u'expanded')
int_57371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 24), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 15), dict_57357, (unicode_57370, int_57371))
# Adding element type (key, value) (line 84)
unicode_57372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 4), 'unicode', u'extra-expanded')
int_57373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 24), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 15), dict_57357, (unicode_57372, int_57373))
# Adding element type (key, value) (line 84)
unicode_57374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 4), 'unicode', u'ultra-expanded')
int_57375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 24), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 15), dict_57357, (unicode_57374, int_57375))

# Assigning a type to the variable 'stretch_dict' (line 84)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 0), 'stretch_dict', dict_57357)

# Assigning a Dict to a Name (line 95):

# Assigning a Dict to a Name (line 95):

# Obtaining an instance of the builtin type 'dict' (line 95)
dict_57376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 14), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 95)
# Adding element type (key, value) (line 95)
unicode_57377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 4), 'unicode', u'ultralight')
int_57378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 19), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 14), dict_57376, (unicode_57377, int_57378))
# Adding element type (key, value) (line 95)
unicode_57379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 4), 'unicode', u'light')
int_57380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 19), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 14), dict_57376, (unicode_57379, int_57380))
# Adding element type (key, value) (line 95)
unicode_57381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 4), 'unicode', u'normal')
int_57382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 19), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 14), dict_57376, (unicode_57381, int_57382))
# Adding element type (key, value) (line 95)
unicode_57383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 4), 'unicode', u'regular')
int_57384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 19), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 14), dict_57376, (unicode_57383, int_57384))
# Adding element type (key, value) (line 95)
unicode_57385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 4), 'unicode', u'book')
int_57386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 19), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 14), dict_57376, (unicode_57385, int_57386))
# Adding element type (key, value) (line 95)
unicode_57387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 4), 'unicode', u'medium')
int_57388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 19), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 14), dict_57376, (unicode_57387, int_57388))
# Adding element type (key, value) (line 95)
unicode_57389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 4), 'unicode', u'roman')
int_57390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 19), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 14), dict_57376, (unicode_57389, int_57390))
# Adding element type (key, value) (line 95)
unicode_57391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 4), 'unicode', u'semibold')
int_57392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 19), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 14), dict_57376, (unicode_57391, int_57392))
# Adding element type (key, value) (line 95)
unicode_57393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 4), 'unicode', u'demibold')
int_57394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 19), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 14), dict_57376, (unicode_57393, int_57394))
# Adding element type (key, value) (line 95)
unicode_57395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 4), 'unicode', u'demi')
int_57396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 19), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 14), dict_57376, (unicode_57395, int_57396))
# Adding element type (key, value) (line 95)
unicode_57397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 4), 'unicode', u'bold')
int_57398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 19), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 14), dict_57376, (unicode_57397, int_57398))
# Adding element type (key, value) (line 95)
unicode_57399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 4), 'unicode', u'heavy')
int_57400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 19), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 14), dict_57376, (unicode_57399, int_57400))
# Adding element type (key, value) (line 95)
unicode_57401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 4), 'unicode', u'extra bold')
int_57402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 19), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 14), dict_57376, (unicode_57401, int_57402))
# Adding element type (key, value) (line 95)
unicode_57403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 4), 'unicode', u'black')
int_57404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 19), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 14), dict_57376, (unicode_57403, int_57404))

# Assigning a type to the variable 'weight_dict' (line 95)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 0), 'weight_dict', dict_57376)

# Assigning a Set to a Name (line 111):

# Assigning a Set to a Name (line 111):

# Obtaining an instance of the builtin type 'set' (line 111)
set_57405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 22), 'set')
# Adding type elements to the builtin type 'set' instance (line 111)
# Adding element type (line 111)
unicode_57406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 4), 'unicode', u'serif')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 22), set_57405, unicode_57406)
# Adding element type (line 111)
unicode_57407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 4), 'unicode', u'sans-serif')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 22), set_57405, unicode_57407)
# Adding element type (line 111)
unicode_57408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 4), 'unicode', u'sans serif')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 22), set_57405, unicode_57408)
# Adding element type (line 111)
unicode_57409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 4), 'unicode', u'cursive')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 22), set_57405, unicode_57409)
# Adding element type (line 111)
unicode_57410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 4), 'unicode', u'fantasy')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 22), set_57405, unicode_57410)
# Adding element type (line 111)
unicode_57411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 4), 'unicode', u'monospace')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 22), set_57405, unicode_57411)
# Adding element type (line 111)
unicode_57412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 4), 'unicode', u'sans')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 22), set_57405, unicode_57412)

# Assigning a type to the variable 'font_family_aliases' (line 111)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 0), 'font_family_aliases', set_57405)

# Assigning a Str to a Name (line 121):

# Assigning a Str to a Name (line 121):
unicode_57413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 4), 'unicode', u'Software\\Microsoft\\Windows\\CurrentVersion\\Explorer\\Shell Folders')
# Assigning a type to the variable 'MSFolders' (line 121)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 0), 'MSFolders', unicode_57413)

# Assigning a List to a Name (line 125):

# Assigning a List to a Name (line 125):

# Obtaining an instance of the builtin type 'list' (line 125)
list_57414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 20), 'list')
# Adding type elements to the builtin type 'list' instance (line 125)
# Adding element type (line 125)
unicode_57415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 4), 'unicode', u'SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion\\Fonts')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 20), list_57414, unicode_57415)
# Adding element type (line 125)
unicode_57416 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 4), 'unicode', u'SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Fonts')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 20), list_57414, unicode_57416)

# Assigning a type to the variable 'MSFontDirectories' (line 125)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 0), 'MSFontDirectories', list_57414)

# Assigning a List to a Name (line 130):

# Assigning a List to a Name (line 130):

# Obtaining an instance of the builtin type 'list' (line 130)
list_57417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 21), 'list')
# Adding type elements to the builtin type 'list' instance (line 130)
# Adding element type (line 130)
unicode_57418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 4), 'unicode', u'/usr/X11R6/lib/X11/fonts/TTF/')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 130, 21), list_57417, unicode_57418)
# Adding element type (line 130)
unicode_57419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 4), 'unicode', u'/usr/X11/lib/X11/fonts')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 130, 21), list_57417, unicode_57419)
# Adding element type (line 130)
unicode_57420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 4), 'unicode', u'/usr/share/fonts/')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 130, 21), list_57417, unicode_57420)
# Adding element type (line 130)
unicode_57421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 4), 'unicode', u'/usr/local/share/fonts/')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 130, 21), list_57417, unicode_57421)
# Adding element type (line 130)
unicode_57422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 4), 'unicode', u'/usr/lib/openoffice/share/fonts/truetype/')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 130, 21), list_57417, unicode_57422)

# Assigning a type to the variable 'X11FontDirectories' (line 130)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 0), 'X11FontDirectories', list_57417)

# Assigning a List to a Name (line 142):

# Assigning a List to a Name (line 142):

# Obtaining an instance of the builtin type 'list' (line 142)
list_57423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 21), 'list')
# Adding type elements to the builtin type 'list' instance (line 142)
# Adding element type (line 142)
unicode_57424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 4), 'unicode', u'/Library/Fonts/')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 142, 21), list_57423, unicode_57424)
# Adding element type (line 142)
unicode_57425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 4), 'unicode', u'/Network/Library/Fonts/')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 142, 21), list_57423, unicode_57425)
# Adding element type (line 142)
unicode_57426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 4), 'unicode', u'/System/Library/Fonts/')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 142, 21), list_57423, unicode_57426)
# Adding element type (line 142)
unicode_57427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 4), 'unicode', u'/opt/local/share/fonts')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 142, 21), list_57423, unicode_57427)

# Assigning a type to the variable 'OSXFontDirectories' (line 142)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 0), 'OSXFontDirectories', list_57423)


# Evaluating a boolean operation

# Getting the type of 'USE_FONTCONFIG' (line 151)
USE_FONTCONFIG_57428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 7), 'USE_FONTCONFIG')
# Applying the 'not' unary operator (line 151)
result_not__57429 = python_operator(stypy.reporting.localization.Localization(__file__, 151, 3), 'not', USE_FONTCONFIG_57428)


# Getting the type of 'sys' (line 151)
sys_57430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 26), 'sys')
# Obtaining the member 'platform' of a type (line 151)
platform_57431 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 26), sys_57430, 'platform')
unicode_57432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 42), 'unicode', u'win32')
# Applying the binary operator '!=' (line 151)
result_ne_57433 = python_operator(stypy.reporting.localization.Localization(__file__, 151, 26), '!=', platform_57431, unicode_57432)

# Applying the binary operator 'and' (line 151)
result_and_keyword_57434 = python_operator(stypy.reporting.localization.Localization(__file__, 151, 3), 'and', result_not__57429, result_ne_57433)

# Testing the type of an if condition (line 151)
if_condition_57435 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 151, 0), result_and_keyword_57434)
# Assigning a type to the variable 'if_condition_57435' (line 151)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 0), 'if_condition_57435', if_condition_57435)
# SSA begins for if statement (line 151)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Assigning a Call to a Name (line 152):

# Assigning a Call to a Name (line 152):

# Call to get(...): (line 152)
# Processing the call arguments (line 152)
unicode_57439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 26), 'unicode', u'HOME')
# Processing the call keyword arguments (line 152)
kwargs_57440 = {}
# Getting the type of 'os' (line 152)
os_57436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 11), 'os', False)
# Obtaining the member 'environ' of a type (line 152)
environ_57437 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 11), os_57436, 'environ')
# Obtaining the member 'get' of a type (line 152)
get_57438 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 11), environ_57437, 'get')
# Calling get(args, kwargs) (line 152)
get_call_result_57441 = invoke(stypy.reporting.localization.Localization(__file__, 152, 11), get_57438, *[unicode_57439], **kwargs_57440)

# Assigning a type to the variable 'home' (line 152)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 4), 'home', get_call_result_57441)

# Type idiom detected: calculating its left and rigth part (line 153)
# Getting the type of 'home' (line 153)
home_57442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 4), 'home')
# Getting the type of 'None' (line 153)
None_57443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 19), 'None')

(may_be_57444, more_types_in_union_57445) = may_not_be_none(home_57442, None_57443)

if may_be_57444:

    if more_types_in_union_57445:
        # Runtime conditional SSA (line 153)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
    else:
        module_type_store = module_type_store

    
    # Assigning a Call to a Name (line 155):
    
    # Assigning a Call to a Name (line 155):
    
    # Call to join(...): (line 155)
    # Processing the call arguments (line 155)
    # Getting the type of 'home' (line 155)
    home_57449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 28), 'home', False)
    unicode_57450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 34), 'unicode', u'Library')
    unicode_57451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 45), 'unicode', u'Fonts')
    # Processing the call keyword arguments (line 155)
    kwargs_57452 = {}
    # Getting the type of 'os' (line 155)
    os_57446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 15), 'os', False)
    # Obtaining the member 'path' of a type (line 155)
    path_57447 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 15), os_57446, 'path')
    # Obtaining the member 'join' of a type (line 155)
    join_57448 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 15), path_57447, 'join')
    # Calling join(args, kwargs) (line 155)
    join_call_result_57453 = invoke(stypy.reporting.localization.Localization(__file__, 155, 15), join_57448, *[home_57449, unicode_57450, unicode_57451], **kwargs_57452)
    
    # Assigning a type to the variable 'path' (line 155)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 8), 'path', join_call_result_57453)
    
    # Call to append(...): (line 156)
    # Processing the call arguments (line 156)
    # Getting the type of 'path' (line 156)
    path_57456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 34), 'path', False)
    # Processing the call keyword arguments (line 156)
    kwargs_57457 = {}
    # Getting the type of 'OSXFontDirectories' (line 156)
    OSXFontDirectories_57454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 8), 'OSXFontDirectories', False)
    # Obtaining the member 'append' of a type (line 156)
    append_57455 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 8), OSXFontDirectories_57454, 'append')
    # Calling append(args, kwargs) (line 156)
    append_call_result_57458 = invoke(stypy.reporting.localization.Localization(__file__, 156, 8), append_57455, *[path_57456], **kwargs_57457)
    
    
    # Assigning a Call to a Name (line 157):
    
    # Assigning a Call to a Name (line 157):
    
    # Call to join(...): (line 157)
    # Processing the call arguments (line 157)
    # Getting the type of 'home' (line 157)
    home_57462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 28), 'home', False)
    unicode_57463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 34), 'unicode', u'.fonts')
    # Processing the call keyword arguments (line 157)
    kwargs_57464 = {}
    # Getting the type of 'os' (line 157)
    os_57459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 15), 'os', False)
    # Obtaining the member 'path' of a type (line 157)
    path_57460 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 15), os_57459, 'path')
    # Obtaining the member 'join' of a type (line 157)
    join_57461 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 15), path_57460, 'join')
    # Calling join(args, kwargs) (line 157)
    join_call_result_57465 = invoke(stypy.reporting.localization.Localization(__file__, 157, 15), join_57461, *[home_57462, unicode_57463], **kwargs_57464)
    
    # Assigning a type to the variable 'path' (line 157)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 8), 'path', join_call_result_57465)
    
    # Call to append(...): (line 158)
    # Processing the call arguments (line 158)
    # Getting the type of 'path' (line 158)
    path_57468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 34), 'path', False)
    # Processing the call keyword arguments (line 158)
    kwargs_57469 = {}
    # Getting the type of 'X11FontDirectories' (line 158)
    X11FontDirectories_57466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'X11FontDirectories', False)
    # Obtaining the member 'append' of a type (line 158)
    append_57467 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 8), X11FontDirectories_57466, 'append')
    # Calling append(args, kwargs) (line 158)
    append_call_result_57470 = invoke(stypy.reporting.localization.Localization(__file__, 158, 8), append_57467, *[path_57468], **kwargs_57469)
    

    if more_types_in_union_57445:
        # SSA join for if statement (line 153)
        module_type_store = module_type_store.join_ssa_context()



# SSA join for if statement (line 151)
module_type_store = module_type_store.join_ssa_context()


@norecursion
def get_fontext_synonyms(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'get_fontext_synonyms'
    module_type_store = module_type_store.open_function_context('get_fontext_synonyms', 161, 0, False)
    
    # Passed parameters checking function
    get_fontext_synonyms.stypy_localization = localization
    get_fontext_synonyms.stypy_type_of_self = None
    get_fontext_synonyms.stypy_type_store = module_type_store
    get_fontext_synonyms.stypy_function_name = 'get_fontext_synonyms'
    get_fontext_synonyms.stypy_param_names_list = ['fontext']
    get_fontext_synonyms.stypy_varargs_param_name = None
    get_fontext_synonyms.stypy_kwargs_param_name = None
    get_fontext_synonyms.stypy_call_defaults = defaults
    get_fontext_synonyms.stypy_call_varargs = varargs
    get_fontext_synonyms.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'get_fontext_synonyms', ['fontext'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'get_fontext_synonyms', localization, ['fontext'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'get_fontext_synonyms(...)' code ##################

    unicode_57471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, (-1)), 'unicode', u'\n    Return a list of file extensions extensions that are synonyms for\n    the given file extension *fileext*.\n    ')
    
    # Obtaining the type of the subscript
    # Getting the type of 'fontext' (line 168)
    fontext_57472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 29), 'fontext')
    
    # Obtaining an instance of the builtin type 'dict' (line 166)
    dict_57473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 11), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 166)
    # Adding element type (key, value) (line 166)
    unicode_57474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 12), 'unicode', u'ttf')
    
    # Obtaining an instance of the builtin type 'tuple' (line 166)
    tuple_57475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 20), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 166)
    # Adding element type (line 166)
    unicode_57476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 20), 'unicode', u'ttf')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 166, 20), tuple_57475, unicode_57476)
    # Adding element type (line 166)
    unicode_57477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 27), 'unicode', u'otf')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 166, 20), tuple_57475, unicode_57477)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 166, 11), dict_57473, (unicode_57474, tuple_57475))
    # Adding element type (key, value) (line 166)
    unicode_57478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 12), 'unicode', u'otf')
    
    # Obtaining an instance of the builtin type 'tuple' (line 167)
    tuple_57479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 20), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 167)
    # Adding element type (line 167)
    unicode_57480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 20), 'unicode', u'ttf')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 20), tuple_57479, unicode_57480)
    # Adding element type (line 167)
    unicode_57481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 27), 'unicode', u'otf')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 20), tuple_57479, unicode_57481)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 166, 11), dict_57473, (unicode_57478, tuple_57479))
    # Adding element type (key, value) (line 166)
    unicode_57482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 12), 'unicode', u'afm')
    
    # Obtaining an instance of the builtin type 'tuple' (line 168)
    tuple_57483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 20), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 168)
    # Adding element type (line 168)
    unicode_57484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 20), 'unicode', u'afm')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 20), tuple_57483, unicode_57484)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 166, 11), dict_57473, (unicode_57482, tuple_57483))
    
    # Obtaining the member '__getitem__' of a type (line 166)
    getitem___57485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 11), dict_57473, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 166)
    subscript_call_result_57486 = invoke(stypy.reporting.localization.Localization(__file__, 166, 11), getitem___57485, fontext_57472)
    
    # Assigning a type to the variable 'stypy_return_type' (line 166)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 4), 'stypy_return_type', subscript_call_result_57486)
    
    # ################# End of 'get_fontext_synonyms(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_fontext_synonyms' in the type store
    # Getting the type of 'stypy_return_type' (line 161)
    stypy_return_type_57487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_57487)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_fontext_synonyms'
    return stypy_return_type_57487

# Assigning a type to the variable 'get_fontext_synonyms' (line 161)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 0), 'get_fontext_synonyms', get_fontext_synonyms)

@norecursion
def list_fonts(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'list_fonts'
    module_type_store = module_type_store.open_function_context('list_fonts', 171, 0, False)
    
    # Passed parameters checking function
    list_fonts.stypy_localization = localization
    list_fonts.stypy_type_of_self = None
    list_fonts.stypy_type_store = module_type_store
    list_fonts.stypy_function_name = 'list_fonts'
    list_fonts.stypy_param_names_list = ['directory', 'extensions']
    list_fonts.stypy_varargs_param_name = None
    list_fonts.stypy_kwargs_param_name = None
    list_fonts.stypy_call_defaults = defaults
    list_fonts.stypy_call_varargs = varargs
    list_fonts.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'list_fonts', ['directory', 'extensions'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'list_fonts', localization, ['directory', 'extensions'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'list_fonts(...)' code ##################

    unicode_57488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, (-1)), 'unicode', u'\n    Return a list of all fonts matching any of the extensions,\n    possibly upper-cased, found recursively under the directory.\n    ')
    
    # Assigning a Call to a Name (line 176):
    
    # Assigning a Call to a Name (line 176):
    
    # Call to join(...): (line 176)
    # Processing the call arguments (line 176)
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'extensions' (line 177)
    extensions_57499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 35), 'extensions', False)
    comprehension_57500 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 24), extensions_57499)
    # Assigning a type to the variable 'ext' (line 176)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 24), 'ext', comprehension_57500)
    unicode_57491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 24), 'unicode', u'*.%s;*.%s')
    
    # Obtaining an instance of the builtin type 'tuple' (line 176)
    tuple_57492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 39), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 176)
    # Adding element type (line 176)
    # Getting the type of 'ext' (line 176)
    ext_57493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 39), 'ext', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 39), tuple_57492, ext_57493)
    # Adding element type (line 176)
    
    # Call to upper(...): (line 176)
    # Processing the call keyword arguments (line 176)
    kwargs_57496 = {}
    # Getting the type of 'ext' (line 176)
    ext_57494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 44), 'ext', False)
    # Obtaining the member 'upper' of a type (line 176)
    upper_57495 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 44), ext_57494, 'upper')
    # Calling upper(args, kwargs) (line 176)
    upper_call_result_57497 = invoke(stypy.reporting.localization.Localization(__file__, 176, 44), upper_57495, *[], **kwargs_57496)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 39), tuple_57492, upper_call_result_57497)
    
    # Applying the binary operator '%' (line 176)
    result_mod_57498 = python_operator(stypy.reporting.localization.Localization(__file__, 176, 24), '%', unicode_57491, tuple_57492)
    
    list_57501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 24), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 24), list_57501, result_mod_57498)
    # Processing the call keyword arguments (line 176)
    kwargs_57502 = {}
    unicode_57489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 14), 'unicode', u';')
    # Obtaining the member 'join' of a type (line 176)
    join_57490 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 14), unicode_57489, 'join')
    # Calling join(args, kwargs) (line 176)
    join_call_result_57503 = invoke(stypy.reporting.localization.Localization(__file__, 176, 14), join_57490, *[list_57501], **kwargs_57502)
    
    # Assigning a type to the variable 'pattern' (line 176)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 4), 'pattern', join_call_result_57503)
    
    # Call to listFiles(...): (line 178)
    # Processing the call arguments (line 178)
    # Getting the type of 'directory' (line 178)
    directory_57506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 27), 'directory', False)
    # Getting the type of 'pattern' (line 178)
    pattern_57507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 38), 'pattern', False)
    # Processing the call keyword arguments (line 178)
    kwargs_57508 = {}
    # Getting the type of 'cbook' (line 178)
    cbook_57504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 11), 'cbook', False)
    # Obtaining the member 'listFiles' of a type (line 178)
    listFiles_57505 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 11), cbook_57504, 'listFiles')
    # Calling listFiles(args, kwargs) (line 178)
    listFiles_call_result_57509 = invoke(stypy.reporting.localization.Localization(__file__, 178, 11), listFiles_57505, *[directory_57506, pattern_57507], **kwargs_57508)
    
    # Assigning a type to the variable 'stypy_return_type' (line 178)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 4), 'stypy_return_type', listFiles_call_result_57509)
    
    # ################# End of 'list_fonts(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'list_fonts' in the type store
    # Getting the type of 'stypy_return_type' (line 171)
    stypy_return_type_57510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_57510)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'list_fonts'
    return stypy_return_type_57510

# Assigning a type to the variable 'list_fonts' (line 171)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 0), 'list_fonts', list_fonts)

@norecursion
def win32FontDirectory(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'win32FontDirectory'
    module_type_store = module_type_store.open_function_context('win32FontDirectory', 181, 0, False)
    
    # Passed parameters checking function
    win32FontDirectory.stypy_localization = localization
    win32FontDirectory.stypy_type_of_self = None
    win32FontDirectory.stypy_type_store = module_type_store
    win32FontDirectory.stypy_function_name = 'win32FontDirectory'
    win32FontDirectory.stypy_param_names_list = []
    win32FontDirectory.stypy_varargs_param_name = None
    win32FontDirectory.stypy_kwargs_param_name = None
    win32FontDirectory.stypy_call_defaults = defaults
    win32FontDirectory.stypy_call_varargs = varargs
    win32FontDirectory.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'win32FontDirectory', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'win32FontDirectory', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'win32FontDirectory(...)' code ##################

    unicode_57511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, (-1)), 'unicode', u'\n    Return the user-specified font directory for Win32.  This is\n    looked up from the registry key::\n\n      \\\\HKEY_CURRENT_USER\\Software\\Microsoft\\Windows\\CurrentVersion\\Explorer\\Shell Folders\\Fonts\n\n    If the key is not found, $WINDIR/Fonts will be returned.\n    ')
    
    
    # SSA begins for try-except statement (line 190)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 191, 8))
    
    # 'from six.moves import winreg' statement (line 191)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
    import_57512 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 191, 8), 'six.moves')

    if (type(import_57512) is not StypyTypeError):

        if (import_57512 != 'pyd_module'):
            __import__(import_57512)
            sys_modules_57513 = sys.modules[import_57512]
            import_from_module(stypy.reporting.localization.Localization(__file__, 191, 8), 'six.moves', sys_modules_57513.module_type_store, module_type_store, ['winreg'])
            nest_module(stypy.reporting.localization.Localization(__file__, 191, 8), __file__, sys_modules_57513, sys_modules_57513.module_type_store, module_type_store)
        else:
            from six.moves import winreg

            import_from_module(stypy.reporting.localization.Localization(__file__, 191, 8), 'six.moves', None, module_type_store, ['winreg'], [winreg])

    else:
        # Assigning a type to the variable 'six.moves' (line 191)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 8), 'six.moves', import_57512)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')
    
    # SSA branch for the except part of a try statement (line 190)
    # SSA branch for the except 'ImportError' branch of a try statement (line 190)
    module_type_store.open_ssa_branch('except')
    pass
    # SSA branch for the else branch of a try statement (line 190)
    module_type_store.open_ssa_branch('except else')
    
    
    # SSA begins for try-except statement (line 195)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Name (line 196):
    
    # Assigning a Call to a Name (line 196):
    
    # Call to OpenKey(...): (line 196)
    # Processing the call arguments (line 196)
    # Getting the type of 'winreg' (line 196)
    winreg_57516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 34), 'winreg', False)
    # Obtaining the member 'HKEY_CURRENT_USER' of a type (line 196)
    HKEY_CURRENT_USER_57517 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 34), winreg_57516, 'HKEY_CURRENT_USER')
    # Getting the type of 'MSFolders' (line 196)
    MSFolders_57518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 60), 'MSFolders', False)
    # Processing the call keyword arguments (line 196)
    kwargs_57519 = {}
    # Getting the type of 'winreg' (line 196)
    winreg_57514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 19), 'winreg', False)
    # Obtaining the member 'OpenKey' of a type (line 196)
    OpenKey_57515 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 19), winreg_57514, 'OpenKey')
    # Calling OpenKey(args, kwargs) (line 196)
    OpenKey_call_result_57520 = invoke(stypy.reporting.localization.Localization(__file__, 196, 19), OpenKey_57515, *[HKEY_CURRENT_USER_57517, MSFolders_57518], **kwargs_57519)
    
    # Assigning a type to the variable 'user' (line 196)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 12), 'user', OpenKey_call_result_57520)
    
    # Try-finally block (line 197)
    
    
    # SSA begins for try-except statement (line 198)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Obtaining the type of the subscript
    int_57521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 62), 'int')
    
    # Call to QueryValueEx(...): (line 199)
    # Processing the call arguments (line 199)
    # Getting the type of 'user' (line 199)
    user_57524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 47), 'user', False)
    unicode_57525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 53), 'unicode', u'Fonts')
    # Processing the call keyword arguments (line 199)
    kwargs_57526 = {}
    # Getting the type of 'winreg' (line 199)
    winreg_57522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 27), 'winreg', False)
    # Obtaining the member 'QueryValueEx' of a type (line 199)
    QueryValueEx_57523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 27), winreg_57522, 'QueryValueEx')
    # Calling QueryValueEx(args, kwargs) (line 199)
    QueryValueEx_call_result_57527 = invoke(stypy.reporting.localization.Localization(__file__, 199, 27), QueryValueEx_57523, *[user_57524, unicode_57525], **kwargs_57526)
    
    # Obtaining the member '__getitem__' of a type (line 199)
    getitem___57528 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 27), QueryValueEx_call_result_57527, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 199)
    subscript_call_result_57529 = invoke(stypy.reporting.localization.Localization(__file__, 199, 27), getitem___57528, int_57521)
    
    # Assigning a type to the variable 'stypy_return_type' (line 199)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 20), 'stypy_return_type', subscript_call_result_57529)
    # SSA branch for the except part of a try statement (line 198)
    # SSA branch for the except 'OSError' branch of a try statement (line 198)
    module_type_store.open_ssa_branch('except')
    pass
    # SSA join for try-except statement (line 198)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # finally branch of the try-finally block (line 197)
    
    # Call to CloseKey(...): (line 203)
    # Processing the call arguments (line 203)
    # Getting the type of 'user' (line 203)
    user_57532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 32), 'user', False)
    # Processing the call keyword arguments (line 203)
    kwargs_57533 = {}
    # Getting the type of 'winreg' (line 203)
    winreg_57530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 16), 'winreg', False)
    # Obtaining the member 'CloseKey' of a type (line 203)
    CloseKey_57531 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 16), winreg_57530, 'CloseKey')
    # Calling CloseKey(args, kwargs) (line 203)
    CloseKey_call_result_57534 = invoke(stypy.reporting.localization.Localization(__file__, 203, 16), CloseKey_57531, *[user_57532], **kwargs_57533)
    
    
    # SSA branch for the except part of a try statement (line 195)
    # SSA branch for the except 'OSError' branch of a try statement (line 195)
    module_type_store.open_ssa_branch('except')
    pass
    # SSA join for try-except statement (line 195)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for try-except statement (line 190)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to join(...): (line 206)
    # Processing the call arguments (line 206)
    
    # Obtaining the type of the subscript
    unicode_57538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 35), 'unicode', u'WINDIR')
    # Getting the type of 'os' (line 206)
    os_57539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 24), 'os', False)
    # Obtaining the member 'environ' of a type (line 206)
    environ_57540 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 24), os_57539, 'environ')
    # Obtaining the member '__getitem__' of a type (line 206)
    getitem___57541 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 24), environ_57540, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 206)
    subscript_call_result_57542 = invoke(stypy.reporting.localization.Localization(__file__, 206, 24), getitem___57541, unicode_57538)
    
    unicode_57543 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 46), 'unicode', u'Fonts')
    # Processing the call keyword arguments (line 206)
    kwargs_57544 = {}
    # Getting the type of 'os' (line 206)
    os_57535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 11), 'os', False)
    # Obtaining the member 'path' of a type (line 206)
    path_57536 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 11), os_57535, 'path')
    # Obtaining the member 'join' of a type (line 206)
    join_57537 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 11), path_57536, 'join')
    # Calling join(args, kwargs) (line 206)
    join_call_result_57545 = invoke(stypy.reporting.localization.Localization(__file__, 206, 11), join_57537, *[subscript_call_result_57542, unicode_57543], **kwargs_57544)
    
    # Assigning a type to the variable 'stypy_return_type' (line 206)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 4), 'stypy_return_type', join_call_result_57545)
    
    # ################# End of 'win32FontDirectory(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'win32FontDirectory' in the type store
    # Getting the type of 'stypy_return_type' (line 181)
    stypy_return_type_57546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_57546)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'win32FontDirectory'
    return stypy_return_type_57546

# Assigning a type to the variable 'win32FontDirectory' (line 181)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 0), 'win32FontDirectory', win32FontDirectory)

@norecursion
def win32InstalledFonts(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 209)
    None_57547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 34), 'None')
    unicode_57548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 48), 'unicode', u'ttf')
    defaults = [None_57547, unicode_57548]
    # Create a new context for function 'win32InstalledFonts'
    module_type_store = module_type_store.open_function_context('win32InstalledFonts', 209, 0, False)
    
    # Passed parameters checking function
    win32InstalledFonts.stypy_localization = localization
    win32InstalledFonts.stypy_type_of_self = None
    win32InstalledFonts.stypy_type_store = module_type_store
    win32InstalledFonts.stypy_function_name = 'win32InstalledFonts'
    win32InstalledFonts.stypy_param_names_list = ['directory', 'fontext']
    win32InstalledFonts.stypy_varargs_param_name = None
    win32InstalledFonts.stypy_kwargs_param_name = None
    win32InstalledFonts.stypy_call_defaults = defaults
    win32InstalledFonts.stypy_call_varargs = varargs
    win32InstalledFonts.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'win32InstalledFonts', ['directory', 'fontext'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'win32InstalledFonts', localization, ['directory', 'fontext'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'win32InstalledFonts(...)' code ##################

    unicode_57549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, (-1)), 'unicode', u"\n    Search for fonts in the specified font directory, or use the\n    system directories if none given.  A list of TrueType font\n    filenames are returned by default, or AFM fonts if *fontext* ==\n    'afm'.\n    ")
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 217, 4))
    
    # 'from six.moves import winreg' statement (line 217)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
    import_57550 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 217, 4), 'six.moves')

    if (type(import_57550) is not StypyTypeError):

        if (import_57550 != 'pyd_module'):
            __import__(import_57550)
            sys_modules_57551 = sys.modules[import_57550]
            import_from_module(stypy.reporting.localization.Localization(__file__, 217, 4), 'six.moves', sys_modules_57551.module_type_store, module_type_store, ['winreg'])
            nest_module(stypy.reporting.localization.Localization(__file__, 217, 4), __file__, sys_modules_57551, sys_modules_57551.module_type_store, module_type_store)
        else:
            from six.moves import winreg

            import_from_module(stypy.reporting.localization.Localization(__file__, 217, 4), 'six.moves', None, module_type_store, ['winreg'], [winreg])

    else:
        # Assigning a type to the variable 'six.moves' (line 217)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 4), 'six.moves', import_57550)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')
    
    
    # Type idiom detected: calculating its left and rigth part (line 218)
    # Getting the type of 'directory' (line 218)
    directory_57552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 7), 'directory')
    # Getting the type of 'None' (line 218)
    None_57553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 20), 'None')
    
    (may_be_57554, more_types_in_union_57555) = may_be_none(directory_57552, None_57553)

    if may_be_57554:

        if more_types_in_union_57555:
            # Runtime conditional SSA (line 218)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 219):
        
        # Assigning a Call to a Name (line 219):
        
        # Call to win32FontDirectory(...): (line 219)
        # Processing the call keyword arguments (line 219)
        kwargs_57557 = {}
        # Getting the type of 'win32FontDirectory' (line 219)
        win32FontDirectory_57556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 20), 'win32FontDirectory', False)
        # Calling win32FontDirectory(args, kwargs) (line 219)
        win32FontDirectory_call_result_57558 = invoke(stypy.reporting.localization.Localization(__file__, 219, 20), win32FontDirectory_57556, *[], **kwargs_57557)
        
        # Assigning a type to the variable 'directory' (line 219)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 8), 'directory', win32FontDirectory_call_result_57558)

        if more_types_in_union_57555:
            # SSA join for if statement (line 218)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 221):
    
    # Assigning a Call to a Name (line 221):
    
    # Call to get_fontext_synonyms(...): (line 221)
    # Processing the call arguments (line 221)
    # Getting the type of 'fontext' (line 221)
    fontext_57560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 35), 'fontext', False)
    # Processing the call keyword arguments (line 221)
    kwargs_57561 = {}
    # Getting the type of 'get_fontext_synonyms' (line 221)
    get_fontext_synonyms_57559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 14), 'get_fontext_synonyms', False)
    # Calling get_fontext_synonyms(args, kwargs) (line 221)
    get_fontext_synonyms_call_result_57562 = invoke(stypy.reporting.localization.Localization(__file__, 221, 14), get_fontext_synonyms_57559, *[fontext_57560], **kwargs_57561)
    
    # Assigning a type to the variable 'fontext' (line 221)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 4), 'fontext', get_fontext_synonyms_call_result_57562)
    
    # Assigning a Tuple to a Tuple (line 223):
    
    # Assigning a Name to a Name (line 223):
    # Getting the type of 'None' (line 223)
    None_57563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 17), 'None')
    # Assigning a type to the variable 'tuple_assignment_57308' (line 223)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 4), 'tuple_assignment_57308', None_57563)
    
    # Assigning a Dict to a Name (line 223):
    
    # Obtaining an instance of the builtin type 'dict' (line 223)
    dict_57564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 23), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 223)
    
    # Assigning a type to the variable 'tuple_assignment_57309' (line 223)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 4), 'tuple_assignment_57309', dict_57564)
    
    # Assigning a Name to a Name (line 223):
    # Getting the type of 'tuple_assignment_57308' (line 223)
    tuple_assignment_57308_57565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 4), 'tuple_assignment_57308')
    # Assigning a type to the variable 'key' (line 223)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 4), 'key', tuple_assignment_57308_57565)
    
    # Assigning a Name to a Name (line 223):
    # Getting the type of 'tuple_assignment_57309' (line 223)
    tuple_assignment_57309_57566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 4), 'tuple_assignment_57309')
    # Assigning a type to the variable 'items' (line 223)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 9), 'items', tuple_assignment_57309_57566)
    
    # Getting the type of 'MSFontDirectories' (line 224)
    MSFontDirectories_57567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 19), 'MSFontDirectories')
    # Testing the type of a for loop iterable (line 224)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 224, 4), MSFontDirectories_57567)
    # Getting the type of the for loop variable (line 224)
    for_loop_var_57568 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 224, 4), MSFontDirectories_57567)
    # Assigning a type to the variable 'fontdir' (line 224)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 4), 'fontdir', for_loop_var_57568)
    # SSA begins for a for statement (line 224)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # SSA begins for try-except statement (line 225)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Name (line 226):
    
    # Assigning a Call to a Name (line 226):
    
    # Call to OpenKey(...): (line 226)
    # Processing the call arguments (line 226)
    # Getting the type of 'winreg' (line 226)
    winreg_57571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 35), 'winreg', False)
    # Obtaining the member 'HKEY_LOCAL_MACHINE' of a type (line 226)
    HKEY_LOCAL_MACHINE_57572 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 35), winreg_57571, 'HKEY_LOCAL_MACHINE')
    # Getting the type of 'fontdir' (line 226)
    fontdir_57573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 62), 'fontdir', False)
    # Processing the call keyword arguments (line 226)
    kwargs_57574 = {}
    # Getting the type of 'winreg' (line 226)
    winreg_57569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 20), 'winreg', False)
    # Obtaining the member 'OpenKey' of a type (line 226)
    OpenKey_57570 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 20), winreg_57569, 'OpenKey')
    # Calling OpenKey(args, kwargs) (line 226)
    OpenKey_call_result_57575 = invoke(stypy.reporting.localization.Localization(__file__, 226, 20), OpenKey_57570, *[HKEY_LOCAL_MACHINE_57572, fontdir_57573], **kwargs_57574)
    
    # Assigning a type to the variable 'local' (line 226)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 12), 'local', OpenKey_call_result_57575)
    # SSA branch for the except part of a try statement (line 225)
    # SSA branch for the except 'OSError' branch of a try statement (line 225)
    module_type_store.open_ssa_branch('except')
    # SSA join for try-except statement (line 225)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'local' (line 230)
    local_57576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 15), 'local')
    # Applying the 'not' unary operator (line 230)
    result_not__57577 = python_operator(stypy.reporting.localization.Localization(__file__, 230, 11), 'not', local_57576)
    
    # Testing the type of an if condition (line 230)
    if_condition_57578 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 230, 8), result_not__57577)
    # Assigning a type to the variable 'if_condition_57578' (line 230)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 8), 'if_condition_57578', if_condition_57578)
    # SSA begins for if statement (line 230)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to list_fonts(...): (line 231)
    # Processing the call arguments (line 231)
    # Getting the type of 'directory' (line 231)
    directory_57580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 30), 'directory', False)
    # Getting the type of 'fontext' (line 231)
    fontext_57581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 41), 'fontext', False)
    # Processing the call keyword arguments (line 231)
    kwargs_57582 = {}
    # Getting the type of 'list_fonts' (line 231)
    list_fonts_57579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 19), 'list_fonts', False)
    # Calling list_fonts(args, kwargs) (line 231)
    list_fonts_call_result_57583 = invoke(stypy.reporting.localization.Localization(__file__, 231, 19), list_fonts_57579, *[directory_57580, fontext_57581], **kwargs_57582)
    
    # Assigning a type to the variable 'stypy_return_type' (line 231)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 12), 'stypy_return_type', list_fonts_call_result_57583)
    # SSA join for if statement (line 230)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Try-finally block (line 232)
    
    
    # Call to range(...): (line 233)
    # Processing the call arguments (line 233)
    
    # Obtaining the type of the subscript
    int_57585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 54), 'int')
    
    # Call to QueryInfoKey(...): (line 233)
    # Processing the call arguments (line 233)
    # Getting the type of 'local' (line 233)
    local_57588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 47), 'local', False)
    # Processing the call keyword arguments (line 233)
    kwargs_57589 = {}
    # Getting the type of 'winreg' (line 233)
    winreg_57586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 27), 'winreg', False)
    # Obtaining the member 'QueryInfoKey' of a type (line 233)
    QueryInfoKey_57587 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 27), winreg_57586, 'QueryInfoKey')
    # Calling QueryInfoKey(args, kwargs) (line 233)
    QueryInfoKey_call_result_57590 = invoke(stypy.reporting.localization.Localization(__file__, 233, 27), QueryInfoKey_57587, *[local_57588], **kwargs_57589)
    
    # Obtaining the member '__getitem__' of a type (line 233)
    getitem___57591 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 27), QueryInfoKey_call_result_57590, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 233)
    subscript_call_result_57592 = invoke(stypy.reporting.localization.Localization(__file__, 233, 27), getitem___57591, int_57585)
    
    # Processing the call keyword arguments (line 233)
    kwargs_57593 = {}
    # Getting the type of 'range' (line 233)
    range_57584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 21), 'range', False)
    # Calling range(args, kwargs) (line 233)
    range_call_result_57594 = invoke(stypy.reporting.localization.Localization(__file__, 233, 21), range_57584, *[subscript_call_result_57592], **kwargs_57593)
    
    # Testing the type of a for loop iterable (line 233)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 233, 12), range_call_result_57594)
    # Getting the type of the for loop variable (line 233)
    for_loop_var_57595 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 233, 12), range_call_result_57594)
    # Assigning a type to the variable 'j' (line 233)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 12), 'j', for_loop_var_57595)
    # SSA begins for a for statement (line 233)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # SSA begins for try-except statement (line 234)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Tuple (line 235):
    
    # Assigning a Call to a Name:
    
    # Call to EnumValue(...): (line 235)
    # Processing the call arguments (line 235)
    # Getting the type of 'local' (line 235)
    local_57598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 56), 'local', False)
    # Getting the type of 'j' (line 235)
    j_57599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 63), 'j', False)
    # Processing the call keyword arguments (line 235)
    kwargs_57600 = {}
    # Getting the type of 'winreg' (line 235)
    winreg_57596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 38), 'winreg', False)
    # Obtaining the member 'EnumValue' of a type (line 235)
    EnumValue_57597 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 38), winreg_57596, 'EnumValue')
    # Calling EnumValue(args, kwargs) (line 235)
    EnumValue_call_result_57601 = invoke(stypy.reporting.localization.Localization(__file__, 235, 38), EnumValue_57597, *[local_57598, j_57599], **kwargs_57600)
    
    # Assigning a type to the variable 'call_assignment_57310' (line 235)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 20), 'call_assignment_57310', EnumValue_call_result_57601)
    
    # Assigning a Call to a Name (line 235):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_57604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 20), 'int')
    # Processing the call keyword arguments
    kwargs_57605 = {}
    # Getting the type of 'call_assignment_57310' (line 235)
    call_assignment_57310_57602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 20), 'call_assignment_57310', False)
    # Obtaining the member '__getitem__' of a type (line 235)
    getitem___57603 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 20), call_assignment_57310_57602, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_57606 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___57603, *[int_57604], **kwargs_57605)
    
    # Assigning a type to the variable 'call_assignment_57311' (line 235)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 20), 'call_assignment_57311', getitem___call_result_57606)
    
    # Assigning a Name to a Name (line 235):
    # Getting the type of 'call_assignment_57311' (line 235)
    call_assignment_57311_57607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 20), 'call_assignment_57311')
    # Assigning a type to the variable 'key' (line 235)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 20), 'key', call_assignment_57311_57607)
    
    # Assigning a Call to a Name (line 235):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_57610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 20), 'int')
    # Processing the call keyword arguments
    kwargs_57611 = {}
    # Getting the type of 'call_assignment_57310' (line 235)
    call_assignment_57310_57608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 20), 'call_assignment_57310', False)
    # Obtaining the member '__getitem__' of a type (line 235)
    getitem___57609 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 20), call_assignment_57310_57608, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_57612 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___57609, *[int_57610], **kwargs_57611)
    
    # Assigning a type to the variable 'call_assignment_57312' (line 235)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 20), 'call_assignment_57312', getitem___call_result_57612)
    
    # Assigning a Name to a Name (line 235):
    # Getting the type of 'call_assignment_57312' (line 235)
    call_assignment_57312_57613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 20), 'call_assignment_57312')
    # Assigning a type to the variable 'direc' (line 235)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 25), 'direc', call_assignment_57312_57613)
    
    # Assigning a Call to a Name (line 235):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_57616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 20), 'int')
    # Processing the call keyword arguments
    kwargs_57617 = {}
    # Getting the type of 'call_assignment_57310' (line 235)
    call_assignment_57310_57614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 20), 'call_assignment_57310', False)
    # Obtaining the member '__getitem__' of a type (line 235)
    getitem___57615 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 20), call_assignment_57310_57614, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_57618 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___57615, *[int_57616], **kwargs_57617)
    
    # Assigning a type to the variable 'call_assignment_57313' (line 235)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 20), 'call_assignment_57313', getitem___call_result_57618)
    
    # Assigning a Name to a Name (line 235):
    # Getting the type of 'call_assignment_57313' (line 235)
    call_assignment_57313_57619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 20), 'call_assignment_57313')
    # Assigning a type to the variable 'any' (line 235)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 32), 'any', call_assignment_57313_57619)
    
    
    
    # Call to isinstance(...): (line 236)
    # Processing the call arguments (line 236)
    # Getting the type of 'direc' (line 236)
    direc_57621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 38), 'direc', False)
    # Getting the type of 'six' (line 236)
    six_57622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 45), 'six', False)
    # Obtaining the member 'string_types' of a type (line 236)
    string_types_57623 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 45), six_57622, 'string_types')
    # Processing the call keyword arguments (line 236)
    kwargs_57624 = {}
    # Getting the type of 'isinstance' (line 236)
    isinstance_57620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 27), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 236)
    isinstance_call_result_57625 = invoke(stypy.reporting.localization.Localization(__file__, 236, 27), isinstance_57620, *[direc_57621, string_types_57623], **kwargs_57624)
    
    # Applying the 'not' unary operator (line 236)
    result_not__57626 = python_operator(stypy.reporting.localization.Localization(__file__, 236, 23), 'not', isinstance_call_result_57625)
    
    # Testing the type of an if condition (line 236)
    if_condition_57627 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 236, 20), result_not__57626)
    # Assigning a type to the variable 'if_condition_57627' (line 236)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 20), 'if_condition_57627', if_condition_57627)
    # SSA begins for if statement (line 236)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 236)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to dirname(...): (line 238)
    # Processing the call arguments (line 238)
    # Getting the type of 'direc' (line 238)
    direc_57631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 43), 'direc', False)
    # Processing the call keyword arguments (line 238)
    kwargs_57632 = {}
    # Getting the type of 'os' (line 238)
    os_57628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 27), 'os', False)
    # Obtaining the member 'path' of a type (line 238)
    path_57629 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 27), os_57628, 'path')
    # Obtaining the member 'dirname' of a type (line 238)
    dirname_57630 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 27), path_57629, 'dirname')
    # Calling dirname(args, kwargs) (line 238)
    dirname_call_result_57633 = invoke(stypy.reporting.localization.Localization(__file__, 238, 27), dirname_57630, *[direc_57631], **kwargs_57632)
    
    # Applying the 'not' unary operator (line 238)
    result_not__57634 = python_operator(stypy.reporting.localization.Localization(__file__, 238, 23), 'not', dirname_call_result_57633)
    
    # Testing the type of an if condition (line 238)
    if_condition_57635 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 238, 20), result_not__57634)
    # Assigning a type to the variable 'if_condition_57635' (line 238)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 20), 'if_condition_57635', if_condition_57635)
    # SSA begins for if statement (line 238)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 239):
    
    # Assigning a Call to a Name (line 239):
    
    # Call to join(...): (line 239)
    # Processing the call arguments (line 239)
    # Getting the type of 'directory' (line 239)
    directory_57639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 45), 'directory', False)
    # Getting the type of 'direc' (line 239)
    direc_57640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 56), 'direc', False)
    # Processing the call keyword arguments (line 239)
    kwargs_57641 = {}
    # Getting the type of 'os' (line 239)
    os_57636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 32), 'os', False)
    # Obtaining the member 'path' of a type (line 239)
    path_57637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 32), os_57636, 'path')
    # Obtaining the member 'join' of a type (line 239)
    join_57638 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 32), path_57637, 'join')
    # Calling join(args, kwargs) (line 239)
    join_call_result_57642 = invoke(stypy.reporting.localization.Localization(__file__, 239, 32), join_57638, *[directory_57639, direc_57640], **kwargs_57641)
    
    # Assigning a type to the variable 'direc' (line 239)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 24), 'direc', join_call_result_57642)
    # SSA join for if statement (line 238)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 240):
    
    # Assigning a Call to a Name (line 240):
    
    # Call to lower(...): (line 240)
    # Processing the call keyword arguments (line 240)
    kwargs_57650 = {}
    
    # Call to abspath(...): (line 240)
    # Processing the call arguments (line 240)
    # Getting the type of 'direc' (line 240)
    direc_57646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 44), 'direc', False)
    # Processing the call keyword arguments (line 240)
    kwargs_57647 = {}
    # Getting the type of 'os' (line 240)
    os_57643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 28), 'os', False)
    # Obtaining the member 'path' of a type (line 240)
    path_57644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 28), os_57643, 'path')
    # Obtaining the member 'abspath' of a type (line 240)
    abspath_57645 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 28), path_57644, 'abspath')
    # Calling abspath(args, kwargs) (line 240)
    abspath_call_result_57648 = invoke(stypy.reporting.localization.Localization(__file__, 240, 28), abspath_57645, *[direc_57646], **kwargs_57647)
    
    # Obtaining the member 'lower' of a type (line 240)
    lower_57649 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 28), abspath_call_result_57648, 'lower')
    # Calling lower(args, kwargs) (line 240)
    lower_call_result_57651 = invoke(stypy.reporting.localization.Localization(__file__, 240, 28), lower_57649, *[], **kwargs_57650)
    
    # Assigning a type to the variable 'direc' (line 240)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 20), 'direc', lower_call_result_57651)
    
    
    
    # Obtaining the type of the subscript
    int_57652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 50), 'int')
    slice_57653 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 241, 23), int_57652, None, None)
    
    # Obtaining the type of the subscript
    int_57654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 47), 'int')
    
    # Call to splitext(...): (line 241)
    # Processing the call arguments (line 241)
    # Getting the type of 'direc' (line 241)
    direc_57658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 40), 'direc', False)
    # Processing the call keyword arguments (line 241)
    kwargs_57659 = {}
    # Getting the type of 'os' (line 241)
    os_57655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 23), 'os', False)
    # Obtaining the member 'path' of a type (line 241)
    path_57656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 23), os_57655, 'path')
    # Obtaining the member 'splitext' of a type (line 241)
    splitext_57657 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 23), path_57656, 'splitext')
    # Calling splitext(args, kwargs) (line 241)
    splitext_call_result_57660 = invoke(stypy.reporting.localization.Localization(__file__, 241, 23), splitext_57657, *[direc_57658], **kwargs_57659)
    
    # Obtaining the member '__getitem__' of a type (line 241)
    getitem___57661 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 23), splitext_call_result_57660, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 241)
    subscript_call_result_57662 = invoke(stypy.reporting.localization.Localization(__file__, 241, 23), getitem___57661, int_57654)
    
    # Obtaining the member '__getitem__' of a type (line 241)
    getitem___57663 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 23), subscript_call_result_57662, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 241)
    subscript_call_result_57664 = invoke(stypy.reporting.localization.Localization(__file__, 241, 23), getitem___57663, slice_57653)
    
    # Getting the type of 'fontext' (line 241)
    fontext_57665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 57), 'fontext')
    # Applying the binary operator 'in' (line 241)
    result_contains_57666 = python_operator(stypy.reporting.localization.Localization(__file__, 241, 23), 'in', subscript_call_result_57664, fontext_57665)
    
    # Testing the type of an if condition (line 241)
    if_condition_57667 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 241, 20), result_contains_57666)
    # Assigning a type to the variable 'if_condition_57667' (line 241)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 20), 'if_condition_57667', if_condition_57667)
    # SSA begins for if statement (line 241)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Subscript (line 242):
    
    # Assigning a Num to a Subscript (line 242):
    int_57668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 39), 'int')
    # Getting the type of 'items' (line 242)
    items_57669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 24), 'items')
    # Getting the type of 'direc' (line 242)
    direc_57670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 30), 'direc')
    # Storing an element on a container (line 242)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 242, 24), items_57669, (direc_57670, int_57668))
    # SSA join for if statement (line 241)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the except part of a try statement (line 234)
    # SSA branch for the except 'EnvironmentError' branch of a try statement (line 234)
    module_type_store.open_ssa_branch('except')
    # SSA branch for the except 'WindowsError' branch of a try statement (line 234)
    module_type_store.open_ssa_branch('except')
    # SSA branch for the except 'MemoryError' branch of a try statement (line 234)
    module_type_store.open_ssa_branch('except')
    # SSA join for try-except statement (line 234)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to list(...): (line 249)
    # Processing the call arguments (line 249)
    # Getting the type of 'items' (line 249)
    items_57672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 24), 'items', False)
    # Processing the call keyword arguments (line 249)
    kwargs_57673 = {}
    # Getting the type of 'list' (line 249)
    list_57671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 19), 'list', False)
    # Calling list(args, kwargs) (line 249)
    list_call_result_57674 = invoke(stypy.reporting.localization.Localization(__file__, 249, 19), list_57671, *[items_57672], **kwargs_57673)
    
    # Assigning a type to the variable 'stypy_return_type' (line 249)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 12), 'stypy_return_type', list_call_result_57674)
    
    # finally branch of the try-finally block (line 232)
    
    # Call to CloseKey(...): (line 251)
    # Processing the call arguments (line 251)
    # Getting the type of 'local' (line 251)
    local_57677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 28), 'local', False)
    # Processing the call keyword arguments (line 251)
    kwargs_57678 = {}
    # Getting the type of 'winreg' (line 251)
    winreg_57675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 12), 'winreg', False)
    # Obtaining the member 'CloseKey' of a type (line 251)
    CloseKey_57676 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 12), winreg_57675, 'CloseKey')
    # Calling CloseKey(args, kwargs) (line 251)
    CloseKey_call_result_57679 = invoke(stypy.reporting.localization.Localization(__file__, 251, 12), CloseKey_57676, *[local_57677], **kwargs_57678)
    
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'None' (line 252)
    None_57680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 11), 'None')
    # Assigning a type to the variable 'stypy_return_type' (line 252)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 4), 'stypy_return_type', None_57680)
    
    # ################# End of 'win32InstalledFonts(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'win32InstalledFonts' in the type store
    # Getting the type of 'stypy_return_type' (line 209)
    stypy_return_type_57681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_57681)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'win32InstalledFonts'
    return stypy_return_type_57681

# Assigning a type to the variable 'win32InstalledFonts' (line 209)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 0), 'win32InstalledFonts', win32InstalledFonts)

@norecursion
def OSXInstalledFonts(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 255)
    None_57682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 34), 'None')
    unicode_57683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 48), 'unicode', u'ttf')
    defaults = [None_57682, unicode_57683]
    # Create a new context for function 'OSXInstalledFonts'
    module_type_store = module_type_store.open_function_context('OSXInstalledFonts', 255, 0, False)
    
    # Passed parameters checking function
    OSXInstalledFonts.stypy_localization = localization
    OSXInstalledFonts.stypy_type_of_self = None
    OSXInstalledFonts.stypy_type_store = module_type_store
    OSXInstalledFonts.stypy_function_name = 'OSXInstalledFonts'
    OSXInstalledFonts.stypy_param_names_list = ['directories', 'fontext']
    OSXInstalledFonts.stypy_varargs_param_name = None
    OSXInstalledFonts.stypy_kwargs_param_name = None
    OSXInstalledFonts.stypy_call_defaults = defaults
    OSXInstalledFonts.stypy_call_varargs = varargs
    OSXInstalledFonts.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'OSXInstalledFonts', ['directories', 'fontext'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'OSXInstalledFonts', localization, ['directories', 'fontext'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'OSXInstalledFonts(...)' code ##################

    unicode_57684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, (-1)), 'unicode', u'\n    Get list of font files on OS X - ignores font suffix by default.\n    ')
    
    # Type idiom detected: calculating its left and rigth part (line 259)
    # Getting the type of 'directories' (line 259)
    directories_57685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 7), 'directories')
    # Getting the type of 'None' (line 259)
    None_57686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 22), 'None')
    
    (may_be_57687, more_types_in_union_57688) = may_be_none(directories_57685, None_57686)

    if may_be_57687:

        if more_types_in_union_57688:
            # Runtime conditional SSA (line 259)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Name to a Name (line 260):
        
        # Assigning a Name to a Name (line 260):
        # Getting the type of 'OSXFontDirectories' (line 260)
        OSXFontDirectories_57689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 22), 'OSXFontDirectories')
        # Assigning a type to the variable 'directories' (line 260)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 8), 'directories', OSXFontDirectories_57689)

        if more_types_in_union_57688:
            # SSA join for if statement (line 259)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 262):
    
    # Assigning a Call to a Name (line 262):
    
    # Call to get_fontext_synonyms(...): (line 262)
    # Processing the call arguments (line 262)
    # Getting the type of 'fontext' (line 262)
    fontext_57691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 35), 'fontext', False)
    # Processing the call keyword arguments (line 262)
    kwargs_57692 = {}
    # Getting the type of 'get_fontext_synonyms' (line 262)
    get_fontext_synonyms_57690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 14), 'get_fontext_synonyms', False)
    # Calling get_fontext_synonyms(args, kwargs) (line 262)
    get_fontext_synonyms_call_result_57693 = invoke(stypy.reporting.localization.Localization(__file__, 262, 14), get_fontext_synonyms_57690, *[fontext_57691], **kwargs_57692)
    
    # Assigning a type to the variable 'fontext' (line 262)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 4), 'fontext', get_fontext_synonyms_call_result_57693)
    
    # Assigning a List to a Name (line 264):
    
    # Assigning a List to a Name (line 264):
    
    # Obtaining an instance of the builtin type 'list' (line 264)
    list_57694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 12), 'list')
    # Adding type elements to the builtin type 'list' instance (line 264)
    
    # Assigning a type to the variable 'files' (line 264)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 4), 'files', list_57694)
    
    # Getting the type of 'directories' (line 265)
    directories_57695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 16), 'directories')
    # Testing the type of a for loop iterable (line 265)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 265, 4), directories_57695)
    # Getting the type of the for loop variable (line 265)
    for_loop_var_57696 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 265, 4), directories_57695)
    # Assigning a type to the variable 'path' (line 265)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 4), 'path', for_loop_var_57696)
    # SSA begins for a for statement (line 265)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Type idiom detected: calculating its left and rigth part (line 266)
    # Getting the type of 'fontext' (line 266)
    fontext_57697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 11), 'fontext')
    # Getting the type of 'None' (line 266)
    None_57698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 22), 'None')
    
    (may_be_57699, more_types_in_union_57700) = may_be_none(fontext_57697, None_57698)

    if may_be_57699:

        if more_types_in_union_57700:
            # Runtime conditional SSA (line 266)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Call to extend(...): (line 267)
        # Processing the call arguments (line 267)
        
        # Call to listFiles(...): (line 267)
        # Processing the call arguments (line 267)
        # Getting the type of 'path' (line 267)
        path_57705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 41), 'path', False)
        unicode_57706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 47), 'unicode', u'*')
        # Processing the call keyword arguments (line 267)
        kwargs_57707 = {}
        # Getting the type of 'cbook' (line 267)
        cbook_57703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 25), 'cbook', False)
        # Obtaining the member 'listFiles' of a type (line 267)
        listFiles_57704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 267, 25), cbook_57703, 'listFiles')
        # Calling listFiles(args, kwargs) (line 267)
        listFiles_call_result_57708 = invoke(stypy.reporting.localization.Localization(__file__, 267, 25), listFiles_57704, *[path_57705, unicode_57706], **kwargs_57707)
        
        # Processing the call keyword arguments (line 267)
        kwargs_57709 = {}
        # Getting the type of 'files' (line 267)
        files_57701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 12), 'files', False)
        # Obtaining the member 'extend' of a type (line 267)
        extend_57702 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 267, 12), files_57701, 'extend')
        # Calling extend(args, kwargs) (line 267)
        extend_call_result_57710 = invoke(stypy.reporting.localization.Localization(__file__, 267, 12), extend_57702, *[listFiles_call_result_57708], **kwargs_57709)
        

        if more_types_in_union_57700:
            # Runtime conditional SSA for else branch (line 266)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_57699) or more_types_in_union_57700):
        
        # Call to extend(...): (line 269)
        # Processing the call arguments (line 269)
        
        # Call to list_fonts(...): (line 269)
        # Processing the call arguments (line 269)
        # Getting the type of 'path' (line 269)
        path_57714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 36), 'path', False)
        # Getting the type of 'fontext' (line 269)
        fontext_57715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 42), 'fontext', False)
        # Processing the call keyword arguments (line 269)
        kwargs_57716 = {}
        # Getting the type of 'list_fonts' (line 269)
        list_fonts_57713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 25), 'list_fonts', False)
        # Calling list_fonts(args, kwargs) (line 269)
        list_fonts_call_result_57717 = invoke(stypy.reporting.localization.Localization(__file__, 269, 25), list_fonts_57713, *[path_57714, fontext_57715], **kwargs_57716)
        
        # Processing the call keyword arguments (line 269)
        kwargs_57718 = {}
        # Getting the type of 'files' (line 269)
        files_57711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 12), 'files', False)
        # Obtaining the member 'extend' of a type (line 269)
        extend_57712 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 12), files_57711, 'extend')
        # Calling extend(args, kwargs) (line 269)
        extend_call_result_57719 = invoke(stypy.reporting.localization.Localization(__file__, 269, 12), extend_57712, *[list_fonts_call_result_57717], **kwargs_57718)
        

        if (may_be_57699 and more_types_in_union_57700):
            # SSA join for if statement (line 266)
            module_type_store = module_type_store.join_ssa_context()


    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'files' (line 270)
    files_57720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 11), 'files')
    # Assigning a type to the variable 'stypy_return_type' (line 270)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 270, 4), 'stypy_return_type', files_57720)
    
    # ################# End of 'OSXInstalledFonts(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'OSXInstalledFonts' in the type store
    # Getting the type of 'stypy_return_type' (line 255)
    stypy_return_type_57721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_57721)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'OSXInstalledFonts'
    return stypy_return_type_57721

# Assigning a type to the variable 'OSXInstalledFonts' (line 255)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 0), 'OSXInstalledFonts', OSXInstalledFonts)

@norecursion
def _call_fc_list(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_call_fc_list'
    module_type_store = module_type_store.open_function_context('_call_fc_list', 273, 0, False)
    
    # Passed parameters checking function
    _call_fc_list.stypy_localization = localization
    _call_fc_list.stypy_type_of_self = None
    _call_fc_list.stypy_type_store = module_type_store
    _call_fc_list.stypy_function_name = '_call_fc_list'
    _call_fc_list.stypy_param_names_list = []
    _call_fc_list.stypy_varargs_param_name = None
    _call_fc_list.stypy_kwargs_param_name = None
    _call_fc_list.stypy_call_defaults = defaults
    _call_fc_list.stypy_call_varargs = varargs
    _call_fc_list.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_call_fc_list', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_call_fc_list', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_call_fc_list(...)' code ##################

    unicode_57722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, (-1)), 'unicode', u'Cache and list the font filenames known to `fc-list`.\n    ')
    
    # Assigning a Call to a Name (line 278):
    
    # Assigning a Call to a Name (line 278):
    
    # Call to Timer(...): (line 278)
    # Processing the call arguments (line 278)
    int_57724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 18), 'int')

    @norecursion
    def _stypy_temp_lambda_16(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_16'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_16', 278, 21, True)
        # Passed parameters checking function
        _stypy_temp_lambda_16.stypy_localization = localization
        _stypy_temp_lambda_16.stypy_type_of_self = None
        _stypy_temp_lambda_16.stypy_type_store = module_type_store
        _stypy_temp_lambda_16.stypy_function_name = '_stypy_temp_lambda_16'
        _stypy_temp_lambda_16.stypy_param_names_list = []
        _stypy_temp_lambda_16.stypy_varargs_param_name = None
        _stypy_temp_lambda_16.stypy_kwargs_param_name = None
        _stypy_temp_lambda_16.stypy_call_defaults = defaults
        _stypy_temp_lambda_16.stypy_call_varargs = varargs
        _stypy_temp_lambda_16.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_16', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Stacktrace push for error reporting
        localization.set_stack_trace('_stypy_temp_lambda_16', [], arguments)
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of the lambda function code ##################

        
        # Call to warn(...): (line 278)
        # Processing the call arguments (line 278)
        unicode_57727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 8), 'unicode', u'Matplotlib is building the font cache using fc-list. This may take a moment.')
        # Processing the call keyword arguments (line 278)
        kwargs_57728 = {}
        # Getting the type of 'warnings' (line 278)
        warnings_57725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 29), 'warnings', False)
        # Obtaining the member 'warn' of a type (line 278)
        warn_57726 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 29), warnings_57725, 'warn')
        # Calling warn(args, kwargs) (line 278)
        warn_call_result_57729 = invoke(stypy.reporting.localization.Localization(__file__, 278, 29), warn_57726, *[unicode_57727], **kwargs_57728)
        
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 278)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 21), 'stypy_return_type', warn_call_result_57729)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_16' in the type store
        # Getting the type of 'stypy_return_type' (line 278)
        stypy_return_type_57730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 21), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_57730)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_16'
        return stypy_return_type_57730

    # Assigning a type to the variable '_stypy_temp_lambda_16' (line 278)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 21), '_stypy_temp_lambda_16', _stypy_temp_lambda_16)
    # Getting the type of '_stypy_temp_lambda_16' (line 278)
    _stypy_temp_lambda_16_57731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 21), '_stypy_temp_lambda_16')
    # Processing the call keyword arguments (line 278)
    kwargs_57732 = {}
    # Getting the type of 'Timer' (line 278)
    Timer_57723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 12), 'Timer', False)
    # Calling Timer(args, kwargs) (line 278)
    Timer_call_result_57733 = invoke(stypy.reporting.localization.Localization(__file__, 278, 12), Timer_57723, *[int_57724, _stypy_temp_lambda_16_57731], **kwargs_57732)
    
    # Assigning a type to the variable 'timer' (line 278)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 4), 'timer', Timer_call_result_57733)
    
    # Call to start(...): (line 281)
    # Processing the call keyword arguments (line 281)
    kwargs_57736 = {}
    # Getting the type of 'timer' (line 281)
    timer_57734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 4), 'timer', False)
    # Obtaining the member 'start' of a type (line 281)
    start_57735 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 4), timer_57734, 'start')
    # Calling start(args, kwargs) (line 281)
    start_call_result_57737 = invoke(stypy.reporting.localization.Localization(__file__, 281, 4), start_57735, *[], **kwargs_57736)
    
    
    # Try-finally block (line 282)
    
    
    # SSA begins for try-except statement (line 282)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Name (line 283):
    
    # Assigning a Call to a Name (line 283):
    
    # Call to check_output(...): (line 283)
    # Processing the call arguments (line 283)
    
    # Obtaining an instance of the builtin type 'list' (line 283)
    list_57740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 38), 'list')
    # Adding type elements to the builtin type 'list' instance (line 283)
    # Adding element type (line 283)
    
    # Call to str(...): (line 283)
    # Processing the call arguments (line 283)
    unicode_57742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 43), 'unicode', u'fc-list')
    # Processing the call keyword arguments (line 283)
    kwargs_57743 = {}
    # Getting the type of 'str' (line 283)
    str_57741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 39), 'str', False)
    # Calling str(args, kwargs) (line 283)
    str_call_result_57744 = invoke(stypy.reporting.localization.Localization(__file__, 283, 39), str_57741, *[unicode_57742], **kwargs_57743)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 283, 38), list_57740, str_call_result_57744)
    # Adding element type (line 283)
    unicode_57745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 55), 'unicode', u'--format=%{file}\\n')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 283, 38), list_57740, unicode_57745)
    
    # Processing the call keyword arguments (line 283)
    kwargs_57746 = {}
    # Getting the type of 'subprocess' (line 283)
    subprocess_57738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 14), 'subprocess', False)
    # Obtaining the member 'check_output' of a type (line 283)
    check_output_57739 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 14), subprocess_57738, 'check_output')
    # Calling check_output(args, kwargs) (line 283)
    check_output_call_result_57747 = invoke(stypy.reporting.localization.Localization(__file__, 283, 14), check_output_57739, *[list_57740], **kwargs_57746)
    
    # Assigning a type to the variable 'out' (line 283)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 8), 'out', check_output_call_result_57747)
    # SSA branch for the except part of a try statement (line 282)
    # SSA branch for the except 'Tuple' branch of a try statement (line 282)
    module_type_store.open_ssa_branch('except')
    
    # Obtaining an instance of the builtin type 'list' (line 285)
    list_57748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 285)
    
    # Assigning a type to the variable 'stypy_return_type' (line 285)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 8), 'stypy_return_type', list_57748)
    # SSA join for try-except statement (line 282)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # finally branch of the try-finally block (line 282)
    
    # Call to cancel(...): (line 287)
    # Processing the call keyword arguments (line 287)
    kwargs_57751 = {}
    # Getting the type of 'timer' (line 287)
    timer_57749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 8), 'timer', False)
    # Obtaining the member 'cancel' of a type (line 287)
    cancel_57750 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 8), timer_57749, 'cancel')
    # Calling cancel(args, kwargs) (line 287)
    cancel_call_result_57752 = invoke(stypy.reporting.localization.Localization(__file__, 287, 8), cancel_57750, *[], **kwargs_57751)
    
    
    
    # Assigning a List to a Name (line 288):
    
    # Assigning a List to a Name (line 288):
    
    # Obtaining an instance of the builtin type 'list' (line 288)
    list_57753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 13), 'list')
    # Adding type elements to the builtin type 'list' instance (line 288)
    
    # Assigning a type to the variable 'fnames' (line 288)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 4), 'fnames', list_57753)
    
    
    # Call to split(...): (line 289)
    # Processing the call arguments (line 289)
    str_57756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 27), 'str', '\n')
    # Processing the call keyword arguments (line 289)
    kwargs_57757 = {}
    # Getting the type of 'out' (line 289)
    out_57754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 17), 'out', False)
    # Obtaining the member 'split' of a type (line 289)
    split_57755 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 17), out_57754, 'split')
    # Calling split(args, kwargs) (line 289)
    split_call_result_57758 = invoke(stypy.reporting.localization.Localization(__file__, 289, 17), split_57755, *[str_57756], **kwargs_57757)
    
    # Testing the type of a for loop iterable (line 289)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 289, 4), split_call_result_57758)
    # Getting the type of the for loop variable (line 289)
    for_loop_var_57759 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 289, 4), split_call_result_57758)
    # Assigning a type to the variable 'fname' (line 289)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 4), 'fname', for_loop_var_57759)
    # SSA begins for a for statement (line 289)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # SSA begins for try-except statement (line 290)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Name (line 291):
    
    # Assigning a Call to a Name (line 291):
    
    # Call to text_type(...): (line 291)
    # Processing the call arguments (line 291)
    # Getting the type of 'fname' (line 291)
    fname_57762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 34), 'fname', False)
    
    # Call to getfilesystemencoding(...): (line 291)
    # Processing the call keyword arguments (line 291)
    kwargs_57765 = {}
    # Getting the type of 'sys' (line 291)
    sys_57763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 41), 'sys', False)
    # Obtaining the member 'getfilesystemencoding' of a type (line 291)
    getfilesystemencoding_57764 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 41), sys_57763, 'getfilesystemencoding')
    # Calling getfilesystemencoding(args, kwargs) (line 291)
    getfilesystemencoding_call_result_57766 = invoke(stypy.reporting.localization.Localization(__file__, 291, 41), getfilesystemencoding_57764, *[], **kwargs_57765)
    
    # Processing the call keyword arguments (line 291)
    kwargs_57767 = {}
    # Getting the type of 'six' (line 291)
    six_57760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 20), 'six', False)
    # Obtaining the member 'text_type' of a type (line 291)
    text_type_57761 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 20), six_57760, 'text_type')
    # Calling text_type(args, kwargs) (line 291)
    text_type_call_result_57768 = invoke(stypy.reporting.localization.Localization(__file__, 291, 20), text_type_57761, *[fname_57762, getfilesystemencoding_call_result_57766], **kwargs_57767)
    
    # Assigning a type to the variable 'fname' (line 291)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 12), 'fname', text_type_call_result_57768)
    # SSA branch for the except part of a try statement (line 290)
    # SSA branch for the except 'UnicodeDecodeError' branch of a try statement (line 290)
    module_type_store.open_ssa_branch('except')
    # SSA join for try-except statement (line 290)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to append(...): (line 294)
    # Processing the call arguments (line 294)
    # Getting the type of 'fname' (line 294)
    fname_57771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 22), 'fname', False)
    # Processing the call keyword arguments (line 294)
    kwargs_57772 = {}
    # Getting the type of 'fnames' (line 294)
    fnames_57769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 8), 'fnames', False)
    # Obtaining the member 'append' of a type (line 294)
    append_57770 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 8), fnames_57769, 'append')
    # Calling append(args, kwargs) (line 294)
    append_call_result_57773 = invoke(stypy.reporting.localization.Localization(__file__, 294, 8), append_57770, *[fname_57771], **kwargs_57772)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'fnames' (line 295)
    fnames_57774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 11), 'fnames')
    # Assigning a type to the variable 'stypy_return_type' (line 295)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 4), 'stypy_return_type', fnames_57774)
    
    # ################# End of '_call_fc_list(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_call_fc_list' in the type store
    # Getting the type of 'stypy_return_type' (line 273)
    stypy_return_type_57775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_57775)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_call_fc_list'
    return stypy_return_type_57775

# Assigning a type to the variable '_call_fc_list' (line 273)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 0), '_call_fc_list', _call_fc_list)

@norecursion
def get_fontconfig_fonts(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    unicode_57776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 33), 'unicode', u'ttf')
    defaults = [unicode_57776]
    # Create a new context for function 'get_fontconfig_fonts'
    module_type_store = module_type_store.open_function_context('get_fontconfig_fonts', 298, 0, False)
    
    # Passed parameters checking function
    get_fontconfig_fonts.stypy_localization = localization
    get_fontconfig_fonts.stypy_type_of_self = None
    get_fontconfig_fonts.stypy_type_store = module_type_store
    get_fontconfig_fonts.stypy_function_name = 'get_fontconfig_fonts'
    get_fontconfig_fonts.stypy_param_names_list = ['fontext']
    get_fontconfig_fonts.stypy_varargs_param_name = None
    get_fontconfig_fonts.stypy_kwargs_param_name = None
    get_fontconfig_fonts.stypy_call_defaults = defaults
    get_fontconfig_fonts.stypy_call_varargs = varargs
    get_fontconfig_fonts.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'get_fontconfig_fonts', ['fontext'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'get_fontconfig_fonts', localization, ['fontext'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'get_fontconfig_fonts(...)' code ##################

    unicode_57777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, (-1)), 'unicode', u'List the font filenames known to `fc-list` having the given extension.\n    ')
    
    # Assigning a Call to a Name (line 301):
    
    # Assigning a Call to a Name (line 301):
    
    # Call to get_fontext_synonyms(...): (line 301)
    # Processing the call arguments (line 301)
    # Getting the type of 'fontext' (line 301)
    fontext_57779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 35), 'fontext', False)
    # Processing the call keyword arguments (line 301)
    kwargs_57780 = {}
    # Getting the type of 'get_fontext_synonyms' (line 301)
    get_fontext_synonyms_57778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 14), 'get_fontext_synonyms', False)
    # Calling get_fontext_synonyms(args, kwargs) (line 301)
    get_fontext_synonyms_call_result_57781 = invoke(stypy.reporting.localization.Localization(__file__, 301, 14), get_fontext_synonyms_57778, *[fontext_57779], **kwargs_57780)
    
    # Assigning a type to the variable 'fontext' (line 301)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 4), 'fontext', get_fontext_synonyms_call_result_57781)
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to _call_fc_list(...): (line 302)
    # Processing the call keyword arguments (line 302)
    kwargs_57799 = {}
    # Getting the type of '_call_fc_list' (line 302)
    _call_fc_list_57798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 31), '_call_fc_list', False)
    # Calling _call_fc_list(args, kwargs) (line 302)
    _call_fc_list_call_result_57800 = invoke(stypy.reporting.localization.Localization(__file__, 302, 31), _call_fc_list_57798, *[], **kwargs_57799)
    
    comprehension_57801 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 302, 12), _call_fc_list_call_result_57800)
    # Assigning a type to the variable 'fname' (line 302)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 302, 12), 'fname', comprehension_57801)
    
    
    # Obtaining the type of the subscript
    int_57783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 42), 'int')
    slice_57784 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 303, 15), int_57783, None, None)
    
    # Obtaining the type of the subscript
    int_57785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 39), 'int')
    
    # Call to splitext(...): (line 303)
    # Processing the call arguments (line 303)
    # Getting the type of 'fname' (line 303)
    fname_57789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 32), 'fname', False)
    # Processing the call keyword arguments (line 303)
    kwargs_57790 = {}
    # Getting the type of 'os' (line 303)
    os_57786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 15), 'os', False)
    # Obtaining the member 'path' of a type (line 303)
    path_57787 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 15), os_57786, 'path')
    # Obtaining the member 'splitext' of a type (line 303)
    splitext_57788 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 15), path_57787, 'splitext')
    # Calling splitext(args, kwargs) (line 303)
    splitext_call_result_57791 = invoke(stypy.reporting.localization.Localization(__file__, 303, 15), splitext_57788, *[fname_57789], **kwargs_57790)
    
    # Obtaining the member '__getitem__' of a type (line 303)
    getitem___57792 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 15), splitext_call_result_57791, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 303)
    subscript_call_result_57793 = invoke(stypy.reporting.localization.Localization(__file__, 303, 15), getitem___57792, int_57785)
    
    # Obtaining the member '__getitem__' of a type (line 303)
    getitem___57794 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 15), subscript_call_result_57793, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 303)
    subscript_call_result_57795 = invoke(stypy.reporting.localization.Localization(__file__, 303, 15), getitem___57794, slice_57784)
    
    # Getting the type of 'fontext' (line 303)
    fontext_57796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 49), 'fontext')
    # Applying the binary operator 'in' (line 303)
    result_contains_57797 = python_operator(stypy.reporting.localization.Localization(__file__, 303, 15), 'in', subscript_call_result_57795, fontext_57796)
    
    # Getting the type of 'fname' (line 302)
    fname_57782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 12), 'fname')
    list_57802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 12), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 302, 12), list_57802, fname_57782)
    # Assigning a type to the variable 'stypy_return_type' (line 302)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 302, 4), 'stypy_return_type', list_57802)
    
    # ################# End of 'get_fontconfig_fonts(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_fontconfig_fonts' in the type store
    # Getting the type of 'stypy_return_type' (line 298)
    stypy_return_type_57803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_57803)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_fontconfig_fonts'
    return stypy_return_type_57803

# Assigning a type to the variable 'get_fontconfig_fonts' (line 298)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 0), 'get_fontconfig_fonts', get_fontconfig_fonts)

@norecursion
def findSystemFonts(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 306)
    None_57804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 30), 'None')
    unicode_57805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 44), 'unicode', u'ttf')
    defaults = [None_57804, unicode_57805]
    # Create a new context for function 'findSystemFonts'
    module_type_store = module_type_store.open_function_context('findSystemFonts', 306, 0, False)
    
    # Passed parameters checking function
    findSystemFonts.stypy_localization = localization
    findSystemFonts.stypy_type_of_self = None
    findSystemFonts.stypy_type_store = module_type_store
    findSystemFonts.stypy_function_name = 'findSystemFonts'
    findSystemFonts.stypy_param_names_list = ['fontpaths', 'fontext']
    findSystemFonts.stypy_varargs_param_name = None
    findSystemFonts.stypy_kwargs_param_name = None
    findSystemFonts.stypy_call_defaults = defaults
    findSystemFonts.stypy_call_varargs = varargs
    findSystemFonts.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'findSystemFonts', ['fontpaths', 'fontext'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'findSystemFonts', localization, ['fontpaths', 'fontext'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'findSystemFonts(...)' code ##################

    unicode_57806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, (-1)), 'unicode', u'\n    Search for fonts in the specified font paths.  If no paths are\n    given, will use a standard set of system paths, as well as the\n    list of fonts tracked by fontconfig if fontconfig is installed and\n    available.  A list of TrueType fonts are returned by default with\n    AFM fonts as an option.\n    ')
    
    # Assigning a Call to a Name (line 314):
    
    # Assigning a Call to a Name (line 314):
    
    # Call to set(...): (line 314)
    # Processing the call keyword arguments (line 314)
    kwargs_57808 = {}
    # Getting the type of 'set' (line 314)
    set_57807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 16), 'set', False)
    # Calling set(args, kwargs) (line 314)
    set_call_result_57809 = invoke(stypy.reporting.localization.Localization(__file__, 314, 16), set_57807, *[], **kwargs_57808)
    
    # Assigning a type to the variable 'fontfiles' (line 314)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 314, 4), 'fontfiles', set_call_result_57809)
    
    # Assigning a Call to a Name (line 315):
    
    # Assigning a Call to a Name (line 315):
    
    # Call to get_fontext_synonyms(...): (line 315)
    # Processing the call arguments (line 315)
    # Getting the type of 'fontext' (line 315)
    fontext_57811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 36), 'fontext', False)
    # Processing the call keyword arguments (line 315)
    kwargs_57812 = {}
    # Getting the type of 'get_fontext_synonyms' (line 315)
    get_fontext_synonyms_57810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 15), 'get_fontext_synonyms', False)
    # Calling get_fontext_synonyms(args, kwargs) (line 315)
    get_fontext_synonyms_call_result_57813 = invoke(stypy.reporting.localization.Localization(__file__, 315, 15), get_fontext_synonyms_57810, *[fontext_57811], **kwargs_57812)
    
    # Assigning a type to the variable 'fontexts' (line 315)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 4), 'fontexts', get_fontext_synonyms_call_result_57813)
    
    # Type idiom detected: calculating its left and rigth part (line 317)
    # Getting the type of 'fontpaths' (line 317)
    fontpaths_57814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 7), 'fontpaths')
    # Getting the type of 'None' (line 317)
    None_57815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 20), 'None')
    
    (may_be_57816, more_types_in_union_57817) = may_be_none(fontpaths_57814, None_57815)

    if may_be_57816:

        if more_types_in_union_57817:
            # Runtime conditional SSA (line 317)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        
        # Getting the type of 'sys' (line 318)
        sys_57818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 11), 'sys')
        # Obtaining the member 'platform' of a type (line 318)
        platform_57819 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 11), sys_57818, 'platform')
        unicode_57820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 318, 27), 'unicode', u'win32')
        # Applying the binary operator '==' (line 318)
        result_eq_57821 = python_operator(stypy.reporting.localization.Localization(__file__, 318, 11), '==', platform_57819, unicode_57820)
        
        # Testing the type of an if condition (line 318)
        if_condition_57822 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 318, 8), result_eq_57821)
        # Assigning a type to the variable 'if_condition_57822' (line 318)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 8), 'if_condition_57822', if_condition_57822)
        # SSA begins for if statement (line 318)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 319):
        
        # Assigning a Call to a Name (line 319):
        
        # Call to win32FontDirectory(...): (line 319)
        # Processing the call keyword arguments (line 319)
        kwargs_57824 = {}
        # Getting the type of 'win32FontDirectory' (line 319)
        win32FontDirectory_57823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 22), 'win32FontDirectory', False)
        # Calling win32FontDirectory(args, kwargs) (line 319)
        win32FontDirectory_call_result_57825 = invoke(stypy.reporting.localization.Localization(__file__, 319, 22), win32FontDirectory_57823, *[], **kwargs_57824)
        
        # Assigning a type to the variable 'fontdir' (line 319)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 319, 12), 'fontdir', win32FontDirectory_call_result_57825)
        
        # Assigning a List to a Name (line 321):
        
        # Assigning a List to a Name (line 321):
        
        # Obtaining an instance of the builtin type 'list' (line 321)
        list_57826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 321, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 321)
        # Adding element type (line 321)
        # Getting the type of 'fontdir' (line 321)
        fontdir_57827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 25), 'fontdir')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 321, 24), list_57826, fontdir_57827)
        
        # Assigning a type to the variable 'fontpaths' (line 321)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 321, 12), 'fontpaths', list_57826)
        
        
        # Call to win32InstalledFonts(...): (line 323)
        # Processing the call arguments (line 323)
        # Getting the type of 'fontdir' (line 323)
        fontdir_57829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 41), 'fontdir', False)
        # Processing the call keyword arguments (line 323)
        kwargs_57830 = {}
        # Getting the type of 'win32InstalledFonts' (line 323)
        win32InstalledFonts_57828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 21), 'win32InstalledFonts', False)
        # Calling win32InstalledFonts(args, kwargs) (line 323)
        win32InstalledFonts_call_result_57831 = invoke(stypy.reporting.localization.Localization(__file__, 323, 21), win32InstalledFonts_57828, *[fontdir_57829], **kwargs_57830)
        
        # Testing the type of a for loop iterable (line 323)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 323, 12), win32InstalledFonts_call_result_57831)
        # Getting the type of the for loop variable (line 323)
        for_loop_var_57832 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 323, 12), win32InstalledFonts_call_result_57831)
        # Assigning a type to the variable 'f' (line 323)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 12), 'f', for_loop_var_57832)
        # SSA begins for a for statement (line 323)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Tuple (line 324):
        
        # Assigning a Call to a Name:
        
        # Call to splitext(...): (line 324)
        # Processing the call arguments (line 324)
        # Getting the type of 'f' (line 324)
        f_57836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 45), 'f', False)
        # Processing the call keyword arguments (line 324)
        kwargs_57837 = {}
        # Getting the type of 'os' (line 324)
        os_57833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 28), 'os', False)
        # Obtaining the member 'path' of a type (line 324)
        path_57834 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 324, 28), os_57833, 'path')
        # Obtaining the member 'splitext' of a type (line 324)
        splitext_57835 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 324, 28), path_57834, 'splitext')
        # Calling splitext(args, kwargs) (line 324)
        splitext_call_result_57838 = invoke(stypy.reporting.localization.Localization(__file__, 324, 28), splitext_57835, *[f_57836], **kwargs_57837)
        
        # Assigning a type to the variable 'call_assignment_57314' (line 324)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 324, 16), 'call_assignment_57314', splitext_call_result_57838)
        
        # Assigning a Call to a Name (line 324):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_57841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 16), 'int')
        # Processing the call keyword arguments
        kwargs_57842 = {}
        # Getting the type of 'call_assignment_57314' (line 324)
        call_assignment_57314_57839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 16), 'call_assignment_57314', False)
        # Obtaining the member '__getitem__' of a type (line 324)
        getitem___57840 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 324, 16), call_assignment_57314_57839, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_57843 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___57840, *[int_57841], **kwargs_57842)
        
        # Assigning a type to the variable 'call_assignment_57315' (line 324)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 324, 16), 'call_assignment_57315', getitem___call_result_57843)
        
        # Assigning a Name to a Name (line 324):
        # Getting the type of 'call_assignment_57315' (line 324)
        call_assignment_57315_57844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 16), 'call_assignment_57315')
        # Assigning a type to the variable 'base' (line 324)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 324, 16), 'base', call_assignment_57315_57844)
        
        # Assigning a Call to a Name (line 324):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_57847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 16), 'int')
        # Processing the call keyword arguments
        kwargs_57848 = {}
        # Getting the type of 'call_assignment_57314' (line 324)
        call_assignment_57314_57845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 16), 'call_assignment_57314', False)
        # Obtaining the member '__getitem__' of a type (line 324)
        getitem___57846 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 324, 16), call_assignment_57314_57845, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_57849 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___57846, *[int_57847], **kwargs_57848)
        
        # Assigning a type to the variable 'call_assignment_57316' (line 324)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 324, 16), 'call_assignment_57316', getitem___call_result_57849)
        
        # Assigning a Name to a Name (line 324):
        # Getting the type of 'call_assignment_57316' (line 324)
        call_assignment_57316_57850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 16), 'call_assignment_57316')
        # Assigning a type to the variable 'ext' (line 324)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 324, 22), 'ext', call_assignment_57316_57850)
        
        
        # Evaluating a boolean operation
        
        
        # Call to len(...): (line 325)
        # Processing the call arguments (line 325)
        # Getting the type of 'ext' (line 325)
        ext_57852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 23), 'ext', False)
        # Processing the call keyword arguments (line 325)
        kwargs_57853 = {}
        # Getting the type of 'len' (line 325)
        len_57851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 19), 'len', False)
        # Calling len(args, kwargs) (line 325)
        len_call_result_57854 = invoke(stypy.reporting.localization.Localization(__file__, 325, 19), len_57851, *[ext_57852], **kwargs_57853)
        
        int_57855 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 28), 'int')
        # Applying the binary operator '>' (line 325)
        result_gt_57856 = python_operator(stypy.reporting.localization.Localization(__file__, 325, 19), '>', len_call_result_57854, int_57855)
        
        
        
        # Call to lower(...): (line 325)
        # Processing the call keyword arguments (line 325)
        kwargs_57863 = {}
        
        # Obtaining the type of the subscript
        int_57857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 38), 'int')
        slice_57858 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 325, 34), int_57857, None, None)
        # Getting the type of 'ext' (line 325)
        ext_57859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 34), 'ext', False)
        # Obtaining the member '__getitem__' of a type (line 325)
        getitem___57860 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 325, 34), ext_57859, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 325)
        subscript_call_result_57861 = invoke(stypy.reporting.localization.Localization(__file__, 325, 34), getitem___57860, slice_57858)
        
        # Obtaining the member 'lower' of a type (line 325)
        lower_57862 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 325, 34), subscript_call_result_57861, 'lower')
        # Calling lower(args, kwargs) (line 325)
        lower_call_result_57864 = invoke(stypy.reporting.localization.Localization(__file__, 325, 34), lower_57862, *[], **kwargs_57863)
        
        # Getting the type of 'fontexts' (line 325)
        fontexts_57865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 53), 'fontexts')
        # Applying the binary operator 'in' (line 325)
        result_contains_57866 = python_operator(stypy.reporting.localization.Localization(__file__, 325, 34), 'in', lower_call_result_57864, fontexts_57865)
        
        # Applying the binary operator 'and' (line 325)
        result_and_keyword_57867 = python_operator(stypy.reporting.localization.Localization(__file__, 325, 19), 'and', result_gt_57856, result_contains_57866)
        
        # Testing the type of an if condition (line 325)
        if_condition_57868 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 325, 16), result_and_keyword_57867)
        # Assigning a type to the variable 'if_condition_57868' (line 325)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 16), 'if_condition_57868', if_condition_57868)
        # SSA begins for if statement (line 325)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to add(...): (line 326)
        # Processing the call arguments (line 326)
        # Getting the type of 'f' (line 326)
        f_57871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 34), 'f', False)
        # Processing the call keyword arguments (line 326)
        kwargs_57872 = {}
        # Getting the type of 'fontfiles' (line 326)
        fontfiles_57869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 20), 'fontfiles', False)
        # Obtaining the member 'add' of a type (line 326)
        add_57870 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 20), fontfiles_57869, 'add')
        # Calling add(args, kwargs) (line 326)
        add_call_result_57873 = invoke(stypy.reporting.localization.Localization(__file__, 326, 20), add_57870, *[f_57871], **kwargs_57872)
        
        # SSA join for if statement (line 325)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 318)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Name to a Name (line 328):
        
        # Assigning a Name to a Name (line 328):
        # Getting the type of 'X11FontDirectories' (line 328)
        X11FontDirectories_57874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 24), 'X11FontDirectories')
        # Assigning a type to the variable 'fontpaths' (line 328)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 328, 12), 'fontpaths', X11FontDirectories_57874)
        
        
        # Getting the type of 'sys' (line 330)
        sys_57875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 15), 'sys')
        # Obtaining the member 'platform' of a type (line 330)
        platform_57876 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 15), sys_57875, 'platform')
        unicode_57877 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 31), 'unicode', u'darwin')
        # Applying the binary operator '==' (line 330)
        result_eq_57878 = python_operator(stypy.reporting.localization.Localization(__file__, 330, 15), '==', platform_57876, unicode_57877)
        
        # Testing the type of an if condition (line 330)
        if_condition_57879 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 330, 12), result_eq_57878)
        # Assigning a type to the variable 'if_condition_57879' (line 330)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 12), 'if_condition_57879', if_condition_57879)
        # SSA begins for if statement (line 330)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Call to OSXInstalledFonts(...): (line 331)
        # Processing the call keyword arguments (line 331)
        # Getting the type of 'fontext' (line 331)
        fontext_57881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 51), 'fontext', False)
        keyword_57882 = fontext_57881
        kwargs_57883 = {'fontext': keyword_57882}
        # Getting the type of 'OSXInstalledFonts' (line 331)
        OSXInstalledFonts_57880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 25), 'OSXInstalledFonts', False)
        # Calling OSXInstalledFonts(args, kwargs) (line 331)
        OSXInstalledFonts_call_result_57884 = invoke(stypy.reporting.localization.Localization(__file__, 331, 25), OSXInstalledFonts_57880, *[], **kwargs_57883)
        
        # Testing the type of a for loop iterable (line 331)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 331, 16), OSXInstalledFonts_call_result_57884)
        # Getting the type of the for loop variable (line 331)
        for_loop_var_57885 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 331, 16), OSXInstalledFonts_call_result_57884)
        # Assigning a type to the variable 'f' (line 331)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 16), 'f', for_loop_var_57885)
        # SSA begins for a for statement (line 331)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to add(...): (line 332)
        # Processing the call arguments (line 332)
        # Getting the type of 'f' (line 332)
        f_57888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 34), 'f', False)
        # Processing the call keyword arguments (line 332)
        kwargs_57889 = {}
        # Getting the type of 'fontfiles' (line 332)
        fontfiles_57886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 20), 'fontfiles', False)
        # Obtaining the member 'add' of a type (line 332)
        add_57887 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 20), fontfiles_57886, 'add')
        # Calling add(args, kwargs) (line 332)
        add_call_result_57890 = invoke(stypy.reporting.localization.Localization(__file__, 332, 20), add_57887, *[f_57888], **kwargs_57889)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 330)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to get_fontconfig_fonts(...): (line 334)
        # Processing the call arguments (line 334)
        # Getting the type of 'fontext' (line 334)
        fontext_57892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 42), 'fontext', False)
        # Processing the call keyword arguments (line 334)
        kwargs_57893 = {}
        # Getting the type of 'get_fontconfig_fonts' (line 334)
        get_fontconfig_fonts_57891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 21), 'get_fontconfig_fonts', False)
        # Calling get_fontconfig_fonts(args, kwargs) (line 334)
        get_fontconfig_fonts_call_result_57894 = invoke(stypy.reporting.localization.Localization(__file__, 334, 21), get_fontconfig_fonts_57891, *[fontext_57892], **kwargs_57893)
        
        # Testing the type of a for loop iterable (line 334)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 334, 12), get_fontconfig_fonts_call_result_57894)
        # Getting the type of the for loop variable (line 334)
        for_loop_var_57895 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 334, 12), get_fontconfig_fonts_call_result_57894)
        # Assigning a type to the variable 'f' (line 334)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 12), 'f', for_loop_var_57895)
        # SSA begins for a for statement (line 334)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to add(...): (line 335)
        # Processing the call arguments (line 335)
        # Getting the type of 'f' (line 335)
        f_57898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 30), 'f', False)
        # Processing the call keyword arguments (line 335)
        kwargs_57899 = {}
        # Getting the type of 'fontfiles' (line 335)
        fontfiles_57896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 16), 'fontfiles', False)
        # Obtaining the member 'add' of a type (line 335)
        add_57897 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 16), fontfiles_57896, 'add')
        # Calling add(args, kwargs) (line 335)
        add_call_result_57900 = invoke(stypy.reporting.localization.Localization(__file__, 335, 16), add_57897, *[f_57898], **kwargs_57899)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 318)
        module_type_store = module_type_store.join_ssa_context()
        

        if more_types_in_union_57817:
            # Runtime conditional SSA for else branch (line 317)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_57816) or more_types_in_union_57817):
        
        
        # Call to isinstance(...): (line 337)
        # Processing the call arguments (line 337)
        # Getting the type of 'fontpaths' (line 337)
        fontpaths_57902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 20), 'fontpaths', False)
        # Getting the type of 'six' (line 337)
        six_57903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 31), 'six', False)
        # Obtaining the member 'string_types' of a type (line 337)
        string_types_57904 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 31), six_57903, 'string_types')
        # Processing the call keyword arguments (line 337)
        kwargs_57905 = {}
        # Getting the type of 'isinstance' (line 337)
        isinstance_57901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 9), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 337)
        isinstance_call_result_57906 = invoke(stypy.reporting.localization.Localization(__file__, 337, 9), isinstance_57901, *[fontpaths_57902, string_types_57904], **kwargs_57905)
        
        # Testing the type of an if condition (line 337)
        if_condition_57907 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 337, 9), isinstance_call_result_57906)
        # Assigning a type to the variable 'if_condition_57907' (line 337)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 337, 9), 'if_condition_57907', if_condition_57907)
        # SSA begins for if statement (line 337)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a List to a Name (line 338):
        
        # Assigning a List to a Name (line 338):
        
        # Obtaining an instance of the builtin type 'list' (line 338)
        list_57908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 338)
        # Adding element type (line 338)
        # Getting the type of 'fontpaths' (line 338)
        fontpaths_57909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 21), 'fontpaths')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 338, 20), list_57908, fontpaths_57909)
        
        # Assigning a type to the variable 'fontpaths' (line 338)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 338, 8), 'fontpaths', list_57908)
        # SSA join for if statement (line 337)
        module_type_store = module_type_store.join_ssa_context()
        

        if (may_be_57816 and more_types_in_union_57817):
            # SSA join for if statement (line 317)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Getting the type of 'fontpaths' (line 340)
    fontpaths_57910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 16), 'fontpaths')
    # Testing the type of a for loop iterable (line 340)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 340, 4), fontpaths_57910)
    # Getting the type of the for loop variable (line 340)
    for_loop_var_57911 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 340, 4), fontpaths_57910)
    # Assigning a type to the variable 'path' (line 340)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 4), 'path', for_loop_var_57911)
    # SSA begins for a for statement (line 340)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 341):
    
    # Assigning a Call to a Name (line 341):
    
    # Call to list_fonts(...): (line 341)
    # Processing the call arguments (line 341)
    # Getting the type of 'path' (line 341)
    path_57913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 27), 'path', False)
    # Getting the type of 'fontexts' (line 341)
    fontexts_57914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 33), 'fontexts', False)
    # Processing the call keyword arguments (line 341)
    kwargs_57915 = {}
    # Getting the type of 'list_fonts' (line 341)
    list_fonts_57912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 16), 'list_fonts', False)
    # Calling list_fonts(args, kwargs) (line 341)
    list_fonts_call_result_57916 = invoke(stypy.reporting.localization.Localization(__file__, 341, 16), list_fonts_57912, *[path_57913, fontexts_57914], **kwargs_57915)
    
    # Assigning a type to the variable 'files' (line 341)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 8), 'files', list_fonts_call_result_57916)
    
    # Getting the type of 'files' (line 342)
    files_57917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 21), 'files')
    # Testing the type of a for loop iterable (line 342)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 342, 8), files_57917)
    # Getting the type of the for loop variable (line 342)
    for_loop_var_57918 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 342, 8), files_57917)
    # Assigning a type to the variable 'fname' (line 342)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 342, 8), 'fname', for_loop_var_57918)
    # SSA begins for a for statement (line 342)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to add(...): (line 343)
    # Processing the call arguments (line 343)
    
    # Call to abspath(...): (line 343)
    # Processing the call arguments (line 343)
    # Getting the type of 'fname' (line 343)
    fname_57924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 42), 'fname', False)
    # Processing the call keyword arguments (line 343)
    kwargs_57925 = {}
    # Getting the type of 'os' (line 343)
    os_57921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 26), 'os', False)
    # Obtaining the member 'path' of a type (line 343)
    path_57922 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 26), os_57921, 'path')
    # Obtaining the member 'abspath' of a type (line 343)
    abspath_57923 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 26), path_57922, 'abspath')
    # Calling abspath(args, kwargs) (line 343)
    abspath_call_result_57926 = invoke(stypy.reporting.localization.Localization(__file__, 343, 26), abspath_57923, *[fname_57924], **kwargs_57925)
    
    # Processing the call keyword arguments (line 343)
    kwargs_57927 = {}
    # Getting the type of 'fontfiles' (line 343)
    fontfiles_57919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 12), 'fontfiles', False)
    # Obtaining the member 'add' of a type (line 343)
    add_57920 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 12), fontfiles_57919, 'add')
    # Calling add(args, kwargs) (line 343)
    add_call_result_57928 = invoke(stypy.reporting.localization.Localization(__file__, 343, 12), add_57920, *[abspath_call_result_57926], **kwargs_57927)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'fontfiles' (line 345)
    fontfiles_57936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 31), 'fontfiles')
    comprehension_57937 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 345, 12), fontfiles_57936)
    # Assigning a type to the variable 'fname' (line 345)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 345, 12), 'fname', comprehension_57937)
    
    # Call to exists(...): (line 345)
    # Processing the call arguments (line 345)
    # Getting the type of 'fname' (line 345)
    fname_57933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 59), 'fname', False)
    # Processing the call keyword arguments (line 345)
    kwargs_57934 = {}
    # Getting the type of 'os' (line 345)
    os_57930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 44), 'os', False)
    # Obtaining the member 'path' of a type (line 345)
    path_57931 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 345, 44), os_57930, 'path')
    # Obtaining the member 'exists' of a type (line 345)
    exists_57932 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 345, 44), path_57931, 'exists')
    # Calling exists(args, kwargs) (line 345)
    exists_call_result_57935 = invoke(stypy.reporting.localization.Localization(__file__, 345, 44), exists_57932, *[fname_57933], **kwargs_57934)
    
    # Getting the type of 'fname' (line 345)
    fname_57929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 12), 'fname')
    list_57938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 12), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 345, 12), list_57938, fname_57929)
    # Assigning a type to the variable 'stypy_return_type' (line 345)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 345, 4), 'stypy_return_type', list_57938)
    
    # ################# End of 'findSystemFonts(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'findSystemFonts' in the type store
    # Getting the type of 'stypy_return_type' (line 306)
    stypy_return_type_57939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_57939)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'findSystemFonts'
    return stypy_return_type_57939

# Assigning a type to the variable 'findSystemFonts' (line 306)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 0), 'findSystemFonts', findSystemFonts)

@norecursion
def weight_as_number(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'weight_as_number'
    module_type_store = module_type_store.open_function_context('weight_as_number', 348, 0, False)
    
    # Passed parameters checking function
    weight_as_number.stypy_localization = localization
    weight_as_number.stypy_type_of_self = None
    weight_as_number.stypy_type_store = module_type_store
    weight_as_number.stypy_function_name = 'weight_as_number'
    weight_as_number.stypy_param_names_list = ['weight']
    weight_as_number.stypy_varargs_param_name = None
    weight_as_number.stypy_kwargs_param_name = None
    weight_as_number.stypy_call_defaults = defaults
    weight_as_number.stypy_call_varargs = varargs
    weight_as_number.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'weight_as_number', ['weight'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'weight_as_number', localization, ['weight'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'weight_as_number(...)' code ##################

    unicode_57940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 353, (-1)), 'unicode', u'\n    Return the weight property as a numeric value.  String values\n    are converted to their corresponding numeric value.\n    ')
    
    
    # Call to isinstance(...): (line 354)
    # Processing the call arguments (line 354)
    # Getting the type of 'weight' (line 354)
    weight_57942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 18), 'weight', False)
    # Getting the type of 'six' (line 354)
    six_57943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 26), 'six', False)
    # Obtaining the member 'string_types' of a type (line 354)
    string_types_57944 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 354, 26), six_57943, 'string_types')
    # Processing the call keyword arguments (line 354)
    kwargs_57945 = {}
    # Getting the type of 'isinstance' (line 354)
    isinstance_57941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 7), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 354)
    isinstance_call_result_57946 = invoke(stypy.reporting.localization.Localization(__file__, 354, 7), isinstance_57941, *[weight_57942, string_types_57944], **kwargs_57945)
    
    # Testing the type of an if condition (line 354)
    if_condition_57947 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 354, 4), isinstance_call_result_57946)
    # Assigning a type to the variable 'if_condition_57947' (line 354)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 354, 4), 'if_condition_57947', if_condition_57947)
    # SSA begins for if statement (line 354)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # SSA begins for try-except statement (line 355)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Subscript to a Name (line 356):
    
    # Assigning a Subscript to a Name (line 356):
    
    # Obtaining the type of the subscript
    
    # Call to lower(...): (line 356)
    # Processing the call keyword arguments (line 356)
    kwargs_57950 = {}
    # Getting the type of 'weight' (line 356)
    weight_57948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 33), 'weight', False)
    # Obtaining the member 'lower' of a type (line 356)
    lower_57949 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 356, 33), weight_57948, 'lower')
    # Calling lower(args, kwargs) (line 356)
    lower_call_result_57951 = invoke(stypy.reporting.localization.Localization(__file__, 356, 33), lower_57949, *[], **kwargs_57950)
    
    # Getting the type of 'weight_dict' (line 356)
    weight_dict_57952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 21), 'weight_dict')
    # Obtaining the member '__getitem__' of a type (line 356)
    getitem___57953 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 356, 21), weight_dict_57952, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 356)
    subscript_call_result_57954 = invoke(stypy.reporting.localization.Localization(__file__, 356, 21), getitem___57953, lower_call_result_57951)
    
    # Assigning a type to the variable 'weight' (line 356)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 356, 12), 'weight', subscript_call_result_57954)
    # SSA branch for the except part of a try statement (line 355)
    # SSA branch for the except 'KeyError' branch of a try statement (line 355)
    module_type_store.open_ssa_branch('except')
    
    # Assigning a Num to a Name (line 358):
    
    # Assigning a Num to a Name (line 358):
    int_57955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 358, 21), 'int')
    # Assigning a type to the variable 'weight' (line 358)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 358, 12), 'weight', int_57955)
    # SSA join for try-except statement (line 355)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 354)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'weight' (line 359)
    weight_57956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 9), 'weight')
    
    # Call to range(...): (line 359)
    # Processing the call arguments (line 359)
    int_57958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, 25), 'int')
    int_57959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, 30), 'int')
    int_57960 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, 36), 'int')
    # Processing the call keyword arguments (line 359)
    kwargs_57961 = {}
    # Getting the type of 'range' (line 359)
    range_57957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 19), 'range', False)
    # Calling range(args, kwargs) (line 359)
    range_call_result_57962 = invoke(stypy.reporting.localization.Localization(__file__, 359, 19), range_57957, *[int_57958, int_57959, int_57960], **kwargs_57961)
    
    # Applying the binary operator 'in' (line 359)
    result_contains_57963 = python_operator(stypy.reporting.localization.Localization(__file__, 359, 9), 'in', weight_57956, range_call_result_57962)
    
    # Testing the type of an if condition (line 359)
    if_condition_57964 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 359, 9), result_contains_57963)
    # Assigning a type to the variable 'if_condition_57964' (line 359)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 359, 9), 'if_condition_57964', if_condition_57964)
    # SSA begins for if statement (line 359)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    pass
    # SSA branch for the else part of an if statement (line 359)
    module_type_store.open_ssa_branch('else')
    
    # Call to ValueError(...): (line 362)
    # Processing the call arguments (line 362)
    unicode_57966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 25), 'unicode', u'weight not a valid integer')
    # Processing the call keyword arguments (line 362)
    kwargs_57967 = {}
    # Getting the type of 'ValueError' (line 362)
    ValueError_57965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 362)
    ValueError_call_result_57968 = invoke(stypy.reporting.localization.Localization(__file__, 362, 14), ValueError_57965, *[unicode_57966], **kwargs_57967)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 362, 8), ValueError_call_result_57968, 'raise parameter', BaseException)
    # SSA join for if statement (line 359)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 354)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'weight' (line 363)
    weight_57969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 11), 'weight')
    # Assigning a type to the variable 'stypy_return_type' (line 363)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 4), 'stypy_return_type', weight_57969)
    
    # ################# End of 'weight_as_number(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'weight_as_number' in the type store
    # Getting the type of 'stypy_return_type' (line 348)
    stypy_return_type_57970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_57970)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'weight_as_number'
    return stypy_return_type_57970

# Assigning a type to the variable 'weight_as_number' (line 348)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 348, 0), 'weight_as_number', weight_as_number)
# Declaration of the 'FontEntry' class

class FontEntry(object, ):
    unicode_57971 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, (-1)), 'unicode', u'\n    A class for storing Font properties.  It is used when populating\n    the font lookup dictionary.\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        unicode_57972 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 372, 25), 'unicode', u'')
        unicode_57973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, 25), 'unicode', u'')
        unicode_57974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, 25), 'unicode', u'normal')
        unicode_57975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 25), 'unicode', u'normal')
        unicode_57976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 376, 25), 'unicode', u'normal')
        unicode_57977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 377, 25), 'unicode', u'normal')
        unicode_57978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 378, 25), 'unicode', u'medium')
        defaults = [unicode_57972, unicode_57973, unicode_57974, unicode_57975, unicode_57976, unicode_57977, unicode_57978]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 371, 4, False)
        # Assigning a type to the variable 'self' (line 372)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 372, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FontEntry.__init__', ['fname', 'name', 'style', 'variant', 'weight', 'stretch', 'size'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['fname', 'name', 'style', 'variant', 'weight', 'stretch', 'size'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 380):
        
        # Assigning a Name to a Attribute (line 380):
        # Getting the type of 'fname' (line 380)
        fname_57979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 23), 'fname')
        # Getting the type of 'self' (line 380)
        self_57980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 8), 'self')
        # Setting the type of the member 'fname' of a type (line 380)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 380, 8), self_57980, 'fname', fname_57979)
        
        # Assigning a Name to a Attribute (line 381):
        
        # Assigning a Name to a Attribute (line 381):
        # Getting the type of 'name' (line 381)
        name_57981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 23), 'name')
        # Getting the type of 'self' (line 381)
        self_57982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 8), 'self')
        # Setting the type of the member 'name' of a type (line 381)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 381, 8), self_57982, 'name', name_57981)
        
        # Assigning a Name to a Attribute (line 382):
        
        # Assigning a Name to a Attribute (line 382):
        # Getting the type of 'style' (line 382)
        style_57983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 23), 'style')
        # Getting the type of 'self' (line 382)
        self_57984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 8), 'self')
        # Setting the type of the member 'style' of a type (line 382)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 382, 8), self_57984, 'style', style_57983)
        
        # Assigning a Name to a Attribute (line 383):
        
        # Assigning a Name to a Attribute (line 383):
        # Getting the type of 'variant' (line 383)
        variant_57985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 23), 'variant')
        # Getting the type of 'self' (line 383)
        self_57986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 8), 'self')
        # Setting the type of the member 'variant' of a type (line 383)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 383, 8), self_57986, 'variant', variant_57985)
        
        # Assigning a Name to a Attribute (line 384):
        
        # Assigning a Name to a Attribute (line 384):
        # Getting the type of 'weight' (line 384)
        weight_57987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 23), 'weight')
        # Getting the type of 'self' (line 384)
        self_57988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 8), 'self')
        # Setting the type of the member 'weight' of a type (line 384)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 384, 8), self_57988, 'weight', weight_57987)
        
        # Assigning a Name to a Attribute (line 385):
        
        # Assigning a Name to a Attribute (line 385):
        # Getting the type of 'stretch' (line 385)
        stretch_57989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 23), 'stretch')
        # Getting the type of 'self' (line 385)
        self_57990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 8), 'self')
        # Setting the type of the member 'stretch' of a type (line 385)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 385, 8), self_57990, 'stretch', stretch_57989)
        
        
        # SSA begins for try-except statement (line 386)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Attribute (line 387):
        
        # Assigning a Call to a Attribute (line 387):
        
        # Call to str(...): (line 387)
        # Processing the call arguments (line 387)
        
        # Call to float(...): (line 387)
        # Processing the call arguments (line 387)
        # Getting the type of 'size' (line 387)
        size_57993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 34), 'size', False)
        # Processing the call keyword arguments (line 387)
        kwargs_57994 = {}
        # Getting the type of 'float' (line 387)
        float_57992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 28), 'float', False)
        # Calling float(args, kwargs) (line 387)
        float_call_result_57995 = invoke(stypy.reporting.localization.Localization(__file__, 387, 28), float_57992, *[size_57993], **kwargs_57994)
        
        # Processing the call keyword arguments (line 387)
        kwargs_57996 = {}
        # Getting the type of 'str' (line 387)
        str_57991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 24), 'str', False)
        # Calling str(args, kwargs) (line 387)
        str_call_result_57997 = invoke(stypy.reporting.localization.Localization(__file__, 387, 24), str_57991, *[float_call_result_57995], **kwargs_57996)
        
        # Getting the type of 'self' (line 387)
        self_57998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 12), 'self')
        # Setting the type of the member 'size' of a type (line 387)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 387, 12), self_57998, 'size', str_call_result_57997)
        # SSA branch for the except part of a try statement (line 386)
        # SSA branch for the except 'ValueError' branch of a try statement (line 386)
        module_type_store.open_ssa_branch('except')
        
        # Assigning a Name to a Attribute (line 389):
        
        # Assigning a Name to a Attribute (line 389):
        # Getting the type of 'size' (line 389)
        size_57999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 24), 'size')
        # Getting the type of 'self' (line 389)
        self_58000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 12), 'self')
        # Setting the type of the member 'size' of a type (line 389)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 12), self_58000, 'size', size_57999)
        # SSA join for try-except statement (line 386)
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
        module_type_store = module_type_store.open_function_context('__repr__', 391, 4, False)
        # Assigning a type to the variable 'self' (line 392)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 392, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FontEntry.stypy__repr__.__dict__.__setitem__('stypy_localization', localization)
        FontEntry.stypy__repr__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FontEntry.stypy__repr__.__dict__.__setitem__('stypy_type_store', module_type_store)
        FontEntry.stypy__repr__.__dict__.__setitem__('stypy_function_name', 'FontEntry.stypy__repr__')
        FontEntry.stypy__repr__.__dict__.__setitem__('stypy_param_names_list', [])
        FontEntry.stypy__repr__.__dict__.__setitem__('stypy_varargs_param_name', None)
        FontEntry.stypy__repr__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FontEntry.stypy__repr__.__dict__.__setitem__('stypy_call_defaults', defaults)
        FontEntry.stypy__repr__.__dict__.__setitem__('stypy_call_varargs', varargs)
        FontEntry.stypy__repr__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FontEntry.stypy__repr__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FontEntry.stypy__repr__', [], None, None, defaults, varargs, kwargs)

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

        unicode_58001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 392, 15), 'unicode', u"<Font '%s' (%s) %s %s %s %s>")
        
        # Obtaining an instance of the builtin type 'tuple' (line 393)
        tuple_58002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 12), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 393)
        # Adding element type (line 393)
        # Getting the type of 'self' (line 393)
        self_58003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 12), 'self')
        # Obtaining the member 'name' of a type (line 393)
        name_58004 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 393, 12), self_58003, 'name')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 393, 12), tuple_58002, name_58004)
        # Adding element type (line 393)
        
        # Call to basename(...): (line 393)
        # Processing the call arguments (line 393)
        # Getting the type of 'self' (line 393)
        self_58008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 40), 'self', False)
        # Obtaining the member 'fname' of a type (line 393)
        fname_58009 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 393, 40), self_58008, 'fname')
        # Processing the call keyword arguments (line 393)
        kwargs_58010 = {}
        # Getting the type of 'os' (line 393)
        os_58005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 23), 'os', False)
        # Obtaining the member 'path' of a type (line 393)
        path_58006 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 393, 23), os_58005, 'path')
        # Obtaining the member 'basename' of a type (line 393)
        basename_58007 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 393, 23), path_58006, 'basename')
        # Calling basename(args, kwargs) (line 393)
        basename_call_result_58011 = invoke(stypy.reporting.localization.Localization(__file__, 393, 23), basename_58007, *[fname_58009], **kwargs_58010)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 393, 12), tuple_58002, basename_call_result_58011)
        # Adding element type (line 393)
        # Getting the type of 'self' (line 393)
        self_58012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 53), 'self')
        # Obtaining the member 'style' of a type (line 393)
        style_58013 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 393, 53), self_58012, 'style')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 393, 12), tuple_58002, style_58013)
        # Adding element type (line 393)
        # Getting the type of 'self' (line 393)
        self_58014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 65), 'self')
        # Obtaining the member 'variant' of a type (line 393)
        variant_58015 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 393, 65), self_58014, 'variant')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 393, 12), tuple_58002, variant_58015)
        # Adding element type (line 393)
        # Getting the type of 'self' (line 394)
        self_58016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 12), 'self')
        # Obtaining the member 'weight' of a type (line 394)
        weight_58017 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 12), self_58016, 'weight')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 393, 12), tuple_58002, weight_58017)
        # Adding element type (line 393)
        # Getting the type of 'self' (line 394)
        self_58018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 25), 'self')
        # Obtaining the member 'stretch' of a type (line 394)
        stretch_58019 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 25), self_58018, 'stretch')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 393, 12), tuple_58002, stretch_58019)
        
        # Applying the binary operator '%' (line 392)
        result_mod_58020 = python_operator(stypy.reporting.localization.Localization(__file__, 392, 15), '%', unicode_58001, tuple_58002)
        
        # Assigning a type to the variable 'stypy_return_type' (line 392)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 392, 8), 'stypy_return_type', result_mod_58020)
        
        # ################# End of '__repr__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__repr__' in the type store
        # Getting the type of 'stypy_return_type' (line 391)
        stypy_return_type_58021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_58021)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__repr__'
        return stypy_return_type_58021


# Assigning a type to the variable 'FontEntry' (line 366)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 366, 0), 'FontEntry', FontEntry)

@norecursion
def ttfFontProperty(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'ttfFontProperty'
    module_type_store = module_type_store.open_function_context('ttfFontProperty', 397, 0, False)
    
    # Passed parameters checking function
    ttfFontProperty.stypy_localization = localization
    ttfFontProperty.stypy_type_of_self = None
    ttfFontProperty.stypy_type_store = module_type_store
    ttfFontProperty.stypy_function_name = 'ttfFontProperty'
    ttfFontProperty.stypy_param_names_list = ['font']
    ttfFontProperty.stypy_varargs_param_name = None
    ttfFontProperty.stypy_kwargs_param_name = None
    ttfFontProperty.stypy_call_defaults = defaults
    ttfFontProperty.stypy_call_varargs = varargs
    ttfFontProperty.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'ttfFontProperty', ['font'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'ttfFontProperty', localization, ['font'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'ttfFontProperty(...)' code ##################

    unicode_58022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 403, (-1)), 'unicode', u'\n    A function for populating the :class:`FontKey` by extracting\n    information from the TrueType font file.\n\n    *font* is a :class:`FT2Font` instance.\n    ')
    
    # Assigning a Attribute to a Name (line 404):
    
    # Assigning a Attribute to a Name (line 404):
    # Getting the type of 'font' (line 404)
    font_58023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 11), 'font')
    # Obtaining the member 'family_name' of a type (line 404)
    family_name_58024 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 404, 11), font_58023, 'family_name')
    # Assigning a type to the variable 'name' (line 404)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 404, 4), 'name', family_name_58024)
    
    # Assigning a Call to a Name (line 408):
    
    # Assigning a Call to a Name (line 408):
    
    # Call to get_sfnt(...): (line 408)
    # Processing the call keyword arguments (line 408)
    kwargs_58027 = {}
    # Getting the type of 'font' (line 408)
    font_58025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 11), 'font', False)
    # Obtaining the member 'get_sfnt' of a type (line 408)
    get_sfnt_58026 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 408, 11), font_58025, 'get_sfnt')
    # Calling get_sfnt(args, kwargs) (line 408)
    get_sfnt_call_result_58028 = invoke(stypy.reporting.localization.Localization(__file__, 408, 11), get_sfnt_58026, *[], **kwargs_58027)
    
    # Assigning a type to the variable 'sfnt' (line 408)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 408, 4), 'sfnt', get_sfnt_call_result_58028)
    
    # Assigning a Call to a Name (line 409):
    
    # Assigning a Call to a Name (line 409):
    
    # Call to get(...): (line 409)
    # Processing the call arguments (line 409)
    
    # Obtaining an instance of the builtin type 'tuple' (line 409)
    tuple_58031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 22), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 409)
    # Adding element type (line 409)
    int_58032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 409, 22), tuple_58031, int_58032)
    # Adding element type (line 409)
    int_58033 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 409, 22), tuple_58031, int_58033)
    # Adding element type (line 409)
    int_58034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 26), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 409, 22), tuple_58031, int_58034)
    # Adding element type (line 409)
    int_58035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 409, 22), tuple_58031, int_58035)
    
    # Processing the call keyword arguments (line 409)
    kwargs_58036 = {}
    # Getting the type of 'sfnt' (line 409)
    sfnt_58029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 12), 'sfnt', False)
    # Obtaining the member 'get' of a type (line 409)
    get_58030 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 409, 12), sfnt_58029, 'get')
    # Calling get(args, kwargs) (line 409)
    get_call_result_58037 = invoke(stypy.reporting.localization.Localization(__file__, 409, 12), get_58030, *[tuple_58031], **kwargs_58036)
    
    # Assigning a type to the variable 'sfnt2' (line 409)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 409, 4), 'sfnt2', get_call_result_58037)
    
    # Assigning a Call to a Name (line 410):
    
    # Assigning a Call to a Name (line 410):
    
    # Call to get(...): (line 410)
    # Processing the call arguments (line 410)
    
    # Obtaining an instance of the builtin type 'tuple' (line 410)
    tuple_58040 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 410, 22), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 410)
    # Adding element type (line 410)
    int_58041 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 410, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 410, 22), tuple_58040, int_58041)
    # Adding element type (line 410)
    int_58042 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 410, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 410, 22), tuple_58040, int_58042)
    # Adding element type (line 410)
    int_58043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 410, 26), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 410, 22), tuple_58040, int_58043)
    # Adding element type (line 410)
    int_58044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 410, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 410, 22), tuple_58040, int_58044)
    
    # Processing the call keyword arguments (line 410)
    kwargs_58045 = {}
    # Getting the type of 'sfnt' (line 410)
    sfnt_58038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 12), 'sfnt', False)
    # Obtaining the member 'get' of a type (line 410)
    get_58039 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 410, 12), sfnt_58038, 'get')
    # Calling get(args, kwargs) (line 410)
    get_call_result_58046 = invoke(stypy.reporting.localization.Localization(__file__, 410, 12), get_58039, *[tuple_58040], **kwargs_58045)
    
    # Assigning a type to the variable 'sfnt4' (line 410)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 410, 4), 'sfnt4', get_call_result_58046)
    
    # Getting the type of 'sfnt2' (line 411)
    sfnt2_58047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 7), 'sfnt2')
    # Testing the type of an if condition (line 411)
    if_condition_58048 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 411, 4), sfnt2_58047)
    # Assigning a type to the variable 'if_condition_58048' (line 411)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 411, 4), 'if_condition_58048', if_condition_58048)
    # SSA begins for if statement (line 411)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 412):
    
    # Assigning a Call to a Name (line 412):
    
    # Call to lower(...): (line 412)
    # Processing the call keyword arguments (line 412)
    kwargs_58055 = {}
    
    # Call to decode(...): (line 412)
    # Processing the call arguments (line 412)
    unicode_58051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 412, 29), 'unicode', u'macroman')
    # Processing the call keyword arguments (line 412)
    kwargs_58052 = {}
    # Getting the type of 'sfnt2' (line 412)
    sfnt2_58049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 16), 'sfnt2', False)
    # Obtaining the member 'decode' of a type (line 412)
    decode_58050 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 412, 16), sfnt2_58049, 'decode')
    # Calling decode(args, kwargs) (line 412)
    decode_call_result_58053 = invoke(stypy.reporting.localization.Localization(__file__, 412, 16), decode_58050, *[unicode_58051], **kwargs_58052)
    
    # Obtaining the member 'lower' of a type (line 412)
    lower_58054 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 412, 16), decode_call_result_58053, 'lower')
    # Calling lower(args, kwargs) (line 412)
    lower_call_result_58056 = invoke(stypy.reporting.localization.Localization(__file__, 412, 16), lower_58054, *[], **kwargs_58055)
    
    # Assigning a type to the variable 'sfnt2' (line 412)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 412, 8), 'sfnt2', lower_call_result_58056)
    # SSA branch for the else part of an if statement (line 411)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Str to a Name (line 414):
    
    # Assigning a Str to a Name (line 414):
    unicode_58057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 414, 16), 'unicode', u'')
    # Assigning a type to the variable 'sfnt2' (line 414)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 414, 8), 'sfnt2', unicode_58057)
    # SSA join for if statement (line 411)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'sfnt4' (line 415)
    sfnt4_58058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 7), 'sfnt4')
    # Testing the type of an if condition (line 415)
    if_condition_58059 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 415, 4), sfnt4_58058)
    # Assigning a type to the variable 'if_condition_58059' (line 415)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 415, 4), 'if_condition_58059', if_condition_58059)
    # SSA begins for if statement (line 415)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 416):
    
    # Assigning a Call to a Name (line 416):
    
    # Call to lower(...): (line 416)
    # Processing the call keyword arguments (line 416)
    kwargs_58066 = {}
    
    # Call to decode(...): (line 416)
    # Processing the call arguments (line 416)
    unicode_58062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 416, 29), 'unicode', u'macroman')
    # Processing the call keyword arguments (line 416)
    kwargs_58063 = {}
    # Getting the type of 'sfnt4' (line 416)
    sfnt4_58060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 16), 'sfnt4', False)
    # Obtaining the member 'decode' of a type (line 416)
    decode_58061 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 416, 16), sfnt4_58060, 'decode')
    # Calling decode(args, kwargs) (line 416)
    decode_call_result_58064 = invoke(stypy.reporting.localization.Localization(__file__, 416, 16), decode_58061, *[unicode_58062], **kwargs_58063)
    
    # Obtaining the member 'lower' of a type (line 416)
    lower_58065 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 416, 16), decode_call_result_58064, 'lower')
    # Calling lower(args, kwargs) (line 416)
    lower_call_result_58067 = invoke(stypy.reporting.localization.Localization(__file__, 416, 16), lower_58065, *[], **kwargs_58066)
    
    # Assigning a type to the variable 'sfnt4' (line 416)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 416, 8), 'sfnt4', lower_call_result_58067)
    # SSA branch for the else part of an if statement (line 415)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Str to a Name (line 418):
    
    # Assigning a Str to a Name (line 418):
    unicode_58068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 16), 'unicode', u'')
    # Assigning a type to the variable 'sfnt4' (line 418)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 418, 8), 'sfnt4', unicode_58068)
    # SSA join for if statement (line 415)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to find(...): (line 419)
    # Processing the call arguments (line 419)
    unicode_58071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 419, 18), 'unicode', u'oblique')
    # Processing the call keyword arguments (line 419)
    kwargs_58072 = {}
    # Getting the type of 'sfnt4' (line 419)
    sfnt4_58069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 7), 'sfnt4', False)
    # Obtaining the member 'find' of a type (line 419)
    find_58070 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 419, 7), sfnt4_58069, 'find')
    # Calling find(args, kwargs) (line 419)
    find_call_result_58073 = invoke(stypy.reporting.localization.Localization(__file__, 419, 7), find_58070, *[unicode_58071], **kwargs_58072)
    
    int_58074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 419, 32), 'int')
    # Applying the binary operator '>=' (line 419)
    result_ge_58075 = python_operator(stypy.reporting.localization.Localization(__file__, 419, 7), '>=', find_call_result_58073, int_58074)
    
    # Testing the type of an if condition (line 419)
    if_condition_58076 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 419, 4), result_ge_58075)
    # Assigning a type to the variable 'if_condition_58076' (line 419)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 419, 4), 'if_condition_58076', if_condition_58076)
    # SSA begins for if statement (line 419)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 420):
    
    # Assigning a Str to a Name (line 420):
    unicode_58077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 420, 16), 'unicode', u'oblique')
    # Assigning a type to the variable 'style' (line 420)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 420, 8), 'style', unicode_58077)
    # SSA branch for the else part of an if statement (line 419)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Call to find(...): (line 421)
    # Processing the call arguments (line 421)
    unicode_58080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 421, 20), 'unicode', u'italic')
    # Processing the call keyword arguments (line 421)
    kwargs_58081 = {}
    # Getting the type of 'sfnt4' (line 421)
    sfnt4_58078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 9), 'sfnt4', False)
    # Obtaining the member 'find' of a type (line 421)
    find_58079 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 421, 9), sfnt4_58078, 'find')
    # Calling find(args, kwargs) (line 421)
    find_call_result_58082 = invoke(stypy.reporting.localization.Localization(__file__, 421, 9), find_58079, *[unicode_58080], **kwargs_58081)
    
    int_58083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 421, 33), 'int')
    # Applying the binary operator '>=' (line 421)
    result_ge_58084 = python_operator(stypy.reporting.localization.Localization(__file__, 421, 9), '>=', find_call_result_58082, int_58083)
    
    # Testing the type of an if condition (line 421)
    if_condition_58085 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 421, 9), result_ge_58084)
    # Assigning a type to the variable 'if_condition_58085' (line 421)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 421, 9), 'if_condition_58085', if_condition_58085)
    # SSA begins for if statement (line 421)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 422):
    
    # Assigning a Str to a Name (line 422):
    unicode_58086 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 422, 16), 'unicode', u'italic')
    # Assigning a type to the variable 'style' (line 422)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 422, 8), 'style', unicode_58086)
    # SSA branch for the else part of an if statement (line 421)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Call to find(...): (line 423)
    # Processing the call arguments (line 423)
    unicode_58089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, 20), 'unicode', u'regular')
    # Processing the call keyword arguments (line 423)
    kwargs_58090 = {}
    # Getting the type of 'sfnt2' (line 423)
    sfnt2_58087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 9), 'sfnt2', False)
    # Obtaining the member 'find' of a type (line 423)
    find_58088 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 423, 9), sfnt2_58087, 'find')
    # Calling find(args, kwargs) (line 423)
    find_call_result_58091 = invoke(stypy.reporting.localization.Localization(__file__, 423, 9), find_58088, *[unicode_58089], **kwargs_58090)
    
    int_58092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, 34), 'int')
    # Applying the binary operator '>=' (line 423)
    result_ge_58093 = python_operator(stypy.reporting.localization.Localization(__file__, 423, 9), '>=', find_call_result_58091, int_58092)
    
    # Testing the type of an if condition (line 423)
    if_condition_58094 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 423, 9), result_ge_58093)
    # Assigning a type to the variable 'if_condition_58094' (line 423)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 423, 9), 'if_condition_58094', if_condition_58094)
    # SSA begins for if statement (line 423)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 424):
    
    # Assigning a Str to a Name (line 424):
    unicode_58095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 424, 16), 'unicode', u'normal')
    # Assigning a type to the variable 'style' (line 424)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 424, 8), 'style', unicode_58095)
    # SSA branch for the else part of an if statement (line 423)
    module_type_store.open_ssa_branch('else')
    
    # Getting the type of 'font' (line 425)
    font_58096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 9), 'font')
    # Obtaining the member 'style_flags' of a type (line 425)
    style_flags_58097 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 425, 9), font_58096, 'style_flags')
    # Getting the type of 'ft2font' (line 425)
    ft2font_58098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 28), 'ft2font')
    # Obtaining the member 'ITALIC' of a type (line 425)
    ITALIC_58099 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 425, 28), ft2font_58098, 'ITALIC')
    # Applying the binary operator '&' (line 425)
    result_and__58100 = python_operator(stypy.reporting.localization.Localization(__file__, 425, 9), '&', style_flags_58097, ITALIC_58099)
    
    # Testing the type of an if condition (line 425)
    if_condition_58101 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 425, 9), result_and__58100)
    # Assigning a type to the variable 'if_condition_58101' (line 425)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 425, 9), 'if_condition_58101', if_condition_58101)
    # SSA begins for if statement (line 425)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 426):
    
    # Assigning a Str to a Name (line 426):
    unicode_58102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 426, 16), 'unicode', u'italic')
    # Assigning a type to the variable 'style' (line 426)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 426, 8), 'style', unicode_58102)
    # SSA branch for the else part of an if statement (line 425)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Str to a Name (line 428):
    
    # Assigning a Str to a Name (line 428):
    unicode_58103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 428, 16), 'unicode', u'normal')
    # Assigning a type to the variable 'style' (line 428)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 428, 8), 'style', unicode_58103)
    # SSA join for if statement (line 425)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 423)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 421)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 419)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to lower(...): (line 433)
    # Processing the call keyword arguments (line 433)
    kwargs_58106 = {}
    # Getting the type of 'name' (line 433)
    name_58104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 7), 'name', False)
    # Obtaining the member 'lower' of a type (line 433)
    lower_58105 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 433, 7), name_58104, 'lower')
    # Calling lower(args, kwargs) (line 433)
    lower_call_result_58107 = invoke(stypy.reporting.localization.Localization(__file__, 433, 7), lower_58105, *[], **kwargs_58106)
    
    
    # Obtaining an instance of the builtin type 'list' (line 433)
    list_58108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 433)
    # Adding element type (line 433)
    unicode_58109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 24), 'unicode', u'capitals')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 433, 23), list_58108, unicode_58109)
    # Adding element type (line 433)
    unicode_58110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 36), 'unicode', u'small-caps')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 433, 23), list_58108, unicode_58110)
    
    # Applying the binary operator 'in' (line 433)
    result_contains_58111 = python_operator(stypy.reporting.localization.Localization(__file__, 433, 7), 'in', lower_call_result_58107, list_58108)
    
    # Testing the type of an if condition (line 433)
    if_condition_58112 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 433, 4), result_contains_58111)
    # Assigning a type to the variable 'if_condition_58112' (line 433)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 433, 4), 'if_condition_58112', if_condition_58112)
    # SSA begins for if statement (line 433)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 434):
    
    # Assigning a Str to a Name (line 434):
    unicode_58113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, 18), 'unicode', u'small-caps')
    # Assigning a type to the variable 'variant' (line 434)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 434, 8), 'variant', unicode_58113)
    # SSA branch for the else part of an if statement (line 433)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Str to a Name (line 436):
    
    # Assigning a Str to a Name (line 436):
    unicode_58114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 436, 18), 'unicode', u'normal')
    # Assigning a type to the variable 'variant' (line 436)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 436, 8), 'variant', unicode_58114)
    # SSA join for if statement (line 433)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 438):
    
    # Assigning a Call to a Name (line 438):
    
    # Call to next(...): (line 438)
    # Processing the call arguments (line 438)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 438, 19, True)
    # Calculating comprehension expression
    # Getting the type of 'weight_dict' (line 438)
    weight_dict_58124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 30), 'weight_dict', False)
    comprehension_58125 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 438, 19), weight_dict_58124)
    # Assigning a type to the variable 'w' (line 438)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 438, 19), 'w', comprehension_58125)
    
    
    # Call to find(...): (line 438)
    # Processing the call arguments (line 438)
    # Getting the type of 'w' (line 438)
    w_58119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 56), 'w', False)
    # Processing the call keyword arguments (line 438)
    kwargs_58120 = {}
    # Getting the type of 'sfnt4' (line 438)
    sfnt4_58117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 45), 'sfnt4', False)
    # Obtaining the member 'find' of a type (line 438)
    find_58118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 438, 45), sfnt4_58117, 'find')
    # Calling find(args, kwargs) (line 438)
    find_call_result_58121 = invoke(stypy.reporting.localization.Localization(__file__, 438, 45), find_58118, *[w_58119], **kwargs_58120)
    
    int_58122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 438, 62), 'int')
    # Applying the binary operator '>=' (line 438)
    result_ge_58123 = python_operator(stypy.reporting.localization.Localization(__file__, 438, 45), '>=', find_call_result_58121, int_58122)
    
    # Getting the type of 'w' (line 438)
    w_58116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 19), 'w', False)
    list_58126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 438, 19), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 438, 19), list_58126, w_58116)
    # Getting the type of 'None' (line 438)
    None_58127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 66), 'None', False)
    # Processing the call keyword arguments (line 438)
    kwargs_58128 = {}
    # Getting the type of 'next' (line 438)
    next_58115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 13), 'next', False)
    # Calling next(args, kwargs) (line 438)
    next_call_result_58129 = invoke(stypy.reporting.localization.Localization(__file__, 438, 13), next_58115, *[list_58126, None_58127], **kwargs_58128)
    
    # Assigning a type to the variable 'weight' (line 438)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 438, 4), 'weight', next_call_result_58129)
    
    
    # Getting the type of 'weight' (line 439)
    weight_58130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 11), 'weight')
    # Applying the 'not' unary operator (line 439)
    result_not__58131 = python_operator(stypy.reporting.localization.Localization(__file__, 439, 7), 'not', weight_58130)
    
    # Testing the type of an if condition (line 439)
    if_condition_58132 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 439, 4), result_not__58131)
    # Assigning a type to the variable 'if_condition_58132' (line 439)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 439, 4), 'if_condition_58132', if_condition_58132)
    # SSA begins for if statement (line 439)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'font' (line 440)
    font_58133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 11), 'font')
    # Obtaining the member 'style_flags' of a type (line 440)
    style_flags_58134 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 440, 11), font_58133, 'style_flags')
    # Getting the type of 'ft2font' (line 440)
    ft2font_58135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 30), 'ft2font')
    # Obtaining the member 'BOLD' of a type (line 440)
    BOLD_58136 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 440, 30), ft2font_58135, 'BOLD')
    # Applying the binary operator '&' (line 440)
    result_and__58137 = python_operator(stypy.reporting.localization.Localization(__file__, 440, 11), '&', style_flags_58134, BOLD_58136)
    
    # Testing the type of an if condition (line 440)
    if_condition_58138 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 440, 8), result_and__58137)
    # Assigning a type to the variable 'if_condition_58138' (line 440)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 440, 8), 'if_condition_58138', if_condition_58138)
    # SSA begins for if statement (line 440)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 441):
    
    # Assigning a Num to a Name (line 441):
    int_58139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 441, 21), 'int')
    # Assigning a type to the variable 'weight' (line 441)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 441, 12), 'weight', int_58139)
    # SSA branch for the else part of an if statement (line 440)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Num to a Name (line 443):
    
    # Assigning a Num to a Name (line 443):
    int_58140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 443, 21), 'int')
    # Assigning a type to the variable 'weight' (line 443)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 443, 12), 'weight', int_58140)
    # SSA join for if statement (line 440)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 439)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    
    # Call to find(...): (line 452)
    # Processing the call arguments (line 452)
    unicode_58143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 452, 19), 'unicode', u'narrow')
    # Processing the call keyword arguments (line 452)
    kwargs_58144 = {}
    # Getting the type of 'sfnt4' (line 452)
    sfnt4_58141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 8), 'sfnt4', False)
    # Obtaining the member 'find' of a type (line 452)
    find_58142 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 452, 8), sfnt4_58141, 'find')
    # Calling find(args, kwargs) (line 452)
    find_call_result_58145 = invoke(stypy.reporting.localization.Localization(__file__, 452, 8), find_58142, *[unicode_58143], **kwargs_58144)
    
    int_58146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 452, 32), 'int')
    # Applying the binary operator '>=' (line 452)
    result_ge_58147 = python_operator(stypy.reporting.localization.Localization(__file__, 452, 8), '>=', find_call_result_58145, int_58146)
    
    
    
    # Call to find(...): (line 452)
    # Processing the call arguments (line 452)
    unicode_58150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 452, 48), 'unicode', u'condensed')
    # Processing the call keyword arguments (line 452)
    kwargs_58151 = {}
    # Getting the type of 'sfnt4' (line 452)
    sfnt4_58148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 37), 'sfnt4', False)
    # Obtaining the member 'find' of a type (line 452)
    find_58149 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 452, 37), sfnt4_58148, 'find')
    # Calling find(args, kwargs) (line 452)
    find_call_result_58152 = invoke(stypy.reporting.localization.Localization(__file__, 452, 37), find_58149, *[unicode_58150], **kwargs_58151)
    
    int_58153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 452, 64), 'int')
    # Applying the binary operator '>=' (line 452)
    result_ge_58154 = python_operator(stypy.reporting.localization.Localization(__file__, 452, 37), '>=', find_call_result_58152, int_58153)
    
    # Applying the binary operator 'or' (line 452)
    result_or_keyword_58155 = python_operator(stypy.reporting.localization.Localization(__file__, 452, 8), 'or', result_ge_58147, result_ge_58154)
    
    
    # Call to find(...): (line 453)
    # Processing the call arguments (line 453)
    unicode_58158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 453, 23), 'unicode', u'cond')
    # Processing the call keyword arguments (line 453)
    kwargs_58159 = {}
    # Getting the type of 'sfnt4' (line 453)
    sfnt4_58156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 12), 'sfnt4', False)
    # Obtaining the member 'find' of a type (line 453)
    find_58157 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 453, 12), sfnt4_58156, 'find')
    # Calling find(args, kwargs) (line 453)
    find_call_result_58160 = invoke(stypy.reporting.localization.Localization(__file__, 453, 12), find_58157, *[unicode_58158], **kwargs_58159)
    
    int_58161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 453, 34), 'int')
    # Applying the binary operator '>=' (line 453)
    result_ge_58162 = python_operator(stypy.reporting.localization.Localization(__file__, 453, 12), '>=', find_call_result_58160, int_58161)
    
    # Applying the binary operator 'or' (line 452)
    result_or_keyword_58163 = python_operator(stypy.reporting.localization.Localization(__file__, 452, 8), 'or', result_or_keyword_58155, result_ge_58162)
    
    # Testing the type of an if condition (line 452)
    if_condition_58164 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 452, 4), result_or_keyword_58163)
    # Assigning a type to the variable 'if_condition_58164' (line 452)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 452, 4), 'if_condition_58164', if_condition_58164)
    # SSA begins for if statement (line 452)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 454):
    
    # Assigning a Str to a Name (line 454):
    unicode_58165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 454, 18), 'unicode', u'condensed')
    # Assigning a type to the variable 'stretch' (line 454)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 454, 8), 'stretch', unicode_58165)
    # SSA branch for the else part of an if statement (line 452)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Call to find(...): (line 455)
    # Processing the call arguments (line 455)
    unicode_58168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 455, 20), 'unicode', u'demi cond')
    # Processing the call keyword arguments (line 455)
    kwargs_58169 = {}
    # Getting the type of 'sfnt4' (line 455)
    sfnt4_58166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 9), 'sfnt4', False)
    # Obtaining the member 'find' of a type (line 455)
    find_58167 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 455, 9), sfnt4_58166, 'find')
    # Calling find(args, kwargs) (line 455)
    find_call_result_58170 = invoke(stypy.reporting.localization.Localization(__file__, 455, 9), find_58167, *[unicode_58168], **kwargs_58169)
    
    int_58171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 455, 36), 'int')
    # Applying the binary operator '>=' (line 455)
    result_ge_58172 = python_operator(stypy.reporting.localization.Localization(__file__, 455, 9), '>=', find_call_result_58170, int_58171)
    
    # Testing the type of an if condition (line 455)
    if_condition_58173 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 455, 9), result_ge_58172)
    # Assigning a type to the variable 'if_condition_58173' (line 455)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 455, 9), 'if_condition_58173', if_condition_58173)
    # SSA begins for if statement (line 455)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 456):
    
    # Assigning a Str to a Name (line 456):
    unicode_58174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 456, 18), 'unicode', u'semi-condensed')
    # Assigning a type to the variable 'stretch' (line 456)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 456, 8), 'stretch', unicode_58174)
    # SSA branch for the else part of an if statement (line 455)
    module_type_store.open_ssa_branch('else')
    
    
    # Evaluating a boolean operation
    
    
    # Call to find(...): (line 457)
    # Processing the call arguments (line 457)
    unicode_58177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 457, 20), 'unicode', u'wide')
    # Processing the call keyword arguments (line 457)
    kwargs_58178 = {}
    # Getting the type of 'sfnt4' (line 457)
    sfnt4_58175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 9), 'sfnt4', False)
    # Obtaining the member 'find' of a type (line 457)
    find_58176 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 457, 9), sfnt4_58175, 'find')
    # Calling find(args, kwargs) (line 457)
    find_call_result_58179 = invoke(stypy.reporting.localization.Localization(__file__, 457, 9), find_58176, *[unicode_58177], **kwargs_58178)
    
    int_58180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 457, 31), 'int')
    # Applying the binary operator '>=' (line 457)
    result_ge_58181 = python_operator(stypy.reporting.localization.Localization(__file__, 457, 9), '>=', find_call_result_58179, int_58180)
    
    
    
    # Call to find(...): (line 457)
    # Processing the call arguments (line 457)
    unicode_58184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 457, 47), 'unicode', u'expanded')
    # Processing the call keyword arguments (line 457)
    kwargs_58185 = {}
    # Getting the type of 'sfnt4' (line 457)
    sfnt4_58182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 36), 'sfnt4', False)
    # Obtaining the member 'find' of a type (line 457)
    find_58183 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 457, 36), sfnt4_58182, 'find')
    # Calling find(args, kwargs) (line 457)
    find_call_result_58186 = invoke(stypy.reporting.localization.Localization(__file__, 457, 36), find_58183, *[unicode_58184], **kwargs_58185)
    
    int_58187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 457, 62), 'int')
    # Applying the binary operator '>=' (line 457)
    result_ge_58188 = python_operator(stypy.reporting.localization.Localization(__file__, 457, 36), '>=', find_call_result_58186, int_58187)
    
    # Applying the binary operator 'or' (line 457)
    result_or_keyword_58189 = python_operator(stypy.reporting.localization.Localization(__file__, 457, 9), 'or', result_ge_58181, result_ge_58188)
    
    # Testing the type of an if condition (line 457)
    if_condition_58190 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 457, 9), result_or_keyword_58189)
    # Assigning a type to the variable 'if_condition_58190' (line 457)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 457, 9), 'if_condition_58190', if_condition_58190)
    # SSA begins for if statement (line 457)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 458):
    
    # Assigning a Str to a Name (line 458):
    unicode_58191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 458, 18), 'unicode', u'expanded')
    # Assigning a type to the variable 'stretch' (line 458)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 458, 8), 'stretch', unicode_58191)
    # SSA branch for the else part of an if statement (line 457)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Str to a Name (line 460):
    
    # Assigning a Str to a Name (line 460):
    unicode_58192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 460, 18), 'unicode', u'normal')
    # Assigning a type to the variable 'stretch' (line 460)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 460, 8), 'stretch', unicode_58192)
    # SSA join for if statement (line 457)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 455)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 452)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'font' (line 470)
    font_58193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 7), 'font')
    # Obtaining the member 'scalable' of a type (line 470)
    scalable_58194 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 470, 7), font_58193, 'scalable')
    # Testing the type of an if condition (line 470)
    if_condition_58195 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 470, 4), scalable_58194)
    # Assigning a type to the variable 'if_condition_58195' (line 470)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 470, 4), 'if_condition_58195', if_condition_58195)
    # SSA begins for if statement (line 470)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 471):
    
    # Assigning a Str to a Name (line 471):
    unicode_58196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 471, 15), 'unicode', u'scalable')
    # Assigning a type to the variable 'size' (line 471)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 471, 8), 'size', unicode_58196)
    # SSA branch for the else part of an if statement (line 470)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 473):
    
    # Assigning a Call to a Name (line 473):
    
    # Call to str(...): (line 473)
    # Processing the call arguments (line 473)
    
    # Call to float(...): (line 473)
    # Processing the call arguments (line 473)
    
    # Call to get_fontsize(...): (line 473)
    # Processing the call keyword arguments (line 473)
    kwargs_58201 = {}
    # Getting the type of 'font' (line 473)
    font_58199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 25), 'font', False)
    # Obtaining the member 'get_fontsize' of a type (line 473)
    get_fontsize_58200 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 473, 25), font_58199, 'get_fontsize')
    # Calling get_fontsize(args, kwargs) (line 473)
    get_fontsize_call_result_58202 = invoke(stypy.reporting.localization.Localization(__file__, 473, 25), get_fontsize_58200, *[], **kwargs_58201)
    
    # Processing the call keyword arguments (line 473)
    kwargs_58203 = {}
    # Getting the type of 'float' (line 473)
    float_58198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 19), 'float', False)
    # Calling float(args, kwargs) (line 473)
    float_call_result_58204 = invoke(stypy.reporting.localization.Localization(__file__, 473, 19), float_58198, *[get_fontsize_call_result_58202], **kwargs_58203)
    
    # Processing the call keyword arguments (line 473)
    kwargs_58205 = {}
    # Getting the type of 'str' (line 473)
    str_58197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 15), 'str', False)
    # Calling str(args, kwargs) (line 473)
    str_call_result_58206 = invoke(stypy.reporting.localization.Localization(__file__, 473, 15), str_58197, *[float_call_result_58204], **kwargs_58205)
    
    # Assigning a type to the variable 'size' (line 473)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 473, 8), 'size', str_call_result_58206)
    # SSA join for if statement (line 470)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Name (line 476):
    
    # Assigning a Name to a Name (line 476):
    # Getting the type of 'None' (line 476)
    None_58207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 18), 'None')
    # Assigning a type to the variable 'size_adjust' (line 476)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 476, 4), 'size_adjust', None_58207)
    
    # Call to FontEntry(...): (line 478)
    # Processing the call arguments (line 478)
    # Getting the type of 'font' (line 478)
    font_58209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 21), 'font', False)
    # Obtaining the member 'fname' of a type (line 478)
    fname_58210 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 478, 21), font_58209, 'fname')
    # Getting the type of 'name' (line 478)
    name_58211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 33), 'name', False)
    # Getting the type of 'style' (line 478)
    style_58212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 39), 'style', False)
    # Getting the type of 'variant' (line 478)
    variant_58213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 46), 'variant', False)
    # Getting the type of 'weight' (line 478)
    weight_58214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 55), 'weight', False)
    # Getting the type of 'stretch' (line 478)
    stretch_58215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 63), 'stretch', False)
    # Getting the type of 'size' (line 478)
    size_58216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 72), 'size', False)
    # Processing the call keyword arguments (line 478)
    kwargs_58217 = {}
    # Getting the type of 'FontEntry' (line 478)
    FontEntry_58208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 11), 'FontEntry', False)
    # Calling FontEntry(args, kwargs) (line 478)
    FontEntry_call_result_58218 = invoke(stypy.reporting.localization.Localization(__file__, 478, 11), FontEntry_58208, *[fname_58210, name_58211, style_58212, variant_58213, weight_58214, stretch_58215, size_58216], **kwargs_58217)
    
    # Assigning a type to the variable 'stypy_return_type' (line 478)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 478, 4), 'stypy_return_type', FontEntry_call_result_58218)
    
    # ################# End of 'ttfFontProperty(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'ttfFontProperty' in the type store
    # Getting the type of 'stypy_return_type' (line 397)
    stypy_return_type_58219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_58219)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'ttfFontProperty'
    return stypy_return_type_58219

# Assigning a type to the variable 'ttfFontProperty' (line 397)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 397, 0), 'ttfFontProperty', ttfFontProperty)

@norecursion
def afmFontProperty(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'afmFontProperty'
    module_type_store = module_type_store.open_function_context('afmFontProperty', 481, 0, False)
    
    # Passed parameters checking function
    afmFontProperty.stypy_localization = localization
    afmFontProperty.stypy_type_of_self = None
    afmFontProperty.stypy_type_store = module_type_store
    afmFontProperty.stypy_function_name = 'afmFontProperty'
    afmFontProperty.stypy_param_names_list = ['fontpath', 'font']
    afmFontProperty.stypy_varargs_param_name = None
    afmFontProperty.stypy_kwargs_param_name = None
    afmFontProperty.stypy_call_defaults = defaults
    afmFontProperty.stypy_call_varargs = varargs
    afmFontProperty.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'afmFontProperty', ['fontpath', 'font'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'afmFontProperty', localization, ['fontpath', 'font'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'afmFontProperty(...)' code ##################

    unicode_58220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 487, (-1)), 'unicode', u'\n    A function for populating a :class:`FontKey` instance by\n    extracting information from the AFM font file.\n\n    *font* is a class:`AFM` instance.\n    ')
    
    # Assigning a Call to a Name (line 489):
    
    # Assigning a Call to a Name (line 489):
    
    # Call to get_familyname(...): (line 489)
    # Processing the call keyword arguments (line 489)
    kwargs_58223 = {}
    # Getting the type of 'font' (line 489)
    font_58221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 11), 'font', False)
    # Obtaining the member 'get_familyname' of a type (line 489)
    get_familyname_58222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 489, 11), font_58221, 'get_familyname')
    # Calling get_familyname(args, kwargs) (line 489)
    get_familyname_call_result_58224 = invoke(stypy.reporting.localization.Localization(__file__, 489, 11), get_familyname_58222, *[], **kwargs_58223)
    
    # Assigning a type to the variable 'name' (line 489)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 489, 4), 'name', get_familyname_call_result_58224)
    
    # Assigning a Call to a Name (line 490):
    
    # Assigning a Call to a Name (line 490):
    
    # Call to lower(...): (line 490)
    # Processing the call keyword arguments (line 490)
    kwargs_58230 = {}
    
    # Call to get_fontname(...): (line 490)
    # Processing the call keyword arguments (line 490)
    kwargs_58227 = {}
    # Getting the type of 'font' (line 490)
    font_58225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 15), 'font', False)
    # Obtaining the member 'get_fontname' of a type (line 490)
    get_fontname_58226 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 490, 15), font_58225, 'get_fontname')
    # Calling get_fontname(args, kwargs) (line 490)
    get_fontname_call_result_58228 = invoke(stypy.reporting.localization.Localization(__file__, 490, 15), get_fontname_58226, *[], **kwargs_58227)
    
    # Obtaining the member 'lower' of a type (line 490)
    lower_58229 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 490, 15), get_fontname_call_result_58228, 'lower')
    # Calling lower(args, kwargs) (line 490)
    lower_call_result_58231 = invoke(stypy.reporting.localization.Localization(__file__, 490, 15), lower_58229, *[], **kwargs_58230)
    
    # Assigning a type to the variable 'fontname' (line 490)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 490, 4), 'fontname', lower_call_result_58231)
    
    
    # Evaluating a boolean operation
    
    
    # Call to get_angle(...): (line 494)
    # Processing the call keyword arguments (line 494)
    kwargs_58234 = {}
    # Getting the type of 'font' (line 494)
    font_58232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 7), 'font', False)
    # Obtaining the member 'get_angle' of a type (line 494)
    get_angle_58233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 494, 7), font_58232, 'get_angle')
    # Calling get_angle(args, kwargs) (line 494)
    get_angle_call_result_58235 = invoke(stypy.reporting.localization.Localization(__file__, 494, 7), get_angle_58233, *[], **kwargs_58234)
    
    int_58236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 494, 27), 'int')
    # Applying the binary operator '!=' (line 494)
    result_ne_58237 = python_operator(stypy.reporting.localization.Localization(__file__, 494, 7), '!=', get_angle_call_result_58235, int_58236)
    
    
    
    # Call to find(...): (line 494)
    # Processing the call arguments (line 494)
    unicode_58243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 494, 50), 'unicode', u'italic')
    # Processing the call keyword arguments (line 494)
    kwargs_58244 = {}
    
    # Call to lower(...): (line 494)
    # Processing the call keyword arguments (line 494)
    kwargs_58240 = {}
    # Getting the type of 'name' (line 494)
    name_58238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 32), 'name', False)
    # Obtaining the member 'lower' of a type (line 494)
    lower_58239 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 494, 32), name_58238, 'lower')
    # Calling lower(args, kwargs) (line 494)
    lower_call_result_58241 = invoke(stypy.reporting.localization.Localization(__file__, 494, 32), lower_58239, *[], **kwargs_58240)
    
    # Obtaining the member 'find' of a type (line 494)
    find_58242 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 494, 32), lower_call_result_58241, 'find')
    # Calling find(args, kwargs) (line 494)
    find_call_result_58245 = invoke(stypy.reporting.localization.Localization(__file__, 494, 32), find_58242, *[unicode_58243], **kwargs_58244)
    
    int_58246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 494, 63), 'int')
    # Applying the binary operator '>=' (line 494)
    result_ge_58247 = python_operator(stypy.reporting.localization.Localization(__file__, 494, 32), '>=', find_call_result_58245, int_58246)
    
    # Applying the binary operator 'or' (line 494)
    result_or_keyword_58248 = python_operator(stypy.reporting.localization.Localization(__file__, 494, 7), 'or', result_ne_58237, result_ge_58247)
    
    # Testing the type of an if condition (line 494)
    if_condition_58249 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 494, 4), result_or_keyword_58248)
    # Assigning a type to the variable 'if_condition_58249' (line 494)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 494, 4), 'if_condition_58249', if_condition_58249)
    # SSA begins for if statement (line 494)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 495):
    
    # Assigning a Str to a Name (line 495):
    unicode_58250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 495, 16), 'unicode', u'italic')
    # Assigning a type to the variable 'style' (line 495)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 495, 8), 'style', unicode_58250)
    # SSA branch for the else part of an if statement (line 494)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Call to find(...): (line 496)
    # Processing the call arguments (line 496)
    unicode_58256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 496, 27), 'unicode', u'oblique')
    # Processing the call keyword arguments (line 496)
    kwargs_58257 = {}
    
    # Call to lower(...): (line 496)
    # Processing the call keyword arguments (line 496)
    kwargs_58253 = {}
    # Getting the type of 'name' (line 496)
    name_58251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 9), 'name', False)
    # Obtaining the member 'lower' of a type (line 496)
    lower_58252 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 496, 9), name_58251, 'lower')
    # Calling lower(args, kwargs) (line 496)
    lower_call_result_58254 = invoke(stypy.reporting.localization.Localization(__file__, 496, 9), lower_58252, *[], **kwargs_58253)
    
    # Obtaining the member 'find' of a type (line 496)
    find_58255 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 496, 9), lower_call_result_58254, 'find')
    # Calling find(args, kwargs) (line 496)
    find_call_result_58258 = invoke(stypy.reporting.localization.Localization(__file__, 496, 9), find_58255, *[unicode_58256], **kwargs_58257)
    
    int_58259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 496, 41), 'int')
    # Applying the binary operator '>=' (line 496)
    result_ge_58260 = python_operator(stypy.reporting.localization.Localization(__file__, 496, 9), '>=', find_call_result_58258, int_58259)
    
    # Testing the type of an if condition (line 496)
    if_condition_58261 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 496, 9), result_ge_58260)
    # Assigning a type to the variable 'if_condition_58261' (line 496)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 496, 9), 'if_condition_58261', if_condition_58261)
    # SSA begins for if statement (line 496)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 497):
    
    # Assigning a Str to a Name (line 497):
    unicode_58262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 497, 16), 'unicode', u'oblique')
    # Assigning a type to the variable 'style' (line 497)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 497, 8), 'style', unicode_58262)
    # SSA branch for the else part of an if statement (line 496)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Str to a Name (line 499):
    
    # Assigning a Str to a Name (line 499):
    unicode_58263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 499, 16), 'unicode', u'normal')
    # Assigning a type to the variable 'style' (line 499)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 499, 8), 'style', unicode_58263)
    # SSA join for if statement (line 496)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 494)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to lower(...): (line 504)
    # Processing the call keyword arguments (line 504)
    kwargs_58266 = {}
    # Getting the type of 'name' (line 504)
    name_58264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 7), 'name', False)
    # Obtaining the member 'lower' of a type (line 504)
    lower_58265 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 504, 7), name_58264, 'lower')
    # Calling lower(args, kwargs) (line 504)
    lower_call_result_58267 = invoke(stypy.reporting.localization.Localization(__file__, 504, 7), lower_58265, *[], **kwargs_58266)
    
    
    # Obtaining an instance of the builtin type 'list' (line 504)
    list_58268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 504, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 504)
    # Adding element type (line 504)
    unicode_58269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 504, 24), 'unicode', u'capitals')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 504, 23), list_58268, unicode_58269)
    # Adding element type (line 504)
    unicode_58270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 504, 36), 'unicode', u'small-caps')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 504, 23), list_58268, unicode_58270)
    
    # Applying the binary operator 'in' (line 504)
    result_contains_58271 = python_operator(stypy.reporting.localization.Localization(__file__, 504, 7), 'in', lower_call_result_58267, list_58268)
    
    # Testing the type of an if condition (line 504)
    if_condition_58272 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 504, 4), result_contains_58271)
    # Assigning a type to the variable 'if_condition_58272' (line 504)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 504, 4), 'if_condition_58272', if_condition_58272)
    # SSA begins for if statement (line 504)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 505):
    
    # Assigning a Str to a Name (line 505):
    unicode_58273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 505, 18), 'unicode', u'small-caps')
    # Assigning a type to the variable 'variant' (line 505)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 505, 8), 'variant', unicode_58273)
    # SSA branch for the else part of an if statement (line 504)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Str to a Name (line 507):
    
    # Assigning a Str to a Name (line 507):
    unicode_58274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 507, 18), 'unicode', u'normal')
    # Assigning a type to the variable 'variant' (line 507)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 507, 8), 'variant', unicode_58274)
    # SSA join for if statement (line 504)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 509):
    
    # Assigning a Call to a Name (line 509):
    
    # Call to lower(...): (line 509)
    # Processing the call keyword arguments (line 509)
    kwargs_58280 = {}
    
    # Call to get_weight(...): (line 509)
    # Processing the call keyword arguments (line 509)
    kwargs_58277 = {}
    # Getting the type of 'font' (line 509)
    font_58275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 13), 'font', False)
    # Obtaining the member 'get_weight' of a type (line 509)
    get_weight_58276 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 509, 13), font_58275, 'get_weight')
    # Calling get_weight(args, kwargs) (line 509)
    get_weight_call_result_58278 = invoke(stypy.reporting.localization.Localization(__file__, 509, 13), get_weight_58276, *[], **kwargs_58277)
    
    # Obtaining the member 'lower' of a type (line 509)
    lower_58279 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 509, 13), get_weight_call_result_58278, 'lower')
    # Calling lower(args, kwargs) (line 509)
    lower_call_result_58281 = invoke(stypy.reporting.localization.Localization(__file__, 509, 13), lower_58279, *[], **kwargs_58280)
    
    # Assigning a type to the variable 'weight' (line 509)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 509, 4), 'weight', lower_call_result_58281)
    
    
    # Evaluating a boolean operation
    
    
    # Call to find(...): (line 517)
    # Processing the call arguments (line 517)
    unicode_58284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 517, 21), 'unicode', u'narrow')
    # Processing the call keyword arguments (line 517)
    kwargs_58285 = {}
    # Getting the type of 'fontname' (line 517)
    fontname_58282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 7), 'fontname', False)
    # Obtaining the member 'find' of a type (line 517)
    find_58283 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 517, 7), fontname_58282, 'find')
    # Calling find(args, kwargs) (line 517)
    find_call_result_58286 = invoke(stypy.reporting.localization.Localization(__file__, 517, 7), find_58283, *[unicode_58284], **kwargs_58285)
    
    int_58287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 517, 34), 'int')
    # Applying the binary operator '>=' (line 517)
    result_ge_58288 = python_operator(stypy.reporting.localization.Localization(__file__, 517, 7), '>=', find_call_result_58286, int_58287)
    
    
    
    # Call to find(...): (line 517)
    # Processing the call arguments (line 517)
    unicode_58291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 517, 53), 'unicode', u'condensed')
    # Processing the call keyword arguments (line 517)
    kwargs_58292 = {}
    # Getting the type of 'fontname' (line 517)
    fontname_58289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 39), 'fontname', False)
    # Obtaining the member 'find' of a type (line 517)
    find_58290 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 517, 39), fontname_58289, 'find')
    # Calling find(args, kwargs) (line 517)
    find_call_result_58293 = invoke(stypy.reporting.localization.Localization(__file__, 517, 39), find_58290, *[unicode_58291], **kwargs_58292)
    
    int_58294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 517, 69), 'int')
    # Applying the binary operator '>=' (line 517)
    result_ge_58295 = python_operator(stypy.reporting.localization.Localization(__file__, 517, 39), '>=', find_call_result_58293, int_58294)
    
    # Applying the binary operator 'or' (line 517)
    result_or_keyword_58296 = python_operator(stypy.reporting.localization.Localization(__file__, 517, 7), 'or', result_ge_58288, result_ge_58295)
    
    
    # Call to find(...): (line 518)
    # Processing the call arguments (line 518)
    unicode_58299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 518, 25), 'unicode', u'cond')
    # Processing the call keyword arguments (line 518)
    kwargs_58300 = {}
    # Getting the type of 'fontname' (line 518)
    fontname_58297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 11), 'fontname', False)
    # Obtaining the member 'find' of a type (line 518)
    find_58298 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 518, 11), fontname_58297, 'find')
    # Calling find(args, kwargs) (line 518)
    find_call_result_58301 = invoke(stypy.reporting.localization.Localization(__file__, 518, 11), find_58298, *[unicode_58299], **kwargs_58300)
    
    int_58302 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 518, 36), 'int')
    # Applying the binary operator '>=' (line 518)
    result_ge_58303 = python_operator(stypy.reporting.localization.Localization(__file__, 518, 11), '>=', find_call_result_58301, int_58302)
    
    # Applying the binary operator 'or' (line 517)
    result_or_keyword_58304 = python_operator(stypy.reporting.localization.Localization(__file__, 517, 7), 'or', result_or_keyword_58296, result_ge_58303)
    
    # Testing the type of an if condition (line 517)
    if_condition_58305 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 517, 4), result_or_keyword_58304)
    # Assigning a type to the variable 'if_condition_58305' (line 517)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 517, 4), 'if_condition_58305', if_condition_58305)
    # SSA begins for if statement (line 517)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 519):
    
    # Assigning a Str to a Name (line 519):
    unicode_58306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 519, 18), 'unicode', u'condensed')
    # Assigning a type to the variable 'stretch' (line 519)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 519, 8), 'stretch', unicode_58306)
    # SSA branch for the else part of an if statement (line 517)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Call to find(...): (line 520)
    # Processing the call arguments (line 520)
    unicode_58309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 520, 23), 'unicode', u'demi cond')
    # Processing the call keyword arguments (line 520)
    kwargs_58310 = {}
    # Getting the type of 'fontname' (line 520)
    fontname_58307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 9), 'fontname', False)
    # Obtaining the member 'find' of a type (line 520)
    find_58308 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 520, 9), fontname_58307, 'find')
    # Calling find(args, kwargs) (line 520)
    find_call_result_58311 = invoke(stypy.reporting.localization.Localization(__file__, 520, 9), find_58308, *[unicode_58309], **kwargs_58310)
    
    int_58312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 520, 39), 'int')
    # Applying the binary operator '>=' (line 520)
    result_ge_58313 = python_operator(stypy.reporting.localization.Localization(__file__, 520, 9), '>=', find_call_result_58311, int_58312)
    
    # Testing the type of an if condition (line 520)
    if_condition_58314 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 520, 9), result_ge_58313)
    # Assigning a type to the variable 'if_condition_58314' (line 520)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 520, 9), 'if_condition_58314', if_condition_58314)
    # SSA begins for if statement (line 520)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 521):
    
    # Assigning a Str to a Name (line 521):
    unicode_58315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 521, 18), 'unicode', u'semi-condensed')
    # Assigning a type to the variable 'stretch' (line 521)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 521, 8), 'stretch', unicode_58315)
    # SSA branch for the else part of an if statement (line 520)
    module_type_store.open_ssa_branch('else')
    
    
    # Evaluating a boolean operation
    
    
    # Call to find(...): (line 522)
    # Processing the call arguments (line 522)
    unicode_58318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 522, 23), 'unicode', u'wide')
    # Processing the call keyword arguments (line 522)
    kwargs_58319 = {}
    # Getting the type of 'fontname' (line 522)
    fontname_58316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 9), 'fontname', False)
    # Obtaining the member 'find' of a type (line 522)
    find_58317 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 522, 9), fontname_58316, 'find')
    # Calling find(args, kwargs) (line 522)
    find_call_result_58320 = invoke(stypy.reporting.localization.Localization(__file__, 522, 9), find_58317, *[unicode_58318], **kwargs_58319)
    
    int_58321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 522, 34), 'int')
    # Applying the binary operator '>=' (line 522)
    result_ge_58322 = python_operator(stypy.reporting.localization.Localization(__file__, 522, 9), '>=', find_call_result_58320, int_58321)
    
    
    
    # Call to find(...): (line 522)
    # Processing the call arguments (line 522)
    unicode_58325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 522, 53), 'unicode', u'expanded')
    # Processing the call keyword arguments (line 522)
    kwargs_58326 = {}
    # Getting the type of 'fontname' (line 522)
    fontname_58323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 39), 'fontname', False)
    # Obtaining the member 'find' of a type (line 522)
    find_58324 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 522, 39), fontname_58323, 'find')
    # Calling find(args, kwargs) (line 522)
    find_call_result_58327 = invoke(stypy.reporting.localization.Localization(__file__, 522, 39), find_58324, *[unicode_58325], **kwargs_58326)
    
    int_58328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 522, 68), 'int')
    # Applying the binary operator '>=' (line 522)
    result_ge_58329 = python_operator(stypy.reporting.localization.Localization(__file__, 522, 39), '>=', find_call_result_58327, int_58328)
    
    # Applying the binary operator 'or' (line 522)
    result_or_keyword_58330 = python_operator(stypy.reporting.localization.Localization(__file__, 522, 9), 'or', result_ge_58322, result_ge_58329)
    
    # Testing the type of an if condition (line 522)
    if_condition_58331 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 522, 9), result_or_keyword_58330)
    # Assigning a type to the variable 'if_condition_58331' (line 522)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 522, 9), 'if_condition_58331', if_condition_58331)
    # SSA begins for if statement (line 522)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 523):
    
    # Assigning a Str to a Name (line 523):
    unicode_58332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 523, 18), 'unicode', u'expanded')
    # Assigning a type to the variable 'stretch' (line 523)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 523, 8), 'stretch', unicode_58332)
    # SSA branch for the else part of an if statement (line 522)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Str to a Name (line 525):
    
    # Assigning a Str to a Name (line 525):
    unicode_58333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 525, 18), 'unicode', u'normal')
    # Assigning a type to the variable 'stretch' (line 525)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 525, 8), 'stretch', unicode_58333)
    # SSA join for if statement (line 522)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 520)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 517)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Str to a Name (line 536):
    
    # Assigning a Str to a Name (line 536):
    unicode_58334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 536, 11), 'unicode', u'scalable')
    # Assigning a type to the variable 'size' (line 536)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 536, 4), 'size', unicode_58334)
    
    # Assigning a Name to a Name (line 539):
    
    # Assigning a Name to a Name (line 539):
    # Getting the type of 'None' (line 539)
    None_58335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 18), 'None')
    # Assigning a type to the variable 'size_adjust' (line 539)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 539, 4), 'size_adjust', None_58335)
    
    # Call to FontEntry(...): (line 541)
    # Processing the call arguments (line 541)
    # Getting the type of 'fontpath' (line 541)
    fontpath_58337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 21), 'fontpath', False)
    # Getting the type of 'name' (line 541)
    name_58338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 31), 'name', False)
    # Getting the type of 'style' (line 541)
    style_58339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 37), 'style', False)
    # Getting the type of 'variant' (line 541)
    variant_58340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 44), 'variant', False)
    # Getting the type of 'weight' (line 541)
    weight_58341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 53), 'weight', False)
    # Getting the type of 'stretch' (line 541)
    stretch_58342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 61), 'stretch', False)
    # Getting the type of 'size' (line 541)
    size_58343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 70), 'size', False)
    # Processing the call keyword arguments (line 541)
    kwargs_58344 = {}
    # Getting the type of 'FontEntry' (line 541)
    FontEntry_58336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 11), 'FontEntry', False)
    # Calling FontEntry(args, kwargs) (line 541)
    FontEntry_call_result_58345 = invoke(stypy.reporting.localization.Localization(__file__, 541, 11), FontEntry_58336, *[fontpath_58337, name_58338, style_58339, variant_58340, weight_58341, stretch_58342, size_58343], **kwargs_58344)
    
    # Assigning a type to the variable 'stypy_return_type' (line 541)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 541, 4), 'stypy_return_type', FontEntry_call_result_58345)
    
    # ################# End of 'afmFontProperty(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'afmFontProperty' in the type store
    # Getting the type of 'stypy_return_type' (line 481)
    stypy_return_type_58346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_58346)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'afmFontProperty'
    return stypy_return_type_58346

# Assigning a type to the variable 'afmFontProperty' (line 481)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 481, 0), 'afmFontProperty', afmFontProperty)

@norecursion
def createFontList(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    unicode_58347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 544, 38), 'unicode', u'ttf')
    defaults = [unicode_58347]
    # Create a new context for function 'createFontList'
    module_type_store = module_type_store.open_function_context('createFontList', 544, 0, False)
    
    # Passed parameters checking function
    createFontList.stypy_localization = localization
    createFontList.stypy_type_of_self = None
    createFontList.stypy_type_store = module_type_store
    createFontList.stypy_function_name = 'createFontList'
    createFontList.stypy_param_names_list = ['fontfiles', 'fontext']
    createFontList.stypy_varargs_param_name = None
    createFontList.stypy_kwargs_param_name = None
    createFontList.stypy_call_defaults = defaults
    createFontList.stypy_call_varargs = varargs
    createFontList.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'createFontList', ['fontfiles', 'fontext'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'createFontList', localization, ['fontfiles', 'fontext'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'createFontList(...)' code ##################

    unicode_58348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 549, (-1)), 'unicode', u'\n    A function to create a font lookup list.  The default is to create\n    a list of TrueType fonts.  An AFM font list can optionally be\n    created.\n    ')
    
    # Assigning a List to a Name (line 551):
    
    # Assigning a List to a Name (line 551):
    
    # Obtaining an instance of the builtin type 'list' (line 551)
    list_58349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 551, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 551)
    
    # Assigning a type to the variable 'fontlist' (line 551)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 551, 4), 'fontlist', list_58349)
    
    # Assigning a Dict to a Name (line 553):
    
    # Assigning a Dict to a Name (line 553):
    
    # Obtaining an instance of the builtin type 'dict' (line 553)
    dict_58350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 553, 11), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 553)
    
    # Assigning a type to the variable 'seen' (line 553)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 553, 4), 'seen', dict_58350)
    
    # Getting the type of 'fontfiles' (line 554)
    fontfiles_58351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 17), 'fontfiles')
    # Testing the type of a for loop iterable (line 554)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 554, 4), fontfiles_58351)
    # Getting the type of the for loop variable (line 554)
    for_loop_var_58352 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 554, 4), fontfiles_58351)
    # Assigning a type to the variable 'fpath' (line 554)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 554, 4), 'fpath', for_loop_var_58352)
    # SSA begins for a for statement (line 554)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to report(...): (line 555)
    # Processing the call arguments (line 555)
    unicode_58355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 555, 23), 'unicode', u'createFontDict: %s')
    # Getting the type of 'fpath' (line 555)
    fpath_58356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 47), 'fpath', False)
    # Applying the binary operator '%' (line 555)
    result_mod_58357 = python_operator(stypy.reporting.localization.Localization(__file__, 555, 23), '%', unicode_58355, fpath_58356)
    
    unicode_58358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 555, 55), 'unicode', u'debug')
    # Processing the call keyword arguments (line 555)
    kwargs_58359 = {}
    # Getting the type of 'verbose' (line 555)
    verbose_58353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 8), 'verbose', False)
    # Obtaining the member 'report' of a type (line 555)
    report_58354 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 555, 8), verbose_58353, 'report')
    # Calling report(args, kwargs) (line 555)
    report_call_result_58360 = invoke(stypy.reporting.localization.Localization(__file__, 555, 8), report_58354, *[result_mod_58357, unicode_58358], **kwargs_58359)
    
    
    # Assigning a Subscript to a Name (line 556):
    
    # Assigning a Subscript to a Name (line 556):
    
    # Obtaining the type of the subscript
    int_58361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 556, 37), 'int')
    
    # Call to split(...): (line 556)
    # Processing the call arguments (line 556)
    # Getting the type of 'fpath' (line 556)
    fpath_58365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 30), 'fpath', False)
    # Processing the call keyword arguments (line 556)
    kwargs_58366 = {}
    # Getting the type of 'os' (line 556)
    os_58362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 16), 'os', False)
    # Obtaining the member 'path' of a type (line 556)
    path_58363 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 556, 16), os_58362, 'path')
    # Obtaining the member 'split' of a type (line 556)
    split_58364 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 556, 16), path_58363, 'split')
    # Calling split(args, kwargs) (line 556)
    split_call_result_58367 = invoke(stypy.reporting.localization.Localization(__file__, 556, 16), split_58364, *[fpath_58365], **kwargs_58366)
    
    # Obtaining the member '__getitem__' of a type (line 556)
    getitem___58368 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 556, 16), split_call_result_58367, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 556)
    subscript_call_result_58369 = invoke(stypy.reporting.localization.Localization(__file__, 556, 16), getitem___58368, int_58361)
    
    # Assigning a type to the variable 'fname' (line 556)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 556, 8), 'fname', subscript_call_result_58369)
    
    
    # Getting the type of 'fname' (line 557)
    fname_58370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 11), 'fname')
    # Getting the type of 'seen' (line 557)
    seen_58371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 20), 'seen')
    # Applying the binary operator 'in' (line 557)
    result_contains_58372 = python_operator(stypy.reporting.localization.Localization(__file__, 557, 11), 'in', fname_58370, seen_58371)
    
    # Testing the type of an if condition (line 557)
    if_condition_58373 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 557, 8), result_contains_58372)
    # Assigning a type to the variable 'if_condition_58373' (line 557)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 557, 8), 'if_condition_58373', if_condition_58373)
    # SSA begins for if statement (line 557)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA branch for the else part of an if statement (line 557)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Num to a Subscript (line 560):
    
    # Assigning a Num to a Subscript (line 560):
    int_58374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 560, 26), 'int')
    # Getting the type of 'seen' (line 560)
    seen_58375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 12), 'seen')
    # Getting the type of 'fname' (line 560)
    fname_58376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 17), 'fname')
    # Storing an element on a container (line 560)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 560, 12), seen_58375, (fname_58376, int_58374))
    # SSA join for if statement (line 557)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'fontext' (line 561)
    fontext_58377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 11), 'fontext')
    unicode_58378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 561, 22), 'unicode', u'afm')
    # Applying the binary operator '==' (line 561)
    result_eq_58379 = python_operator(stypy.reporting.localization.Localization(__file__, 561, 11), '==', fontext_58377, unicode_58378)
    
    # Testing the type of an if condition (line 561)
    if_condition_58380 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 561, 8), result_eq_58379)
    # Assigning a type to the variable 'if_condition_58380' (line 561)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 561, 8), 'if_condition_58380', if_condition_58380)
    # SSA begins for if statement (line 561)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # SSA begins for try-except statement (line 562)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Name (line 563):
    
    # Assigning a Call to a Name (line 563):
    
    # Call to open(...): (line 563)
    # Processing the call arguments (line 563)
    # Getting the type of 'fpath' (line 563)
    fpath_58382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 26), 'fpath', False)
    unicode_58383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 563, 33), 'unicode', u'rb')
    # Processing the call keyword arguments (line 563)
    kwargs_58384 = {}
    # Getting the type of 'open' (line 563)
    open_58381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 21), 'open', False)
    # Calling open(args, kwargs) (line 563)
    open_call_result_58385 = invoke(stypy.reporting.localization.Localization(__file__, 563, 21), open_58381, *[fpath_58382, unicode_58383], **kwargs_58384)
    
    # Assigning a type to the variable 'fh' (line 563)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 563, 16), 'fh', open_call_result_58385)
    # SSA branch for the except part of a try statement (line 562)
    # SSA branch for the except 'EnvironmentError' branch of a try statement (line 562)
    module_type_store.open_ssa_branch('except')
    
    # Call to report(...): (line 565)
    # Processing the call arguments (line 565)
    unicode_58388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 565, 31), 'unicode', u'Could not open font file %s')
    # Getting the type of 'fpath' (line 565)
    fpath_58389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 63), 'fpath', False)
    # Applying the binary operator '%' (line 565)
    result_mod_58390 = python_operator(stypy.reporting.localization.Localization(__file__, 565, 31), '%', unicode_58388, fpath_58389)
    
    # Processing the call keyword arguments (line 565)
    kwargs_58391 = {}
    # Getting the type of 'verbose' (line 565)
    verbose_58386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 16), 'verbose', False)
    # Obtaining the member 'report' of a type (line 565)
    report_58387 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 565, 16), verbose_58386, 'report')
    # Calling report(args, kwargs) (line 565)
    report_call_result_58392 = invoke(stypy.reporting.localization.Localization(__file__, 565, 16), report_58387, *[result_mod_58390], **kwargs_58391)
    
    # SSA join for try-except statement (line 562)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Try-finally block (line 567)
    
    
    # SSA begins for try-except statement (line 567)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Name (line 568):
    
    # Assigning a Call to a Name (line 568):
    
    # Call to AFM(...): (line 568)
    # Processing the call arguments (line 568)
    # Getting the type of 'fh' (line 568)
    fh_58395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 31), 'fh', False)
    # Processing the call keyword arguments (line 568)
    kwargs_58396 = {}
    # Getting the type of 'afm' (line 568)
    afm_58393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 23), 'afm', False)
    # Obtaining the member 'AFM' of a type (line 568)
    AFM_58394 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 568, 23), afm_58393, 'AFM')
    # Calling AFM(args, kwargs) (line 568)
    AFM_call_result_58397 = invoke(stypy.reporting.localization.Localization(__file__, 568, 23), AFM_58394, *[fh_58395], **kwargs_58396)
    
    # Assigning a type to the variable 'font' (line 568)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 568, 16), 'font', AFM_call_result_58397)
    # SSA branch for the except part of a try statement (line 567)
    # SSA branch for the except 'RuntimeError' branch of a try statement (line 567)
    module_type_store.open_ssa_branch('except')
    
    # Call to report(...): (line 570)
    # Processing the call arguments (line 570)
    unicode_58400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 570, 31), 'unicode', u'Could not parse font file %s')
    # Getting the type of 'fpath' (line 570)
    fpath_58401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 64), 'fpath', False)
    # Applying the binary operator '%' (line 570)
    result_mod_58402 = python_operator(stypy.reporting.localization.Localization(__file__, 570, 31), '%', unicode_58400, fpath_58401)
    
    # Processing the call keyword arguments (line 570)
    kwargs_58403 = {}
    # Getting the type of 'verbose' (line 570)
    verbose_58398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 16), 'verbose', False)
    # Obtaining the member 'report' of a type (line 570)
    report_58399 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 570, 16), verbose_58398, 'report')
    # Calling report(args, kwargs) (line 570)
    report_call_result_58404 = invoke(stypy.reporting.localization.Localization(__file__, 570, 16), report_58399, *[result_mod_58402], **kwargs_58403)
    
    # SSA join for try-except statement (line 567)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # finally branch of the try-finally block (line 567)
    
    # Call to close(...): (line 573)
    # Processing the call keyword arguments (line 573)
    kwargs_58407 = {}
    # Getting the type of 'fh' (line 573)
    fh_58405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 16), 'fh', False)
    # Obtaining the member 'close' of a type (line 573)
    close_58406 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 573, 16), fh_58405, 'close')
    # Calling close(args, kwargs) (line 573)
    close_call_result_58408 = invoke(stypy.reporting.localization.Localization(__file__, 573, 16), close_58406, *[], **kwargs_58407)
    
    
    
    
    # SSA begins for try-except statement (line 574)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Name (line 575):
    
    # Assigning a Call to a Name (line 575):
    
    # Call to afmFontProperty(...): (line 575)
    # Processing the call arguments (line 575)
    # Getting the type of 'fpath' (line 575)
    fpath_58410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 39), 'fpath', False)
    # Getting the type of 'font' (line 575)
    font_58411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 46), 'font', False)
    # Processing the call keyword arguments (line 575)
    kwargs_58412 = {}
    # Getting the type of 'afmFontProperty' (line 575)
    afmFontProperty_58409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 23), 'afmFontProperty', False)
    # Calling afmFontProperty(args, kwargs) (line 575)
    afmFontProperty_call_result_58413 = invoke(stypy.reporting.localization.Localization(__file__, 575, 23), afmFontProperty_58409, *[fpath_58410, font_58411], **kwargs_58412)
    
    # Assigning a type to the variable 'prop' (line 575)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 575, 16), 'prop', afmFontProperty_call_result_58413)
    # SSA branch for the except part of a try statement (line 574)
    # SSA branch for the except 'KeyError' branch of a try statement (line 574)
    module_type_store.open_ssa_branch('except')
    # SSA join for try-except statement (line 574)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 561)
    module_type_store.open_ssa_branch('else')
    
    
    # SSA begins for try-except statement (line 579)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Name (line 580):
    
    # Assigning a Call to a Name (line 580):
    
    # Call to FT2Font(...): (line 580)
    # Processing the call arguments (line 580)
    # Getting the type of 'fpath' (line 580)
    fpath_58416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 39), 'fpath', False)
    # Processing the call keyword arguments (line 580)
    kwargs_58417 = {}
    # Getting the type of 'ft2font' (line 580)
    ft2font_58414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 23), 'ft2font', False)
    # Obtaining the member 'FT2Font' of a type (line 580)
    FT2Font_58415 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 580, 23), ft2font_58414, 'FT2Font')
    # Calling FT2Font(args, kwargs) (line 580)
    FT2Font_call_result_58418 = invoke(stypy.reporting.localization.Localization(__file__, 580, 23), FT2Font_58415, *[fpath_58416], **kwargs_58417)
    
    # Assigning a type to the variable 'font' (line 580)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 580, 16), 'font', FT2Font_call_result_58418)
    # SSA branch for the except part of a try statement (line 579)
    # SSA branch for the except 'RuntimeError' branch of a try statement (line 579)
    module_type_store.open_ssa_branch('except')
    
    # Call to report(...): (line 582)
    # Processing the call arguments (line 582)
    unicode_58421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 582, 31), 'unicode', u'Could not open font file %s')
    # Getting the type of 'fpath' (line 582)
    fpath_58422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 63), 'fpath', False)
    # Applying the binary operator '%' (line 582)
    result_mod_58423 = python_operator(stypy.reporting.localization.Localization(__file__, 582, 31), '%', unicode_58421, fpath_58422)
    
    # Processing the call keyword arguments (line 582)
    kwargs_58424 = {}
    # Getting the type of 'verbose' (line 582)
    verbose_58419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 16), 'verbose', False)
    # Obtaining the member 'report' of a type (line 582)
    report_58420 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 582, 16), verbose_58419, 'report')
    # Calling report(args, kwargs) (line 582)
    report_call_result_58425 = invoke(stypy.reporting.localization.Localization(__file__, 582, 16), report_58420, *[result_mod_58423], **kwargs_58424)
    
    # SSA branch for the except 'UnicodeError' branch of a try statement (line 579)
    module_type_store.open_ssa_branch('except')
    
    # Call to report(...): (line 585)
    # Processing the call arguments (line 585)
    unicode_58428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 585, 31), 'unicode', u'Cannot handle unicode filenames')
    # Processing the call keyword arguments (line 585)
    kwargs_58429 = {}
    # Getting the type of 'verbose' (line 585)
    verbose_58426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 16), 'verbose', False)
    # Obtaining the member 'report' of a type (line 585)
    report_58427 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 585, 16), verbose_58426, 'report')
    # Calling report(args, kwargs) (line 585)
    report_call_result_58430 = invoke(stypy.reporting.localization.Localization(__file__, 585, 16), report_58427, *[unicode_58428], **kwargs_58429)
    
    # SSA branch for the except 'IOError' branch of a try statement (line 579)
    module_type_store.open_ssa_branch('except')
    
    # Call to report(...): (line 589)
    # Processing the call arguments (line 589)
    unicode_58433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 589, 31), 'unicode', u'IO error - cannot open font file %s')
    # Getting the type of 'fpath' (line 589)
    fpath_58434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 71), 'fpath', False)
    # Applying the binary operator '%' (line 589)
    result_mod_58435 = python_operator(stypy.reporting.localization.Localization(__file__, 589, 31), '%', unicode_58433, fpath_58434)
    
    # Processing the call keyword arguments (line 589)
    kwargs_58436 = {}
    # Getting the type of 'verbose' (line 589)
    verbose_58431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 16), 'verbose', False)
    # Obtaining the member 'report' of a type (line 589)
    report_58432 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 589, 16), verbose_58431, 'report')
    # Calling report(args, kwargs) (line 589)
    report_call_result_58437 = invoke(stypy.reporting.localization.Localization(__file__, 589, 16), report_58432, *[result_mod_58435], **kwargs_58436)
    
    # SSA join for try-except statement (line 579)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # SSA begins for try-except statement (line 591)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Name (line 592):
    
    # Assigning a Call to a Name (line 592):
    
    # Call to ttfFontProperty(...): (line 592)
    # Processing the call arguments (line 592)
    # Getting the type of 'font' (line 592)
    font_58439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 39), 'font', False)
    # Processing the call keyword arguments (line 592)
    kwargs_58440 = {}
    # Getting the type of 'ttfFontProperty' (line 592)
    ttfFontProperty_58438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 23), 'ttfFontProperty', False)
    # Calling ttfFontProperty(args, kwargs) (line 592)
    ttfFontProperty_call_result_58441 = invoke(stypy.reporting.localization.Localization(__file__, 592, 23), ttfFontProperty_58438, *[font_58439], **kwargs_58440)
    
    # Assigning a type to the variable 'prop' (line 592)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 592, 16), 'prop', ttfFontProperty_call_result_58441)
    # SSA branch for the except part of a try statement (line 591)
    # SSA branch for the except 'Tuple' branch of a try statement (line 591)
    module_type_store.open_ssa_branch('except')
    # SSA join for try-except statement (line 591)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 561)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to append(...): (line 596)
    # Processing the call arguments (line 596)
    # Getting the type of 'prop' (line 596)
    prop_58444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 24), 'prop', False)
    # Processing the call keyword arguments (line 596)
    kwargs_58445 = {}
    # Getting the type of 'fontlist' (line 596)
    fontlist_58442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 8), 'fontlist', False)
    # Obtaining the member 'append' of a type (line 596)
    append_58443 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 596, 8), fontlist_58442, 'append')
    # Calling append(args, kwargs) (line 596)
    append_call_result_58446 = invoke(stypy.reporting.localization.Localization(__file__, 596, 8), append_58443, *[prop_58444], **kwargs_58445)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'fontlist' (line 597)
    fontlist_58447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 11), 'fontlist')
    # Assigning a type to the variable 'stypy_return_type' (line 597)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 597, 4), 'stypy_return_type', fontlist_58447)
    
    # ################# End of 'createFontList(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'createFontList' in the type store
    # Getting the type of 'stypy_return_type' (line 544)
    stypy_return_type_58448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_58448)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'createFontList'
    return stypy_return_type_58448

# Assigning a type to the variable 'createFontList' (line 544)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 544, 0), 'createFontList', createFontList)
# Declaration of the 'FontProperties' class

class FontProperties(object, ):
    unicode_58449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 657, (-1)), 'unicode', u"\n    A class for storing and manipulating font properties.\n\n    The font properties are those described in the `W3C Cascading\n    Style Sheet, Level 1\n    <http://www.w3.org/TR/1998/REC-CSS2-19980512/>`_ font\n    specification.  The six properties are:\n\n      - family: A list of font names in decreasing order of priority.\n        The items may include a generic font family name, either\n        'serif', 'sans-serif', 'cursive', 'fantasy', or 'monospace'.\n        In that case, the actual font to be used will be looked up\n        from the associated rcParam in :file:`matplotlibrc`.\n\n      - style: Either 'normal', 'italic' or 'oblique'.\n\n      - variant: Either 'normal' or 'small-caps'.\n\n      - stretch: A numeric value in the range 0-1000 or one of\n        'ultra-condensed', 'extra-condensed', 'condensed',\n        'semi-condensed', 'normal', 'semi-expanded', 'expanded',\n        'extra-expanded' or 'ultra-expanded'\n\n      - weight: A numeric value in the range 0-1000 or one of\n        'ultralight', 'light', 'normal', 'regular', 'book', 'medium',\n        'roman', 'semibold', 'demibold', 'demi', 'bold', 'heavy',\n        'extra bold', 'black'\n\n      - size: Either an relative value of 'xx-small', 'x-small',\n        'small', 'medium', 'large', 'x-large', 'xx-large' or an\n        absolute font size, e.g., 12\n\n    The default font property for TrueType fonts (as specified in the\n    default :file:`matplotlibrc` file) is::\n\n      sans-serif, normal, normal, normal, normal, scalable.\n\n    Alternatively, a font may be specified using an absolute path to a\n    .ttf file, by using the *fname* kwarg.\n\n    The preferred usage of font sizes is to use the relative values,\n    e.g.,  'large', instead of absolute font sizes, e.g., 12.  This\n    approach allows all text sizes to be made larger or smaller based\n    on the font manager's default font size.\n\n    This class will also accept a `fontconfig\n    <https://www.freedesktop.org/wiki/Software/fontconfig/>`_ pattern, if it is\n    the only argument provided.  See the documentation on `fontconfig patterns\n    <https://www.freedesktop.org/software/fontconfig/fontconfig-user.html>`_.\n    This support does not require fontconfig to be installed.  We are merely\n    borrowing its pattern syntax for use here.\n\n    Note that matplotlib's internal font manager and fontconfig use a\n    different algorithm to lookup fonts, so the results of the same pattern\n    may be different in matplotlib than in other applications that use\n    fontconfig.\n    ")

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 660)
        None_58450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 26), 'None')
        # Getting the type of 'None' (line 661)
        None_58451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 26), 'None')
        # Getting the type of 'None' (line 662)
        None_58452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 26), 'None')
        # Getting the type of 'None' (line 663)
        None_58453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 26), 'None')
        # Getting the type of 'None' (line 664)
        None_58454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 664, 26), 'None')
        # Getting the type of 'None' (line 665)
        None_58455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 665, 26), 'None')
        # Getting the type of 'None' (line 666)
        None_58456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 26), 'None')
        # Getting the type of 'None' (line 667)
        None_58457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 27), 'None')
        defaults = [None_58450, None_58451, None_58452, None_58453, None_58454, None_58455, None_58456, None_58457]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 659, 4, False)
        # Assigning a type to the variable 'self' (line 660)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 660, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FontProperties.__init__', ['family', 'style', 'variant', 'weight', 'stretch', 'size', 'fname', '_init'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['family', 'style', 'variant', 'weight', 'stretch', 'size', 'fname', '_init'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Call to a Attribute (line 669):
        
        # Assigning a Call to a Attribute (line 669):
        
        # Call to _normalize_font_family(...): (line 669)
        # Processing the call arguments (line 669)
        
        # Obtaining the type of the subscript
        unicode_58459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 669, 55), 'unicode', u'font.family')
        # Getting the type of 'rcParams' (line 669)
        rcParams_58460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 669, 46), 'rcParams', False)
        # Obtaining the member '__getitem__' of a type (line 669)
        getitem___58461 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 669, 46), rcParams_58460, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 669)
        subscript_call_result_58462 = invoke(stypy.reporting.localization.Localization(__file__, 669, 46), getitem___58461, unicode_58459)
        
        # Processing the call keyword arguments (line 669)
        kwargs_58463 = {}
        # Getting the type of '_normalize_font_family' (line 669)
        _normalize_font_family_58458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 669, 23), '_normalize_font_family', False)
        # Calling _normalize_font_family(args, kwargs) (line 669)
        _normalize_font_family_call_result_58464 = invoke(stypy.reporting.localization.Localization(__file__, 669, 23), _normalize_font_family_58458, *[subscript_call_result_58462], **kwargs_58463)
        
        # Getting the type of 'self' (line 669)
        self_58465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 669, 8), 'self')
        # Setting the type of the member '_family' of a type (line 669)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 669, 8), self_58465, '_family', _normalize_font_family_call_result_58464)
        
        # Assigning a Subscript to a Attribute (line 670):
        
        # Assigning a Subscript to a Attribute (line 670):
        
        # Obtaining the type of the subscript
        unicode_58466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 670, 31), 'unicode', u'font.style')
        # Getting the type of 'rcParams' (line 670)
        rcParams_58467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 670, 22), 'rcParams')
        # Obtaining the member '__getitem__' of a type (line 670)
        getitem___58468 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 670, 22), rcParams_58467, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 670)
        subscript_call_result_58469 = invoke(stypy.reporting.localization.Localization(__file__, 670, 22), getitem___58468, unicode_58466)
        
        # Getting the type of 'self' (line 670)
        self_58470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 670, 8), 'self')
        # Setting the type of the member '_slant' of a type (line 670)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 670, 8), self_58470, '_slant', subscript_call_result_58469)
        
        # Assigning a Subscript to a Attribute (line 671):
        
        # Assigning a Subscript to a Attribute (line 671):
        
        # Obtaining the type of the subscript
        unicode_58471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 671, 33), 'unicode', u'font.variant')
        # Getting the type of 'rcParams' (line 671)
        rcParams_58472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 671, 24), 'rcParams')
        # Obtaining the member '__getitem__' of a type (line 671)
        getitem___58473 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 671, 24), rcParams_58472, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 671)
        subscript_call_result_58474 = invoke(stypy.reporting.localization.Localization(__file__, 671, 24), getitem___58473, unicode_58471)
        
        # Getting the type of 'self' (line 671)
        self_58475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 671, 8), 'self')
        # Setting the type of the member '_variant' of a type (line 671)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 671, 8), self_58475, '_variant', subscript_call_result_58474)
        
        # Assigning a Subscript to a Attribute (line 672):
        
        # Assigning a Subscript to a Attribute (line 672):
        
        # Obtaining the type of the subscript
        unicode_58476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 672, 32), 'unicode', u'font.weight')
        # Getting the type of 'rcParams' (line 672)
        rcParams_58477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 672, 23), 'rcParams')
        # Obtaining the member '__getitem__' of a type (line 672)
        getitem___58478 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 672, 23), rcParams_58477, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 672)
        subscript_call_result_58479 = invoke(stypy.reporting.localization.Localization(__file__, 672, 23), getitem___58478, unicode_58476)
        
        # Getting the type of 'self' (line 672)
        self_58480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 672, 8), 'self')
        # Setting the type of the member '_weight' of a type (line 672)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 672, 8), self_58480, '_weight', subscript_call_result_58479)
        
        # Assigning a Subscript to a Attribute (line 673):
        
        # Assigning a Subscript to a Attribute (line 673):
        
        # Obtaining the type of the subscript
        unicode_58481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 673, 33), 'unicode', u'font.stretch')
        # Getting the type of 'rcParams' (line 673)
        rcParams_58482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 673, 24), 'rcParams')
        # Obtaining the member '__getitem__' of a type (line 673)
        getitem___58483 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 673, 24), rcParams_58482, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 673)
        subscript_call_result_58484 = invoke(stypy.reporting.localization.Localization(__file__, 673, 24), getitem___58483, unicode_58481)
        
        # Getting the type of 'self' (line 673)
        self_58485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 673, 8), 'self')
        # Setting the type of the member '_stretch' of a type (line 673)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 673, 8), self_58485, '_stretch', subscript_call_result_58484)
        
        # Assigning a Subscript to a Attribute (line 674):
        
        # Assigning a Subscript to a Attribute (line 674):
        
        # Obtaining the type of the subscript
        unicode_58486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 674, 30), 'unicode', u'font.size')
        # Getting the type of 'rcParams' (line 674)
        rcParams_58487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 674, 21), 'rcParams')
        # Obtaining the member '__getitem__' of a type (line 674)
        getitem___58488 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 674, 21), rcParams_58487, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 674)
        subscript_call_result_58489 = invoke(stypy.reporting.localization.Localization(__file__, 674, 21), getitem___58488, unicode_58486)
        
        # Getting the type of 'self' (line 674)
        self_58490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 674, 8), 'self')
        # Setting the type of the member '_size' of a type (line 674)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 674, 8), self_58490, '_size', subscript_call_result_58489)
        
        # Assigning a Name to a Attribute (line 675):
        
        # Assigning a Name to a Attribute (line 675):
        # Getting the type of 'None' (line 675)
        None_58491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 21), 'None')
        # Getting the type of 'self' (line 675)
        self_58492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 8), 'self')
        # Setting the type of the member '_file' of a type (line 675)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 675, 8), self_58492, '_file', None_58491)
        
        # Type idiom detected: calculating its left and rigth part (line 678)
        # Getting the type of '_init' (line 678)
        _init_58493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 678, 8), '_init')
        # Getting the type of 'None' (line 678)
        None_58494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 678, 24), 'None')
        
        (may_be_58495, more_types_in_union_58496) = may_not_be_none(_init_58493, None_58494)

        if may_be_58495:

            if more_types_in_union_58496:
                # Runtime conditional SSA (line 678)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to update(...): (line 679)
            # Processing the call arguments (line 679)
            # Getting the type of '_init' (line 679)
            _init_58500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 679, 33), '_init', False)
            # Obtaining the member '__dict__' of a type (line 679)
            dict___58501 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 679, 33), _init_58500, '__dict__')
            # Processing the call keyword arguments (line 679)
            kwargs_58502 = {}
            # Getting the type of 'self' (line 679)
            self_58497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 679, 12), 'self', False)
            # Obtaining the member '__dict__' of a type (line 679)
            dict___58498 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 679, 12), self_58497, '__dict__')
            # Obtaining the member 'update' of a type (line 679)
            update_58499 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 679, 12), dict___58498, 'update')
            # Calling update(args, kwargs) (line 679)
            update_call_result_58503 = invoke(stypy.reporting.localization.Localization(__file__, 679, 12), update_58499, *[dict___58501], **kwargs_58502)
            
            # Assigning a type to the variable 'stypy_return_type' (line 680)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 680, 12), 'stypy_return_type', types.NoneType)

            if more_types_in_union_58496:
                # SSA join for if statement (line 678)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        # Call to isinstance(...): (line 682)
        # Processing the call arguments (line 682)
        # Getting the type of 'family' (line 682)
        family_58505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 682, 22), 'family', False)
        # Getting the type of 'six' (line 682)
        six_58506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 682, 30), 'six', False)
        # Obtaining the member 'string_types' of a type (line 682)
        string_types_58507 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 682, 30), six_58506, 'string_types')
        # Processing the call keyword arguments (line 682)
        kwargs_58508 = {}
        # Getting the type of 'isinstance' (line 682)
        isinstance_58504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 682, 11), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 682)
        isinstance_call_result_58509 = invoke(stypy.reporting.localization.Localization(__file__, 682, 11), isinstance_58504, *[family_58505, string_types_58507], **kwargs_58508)
        
        # Testing the type of an if condition (line 682)
        if_condition_58510 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 682, 8), isinstance_call_result_58509)
        # Assigning a type to the variable 'if_condition_58510' (line 682)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 682, 8), 'if_condition_58510', if_condition_58510)
        # SSA begins for if statement (line 682)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'style' (line 685)
        style_58511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 685, 16), 'style')
        # Getting the type of 'None' (line 685)
        None_58512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 685, 25), 'None')
        # Applying the binary operator 'is' (line 685)
        result_is__58513 = python_operator(stypy.reporting.localization.Localization(__file__, 685, 16), 'is', style_58511, None_58512)
        
        
        # Getting the type of 'variant' (line 686)
        variant_58514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 686, 16), 'variant')
        # Getting the type of 'None' (line 686)
        None_58515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 686, 27), 'None')
        # Applying the binary operator 'is' (line 686)
        result_is__58516 = python_operator(stypy.reporting.localization.Localization(__file__, 686, 16), 'is', variant_58514, None_58515)
        
        # Applying the binary operator 'and' (line 685)
        result_and_keyword_58517 = python_operator(stypy.reporting.localization.Localization(__file__, 685, 16), 'and', result_is__58513, result_is__58516)
        
        # Getting the type of 'weight' (line 687)
        weight_58518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 687, 16), 'weight')
        # Getting the type of 'None' (line 687)
        None_58519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 687, 26), 'None')
        # Applying the binary operator 'is' (line 687)
        result_is__58520 = python_operator(stypy.reporting.localization.Localization(__file__, 687, 16), 'is', weight_58518, None_58519)
        
        # Applying the binary operator 'and' (line 685)
        result_and_keyword_58521 = python_operator(stypy.reporting.localization.Localization(__file__, 685, 16), 'and', result_and_keyword_58517, result_is__58520)
        
        # Getting the type of 'stretch' (line 688)
        stretch_58522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 688, 16), 'stretch')
        # Getting the type of 'None' (line 688)
        None_58523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 688, 27), 'None')
        # Applying the binary operator 'is' (line 688)
        result_is__58524 = python_operator(stypy.reporting.localization.Localization(__file__, 688, 16), 'is', stretch_58522, None_58523)
        
        # Applying the binary operator 'and' (line 685)
        result_and_keyword_58525 = python_operator(stypy.reporting.localization.Localization(__file__, 685, 16), 'and', result_and_keyword_58521, result_is__58524)
        
        # Getting the type of 'size' (line 689)
        size_58526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 16), 'size')
        # Getting the type of 'None' (line 689)
        None_58527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 24), 'None')
        # Applying the binary operator 'is' (line 689)
        result_is__58528 = python_operator(stypy.reporting.localization.Localization(__file__, 689, 16), 'is', size_58526, None_58527)
        
        # Applying the binary operator 'and' (line 685)
        result_and_keyword_58529 = python_operator(stypy.reporting.localization.Localization(__file__, 685, 16), 'and', result_and_keyword_58525, result_is__58528)
        
        # Getting the type of 'fname' (line 690)
        fname_58530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 16), 'fname')
        # Getting the type of 'None' (line 690)
        None_58531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 25), 'None')
        # Applying the binary operator 'is' (line 690)
        result_is__58532 = python_operator(stypy.reporting.localization.Localization(__file__, 690, 16), 'is', fname_58530, None_58531)
        
        # Applying the binary operator 'and' (line 685)
        result_and_keyword_58533 = python_operator(stypy.reporting.localization.Localization(__file__, 685, 16), 'and', result_and_keyword_58529, result_is__58532)
        
        # Testing the type of an if condition (line 685)
        if_condition_58534 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 685, 12), result_and_keyword_58533)
        # Assigning a type to the variable 'if_condition_58534' (line 685)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 685, 12), 'if_condition_58534', if_condition_58534)
        # SSA begins for if statement (line 685)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to set_fontconfig_pattern(...): (line 691)
        # Processing the call arguments (line 691)
        # Getting the type of 'family' (line 691)
        family_58537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 691, 44), 'family', False)
        # Processing the call keyword arguments (line 691)
        kwargs_58538 = {}
        # Getting the type of 'self' (line 691)
        self_58535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 691, 16), 'self', False)
        # Obtaining the member 'set_fontconfig_pattern' of a type (line 691)
        set_fontconfig_pattern_58536 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 691, 16), self_58535, 'set_fontconfig_pattern')
        # Calling set_fontconfig_pattern(args, kwargs) (line 691)
        set_fontconfig_pattern_call_result_58539 = invoke(stypy.reporting.localization.Localization(__file__, 691, 16), set_fontconfig_pattern_58536, *[family_58537], **kwargs_58538)
        
        # Assigning a type to the variable 'stypy_return_type' (line 692)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 692, 16), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 685)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 682)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to set_family(...): (line 694)
        # Processing the call arguments (line 694)
        # Getting the type of 'family' (line 694)
        family_58542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 24), 'family', False)
        # Processing the call keyword arguments (line 694)
        kwargs_58543 = {}
        # Getting the type of 'self' (line 694)
        self_58540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 8), 'self', False)
        # Obtaining the member 'set_family' of a type (line 694)
        set_family_58541 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 694, 8), self_58540, 'set_family')
        # Calling set_family(args, kwargs) (line 694)
        set_family_call_result_58544 = invoke(stypy.reporting.localization.Localization(__file__, 694, 8), set_family_58541, *[family_58542], **kwargs_58543)
        
        
        # Call to set_style(...): (line 695)
        # Processing the call arguments (line 695)
        # Getting the type of 'style' (line 695)
        style_58547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 23), 'style', False)
        # Processing the call keyword arguments (line 695)
        kwargs_58548 = {}
        # Getting the type of 'self' (line 695)
        self_58545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 8), 'self', False)
        # Obtaining the member 'set_style' of a type (line 695)
        set_style_58546 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 695, 8), self_58545, 'set_style')
        # Calling set_style(args, kwargs) (line 695)
        set_style_call_result_58549 = invoke(stypy.reporting.localization.Localization(__file__, 695, 8), set_style_58546, *[style_58547], **kwargs_58548)
        
        
        # Call to set_variant(...): (line 696)
        # Processing the call arguments (line 696)
        # Getting the type of 'variant' (line 696)
        variant_58552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 696, 25), 'variant', False)
        # Processing the call keyword arguments (line 696)
        kwargs_58553 = {}
        # Getting the type of 'self' (line 696)
        self_58550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 696, 8), 'self', False)
        # Obtaining the member 'set_variant' of a type (line 696)
        set_variant_58551 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 696, 8), self_58550, 'set_variant')
        # Calling set_variant(args, kwargs) (line 696)
        set_variant_call_result_58554 = invoke(stypy.reporting.localization.Localization(__file__, 696, 8), set_variant_58551, *[variant_58552], **kwargs_58553)
        
        
        # Call to set_weight(...): (line 697)
        # Processing the call arguments (line 697)
        # Getting the type of 'weight' (line 697)
        weight_58557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 24), 'weight', False)
        # Processing the call keyword arguments (line 697)
        kwargs_58558 = {}
        # Getting the type of 'self' (line 697)
        self_58555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 8), 'self', False)
        # Obtaining the member 'set_weight' of a type (line 697)
        set_weight_58556 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 697, 8), self_58555, 'set_weight')
        # Calling set_weight(args, kwargs) (line 697)
        set_weight_call_result_58559 = invoke(stypy.reporting.localization.Localization(__file__, 697, 8), set_weight_58556, *[weight_58557], **kwargs_58558)
        
        
        # Call to set_stretch(...): (line 698)
        # Processing the call arguments (line 698)
        # Getting the type of 'stretch' (line 698)
        stretch_58562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 25), 'stretch', False)
        # Processing the call keyword arguments (line 698)
        kwargs_58563 = {}
        # Getting the type of 'self' (line 698)
        self_58560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 8), 'self', False)
        # Obtaining the member 'set_stretch' of a type (line 698)
        set_stretch_58561 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 698, 8), self_58560, 'set_stretch')
        # Calling set_stretch(args, kwargs) (line 698)
        set_stretch_call_result_58564 = invoke(stypy.reporting.localization.Localization(__file__, 698, 8), set_stretch_58561, *[stretch_58562], **kwargs_58563)
        
        
        # Call to set_file(...): (line 699)
        # Processing the call arguments (line 699)
        # Getting the type of 'fname' (line 699)
        fname_58567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 699, 22), 'fname', False)
        # Processing the call keyword arguments (line 699)
        kwargs_58568 = {}
        # Getting the type of 'self' (line 699)
        self_58565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 699, 8), 'self', False)
        # Obtaining the member 'set_file' of a type (line 699)
        set_file_58566 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 699, 8), self_58565, 'set_file')
        # Calling set_file(args, kwargs) (line 699)
        set_file_call_result_58569 = invoke(stypy.reporting.localization.Localization(__file__, 699, 8), set_file_58566, *[fname_58567], **kwargs_58568)
        
        
        # Call to set_size(...): (line 700)
        # Processing the call arguments (line 700)
        # Getting the type of 'size' (line 700)
        size_58572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 700, 22), 'size', False)
        # Processing the call keyword arguments (line 700)
        kwargs_58573 = {}
        # Getting the type of 'self' (line 700)
        self_58570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 700, 8), 'self', False)
        # Obtaining the member 'set_size' of a type (line 700)
        set_size_58571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 700, 8), self_58570, 'set_size')
        # Calling set_size(args, kwargs) (line 700)
        set_size_call_result_58574 = invoke(stypy.reporting.localization.Localization(__file__, 700, 8), set_size_58571, *[size_58572], **kwargs_58573)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def _parse_fontconfig_pattern(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_parse_fontconfig_pattern'
        module_type_store = module_type_store.open_function_context('_parse_fontconfig_pattern', 702, 4, False)
        # Assigning a type to the variable 'self' (line 703)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 703, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FontProperties._parse_fontconfig_pattern.__dict__.__setitem__('stypy_localization', localization)
        FontProperties._parse_fontconfig_pattern.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FontProperties._parse_fontconfig_pattern.__dict__.__setitem__('stypy_type_store', module_type_store)
        FontProperties._parse_fontconfig_pattern.__dict__.__setitem__('stypy_function_name', 'FontProperties._parse_fontconfig_pattern')
        FontProperties._parse_fontconfig_pattern.__dict__.__setitem__('stypy_param_names_list', ['pattern'])
        FontProperties._parse_fontconfig_pattern.__dict__.__setitem__('stypy_varargs_param_name', None)
        FontProperties._parse_fontconfig_pattern.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FontProperties._parse_fontconfig_pattern.__dict__.__setitem__('stypy_call_defaults', defaults)
        FontProperties._parse_fontconfig_pattern.__dict__.__setitem__('stypy_call_varargs', varargs)
        FontProperties._parse_fontconfig_pattern.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FontProperties._parse_fontconfig_pattern.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FontProperties._parse_fontconfig_pattern', ['pattern'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_parse_fontconfig_pattern', localization, ['pattern'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_parse_fontconfig_pattern(...)' code ##################

        
        # Call to parse_fontconfig_pattern(...): (line 703)
        # Processing the call arguments (line 703)
        # Getting the type of 'pattern' (line 703)
        pattern_58576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 703, 40), 'pattern', False)
        # Processing the call keyword arguments (line 703)
        kwargs_58577 = {}
        # Getting the type of 'parse_fontconfig_pattern' (line 703)
        parse_fontconfig_pattern_58575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 703, 15), 'parse_fontconfig_pattern', False)
        # Calling parse_fontconfig_pattern(args, kwargs) (line 703)
        parse_fontconfig_pattern_call_result_58578 = invoke(stypy.reporting.localization.Localization(__file__, 703, 15), parse_fontconfig_pattern_58575, *[pattern_58576], **kwargs_58577)
        
        # Assigning a type to the variable 'stypy_return_type' (line 703)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 703, 8), 'stypy_return_type', parse_fontconfig_pattern_call_result_58578)
        
        # ################# End of '_parse_fontconfig_pattern(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_parse_fontconfig_pattern' in the type store
        # Getting the type of 'stypy_return_type' (line 702)
        stypy_return_type_58579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_58579)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_parse_fontconfig_pattern'
        return stypy_return_type_58579


    @norecursion
    def stypy__hash__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__hash__'
        module_type_store = module_type_store.open_function_context('__hash__', 705, 4, False)
        # Assigning a type to the variable 'self' (line 706)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 706, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FontProperties.stypy__hash__.__dict__.__setitem__('stypy_localization', localization)
        FontProperties.stypy__hash__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FontProperties.stypy__hash__.__dict__.__setitem__('stypy_type_store', module_type_store)
        FontProperties.stypy__hash__.__dict__.__setitem__('stypy_function_name', 'FontProperties.stypy__hash__')
        FontProperties.stypy__hash__.__dict__.__setitem__('stypy_param_names_list', [])
        FontProperties.stypy__hash__.__dict__.__setitem__('stypy_varargs_param_name', None)
        FontProperties.stypy__hash__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FontProperties.stypy__hash__.__dict__.__setitem__('stypy_call_defaults', defaults)
        FontProperties.stypy__hash__.__dict__.__setitem__('stypy_call_varargs', varargs)
        FontProperties.stypy__hash__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FontProperties.stypy__hash__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FontProperties.stypy__hash__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__hash__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__hash__(...)' code ##################

        
        # Assigning a Tuple to a Name (line 706):
        
        # Assigning a Tuple to a Name (line 706):
        
        # Obtaining an instance of the builtin type 'tuple' (line 706)
        tuple_58580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 706, 13), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 706)
        # Adding element type (line 706)
        
        # Call to tuple(...): (line 706)
        # Processing the call arguments (line 706)
        
        # Call to get_family(...): (line 706)
        # Processing the call keyword arguments (line 706)
        kwargs_58584 = {}
        # Getting the type of 'self' (line 706)
        self_58582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 706, 19), 'self', False)
        # Obtaining the member 'get_family' of a type (line 706)
        get_family_58583 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 706, 19), self_58582, 'get_family')
        # Calling get_family(args, kwargs) (line 706)
        get_family_call_result_58585 = invoke(stypy.reporting.localization.Localization(__file__, 706, 19), get_family_58583, *[], **kwargs_58584)
        
        # Processing the call keyword arguments (line 706)
        kwargs_58586 = {}
        # Getting the type of 'tuple' (line 706)
        tuple_58581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 706, 13), 'tuple', False)
        # Calling tuple(args, kwargs) (line 706)
        tuple_call_result_58587 = invoke(stypy.reporting.localization.Localization(__file__, 706, 13), tuple_58581, *[get_family_call_result_58585], **kwargs_58586)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 706, 13), tuple_58580, tuple_call_result_58587)
        # Adding element type (line 706)
        
        # Call to get_slant(...): (line 707)
        # Processing the call keyword arguments (line 707)
        kwargs_58590 = {}
        # Getting the type of 'self' (line 707)
        self_58588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 707, 13), 'self', False)
        # Obtaining the member 'get_slant' of a type (line 707)
        get_slant_58589 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 707, 13), self_58588, 'get_slant')
        # Calling get_slant(args, kwargs) (line 707)
        get_slant_call_result_58591 = invoke(stypy.reporting.localization.Localization(__file__, 707, 13), get_slant_58589, *[], **kwargs_58590)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 706, 13), tuple_58580, get_slant_call_result_58591)
        # Adding element type (line 706)
        
        # Call to get_variant(...): (line 708)
        # Processing the call keyword arguments (line 708)
        kwargs_58594 = {}
        # Getting the type of 'self' (line 708)
        self_58592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 708, 13), 'self', False)
        # Obtaining the member 'get_variant' of a type (line 708)
        get_variant_58593 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 708, 13), self_58592, 'get_variant')
        # Calling get_variant(args, kwargs) (line 708)
        get_variant_call_result_58595 = invoke(stypy.reporting.localization.Localization(__file__, 708, 13), get_variant_58593, *[], **kwargs_58594)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 706, 13), tuple_58580, get_variant_call_result_58595)
        # Adding element type (line 706)
        
        # Call to get_weight(...): (line 709)
        # Processing the call keyword arguments (line 709)
        kwargs_58598 = {}
        # Getting the type of 'self' (line 709)
        self_58596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 709, 13), 'self', False)
        # Obtaining the member 'get_weight' of a type (line 709)
        get_weight_58597 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 709, 13), self_58596, 'get_weight')
        # Calling get_weight(args, kwargs) (line 709)
        get_weight_call_result_58599 = invoke(stypy.reporting.localization.Localization(__file__, 709, 13), get_weight_58597, *[], **kwargs_58598)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 706, 13), tuple_58580, get_weight_call_result_58599)
        # Adding element type (line 706)
        
        # Call to get_stretch(...): (line 710)
        # Processing the call keyword arguments (line 710)
        kwargs_58602 = {}
        # Getting the type of 'self' (line 710)
        self_58600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 710, 13), 'self', False)
        # Obtaining the member 'get_stretch' of a type (line 710)
        get_stretch_58601 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 710, 13), self_58600, 'get_stretch')
        # Calling get_stretch(args, kwargs) (line 710)
        get_stretch_call_result_58603 = invoke(stypy.reporting.localization.Localization(__file__, 710, 13), get_stretch_58601, *[], **kwargs_58602)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 706, 13), tuple_58580, get_stretch_call_result_58603)
        # Adding element type (line 706)
        
        # Call to get_size_in_points(...): (line 711)
        # Processing the call keyword arguments (line 711)
        kwargs_58606 = {}
        # Getting the type of 'self' (line 711)
        self_58604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 711, 13), 'self', False)
        # Obtaining the member 'get_size_in_points' of a type (line 711)
        get_size_in_points_58605 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 711, 13), self_58604, 'get_size_in_points')
        # Calling get_size_in_points(args, kwargs) (line 711)
        get_size_in_points_call_result_58607 = invoke(stypy.reporting.localization.Localization(__file__, 711, 13), get_size_in_points_58605, *[], **kwargs_58606)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 706, 13), tuple_58580, get_size_in_points_call_result_58607)
        # Adding element type (line 706)
        
        # Call to get_file(...): (line 712)
        # Processing the call keyword arguments (line 712)
        kwargs_58610 = {}
        # Getting the type of 'self' (line 712)
        self_58608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 712, 13), 'self', False)
        # Obtaining the member 'get_file' of a type (line 712)
        get_file_58609 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 712, 13), self_58608, 'get_file')
        # Calling get_file(args, kwargs) (line 712)
        get_file_call_result_58611 = invoke(stypy.reporting.localization.Localization(__file__, 712, 13), get_file_58609, *[], **kwargs_58610)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 706, 13), tuple_58580, get_file_call_result_58611)
        
        # Assigning a type to the variable 'l' (line 706)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 706, 8), 'l', tuple_58580)
        
        # Call to hash(...): (line 713)
        # Processing the call arguments (line 713)
        # Getting the type of 'l' (line 713)
        l_58613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 713, 20), 'l', False)
        # Processing the call keyword arguments (line 713)
        kwargs_58614 = {}
        # Getting the type of 'hash' (line 713)
        hash_58612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 713, 15), 'hash', False)
        # Calling hash(args, kwargs) (line 713)
        hash_call_result_58615 = invoke(stypy.reporting.localization.Localization(__file__, 713, 15), hash_58612, *[l_58613], **kwargs_58614)
        
        # Assigning a type to the variable 'stypy_return_type' (line 713)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 713, 8), 'stypy_return_type', hash_call_result_58615)
        
        # ################# End of '__hash__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__hash__' in the type store
        # Getting the type of 'stypy_return_type' (line 705)
        stypy_return_type_58616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_58616)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__hash__'
        return stypy_return_type_58616


    @norecursion
    def stypy__eq__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__eq__'
        module_type_store = module_type_store.open_function_context('__eq__', 715, 4, False)
        # Assigning a type to the variable 'self' (line 716)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 716, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FontProperties.stypy__eq__.__dict__.__setitem__('stypy_localization', localization)
        FontProperties.stypy__eq__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FontProperties.stypy__eq__.__dict__.__setitem__('stypy_type_store', module_type_store)
        FontProperties.stypy__eq__.__dict__.__setitem__('stypy_function_name', 'FontProperties.stypy__eq__')
        FontProperties.stypy__eq__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        FontProperties.stypy__eq__.__dict__.__setitem__('stypy_varargs_param_name', None)
        FontProperties.stypy__eq__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FontProperties.stypy__eq__.__dict__.__setitem__('stypy_call_defaults', defaults)
        FontProperties.stypy__eq__.__dict__.__setitem__('stypy_call_varargs', varargs)
        FontProperties.stypy__eq__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FontProperties.stypy__eq__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FontProperties.stypy__eq__', ['other'], None, None, defaults, varargs, kwargs)

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

        
        
        # Call to hash(...): (line 716)
        # Processing the call arguments (line 716)
        # Getting the type of 'self' (line 716)
        self_58618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 716, 20), 'self', False)
        # Processing the call keyword arguments (line 716)
        kwargs_58619 = {}
        # Getting the type of 'hash' (line 716)
        hash_58617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 716, 15), 'hash', False)
        # Calling hash(args, kwargs) (line 716)
        hash_call_result_58620 = invoke(stypy.reporting.localization.Localization(__file__, 716, 15), hash_58617, *[self_58618], **kwargs_58619)
        
        
        # Call to hash(...): (line 716)
        # Processing the call arguments (line 716)
        # Getting the type of 'other' (line 716)
        other_58622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 716, 34), 'other', False)
        # Processing the call keyword arguments (line 716)
        kwargs_58623 = {}
        # Getting the type of 'hash' (line 716)
        hash_58621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 716, 29), 'hash', False)
        # Calling hash(args, kwargs) (line 716)
        hash_call_result_58624 = invoke(stypy.reporting.localization.Localization(__file__, 716, 29), hash_58621, *[other_58622], **kwargs_58623)
        
        # Applying the binary operator '==' (line 716)
        result_eq_58625 = python_operator(stypy.reporting.localization.Localization(__file__, 716, 15), '==', hash_call_result_58620, hash_call_result_58624)
        
        # Assigning a type to the variable 'stypy_return_type' (line 716)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 716, 8), 'stypy_return_type', result_eq_58625)
        
        # ################# End of '__eq__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__eq__' in the type store
        # Getting the type of 'stypy_return_type' (line 715)
        stypy_return_type_58626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 715, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_58626)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__eq__'
        return stypy_return_type_58626


    @norecursion
    def __ne__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__ne__'
        module_type_store = module_type_store.open_function_context('__ne__', 718, 4, False)
        # Assigning a type to the variable 'self' (line 719)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 719, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FontProperties.__ne__.__dict__.__setitem__('stypy_localization', localization)
        FontProperties.__ne__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FontProperties.__ne__.__dict__.__setitem__('stypy_type_store', module_type_store)
        FontProperties.__ne__.__dict__.__setitem__('stypy_function_name', 'FontProperties.__ne__')
        FontProperties.__ne__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        FontProperties.__ne__.__dict__.__setitem__('stypy_varargs_param_name', None)
        FontProperties.__ne__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FontProperties.__ne__.__dict__.__setitem__('stypy_call_defaults', defaults)
        FontProperties.__ne__.__dict__.__setitem__('stypy_call_varargs', varargs)
        FontProperties.__ne__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FontProperties.__ne__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FontProperties.__ne__', ['other'], None, None, defaults, varargs, kwargs)

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

        
        
        # Call to hash(...): (line 719)
        # Processing the call arguments (line 719)
        # Getting the type of 'self' (line 719)
        self_58628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 719, 20), 'self', False)
        # Processing the call keyword arguments (line 719)
        kwargs_58629 = {}
        # Getting the type of 'hash' (line 719)
        hash_58627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 719, 15), 'hash', False)
        # Calling hash(args, kwargs) (line 719)
        hash_call_result_58630 = invoke(stypy.reporting.localization.Localization(__file__, 719, 15), hash_58627, *[self_58628], **kwargs_58629)
        
        
        # Call to hash(...): (line 719)
        # Processing the call arguments (line 719)
        # Getting the type of 'other' (line 719)
        other_58632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 719, 34), 'other', False)
        # Processing the call keyword arguments (line 719)
        kwargs_58633 = {}
        # Getting the type of 'hash' (line 719)
        hash_58631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 719, 29), 'hash', False)
        # Calling hash(args, kwargs) (line 719)
        hash_call_result_58634 = invoke(stypy.reporting.localization.Localization(__file__, 719, 29), hash_58631, *[other_58632], **kwargs_58633)
        
        # Applying the binary operator '!=' (line 719)
        result_ne_58635 = python_operator(stypy.reporting.localization.Localization(__file__, 719, 15), '!=', hash_call_result_58630, hash_call_result_58634)
        
        # Assigning a type to the variable 'stypy_return_type' (line 719)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 719, 8), 'stypy_return_type', result_ne_58635)
        
        # ################# End of '__ne__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__ne__' in the type store
        # Getting the type of 'stypy_return_type' (line 718)
        stypy_return_type_58636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_58636)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__ne__'
        return stypy_return_type_58636


    @norecursion
    def stypy__str__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__str__'
        module_type_store = module_type_store.open_function_context('__str__', 721, 4, False)
        # Assigning a type to the variable 'self' (line 722)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 722, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FontProperties.stypy__str__.__dict__.__setitem__('stypy_localization', localization)
        FontProperties.stypy__str__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FontProperties.stypy__str__.__dict__.__setitem__('stypy_type_store', module_type_store)
        FontProperties.stypy__str__.__dict__.__setitem__('stypy_function_name', 'FontProperties.stypy__str__')
        FontProperties.stypy__str__.__dict__.__setitem__('stypy_param_names_list', [])
        FontProperties.stypy__str__.__dict__.__setitem__('stypy_varargs_param_name', None)
        FontProperties.stypy__str__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FontProperties.stypy__str__.__dict__.__setitem__('stypy_call_defaults', defaults)
        FontProperties.stypy__str__.__dict__.__setitem__('stypy_call_varargs', varargs)
        FontProperties.stypy__str__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FontProperties.stypy__str__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FontProperties.stypy__str__', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to get_fontconfig_pattern(...): (line 722)
        # Processing the call keyword arguments (line 722)
        kwargs_58639 = {}
        # Getting the type of 'self' (line 722)
        self_58637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 722, 15), 'self', False)
        # Obtaining the member 'get_fontconfig_pattern' of a type (line 722)
        get_fontconfig_pattern_58638 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 722, 15), self_58637, 'get_fontconfig_pattern')
        # Calling get_fontconfig_pattern(args, kwargs) (line 722)
        get_fontconfig_pattern_call_result_58640 = invoke(stypy.reporting.localization.Localization(__file__, 722, 15), get_fontconfig_pattern_58638, *[], **kwargs_58639)
        
        # Assigning a type to the variable 'stypy_return_type' (line 722)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 722, 8), 'stypy_return_type', get_fontconfig_pattern_call_result_58640)
        
        # ################# End of '__str__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__str__' in the type store
        # Getting the type of 'stypy_return_type' (line 721)
        stypy_return_type_58641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 721, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_58641)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__str__'
        return stypy_return_type_58641


    @norecursion
    def get_family(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_family'
        module_type_store = module_type_store.open_function_context('get_family', 724, 4, False)
        # Assigning a type to the variable 'self' (line 725)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 725, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FontProperties.get_family.__dict__.__setitem__('stypy_localization', localization)
        FontProperties.get_family.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FontProperties.get_family.__dict__.__setitem__('stypy_type_store', module_type_store)
        FontProperties.get_family.__dict__.__setitem__('stypy_function_name', 'FontProperties.get_family')
        FontProperties.get_family.__dict__.__setitem__('stypy_param_names_list', [])
        FontProperties.get_family.__dict__.__setitem__('stypy_varargs_param_name', None)
        FontProperties.get_family.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FontProperties.get_family.__dict__.__setitem__('stypy_call_defaults', defaults)
        FontProperties.get_family.__dict__.__setitem__('stypy_call_varargs', varargs)
        FontProperties.get_family.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FontProperties.get_family.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FontProperties.get_family', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_family', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_family(...)' code ##################

        unicode_58642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 727, (-1)), 'unicode', u'\n        Return a list of font names that comprise the font family.\n        ')
        # Getting the type of 'self' (line 728)
        self_58643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 728, 15), 'self')
        # Obtaining the member '_family' of a type (line 728)
        _family_58644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 728, 15), self_58643, '_family')
        # Assigning a type to the variable 'stypy_return_type' (line 728)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 728, 8), 'stypy_return_type', _family_58644)
        
        # ################# End of 'get_family(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_family' in the type store
        # Getting the type of 'stypy_return_type' (line 724)
        stypy_return_type_58645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 724, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_58645)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_family'
        return stypy_return_type_58645


    @norecursion
    def get_name(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_name'
        module_type_store = module_type_store.open_function_context('get_name', 730, 4, False)
        # Assigning a type to the variable 'self' (line 731)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 731, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FontProperties.get_name.__dict__.__setitem__('stypy_localization', localization)
        FontProperties.get_name.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FontProperties.get_name.__dict__.__setitem__('stypy_type_store', module_type_store)
        FontProperties.get_name.__dict__.__setitem__('stypy_function_name', 'FontProperties.get_name')
        FontProperties.get_name.__dict__.__setitem__('stypy_param_names_list', [])
        FontProperties.get_name.__dict__.__setitem__('stypy_varargs_param_name', None)
        FontProperties.get_name.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FontProperties.get_name.__dict__.__setitem__('stypy_call_defaults', defaults)
        FontProperties.get_name.__dict__.__setitem__('stypy_call_varargs', varargs)
        FontProperties.get_name.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FontProperties.get_name.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FontProperties.get_name', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_name', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_name(...)' code ##################

        unicode_58646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 734, (-1)), 'unicode', u'\n        Return the name of the font that best matches the font\n        properties.\n        ')
        
        # Call to get_font(...): (line 735)
        # Processing the call arguments (line 735)
        
        # Call to findfont(...): (line 735)
        # Processing the call arguments (line 735)
        # Getting the type of 'self' (line 735)
        self_58649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 735, 33), 'self', False)
        # Processing the call keyword arguments (line 735)
        kwargs_58650 = {}
        # Getting the type of 'findfont' (line 735)
        findfont_58648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 735, 24), 'findfont', False)
        # Calling findfont(args, kwargs) (line 735)
        findfont_call_result_58651 = invoke(stypy.reporting.localization.Localization(__file__, 735, 24), findfont_58648, *[self_58649], **kwargs_58650)
        
        # Processing the call keyword arguments (line 735)
        kwargs_58652 = {}
        # Getting the type of 'get_font' (line 735)
        get_font_58647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 735, 15), 'get_font', False)
        # Calling get_font(args, kwargs) (line 735)
        get_font_call_result_58653 = invoke(stypy.reporting.localization.Localization(__file__, 735, 15), get_font_58647, *[findfont_call_result_58651], **kwargs_58652)
        
        # Obtaining the member 'family_name' of a type (line 735)
        family_name_58654 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 735, 15), get_font_call_result_58653, 'family_name')
        # Assigning a type to the variable 'stypy_return_type' (line 735)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 735, 8), 'stypy_return_type', family_name_58654)
        
        # ################# End of 'get_name(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_name' in the type store
        # Getting the type of 'stypy_return_type' (line 730)
        stypy_return_type_58655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 730, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_58655)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_name'
        return stypy_return_type_58655


    @norecursion
    def get_style(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_style'
        module_type_store = module_type_store.open_function_context('get_style', 737, 4, False)
        # Assigning a type to the variable 'self' (line 738)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 738, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FontProperties.get_style.__dict__.__setitem__('stypy_localization', localization)
        FontProperties.get_style.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FontProperties.get_style.__dict__.__setitem__('stypy_type_store', module_type_store)
        FontProperties.get_style.__dict__.__setitem__('stypy_function_name', 'FontProperties.get_style')
        FontProperties.get_style.__dict__.__setitem__('stypy_param_names_list', [])
        FontProperties.get_style.__dict__.__setitem__('stypy_varargs_param_name', None)
        FontProperties.get_style.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FontProperties.get_style.__dict__.__setitem__('stypy_call_defaults', defaults)
        FontProperties.get_style.__dict__.__setitem__('stypy_call_varargs', varargs)
        FontProperties.get_style.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FontProperties.get_style.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FontProperties.get_style', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_style', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_style(...)' code ##################

        unicode_58656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 741, (-1)), 'unicode', u"\n        Return the font style.  Values are: 'normal', 'italic' or\n        'oblique'.\n        ")
        # Getting the type of 'self' (line 742)
        self_58657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 15), 'self')
        # Obtaining the member '_slant' of a type (line 742)
        _slant_58658 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 742, 15), self_58657, '_slant')
        # Assigning a type to the variable 'stypy_return_type' (line 742)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 742, 8), 'stypy_return_type', _slant_58658)
        
        # ################# End of 'get_style(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_style' in the type store
        # Getting the type of 'stypy_return_type' (line 737)
        stypy_return_type_58659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 737, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_58659)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_style'
        return stypy_return_type_58659

    
    # Assigning a Name to a Name (line 743):

    @norecursion
    def get_variant(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_variant'
        module_type_store = module_type_store.open_function_context('get_variant', 745, 4, False)
        # Assigning a type to the variable 'self' (line 746)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 746, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FontProperties.get_variant.__dict__.__setitem__('stypy_localization', localization)
        FontProperties.get_variant.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FontProperties.get_variant.__dict__.__setitem__('stypy_type_store', module_type_store)
        FontProperties.get_variant.__dict__.__setitem__('stypy_function_name', 'FontProperties.get_variant')
        FontProperties.get_variant.__dict__.__setitem__('stypy_param_names_list', [])
        FontProperties.get_variant.__dict__.__setitem__('stypy_varargs_param_name', None)
        FontProperties.get_variant.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FontProperties.get_variant.__dict__.__setitem__('stypy_call_defaults', defaults)
        FontProperties.get_variant.__dict__.__setitem__('stypy_call_varargs', varargs)
        FontProperties.get_variant.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FontProperties.get_variant.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FontProperties.get_variant', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_variant', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_variant(...)' code ##################

        unicode_58660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 749, (-1)), 'unicode', u"\n        Return the font variant.  Values are: 'normal' or\n        'small-caps'.\n        ")
        # Getting the type of 'self' (line 750)
        self_58661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 750, 15), 'self')
        # Obtaining the member '_variant' of a type (line 750)
        _variant_58662 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 750, 15), self_58661, '_variant')
        # Assigning a type to the variable 'stypy_return_type' (line 750)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 750, 8), 'stypy_return_type', _variant_58662)
        
        # ################# End of 'get_variant(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_variant' in the type store
        # Getting the type of 'stypy_return_type' (line 745)
        stypy_return_type_58663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 745, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_58663)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_variant'
        return stypy_return_type_58663


    @norecursion
    def get_weight(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_weight'
        module_type_store = module_type_store.open_function_context('get_weight', 752, 4, False)
        # Assigning a type to the variable 'self' (line 753)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 753, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FontProperties.get_weight.__dict__.__setitem__('stypy_localization', localization)
        FontProperties.get_weight.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FontProperties.get_weight.__dict__.__setitem__('stypy_type_store', module_type_store)
        FontProperties.get_weight.__dict__.__setitem__('stypy_function_name', 'FontProperties.get_weight')
        FontProperties.get_weight.__dict__.__setitem__('stypy_param_names_list', [])
        FontProperties.get_weight.__dict__.__setitem__('stypy_varargs_param_name', None)
        FontProperties.get_weight.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FontProperties.get_weight.__dict__.__setitem__('stypy_call_defaults', defaults)
        FontProperties.get_weight.__dict__.__setitem__('stypy_call_varargs', varargs)
        FontProperties.get_weight.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FontProperties.get_weight.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FontProperties.get_weight', [], None, None, defaults, varargs, kwargs)

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

        unicode_58664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 758, (-1)), 'unicode', u"\n        Set the font weight.  Options are: A numeric value in the\n        range 0-1000 or one of 'light', 'normal', 'regular', 'book',\n        'medium', 'roman', 'semibold', 'demibold', 'demi', 'bold',\n        'heavy', 'extra bold', 'black'\n        ")
        # Getting the type of 'self' (line 759)
        self_58665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 759, 15), 'self')
        # Obtaining the member '_weight' of a type (line 759)
        _weight_58666 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 759, 15), self_58665, '_weight')
        # Assigning a type to the variable 'stypy_return_type' (line 759)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 759, 8), 'stypy_return_type', _weight_58666)
        
        # ################# End of 'get_weight(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_weight' in the type store
        # Getting the type of 'stypy_return_type' (line 752)
        stypy_return_type_58667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 752, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_58667)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_weight'
        return stypy_return_type_58667


    @norecursion
    def get_stretch(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_stretch'
        module_type_store = module_type_store.open_function_context('get_stretch', 761, 4, False)
        # Assigning a type to the variable 'self' (line 762)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 762, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FontProperties.get_stretch.__dict__.__setitem__('stypy_localization', localization)
        FontProperties.get_stretch.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FontProperties.get_stretch.__dict__.__setitem__('stypy_type_store', module_type_store)
        FontProperties.get_stretch.__dict__.__setitem__('stypy_function_name', 'FontProperties.get_stretch')
        FontProperties.get_stretch.__dict__.__setitem__('stypy_param_names_list', [])
        FontProperties.get_stretch.__dict__.__setitem__('stypy_varargs_param_name', None)
        FontProperties.get_stretch.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FontProperties.get_stretch.__dict__.__setitem__('stypy_call_defaults', defaults)
        FontProperties.get_stretch.__dict__.__setitem__('stypy_call_varargs', varargs)
        FontProperties.get_stretch.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FontProperties.get_stretch.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FontProperties.get_stretch', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_stretch', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_stretch(...)' code ##################

        unicode_58668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 766, (-1)), 'unicode', u"\n        Return the font stretch or width.  Options are: 'ultra-condensed',\n        'extra-condensed', 'condensed', 'semi-condensed', 'normal',\n        'semi-expanded', 'expanded', 'extra-expanded', 'ultra-expanded'.\n        ")
        # Getting the type of 'self' (line 767)
        self_58669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 767, 15), 'self')
        # Obtaining the member '_stretch' of a type (line 767)
        _stretch_58670 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 767, 15), self_58669, '_stretch')
        # Assigning a type to the variable 'stypy_return_type' (line 767)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 767, 8), 'stypy_return_type', _stretch_58670)
        
        # ################# End of 'get_stretch(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_stretch' in the type store
        # Getting the type of 'stypy_return_type' (line 761)
        stypy_return_type_58671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 761, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_58671)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_stretch'
        return stypy_return_type_58671


    @norecursion
    def get_size(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_size'
        module_type_store = module_type_store.open_function_context('get_size', 769, 4, False)
        # Assigning a type to the variable 'self' (line 770)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 770, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FontProperties.get_size.__dict__.__setitem__('stypy_localization', localization)
        FontProperties.get_size.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FontProperties.get_size.__dict__.__setitem__('stypy_type_store', module_type_store)
        FontProperties.get_size.__dict__.__setitem__('stypy_function_name', 'FontProperties.get_size')
        FontProperties.get_size.__dict__.__setitem__('stypy_param_names_list', [])
        FontProperties.get_size.__dict__.__setitem__('stypy_varargs_param_name', None)
        FontProperties.get_size.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FontProperties.get_size.__dict__.__setitem__('stypy_call_defaults', defaults)
        FontProperties.get_size.__dict__.__setitem__('stypy_call_varargs', varargs)
        FontProperties.get_size.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FontProperties.get_size.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FontProperties.get_size', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_size', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_size(...)' code ##################

        unicode_58672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 772, (-1)), 'unicode', u'\n        Return the font size.\n        ')
        # Getting the type of 'self' (line 773)
        self_58673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 773, 15), 'self')
        # Obtaining the member '_size' of a type (line 773)
        _size_58674 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 773, 15), self_58673, '_size')
        # Assigning a type to the variable 'stypy_return_type' (line 773)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 773, 8), 'stypy_return_type', _size_58674)
        
        # ################# End of 'get_size(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_size' in the type store
        # Getting the type of 'stypy_return_type' (line 769)
        stypy_return_type_58675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 769, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_58675)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_size'
        return stypy_return_type_58675


    @norecursion
    def get_size_in_points(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_size_in_points'
        module_type_store = module_type_store.open_function_context('get_size_in_points', 775, 4, False)
        # Assigning a type to the variable 'self' (line 776)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 776, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FontProperties.get_size_in_points.__dict__.__setitem__('stypy_localization', localization)
        FontProperties.get_size_in_points.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FontProperties.get_size_in_points.__dict__.__setitem__('stypy_type_store', module_type_store)
        FontProperties.get_size_in_points.__dict__.__setitem__('stypy_function_name', 'FontProperties.get_size_in_points')
        FontProperties.get_size_in_points.__dict__.__setitem__('stypy_param_names_list', [])
        FontProperties.get_size_in_points.__dict__.__setitem__('stypy_varargs_param_name', None)
        FontProperties.get_size_in_points.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FontProperties.get_size_in_points.__dict__.__setitem__('stypy_call_defaults', defaults)
        FontProperties.get_size_in_points.__dict__.__setitem__('stypy_call_varargs', varargs)
        FontProperties.get_size_in_points.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FontProperties.get_size_in_points.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FontProperties.get_size_in_points', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_size_in_points', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_size_in_points(...)' code ##################

        # Getting the type of 'self' (line 776)
        self_58676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 776, 15), 'self')
        # Obtaining the member '_size' of a type (line 776)
        _size_58677 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 776, 15), self_58676, '_size')
        # Assigning a type to the variable 'stypy_return_type' (line 776)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 776, 8), 'stypy_return_type', _size_58677)
        
        # ################# End of 'get_size_in_points(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_size_in_points' in the type store
        # Getting the type of 'stypy_return_type' (line 775)
        stypy_return_type_58678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 775, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_58678)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_size_in_points'
        return stypy_return_type_58678


    @norecursion
    def get_file(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_file'
        module_type_store = module_type_store.open_function_context('get_file', 778, 4, False)
        # Assigning a type to the variable 'self' (line 779)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 779, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FontProperties.get_file.__dict__.__setitem__('stypy_localization', localization)
        FontProperties.get_file.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FontProperties.get_file.__dict__.__setitem__('stypy_type_store', module_type_store)
        FontProperties.get_file.__dict__.__setitem__('stypy_function_name', 'FontProperties.get_file')
        FontProperties.get_file.__dict__.__setitem__('stypy_param_names_list', [])
        FontProperties.get_file.__dict__.__setitem__('stypy_varargs_param_name', None)
        FontProperties.get_file.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FontProperties.get_file.__dict__.__setitem__('stypy_call_defaults', defaults)
        FontProperties.get_file.__dict__.__setitem__('stypy_call_varargs', varargs)
        FontProperties.get_file.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FontProperties.get_file.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FontProperties.get_file', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_file', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_file(...)' code ##################

        unicode_58679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 781, (-1)), 'unicode', u'\n        Return the filename of the associated font.\n        ')
        # Getting the type of 'self' (line 782)
        self_58680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 782, 15), 'self')
        # Obtaining the member '_file' of a type (line 782)
        _file_58681 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 782, 15), self_58680, '_file')
        # Assigning a type to the variable 'stypy_return_type' (line 782)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 782, 8), 'stypy_return_type', _file_58681)
        
        # ################# End of 'get_file(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_file' in the type store
        # Getting the type of 'stypy_return_type' (line 778)
        stypy_return_type_58682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 778, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_58682)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_file'
        return stypy_return_type_58682


    @norecursion
    def get_fontconfig_pattern(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_fontconfig_pattern'
        module_type_store = module_type_store.open_function_context('get_fontconfig_pattern', 784, 4, False)
        # Assigning a type to the variable 'self' (line 785)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 785, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FontProperties.get_fontconfig_pattern.__dict__.__setitem__('stypy_localization', localization)
        FontProperties.get_fontconfig_pattern.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FontProperties.get_fontconfig_pattern.__dict__.__setitem__('stypy_type_store', module_type_store)
        FontProperties.get_fontconfig_pattern.__dict__.__setitem__('stypy_function_name', 'FontProperties.get_fontconfig_pattern')
        FontProperties.get_fontconfig_pattern.__dict__.__setitem__('stypy_param_names_list', [])
        FontProperties.get_fontconfig_pattern.__dict__.__setitem__('stypy_varargs_param_name', None)
        FontProperties.get_fontconfig_pattern.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FontProperties.get_fontconfig_pattern.__dict__.__setitem__('stypy_call_defaults', defaults)
        FontProperties.get_fontconfig_pattern.__dict__.__setitem__('stypy_call_varargs', varargs)
        FontProperties.get_fontconfig_pattern.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FontProperties.get_fontconfig_pattern.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FontProperties.get_fontconfig_pattern', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_fontconfig_pattern', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_fontconfig_pattern(...)' code ##################

        unicode_58683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 795, (-1)), 'unicode', u"\n        Get a fontconfig pattern suitable for looking up the font as\n        specified with fontconfig's ``fc-match`` utility.\n\n        See the documentation on `fontconfig patterns\n        <https://www.freedesktop.org/software/fontconfig/fontconfig-user.html>`_.\n\n        This support does not require fontconfig to be installed or\n        support for it to be enabled.  We are merely borrowing its\n        pattern syntax for use here.\n        ")
        
        # Call to generate_fontconfig_pattern(...): (line 796)
        # Processing the call arguments (line 796)
        # Getting the type of 'self' (line 796)
        self_58685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 796, 43), 'self', False)
        # Processing the call keyword arguments (line 796)
        kwargs_58686 = {}
        # Getting the type of 'generate_fontconfig_pattern' (line 796)
        generate_fontconfig_pattern_58684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 796, 15), 'generate_fontconfig_pattern', False)
        # Calling generate_fontconfig_pattern(args, kwargs) (line 796)
        generate_fontconfig_pattern_call_result_58687 = invoke(stypy.reporting.localization.Localization(__file__, 796, 15), generate_fontconfig_pattern_58684, *[self_58685], **kwargs_58686)
        
        # Assigning a type to the variable 'stypy_return_type' (line 796)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 796, 8), 'stypy_return_type', generate_fontconfig_pattern_call_result_58687)
        
        # ################# End of 'get_fontconfig_pattern(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_fontconfig_pattern' in the type store
        # Getting the type of 'stypy_return_type' (line 784)
        stypy_return_type_58688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 784, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_58688)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_fontconfig_pattern'
        return stypy_return_type_58688


    @norecursion
    def set_family(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_family'
        module_type_store = module_type_store.open_function_context('set_family', 798, 4, False)
        # Assigning a type to the variable 'self' (line 799)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 799, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FontProperties.set_family.__dict__.__setitem__('stypy_localization', localization)
        FontProperties.set_family.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FontProperties.set_family.__dict__.__setitem__('stypy_type_store', module_type_store)
        FontProperties.set_family.__dict__.__setitem__('stypy_function_name', 'FontProperties.set_family')
        FontProperties.set_family.__dict__.__setitem__('stypy_param_names_list', ['family'])
        FontProperties.set_family.__dict__.__setitem__('stypy_varargs_param_name', None)
        FontProperties.set_family.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FontProperties.set_family.__dict__.__setitem__('stypy_call_defaults', defaults)
        FontProperties.set_family.__dict__.__setitem__('stypy_call_varargs', varargs)
        FontProperties.set_family.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FontProperties.set_family.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FontProperties.set_family', ['family'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_family', localization, ['family'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_family(...)' code ##################

        unicode_58689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 805, (-1)), 'unicode', u"\n        Change the font family.  May be either an alias (generic name\n        is CSS parlance), such as: 'serif', 'sans-serif', 'cursive',\n        'fantasy', or 'monospace', a real font name or a list of real\n        font names.  Real font names are not supported when\n        `text.usetex` is `True`.\n        ")
        
        # Type idiom detected: calculating its left and rigth part (line 806)
        # Getting the type of 'family' (line 806)
        family_58690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 806, 11), 'family')
        # Getting the type of 'None' (line 806)
        None_58691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 806, 21), 'None')
        
        (may_be_58692, more_types_in_union_58693) = may_be_none(family_58690, None_58691)

        if may_be_58692:

            if more_types_in_union_58693:
                # Runtime conditional SSA (line 806)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Subscript to a Name (line 807):
            
            # Assigning a Subscript to a Name (line 807):
            
            # Obtaining the type of the subscript
            unicode_58694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 807, 30), 'unicode', u'font.family')
            # Getting the type of 'rcParams' (line 807)
            rcParams_58695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 807, 21), 'rcParams')
            # Obtaining the member '__getitem__' of a type (line 807)
            getitem___58696 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 807, 21), rcParams_58695, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 807)
            subscript_call_result_58697 = invoke(stypy.reporting.localization.Localization(__file__, 807, 21), getitem___58696, unicode_58694)
            
            # Assigning a type to the variable 'family' (line 807)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 807, 12), 'family', subscript_call_result_58697)

            if more_types_in_union_58693:
                # SSA join for if statement (line 806)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Attribute (line 808):
        
        # Assigning a Call to a Attribute (line 808):
        
        # Call to _normalize_font_family(...): (line 808)
        # Processing the call arguments (line 808)
        # Getting the type of 'family' (line 808)
        family_58699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 808, 46), 'family', False)
        # Processing the call keyword arguments (line 808)
        kwargs_58700 = {}
        # Getting the type of '_normalize_font_family' (line 808)
        _normalize_font_family_58698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 808, 23), '_normalize_font_family', False)
        # Calling _normalize_font_family(args, kwargs) (line 808)
        _normalize_font_family_call_result_58701 = invoke(stypy.reporting.localization.Localization(__file__, 808, 23), _normalize_font_family_58698, *[family_58699], **kwargs_58700)
        
        # Getting the type of 'self' (line 808)
        self_58702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 808, 8), 'self')
        # Setting the type of the member '_family' of a type (line 808)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 808, 8), self_58702, '_family', _normalize_font_family_call_result_58701)
        
        # ################# End of 'set_family(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_family' in the type store
        # Getting the type of 'stypy_return_type' (line 798)
        stypy_return_type_58703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 798, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_58703)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_family'
        return stypy_return_type_58703

    
    # Assigning a Name to a Name (line 809):

    @norecursion
    def set_style(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_style'
        module_type_store = module_type_store.open_function_context('set_style', 811, 4, False)
        # Assigning a type to the variable 'self' (line 812)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 812, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FontProperties.set_style.__dict__.__setitem__('stypy_localization', localization)
        FontProperties.set_style.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FontProperties.set_style.__dict__.__setitem__('stypy_type_store', module_type_store)
        FontProperties.set_style.__dict__.__setitem__('stypy_function_name', 'FontProperties.set_style')
        FontProperties.set_style.__dict__.__setitem__('stypy_param_names_list', ['style'])
        FontProperties.set_style.__dict__.__setitem__('stypy_varargs_param_name', None)
        FontProperties.set_style.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FontProperties.set_style.__dict__.__setitem__('stypy_call_defaults', defaults)
        FontProperties.set_style.__dict__.__setitem__('stypy_call_varargs', varargs)
        FontProperties.set_style.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FontProperties.set_style.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FontProperties.set_style', ['style'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_style', localization, ['style'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_style(...)' code ##################

        unicode_58704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 815, (-1)), 'unicode', u"\n        Set the font style.  Values are: 'normal', 'italic' or\n        'oblique'.\n        ")
        
        # Type idiom detected: calculating its left and rigth part (line 816)
        # Getting the type of 'style' (line 816)
        style_58705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 816, 11), 'style')
        # Getting the type of 'None' (line 816)
        None_58706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 816, 20), 'None')
        
        (may_be_58707, more_types_in_union_58708) = may_be_none(style_58705, None_58706)

        if may_be_58707:

            if more_types_in_union_58708:
                # Runtime conditional SSA (line 816)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Subscript to a Name (line 817):
            
            # Assigning a Subscript to a Name (line 817):
            
            # Obtaining the type of the subscript
            unicode_58709 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 817, 29), 'unicode', u'font.style')
            # Getting the type of 'rcParams' (line 817)
            rcParams_58710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 817, 20), 'rcParams')
            # Obtaining the member '__getitem__' of a type (line 817)
            getitem___58711 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 817, 20), rcParams_58710, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 817)
            subscript_call_result_58712 = invoke(stypy.reporting.localization.Localization(__file__, 817, 20), getitem___58711, unicode_58709)
            
            # Assigning a type to the variable 'style' (line 817)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 817, 12), 'style', subscript_call_result_58712)

            if more_types_in_union_58708:
                # SSA join for if statement (line 816)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        # Getting the type of 'style' (line 818)
        style_58713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 818, 11), 'style')
        
        # Obtaining an instance of the builtin type 'tuple' (line 818)
        tuple_58714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 818, 25), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 818)
        # Adding element type (line 818)
        unicode_58715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 818, 25), 'unicode', u'normal')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 818, 25), tuple_58714, unicode_58715)
        # Adding element type (line 818)
        unicode_58716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 818, 35), 'unicode', u'italic')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 818, 25), tuple_58714, unicode_58716)
        # Adding element type (line 818)
        unicode_58717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 818, 45), 'unicode', u'oblique')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 818, 25), tuple_58714, unicode_58717)
        
        # Applying the binary operator 'notin' (line 818)
        result_contains_58718 = python_operator(stypy.reporting.localization.Localization(__file__, 818, 11), 'notin', style_58713, tuple_58714)
        
        # Testing the type of an if condition (line 818)
        if_condition_58719 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 818, 8), result_contains_58718)
        # Assigning a type to the variable 'if_condition_58719' (line 818)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 818, 8), 'if_condition_58719', if_condition_58719)
        # SSA begins for if statement (line 818)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 819)
        # Processing the call arguments (line 819)
        unicode_58721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 819, 29), 'unicode', u'style must be normal, italic or oblique')
        # Processing the call keyword arguments (line 819)
        kwargs_58722 = {}
        # Getting the type of 'ValueError' (line 819)
        ValueError_58720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 819, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 819)
        ValueError_call_result_58723 = invoke(stypy.reporting.localization.Localization(__file__, 819, 18), ValueError_58720, *[unicode_58721], **kwargs_58722)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 819, 12), ValueError_call_result_58723, 'raise parameter', BaseException)
        # SSA join for if statement (line 818)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 820):
        
        # Assigning a Name to a Attribute (line 820):
        # Getting the type of 'style' (line 820)
        style_58724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 820, 22), 'style')
        # Getting the type of 'self' (line 820)
        self_58725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 820, 8), 'self')
        # Setting the type of the member '_slant' of a type (line 820)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 820, 8), self_58725, '_slant', style_58724)
        
        # ################# End of 'set_style(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_style' in the type store
        # Getting the type of 'stypy_return_type' (line 811)
        stypy_return_type_58726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 811, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_58726)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_style'
        return stypy_return_type_58726

    
    # Assigning a Name to a Name (line 821):

    @norecursion
    def set_variant(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_variant'
        module_type_store = module_type_store.open_function_context('set_variant', 823, 4, False)
        # Assigning a type to the variable 'self' (line 824)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 824, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FontProperties.set_variant.__dict__.__setitem__('stypy_localization', localization)
        FontProperties.set_variant.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FontProperties.set_variant.__dict__.__setitem__('stypy_type_store', module_type_store)
        FontProperties.set_variant.__dict__.__setitem__('stypy_function_name', 'FontProperties.set_variant')
        FontProperties.set_variant.__dict__.__setitem__('stypy_param_names_list', ['variant'])
        FontProperties.set_variant.__dict__.__setitem__('stypy_varargs_param_name', None)
        FontProperties.set_variant.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FontProperties.set_variant.__dict__.__setitem__('stypy_call_defaults', defaults)
        FontProperties.set_variant.__dict__.__setitem__('stypy_call_varargs', varargs)
        FontProperties.set_variant.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FontProperties.set_variant.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FontProperties.set_variant', ['variant'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_variant', localization, ['variant'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_variant(...)' code ##################

        unicode_58727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 826, (-1)), 'unicode', u"\n        Set the font variant.  Values are: 'normal' or 'small-caps'.\n        ")
        
        # Type idiom detected: calculating its left and rigth part (line 827)
        # Getting the type of 'variant' (line 827)
        variant_58728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 827, 11), 'variant')
        # Getting the type of 'None' (line 827)
        None_58729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 827, 22), 'None')
        
        (may_be_58730, more_types_in_union_58731) = may_be_none(variant_58728, None_58729)

        if may_be_58730:

            if more_types_in_union_58731:
                # Runtime conditional SSA (line 827)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Subscript to a Name (line 828):
            
            # Assigning a Subscript to a Name (line 828):
            
            # Obtaining the type of the subscript
            unicode_58732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 828, 31), 'unicode', u'font.variant')
            # Getting the type of 'rcParams' (line 828)
            rcParams_58733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 828, 22), 'rcParams')
            # Obtaining the member '__getitem__' of a type (line 828)
            getitem___58734 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 828, 22), rcParams_58733, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 828)
            subscript_call_result_58735 = invoke(stypy.reporting.localization.Localization(__file__, 828, 22), getitem___58734, unicode_58732)
            
            # Assigning a type to the variable 'variant' (line 828)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 828, 12), 'variant', subscript_call_result_58735)

            if more_types_in_union_58731:
                # SSA join for if statement (line 827)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        # Getting the type of 'variant' (line 829)
        variant_58736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 829, 11), 'variant')
        
        # Obtaining an instance of the builtin type 'tuple' (line 829)
        tuple_58737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 829, 27), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 829)
        # Adding element type (line 829)
        unicode_58738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 829, 27), 'unicode', u'normal')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 829, 27), tuple_58737, unicode_58738)
        # Adding element type (line 829)
        unicode_58739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 829, 37), 'unicode', u'small-caps')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 829, 27), tuple_58737, unicode_58739)
        
        # Applying the binary operator 'notin' (line 829)
        result_contains_58740 = python_operator(stypy.reporting.localization.Localization(__file__, 829, 11), 'notin', variant_58736, tuple_58737)
        
        # Testing the type of an if condition (line 829)
        if_condition_58741 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 829, 8), result_contains_58740)
        # Assigning a type to the variable 'if_condition_58741' (line 829)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 829, 8), 'if_condition_58741', if_condition_58741)
        # SSA begins for if statement (line 829)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 830)
        # Processing the call arguments (line 830)
        unicode_58743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 830, 29), 'unicode', u'variant must be normal or small-caps')
        # Processing the call keyword arguments (line 830)
        kwargs_58744 = {}
        # Getting the type of 'ValueError' (line 830)
        ValueError_58742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 830, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 830)
        ValueError_call_result_58745 = invoke(stypy.reporting.localization.Localization(__file__, 830, 18), ValueError_58742, *[unicode_58743], **kwargs_58744)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 830, 12), ValueError_call_result_58745, 'raise parameter', BaseException)
        # SSA join for if statement (line 829)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 831):
        
        # Assigning a Name to a Attribute (line 831):
        # Getting the type of 'variant' (line 831)
        variant_58746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 831, 24), 'variant')
        # Getting the type of 'self' (line 831)
        self_58747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 831, 8), 'self')
        # Setting the type of the member '_variant' of a type (line 831)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 831, 8), self_58747, '_variant', variant_58746)
        
        # ################# End of 'set_variant(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_variant' in the type store
        # Getting the type of 'stypy_return_type' (line 823)
        stypy_return_type_58748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 823, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_58748)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_variant'
        return stypy_return_type_58748


    @norecursion
    def set_weight(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_weight'
        module_type_store = module_type_store.open_function_context('set_weight', 833, 4, False)
        # Assigning a type to the variable 'self' (line 834)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 834, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FontProperties.set_weight.__dict__.__setitem__('stypy_localization', localization)
        FontProperties.set_weight.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FontProperties.set_weight.__dict__.__setitem__('stypy_type_store', module_type_store)
        FontProperties.set_weight.__dict__.__setitem__('stypy_function_name', 'FontProperties.set_weight')
        FontProperties.set_weight.__dict__.__setitem__('stypy_param_names_list', ['weight'])
        FontProperties.set_weight.__dict__.__setitem__('stypy_varargs_param_name', None)
        FontProperties.set_weight.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FontProperties.set_weight.__dict__.__setitem__('stypy_call_defaults', defaults)
        FontProperties.set_weight.__dict__.__setitem__('stypy_call_varargs', varargs)
        FontProperties.set_weight.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FontProperties.set_weight.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FontProperties.set_weight', ['weight'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_weight', localization, ['weight'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_weight(...)' code ##################

        unicode_58749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 839, (-1)), 'unicode', u"\n        Set the font weight.  May be either a numeric value in the\n        range 0-1000 or one of 'ultralight', 'light', 'normal',\n        'regular', 'book', 'medium', 'roman', 'semibold', 'demibold',\n        'demi', 'bold', 'heavy', 'extra bold', 'black'\n        ")
        
        # Type idiom detected: calculating its left and rigth part (line 840)
        # Getting the type of 'weight' (line 840)
        weight_58750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 840, 11), 'weight')
        # Getting the type of 'None' (line 840)
        None_58751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 840, 21), 'None')
        
        (may_be_58752, more_types_in_union_58753) = may_be_none(weight_58750, None_58751)

        if may_be_58752:

            if more_types_in_union_58753:
                # Runtime conditional SSA (line 840)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Subscript to a Name (line 841):
            
            # Assigning a Subscript to a Name (line 841):
            
            # Obtaining the type of the subscript
            unicode_58754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 841, 30), 'unicode', u'font.weight')
            # Getting the type of 'rcParams' (line 841)
            rcParams_58755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 841, 21), 'rcParams')
            # Obtaining the member '__getitem__' of a type (line 841)
            getitem___58756 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 841, 21), rcParams_58755, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 841)
            subscript_call_result_58757 = invoke(stypy.reporting.localization.Localization(__file__, 841, 21), getitem___58756, unicode_58754)
            
            # Assigning a type to the variable 'weight' (line 841)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 841, 12), 'weight', subscript_call_result_58757)

            if more_types_in_union_58753:
                # SSA join for if statement (line 840)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        # SSA begins for try-except statement (line 842)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Name (line 843):
        
        # Assigning a Call to a Name (line 843):
        
        # Call to int(...): (line 843)
        # Processing the call arguments (line 843)
        # Getting the type of 'weight' (line 843)
        weight_58759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 843, 25), 'weight', False)
        # Processing the call keyword arguments (line 843)
        kwargs_58760 = {}
        # Getting the type of 'int' (line 843)
        int_58758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 843, 21), 'int', False)
        # Calling int(args, kwargs) (line 843)
        int_call_result_58761 = invoke(stypy.reporting.localization.Localization(__file__, 843, 21), int_58758, *[weight_58759], **kwargs_58760)
        
        # Assigning a type to the variable 'weight' (line 843)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 843, 12), 'weight', int_call_result_58761)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'weight' (line 844)
        weight_58762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 844, 15), 'weight')
        int_58763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 844, 24), 'int')
        # Applying the binary operator '<' (line 844)
        result_lt_58764 = python_operator(stypy.reporting.localization.Localization(__file__, 844, 15), '<', weight_58762, int_58763)
        
        
        # Getting the type of 'weight' (line 844)
        weight_58765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 844, 29), 'weight')
        int_58766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 844, 38), 'int')
        # Applying the binary operator '>' (line 844)
        result_gt_58767 = python_operator(stypy.reporting.localization.Localization(__file__, 844, 29), '>', weight_58765, int_58766)
        
        # Applying the binary operator 'or' (line 844)
        result_or_keyword_58768 = python_operator(stypy.reporting.localization.Localization(__file__, 844, 15), 'or', result_lt_58764, result_gt_58767)
        
        # Testing the type of an if condition (line 844)
        if_condition_58769 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 844, 12), result_or_keyword_58768)
        # Assigning a type to the variable 'if_condition_58769' (line 844)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 844, 12), 'if_condition_58769', if_condition_58769)
        # SSA begins for if statement (line 844)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 845)
        # Processing the call keyword arguments (line 845)
        kwargs_58771 = {}
        # Getting the type of 'ValueError' (line 845)
        ValueError_58770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 845, 22), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 845)
        ValueError_call_result_58772 = invoke(stypy.reporting.localization.Localization(__file__, 845, 22), ValueError_58770, *[], **kwargs_58771)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 845, 16), ValueError_call_result_58772, 'raise parameter', BaseException)
        # SSA join for if statement (line 844)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the except part of a try statement (line 842)
        # SSA branch for the except 'ValueError' branch of a try statement (line 842)
        module_type_store.open_ssa_branch('except')
        
        
        # Getting the type of 'weight' (line 847)
        weight_58773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 847, 15), 'weight')
        # Getting the type of 'weight_dict' (line 847)
        weight_dict_58774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 847, 29), 'weight_dict')
        # Applying the binary operator 'notin' (line 847)
        result_contains_58775 = python_operator(stypy.reporting.localization.Localization(__file__, 847, 15), 'notin', weight_58773, weight_dict_58774)
        
        # Testing the type of an if condition (line 847)
        if_condition_58776 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 847, 12), result_contains_58775)
        # Assigning a type to the variable 'if_condition_58776' (line 847)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 847, 12), 'if_condition_58776', if_condition_58776)
        # SSA begins for if statement (line 847)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 848)
        # Processing the call arguments (line 848)
        unicode_58778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 848, 33), 'unicode', u'weight is invalid')
        # Processing the call keyword arguments (line 848)
        kwargs_58779 = {}
        # Getting the type of 'ValueError' (line 848)
        ValueError_58777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 848, 22), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 848)
        ValueError_call_result_58780 = invoke(stypy.reporting.localization.Localization(__file__, 848, 22), ValueError_58777, *[unicode_58778], **kwargs_58779)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 848, 16), ValueError_call_result_58780, 'raise parameter', BaseException)
        # SSA join for if statement (line 847)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for try-except statement (line 842)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 849):
        
        # Assigning a Name to a Attribute (line 849):
        # Getting the type of 'weight' (line 849)
        weight_58781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 849, 23), 'weight')
        # Getting the type of 'self' (line 849)
        self_58782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 849, 8), 'self')
        # Setting the type of the member '_weight' of a type (line 849)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 849, 8), self_58782, '_weight', weight_58781)
        
        # ################# End of 'set_weight(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_weight' in the type store
        # Getting the type of 'stypy_return_type' (line 833)
        stypy_return_type_58783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 833, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_58783)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_weight'
        return stypy_return_type_58783


    @norecursion
    def set_stretch(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_stretch'
        module_type_store = module_type_store.open_function_context('set_stretch', 851, 4, False)
        # Assigning a type to the variable 'self' (line 852)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 852, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FontProperties.set_stretch.__dict__.__setitem__('stypy_localization', localization)
        FontProperties.set_stretch.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FontProperties.set_stretch.__dict__.__setitem__('stypy_type_store', module_type_store)
        FontProperties.set_stretch.__dict__.__setitem__('stypy_function_name', 'FontProperties.set_stretch')
        FontProperties.set_stretch.__dict__.__setitem__('stypy_param_names_list', ['stretch'])
        FontProperties.set_stretch.__dict__.__setitem__('stypy_varargs_param_name', None)
        FontProperties.set_stretch.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FontProperties.set_stretch.__dict__.__setitem__('stypy_call_defaults', defaults)
        FontProperties.set_stretch.__dict__.__setitem__('stypy_call_varargs', varargs)
        FontProperties.set_stretch.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FontProperties.set_stretch.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FontProperties.set_stretch', ['stretch'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_stretch', localization, ['stretch'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_stretch(...)' code ##################

        unicode_58784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 857, (-1)), 'unicode', u"\n        Set the font stretch or width.  Options are: 'ultra-condensed',\n        'extra-condensed', 'condensed', 'semi-condensed', 'normal',\n        'semi-expanded', 'expanded', 'extra-expanded' or\n        'ultra-expanded', or a numeric value in the range 0-1000.\n        ")
        
        # Type idiom detected: calculating its left and rigth part (line 858)
        # Getting the type of 'stretch' (line 858)
        stretch_58785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 858, 11), 'stretch')
        # Getting the type of 'None' (line 858)
        None_58786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 858, 22), 'None')
        
        (may_be_58787, more_types_in_union_58788) = may_be_none(stretch_58785, None_58786)

        if may_be_58787:

            if more_types_in_union_58788:
                # Runtime conditional SSA (line 858)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Subscript to a Name (line 859):
            
            # Assigning a Subscript to a Name (line 859):
            
            # Obtaining the type of the subscript
            unicode_58789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 859, 31), 'unicode', u'font.stretch')
            # Getting the type of 'rcParams' (line 859)
            rcParams_58790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 859, 22), 'rcParams')
            # Obtaining the member '__getitem__' of a type (line 859)
            getitem___58791 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 859, 22), rcParams_58790, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 859)
            subscript_call_result_58792 = invoke(stypy.reporting.localization.Localization(__file__, 859, 22), getitem___58791, unicode_58789)
            
            # Assigning a type to the variable 'stretch' (line 859)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 859, 12), 'stretch', subscript_call_result_58792)

            if more_types_in_union_58788:
                # SSA join for if statement (line 858)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        # SSA begins for try-except statement (line 860)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Name (line 861):
        
        # Assigning a Call to a Name (line 861):
        
        # Call to int(...): (line 861)
        # Processing the call arguments (line 861)
        # Getting the type of 'stretch' (line 861)
        stretch_58794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 861, 26), 'stretch', False)
        # Processing the call keyword arguments (line 861)
        kwargs_58795 = {}
        # Getting the type of 'int' (line 861)
        int_58793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 861, 22), 'int', False)
        # Calling int(args, kwargs) (line 861)
        int_call_result_58796 = invoke(stypy.reporting.localization.Localization(__file__, 861, 22), int_58793, *[stretch_58794], **kwargs_58795)
        
        # Assigning a type to the variable 'stretch' (line 861)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 861, 12), 'stretch', int_call_result_58796)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'stretch' (line 862)
        stretch_58797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 862, 15), 'stretch')
        int_58798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 862, 25), 'int')
        # Applying the binary operator '<' (line 862)
        result_lt_58799 = python_operator(stypy.reporting.localization.Localization(__file__, 862, 15), '<', stretch_58797, int_58798)
        
        
        # Getting the type of 'stretch' (line 862)
        stretch_58800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 862, 30), 'stretch')
        int_58801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 862, 40), 'int')
        # Applying the binary operator '>' (line 862)
        result_gt_58802 = python_operator(stypy.reporting.localization.Localization(__file__, 862, 30), '>', stretch_58800, int_58801)
        
        # Applying the binary operator 'or' (line 862)
        result_or_keyword_58803 = python_operator(stypy.reporting.localization.Localization(__file__, 862, 15), 'or', result_lt_58799, result_gt_58802)
        
        # Testing the type of an if condition (line 862)
        if_condition_58804 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 862, 12), result_or_keyword_58803)
        # Assigning a type to the variable 'if_condition_58804' (line 862)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 862, 12), 'if_condition_58804', if_condition_58804)
        # SSA begins for if statement (line 862)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 863)
        # Processing the call keyword arguments (line 863)
        kwargs_58806 = {}
        # Getting the type of 'ValueError' (line 863)
        ValueError_58805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 863, 22), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 863)
        ValueError_call_result_58807 = invoke(stypy.reporting.localization.Localization(__file__, 863, 22), ValueError_58805, *[], **kwargs_58806)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 863, 16), ValueError_call_result_58807, 'raise parameter', BaseException)
        # SSA join for if statement (line 862)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the except part of a try statement (line 860)
        # SSA branch for the except 'ValueError' branch of a try statement (line 860)
        module_type_store.open_ssa_branch('except')
        
        
        # Getting the type of 'stretch' (line 865)
        stretch_58808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 865, 15), 'stretch')
        # Getting the type of 'stretch_dict' (line 865)
        stretch_dict_58809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 865, 30), 'stretch_dict')
        # Applying the binary operator 'notin' (line 865)
        result_contains_58810 = python_operator(stypy.reporting.localization.Localization(__file__, 865, 15), 'notin', stretch_58808, stretch_dict_58809)
        
        # Testing the type of an if condition (line 865)
        if_condition_58811 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 865, 12), result_contains_58810)
        # Assigning a type to the variable 'if_condition_58811' (line 865)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 865, 12), 'if_condition_58811', if_condition_58811)
        # SSA begins for if statement (line 865)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 866)
        # Processing the call arguments (line 866)
        unicode_58813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 866, 33), 'unicode', u'stretch is invalid')
        # Processing the call keyword arguments (line 866)
        kwargs_58814 = {}
        # Getting the type of 'ValueError' (line 866)
        ValueError_58812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 866, 22), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 866)
        ValueError_call_result_58815 = invoke(stypy.reporting.localization.Localization(__file__, 866, 22), ValueError_58812, *[unicode_58813], **kwargs_58814)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 866, 16), ValueError_call_result_58815, 'raise parameter', BaseException)
        # SSA join for if statement (line 865)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for try-except statement (line 860)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 867):
        
        # Assigning a Name to a Attribute (line 867):
        # Getting the type of 'stretch' (line 867)
        stretch_58816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 867, 24), 'stretch')
        # Getting the type of 'self' (line 867)
        self_58817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 867, 8), 'self')
        # Setting the type of the member '_stretch' of a type (line 867)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 867, 8), self_58817, '_stretch', stretch_58816)
        
        # ################# End of 'set_stretch(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_stretch' in the type store
        # Getting the type of 'stypy_return_type' (line 851)
        stypy_return_type_58818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 851, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_58818)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_stretch'
        return stypy_return_type_58818


    @norecursion
    def set_size(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_size'
        module_type_store = module_type_store.open_function_context('set_size', 869, 4, False)
        # Assigning a type to the variable 'self' (line 870)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 870, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FontProperties.set_size.__dict__.__setitem__('stypy_localization', localization)
        FontProperties.set_size.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FontProperties.set_size.__dict__.__setitem__('stypy_type_store', module_type_store)
        FontProperties.set_size.__dict__.__setitem__('stypy_function_name', 'FontProperties.set_size')
        FontProperties.set_size.__dict__.__setitem__('stypy_param_names_list', ['size'])
        FontProperties.set_size.__dict__.__setitem__('stypy_varargs_param_name', None)
        FontProperties.set_size.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FontProperties.set_size.__dict__.__setitem__('stypy_call_defaults', defaults)
        FontProperties.set_size.__dict__.__setitem__('stypy_call_varargs', varargs)
        FontProperties.set_size.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FontProperties.set_size.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FontProperties.set_size', ['size'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_size', localization, ['size'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_size(...)' code ##################

        unicode_58819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 874, (-1)), 'unicode', u"\n        Set the font size.  Either an relative value of 'xx-small',\n        'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'\n        or an absolute font size, e.g., 12.\n        ")
        
        # Type idiom detected: calculating its left and rigth part (line 875)
        # Getting the type of 'size' (line 875)
        size_58820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 875, 11), 'size')
        # Getting the type of 'None' (line 875)
        None_58821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 875, 19), 'None')
        
        (may_be_58822, more_types_in_union_58823) = may_be_none(size_58820, None_58821)

        if may_be_58822:

            if more_types_in_union_58823:
                # Runtime conditional SSA (line 875)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Subscript to a Name (line 876):
            
            # Assigning a Subscript to a Name (line 876):
            
            # Obtaining the type of the subscript
            unicode_58824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 876, 28), 'unicode', u'font.size')
            # Getting the type of 'rcParams' (line 876)
            rcParams_58825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 876, 19), 'rcParams')
            # Obtaining the member '__getitem__' of a type (line 876)
            getitem___58826 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 876, 19), rcParams_58825, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 876)
            subscript_call_result_58827 = invoke(stypy.reporting.localization.Localization(__file__, 876, 19), getitem___58826, unicode_58824)
            
            # Assigning a type to the variable 'size' (line 876)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 876, 12), 'size', subscript_call_result_58827)

            if more_types_in_union_58823:
                # SSA join for if statement (line 875)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        # SSA begins for try-except statement (line 877)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Name (line 878):
        
        # Assigning a Call to a Name (line 878):
        
        # Call to float(...): (line 878)
        # Processing the call arguments (line 878)
        # Getting the type of 'size' (line 878)
        size_58829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 878, 25), 'size', False)
        # Processing the call keyword arguments (line 878)
        kwargs_58830 = {}
        # Getting the type of 'float' (line 878)
        float_58828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 878, 19), 'float', False)
        # Calling float(args, kwargs) (line 878)
        float_call_result_58831 = invoke(stypy.reporting.localization.Localization(__file__, 878, 19), float_58828, *[size_58829], **kwargs_58830)
        
        # Assigning a type to the variable 'size' (line 878)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 878, 12), 'size', float_call_result_58831)
        # SSA branch for the except part of a try statement (line 877)
        # SSA branch for the except 'ValueError' branch of a try statement (line 877)
        module_type_store.open_ssa_branch('except')
        
        
        # SSA begins for try-except statement (line 880)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Subscript to a Name (line 881):
        
        # Assigning a Subscript to a Name (line 881):
        
        # Obtaining the type of the subscript
        # Getting the type of 'size' (line 881)
        size_58832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 881, 38), 'size')
        # Getting the type of 'font_scalings' (line 881)
        font_scalings_58833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 881, 24), 'font_scalings')
        # Obtaining the member '__getitem__' of a type (line 881)
        getitem___58834 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 881, 24), font_scalings_58833, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 881)
        subscript_call_result_58835 = invoke(stypy.reporting.localization.Localization(__file__, 881, 24), getitem___58834, size_58832)
        
        # Assigning a type to the variable 'scale' (line 881)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 881, 16), 'scale', subscript_call_result_58835)
        # SSA branch for the except part of a try statement (line 880)
        # SSA branch for the except 'KeyError' branch of a try statement (line 880)
        module_type_store.open_ssa_branch('except')
        
        # Call to ValueError(...): (line 883)
        # Processing the call arguments (line 883)
        unicode_58837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 884, 20), 'unicode', u'Size is invalid. Valid font size are ')
        
        # Call to join(...): (line 885)
        # Processing the call arguments (line 885)
        
        # Call to map(...): (line 885)
        # Processing the call arguments (line 885)
        # Getting the type of 'str' (line 885)
        str_58841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 885, 36), 'str', False)
        # Getting the type of 'font_scalings' (line 885)
        font_scalings_58842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 885, 41), 'font_scalings', False)
        # Processing the call keyword arguments (line 885)
        kwargs_58843 = {}
        # Getting the type of 'map' (line 885)
        map_58840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 885, 32), 'map', False)
        # Calling map(args, kwargs) (line 885)
        map_call_result_58844 = invoke(stypy.reporting.localization.Localization(__file__, 885, 32), map_58840, *[str_58841, font_scalings_58842], **kwargs_58843)
        
        # Processing the call keyword arguments (line 885)
        kwargs_58845 = {}
        unicode_58838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 885, 22), 'unicode', u', ')
        # Obtaining the member 'join' of a type (line 885)
        join_58839 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 885, 22), unicode_58838, 'join')
        # Calling join(args, kwargs) (line 885)
        join_call_result_58846 = invoke(stypy.reporting.localization.Localization(__file__, 885, 22), join_58839, *[map_call_result_58844], **kwargs_58845)
        
        # Applying the binary operator '+' (line 884)
        result_add_58847 = python_operator(stypy.reporting.localization.Localization(__file__, 884, 20), '+', unicode_58837, join_call_result_58846)
        
        # Processing the call keyword arguments (line 883)
        kwargs_58848 = {}
        # Getting the type of 'ValueError' (line 883)
        ValueError_58836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 883, 22), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 883)
        ValueError_call_result_58849 = invoke(stypy.reporting.localization.Localization(__file__, 883, 22), ValueError_58836, *[result_add_58847], **kwargs_58848)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 883, 16), ValueError_call_result_58849, 'raise parameter', BaseException)
        # SSA branch for the else branch of a try statement (line 880)
        module_type_store.open_ssa_branch('except else')
        
        # Assigning a BinOp to a Name (line 887):
        
        # Assigning a BinOp to a Name (line 887):
        # Getting the type of 'scale' (line 887)
        scale_58850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 887, 23), 'scale')
        
        # Call to get_default_size(...): (line 887)
        # Processing the call keyword arguments (line 887)
        kwargs_58853 = {}
        # Getting the type of 'FontManager' (line 887)
        FontManager_58851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 887, 31), 'FontManager', False)
        # Obtaining the member 'get_default_size' of a type (line 887)
        get_default_size_58852 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 887, 31), FontManager_58851, 'get_default_size')
        # Calling get_default_size(args, kwargs) (line 887)
        get_default_size_call_result_58854 = invoke(stypy.reporting.localization.Localization(__file__, 887, 31), get_default_size_58852, *[], **kwargs_58853)
        
        # Applying the binary operator '*' (line 887)
        result_mul_58855 = python_operator(stypy.reporting.localization.Localization(__file__, 887, 23), '*', scale_58850, get_default_size_call_result_58854)
        
        # Assigning a type to the variable 'size' (line 887)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 887, 16), 'size', result_mul_58855)
        # SSA join for try-except statement (line 880)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for try-except statement (line 877)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 888):
        
        # Assigning a Name to a Attribute (line 888):
        # Getting the type of 'size' (line 888)
        size_58856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 888, 21), 'size')
        # Getting the type of 'self' (line 888)
        self_58857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 888, 8), 'self')
        # Setting the type of the member '_size' of a type (line 888)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 888, 8), self_58857, '_size', size_58856)
        
        # ################# End of 'set_size(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_size' in the type store
        # Getting the type of 'stypy_return_type' (line 869)
        stypy_return_type_58858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 869, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_58858)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_size'
        return stypy_return_type_58858


    @norecursion
    def set_file(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_file'
        module_type_store = module_type_store.open_function_context('set_file', 890, 4, False)
        # Assigning a type to the variable 'self' (line 891)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 891, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FontProperties.set_file.__dict__.__setitem__('stypy_localization', localization)
        FontProperties.set_file.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FontProperties.set_file.__dict__.__setitem__('stypy_type_store', module_type_store)
        FontProperties.set_file.__dict__.__setitem__('stypy_function_name', 'FontProperties.set_file')
        FontProperties.set_file.__dict__.__setitem__('stypy_param_names_list', ['file'])
        FontProperties.set_file.__dict__.__setitem__('stypy_varargs_param_name', None)
        FontProperties.set_file.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FontProperties.set_file.__dict__.__setitem__('stypy_call_defaults', defaults)
        FontProperties.set_file.__dict__.__setitem__('stypy_call_varargs', varargs)
        FontProperties.set_file.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FontProperties.set_file.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FontProperties.set_file', ['file'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_file', localization, ['file'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_file(...)' code ##################

        unicode_58859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 894, (-1)), 'unicode', u'\n        Set the filename of the fontfile to use.  In this case, all\n        other properties will be ignored.\n        ')
        
        # Assigning a Name to a Attribute (line 895):
        
        # Assigning a Name to a Attribute (line 895):
        # Getting the type of 'file' (line 895)
        file_58860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 895, 21), 'file')
        # Getting the type of 'self' (line 895)
        self_58861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 895, 8), 'self')
        # Setting the type of the member '_file' of a type (line 895)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 895, 8), self_58861, '_file', file_58860)
        
        # ################# End of 'set_file(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_file' in the type store
        # Getting the type of 'stypy_return_type' (line 890)
        stypy_return_type_58862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 890, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_58862)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_file'
        return stypy_return_type_58862


    @norecursion
    def set_fontconfig_pattern(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_fontconfig_pattern'
        module_type_store = module_type_store.open_function_context('set_fontconfig_pattern', 897, 4, False)
        # Assigning a type to the variable 'self' (line 898)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 898, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FontProperties.set_fontconfig_pattern.__dict__.__setitem__('stypy_localization', localization)
        FontProperties.set_fontconfig_pattern.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FontProperties.set_fontconfig_pattern.__dict__.__setitem__('stypy_type_store', module_type_store)
        FontProperties.set_fontconfig_pattern.__dict__.__setitem__('stypy_function_name', 'FontProperties.set_fontconfig_pattern')
        FontProperties.set_fontconfig_pattern.__dict__.__setitem__('stypy_param_names_list', ['pattern'])
        FontProperties.set_fontconfig_pattern.__dict__.__setitem__('stypy_varargs_param_name', None)
        FontProperties.set_fontconfig_pattern.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FontProperties.set_fontconfig_pattern.__dict__.__setitem__('stypy_call_defaults', defaults)
        FontProperties.set_fontconfig_pattern.__dict__.__setitem__('stypy_call_varargs', varargs)
        FontProperties.set_fontconfig_pattern.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FontProperties.set_fontconfig_pattern.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FontProperties.set_fontconfig_pattern', ['pattern'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_fontconfig_pattern', localization, ['pattern'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_fontconfig_pattern(...)' code ##################

        unicode_58863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 907, (-1)), 'unicode', u'\n        Set the properties by parsing a fontconfig *pattern*.\n\n        See the documentation on `fontconfig patterns\n        <https://www.freedesktop.org/software/fontconfig/fontconfig-user.html>`_.\n\n        This support does not require fontconfig to be installed or\n        support for it to be enabled.  We are merely borrowing its\n        pattern syntax for use here.\n        ')
        
        
        # Call to iteritems(...): (line 908)
        # Processing the call arguments (line 908)
        
        # Call to _parse_fontconfig_pattern(...): (line 908)
        # Processing the call arguments (line 908)
        # Getting the type of 'pattern' (line 908)
        pattern_58868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 908, 69), 'pattern', False)
        # Processing the call keyword arguments (line 908)
        kwargs_58869 = {}
        # Getting the type of 'self' (line 908)
        self_58866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 908, 38), 'self', False)
        # Obtaining the member '_parse_fontconfig_pattern' of a type (line 908)
        _parse_fontconfig_pattern_58867 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 908, 38), self_58866, '_parse_fontconfig_pattern')
        # Calling _parse_fontconfig_pattern(args, kwargs) (line 908)
        _parse_fontconfig_pattern_call_result_58870 = invoke(stypy.reporting.localization.Localization(__file__, 908, 38), _parse_fontconfig_pattern_58867, *[pattern_58868], **kwargs_58869)
        
        # Processing the call keyword arguments (line 908)
        kwargs_58871 = {}
        # Getting the type of 'six' (line 908)
        six_58864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 908, 24), 'six', False)
        # Obtaining the member 'iteritems' of a type (line 908)
        iteritems_58865 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 908, 24), six_58864, 'iteritems')
        # Calling iteritems(args, kwargs) (line 908)
        iteritems_call_result_58872 = invoke(stypy.reporting.localization.Localization(__file__, 908, 24), iteritems_58865, *[_parse_fontconfig_pattern_call_result_58870], **kwargs_58871)
        
        # Testing the type of a for loop iterable (line 908)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 908, 8), iteritems_call_result_58872)
        # Getting the type of the for loop variable (line 908)
        for_loop_var_58873 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 908, 8), iteritems_call_result_58872)
        # Assigning a type to the variable 'key' (line 908)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 908, 8), 'key', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 908, 8), for_loop_var_58873))
        # Assigning a type to the variable 'val' (line 908)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 908, 8), 'val', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 908, 8), for_loop_var_58873))
        # SSA begins for a for statement (line 908)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Type idiom detected: calculating its left and rigth part (line 909)
        # Getting the type of 'val' (line 909)
        val_58874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 909, 20), 'val')
        # Getting the type of 'list' (line 909)
        list_58875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 909, 28), 'list')
        
        (may_be_58876, more_types_in_union_58877) = may_be_type(val_58874, list_58875)

        if may_be_58876:

            if more_types_in_union_58877:
                # Runtime conditional SSA (line 909)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'val' (line 909)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 909, 12), 'val', list_58875())
            
            # Call to (...): (line 910)
            # Processing the call arguments (line 910)
            
            # Obtaining the type of the subscript
            int_58885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 910, 48), 'int')
            # Getting the type of 'val' (line 910)
            val_58886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 910, 44), 'val', False)
            # Obtaining the member '__getitem__' of a type (line 910)
            getitem___58887 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 910, 44), val_58886, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 910)
            subscript_call_result_58888 = invoke(stypy.reporting.localization.Localization(__file__, 910, 44), getitem___58887, int_58885)
            
            # Processing the call keyword arguments (line 910)
            kwargs_58889 = {}
            
            # Call to getattr(...): (line 910)
            # Processing the call arguments (line 910)
            # Getting the type of 'self' (line 910)
            self_58879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 910, 24), 'self', False)
            unicode_58880 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 910, 30), 'unicode', u'set_')
            # Getting the type of 'key' (line 910)
            key_58881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 910, 39), 'key', False)
            # Applying the binary operator '+' (line 910)
            result_add_58882 = python_operator(stypy.reporting.localization.Localization(__file__, 910, 30), '+', unicode_58880, key_58881)
            
            # Processing the call keyword arguments (line 910)
            kwargs_58883 = {}
            # Getting the type of 'getattr' (line 910)
            getattr_58878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 910, 16), 'getattr', False)
            # Calling getattr(args, kwargs) (line 910)
            getattr_call_result_58884 = invoke(stypy.reporting.localization.Localization(__file__, 910, 16), getattr_58878, *[self_58879, result_add_58882], **kwargs_58883)
            
            # Calling (args, kwargs) (line 910)
            _call_result_58890 = invoke(stypy.reporting.localization.Localization(__file__, 910, 16), getattr_call_result_58884, *[subscript_call_result_58888], **kwargs_58889)
            

            if more_types_in_union_58877:
                # Runtime conditional SSA for else branch (line 909)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_58876) or more_types_in_union_58877):
            # Getting the type of 'val' (line 909)
            val_58891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 909, 12), 'val')
            # Assigning a type to the variable 'val' (line 909)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 909, 12), 'val', remove_type_from_union(val_58891, list_58875))
            
            # Call to (...): (line 912)
            # Processing the call arguments (line 912)
            # Getting the type of 'val' (line 912)
            val_58899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 912, 44), 'val', False)
            # Processing the call keyword arguments (line 912)
            kwargs_58900 = {}
            
            # Call to getattr(...): (line 912)
            # Processing the call arguments (line 912)
            # Getting the type of 'self' (line 912)
            self_58893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 912, 24), 'self', False)
            unicode_58894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 912, 30), 'unicode', u'set_')
            # Getting the type of 'key' (line 912)
            key_58895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 912, 39), 'key', False)
            # Applying the binary operator '+' (line 912)
            result_add_58896 = python_operator(stypy.reporting.localization.Localization(__file__, 912, 30), '+', unicode_58894, key_58895)
            
            # Processing the call keyword arguments (line 912)
            kwargs_58897 = {}
            # Getting the type of 'getattr' (line 912)
            getattr_58892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 912, 16), 'getattr', False)
            # Calling getattr(args, kwargs) (line 912)
            getattr_call_result_58898 = invoke(stypy.reporting.localization.Localization(__file__, 912, 16), getattr_58892, *[self_58893, result_add_58896], **kwargs_58897)
            
            # Calling (args, kwargs) (line 912)
            _call_result_58901 = invoke(stypy.reporting.localization.Localization(__file__, 912, 16), getattr_call_result_58898, *[val_58899], **kwargs_58900)
            

            if (may_be_58876 and more_types_in_union_58877):
                # SSA join for if statement (line 909)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'set_fontconfig_pattern(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_fontconfig_pattern' in the type store
        # Getting the type of 'stypy_return_type' (line 897)
        stypy_return_type_58902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 897, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_58902)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_fontconfig_pattern'
        return stypy_return_type_58902


    @norecursion
    def copy(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'copy'
        module_type_store = module_type_store.open_function_context('copy', 914, 4, False)
        # Assigning a type to the variable 'self' (line 915)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 915, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FontProperties.copy.__dict__.__setitem__('stypy_localization', localization)
        FontProperties.copy.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FontProperties.copy.__dict__.__setitem__('stypy_type_store', module_type_store)
        FontProperties.copy.__dict__.__setitem__('stypy_function_name', 'FontProperties.copy')
        FontProperties.copy.__dict__.__setitem__('stypy_param_names_list', [])
        FontProperties.copy.__dict__.__setitem__('stypy_varargs_param_name', None)
        FontProperties.copy.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FontProperties.copy.__dict__.__setitem__('stypy_call_defaults', defaults)
        FontProperties.copy.__dict__.__setitem__('stypy_call_varargs', varargs)
        FontProperties.copy.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FontProperties.copy.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FontProperties.copy', [], None, None, defaults, varargs, kwargs)

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

        unicode_58903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 915, 8), 'unicode', u'Return a deep copy of self')
        
        # Call to FontProperties(...): (line 916)
        # Processing the call keyword arguments (line 916)
        # Getting the type of 'self' (line 916)
        self_58905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 916, 36), 'self', False)
        keyword_58906 = self_58905
        kwargs_58907 = {'_init': keyword_58906}
        # Getting the type of 'FontProperties' (line 916)
        FontProperties_58904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 916, 15), 'FontProperties', False)
        # Calling FontProperties(args, kwargs) (line 916)
        FontProperties_call_result_58908 = invoke(stypy.reporting.localization.Localization(__file__, 916, 15), FontProperties_58904, *[], **kwargs_58907)
        
        # Assigning a type to the variable 'stypy_return_type' (line 916)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 916, 8), 'stypy_return_type', FontProperties_call_result_58908)
        
        # ################# End of 'copy(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'copy' in the type store
        # Getting the type of 'stypy_return_type' (line 914)
        stypy_return_type_58909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 914, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_58909)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'copy'
        return stypy_return_type_58909


# Assigning a type to the variable 'FontProperties' (line 600)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 600, 0), 'FontProperties', FontProperties)

# Assigning a Name to a Name (line 743):
# Getting the type of 'FontProperties'
FontProperties_58910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FontProperties')
# Obtaining the member 'get_style' of a type
get_style_58911 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FontProperties_58910, 'get_style')
# Getting the type of 'FontProperties'
FontProperties_58912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FontProperties')
# Setting the type of the member 'get_slant' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FontProperties_58912, 'get_slant', get_style_58911)

# Assigning a Name to a Name (line 809):
# Getting the type of 'FontProperties'
FontProperties_58913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FontProperties')
# Obtaining the member 'set_family' of a type
set_family_58914 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FontProperties_58913, 'set_family')
# Getting the type of 'FontProperties'
FontProperties_58915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FontProperties')
# Setting the type of the member 'set_name' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FontProperties_58915, 'set_name', set_family_58914)

# Assigning a Name to a Name (line 821):
# Getting the type of 'FontProperties'
FontProperties_58916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FontProperties')
# Obtaining the member 'set_style' of a type
set_style_58917 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FontProperties_58916, 'set_style')
# Getting the type of 'FontProperties'
FontProperties_58918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FontProperties')
# Setting the type of the member 'set_slant' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FontProperties_58918, 'set_slant', set_style_58917)

@norecursion
def ttfdict_to_fnames(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'ttfdict_to_fnames'
    module_type_store = module_type_store.open_function_context('ttfdict_to_fnames', 919, 0, False)
    
    # Passed parameters checking function
    ttfdict_to_fnames.stypy_localization = localization
    ttfdict_to_fnames.stypy_type_of_self = None
    ttfdict_to_fnames.stypy_type_store = module_type_store
    ttfdict_to_fnames.stypy_function_name = 'ttfdict_to_fnames'
    ttfdict_to_fnames.stypy_param_names_list = ['d']
    ttfdict_to_fnames.stypy_varargs_param_name = None
    ttfdict_to_fnames.stypy_kwargs_param_name = None
    ttfdict_to_fnames.stypy_call_defaults = defaults
    ttfdict_to_fnames.stypy_call_varargs = varargs
    ttfdict_to_fnames.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'ttfdict_to_fnames', ['d'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'ttfdict_to_fnames', localization, ['d'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'ttfdict_to_fnames(...)' code ##################

    unicode_58919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 923, (-1)), 'unicode', u'\n    flatten a ttfdict to all the filenames it contains\n    ')
    
    # Assigning a List to a Name (line 924):
    
    # Assigning a List to a Name (line 924):
    
    # Obtaining an instance of the builtin type 'list' (line 924)
    list_58920 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 924, 13), 'list')
    # Adding type elements to the builtin type 'list' instance (line 924)
    
    # Assigning a type to the variable 'fnames' (line 924)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 924, 4), 'fnames', list_58920)
    
    
    # Call to itervalues(...): (line 925)
    # Processing the call arguments (line 925)
    # Getting the type of 'd' (line 925)
    d_58923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 925, 32), 'd', False)
    # Processing the call keyword arguments (line 925)
    kwargs_58924 = {}
    # Getting the type of 'six' (line 925)
    six_58921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 925, 17), 'six', False)
    # Obtaining the member 'itervalues' of a type (line 925)
    itervalues_58922 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 925, 17), six_58921, 'itervalues')
    # Calling itervalues(args, kwargs) (line 925)
    itervalues_call_result_58925 = invoke(stypy.reporting.localization.Localization(__file__, 925, 17), itervalues_58922, *[d_58923], **kwargs_58924)
    
    # Testing the type of a for loop iterable (line 925)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 925, 4), itervalues_call_result_58925)
    # Getting the type of the for loop variable (line 925)
    for_loop_var_58926 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 925, 4), itervalues_call_result_58925)
    # Assigning a type to the variable 'named' (line 925)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 925, 4), 'named', for_loop_var_58926)
    # SSA begins for a for statement (line 925)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Call to itervalues(...): (line 926)
    # Processing the call arguments (line 926)
    # Getting the type of 'named' (line 926)
    named_58929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 926, 37), 'named', False)
    # Processing the call keyword arguments (line 926)
    kwargs_58930 = {}
    # Getting the type of 'six' (line 926)
    six_58927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 926, 22), 'six', False)
    # Obtaining the member 'itervalues' of a type (line 926)
    itervalues_58928 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 926, 22), six_58927, 'itervalues')
    # Calling itervalues(args, kwargs) (line 926)
    itervalues_call_result_58931 = invoke(stypy.reporting.localization.Localization(__file__, 926, 22), itervalues_58928, *[named_58929], **kwargs_58930)
    
    # Testing the type of a for loop iterable (line 926)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 926, 8), itervalues_call_result_58931)
    # Getting the type of the for loop variable (line 926)
    for_loop_var_58932 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 926, 8), itervalues_call_result_58931)
    # Assigning a type to the variable 'styled' (line 926)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 926, 8), 'styled', for_loop_var_58932)
    # SSA begins for a for statement (line 926)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Call to itervalues(...): (line 927)
    # Processing the call arguments (line 927)
    # Getting the type of 'styled' (line 927)
    styled_58935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 927, 43), 'styled', False)
    # Processing the call keyword arguments (line 927)
    kwargs_58936 = {}
    # Getting the type of 'six' (line 927)
    six_58933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 927, 28), 'six', False)
    # Obtaining the member 'itervalues' of a type (line 927)
    itervalues_58934 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 927, 28), six_58933, 'itervalues')
    # Calling itervalues(args, kwargs) (line 927)
    itervalues_call_result_58937 = invoke(stypy.reporting.localization.Localization(__file__, 927, 28), itervalues_58934, *[styled_58935], **kwargs_58936)
    
    # Testing the type of a for loop iterable (line 927)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 927, 12), itervalues_call_result_58937)
    # Getting the type of the for loop variable (line 927)
    for_loop_var_58938 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 927, 12), itervalues_call_result_58937)
    # Assigning a type to the variable 'variantd' (line 927)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 927, 12), 'variantd', for_loop_var_58938)
    # SSA begins for a for statement (line 927)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Call to itervalues(...): (line 928)
    # Processing the call arguments (line 928)
    # Getting the type of 'variantd' (line 928)
    variantd_58941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 928, 46), 'variantd', False)
    # Processing the call keyword arguments (line 928)
    kwargs_58942 = {}
    # Getting the type of 'six' (line 928)
    six_58939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 928, 31), 'six', False)
    # Obtaining the member 'itervalues' of a type (line 928)
    itervalues_58940 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 928, 31), six_58939, 'itervalues')
    # Calling itervalues(args, kwargs) (line 928)
    itervalues_call_result_58943 = invoke(stypy.reporting.localization.Localization(__file__, 928, 31), itervalues_58940, *[variantd_58941], **kwargs_58942)
    
    # Testing the type of a for loop iterable (line 928)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 928, 16), itervalues_call_result_58943)
    # Getting the type of the for loop variable (line 928)
    for_loop_var_58944 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 928, 16), itervalues_call_result_58943)
    # Assigning a type to the variable 'weightd' (line 928)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 928, 16), 'weightd', for_loop_var_58944)
    # SSA begins for a for statement (line 928)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Call to itervalues(...): (line 929)
    # Processing the call arguments (line 929)
    # Getting the type of 'weightd' (line 929)
    weightd_58947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 929, 51), 'weightd', False)
    # Processing the call keyword arguments (line 929)
    kwargs_58948 = {}
    # Getting the type of 'six' (line 929)
    six_58945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 929, 36), 'six', False)
    # Obtaining the member 'itervalues' of a type (line 929)
    itervalues_58946 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 929, 36), six_58945, 'itervalues')
    # Calling itervalues(args, kwargs) (line 929)
    itervalues_call_result_58949 = invoke(stypy.reporting.localization.Localization(__file__, 929, 36), itervalues_58946, *[weightd_58947], **kwargs_58948)
    
    # Testing the type of a for loop iterable (line 929)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 929, 20), itervalues_call_result_58949)
    # Getting the type of the for loop variable (line 929)
    for_loop_var_58950 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 929, 20), itervalues_call_result_58949)
    # Assigning a type to the variable 'stretchd' (line 929)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 929, 20), 'stretchd', for_loop_var_58950)
    # SSA begins for a for statement (line 929)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Call to itervalues(...): (line 930)
    # Processing the call arguments (line 930)
    # Getting the type of 'stretchd' (line 930)
    stretchd_58953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 930, 52), 'stretchd', False)
    # Processing the call keyword arguments (line 930)
    kwargs_58954 = {}
    # Getting the type of 'six' (line 930)
    six_58951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 930, 37), 'six', False)
    # Obtaining the member 'itervalues' of a type (line 930)
    itervalues_58952 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 930, 37), six_58951, 'itervalues')
    # Calling itervalues(args, kwargs) (line 930)
    itervalues_call_result_58955 = invoke(stypy.reporting.localization.Localization(__file__, 930, 37), itervalues_58952, *[stretchd_58953], **kwargs_58954)
    
    # Testing the type of a for loop iterable (line 930)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 930, 24), itervalues_call_result_58955)
    # Getting the type of the for loop variable (line 930)
    for_loop_var_58956 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 930, 24), itervalues_call_result_58955)
    # Assigning a type to the variable 'fname' (line 930)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 930, 24), 'fname', for_loop_var_58956)
    # SSA begins for a for statement (line 930)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to append(...): (line 931)
    # Processing the call arguments (line 931)
    # Getting the type of 'fname' (line 931)
    fname_58959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 931, 42), 'fname', False)
    # Processing the call keyword arguments (line 931)
    kwargs_58960 = {}
    # Getting the type of 'fnames' (line 931)
    fnames_58957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 931, 28), 'fnames', False)
    # Obtaining the member 'append' of a type (line 931)
    append_58958 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 931, 28), fnames_58957, 'append')
    # Calling append(args, kwargs) (line 931)
    append_call_result_58961 = invoke(stypy.reporting.localization.Localization(__file__, 931, 28), append_58958, *[fname_58959], **kwargs_58960)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'fnames' (line 932)
    fnames_58962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 932, 11), 'fnames')
    # Assigning a type to the variable 'stypy_return_type' (line 932)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 932, 4), 'stypy_return_type', fnames_58962)
    
    # ################# End of 'ttfdict_to_fnames(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'ttfdict_to_fnames' in the type store
    # Getting the type of 'stypy_return_type' (line 919)
    stypy_return_type_58963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 919, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_58963)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'ttfdict_to_fnames'
    return stypy_return_type_58963

# Assigning a type to the variable 'ttfdict_to_fnames' (line 919)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 919, 0), 'ttfdict_to_fnames', ttfdict_to_fnames)
# Declaration of the 'JSONEncoder' class
# Getting the type of 'json' (line 935)
json_58964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 935, 18), 'json')
# Obtaining the member 'JSONEncoder' of a type (line 935)
JSONEncoder_58965 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 935, 18), json_58964, 'JSONEncoder')

class JSONEncoder(JSONEncoder_58965, ):

    @norecursion
    def default(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'default'
        module_type_store = module_type_store.open_function_context('default', 936, 4, False)
        # Assigning a type to the variable 'self' (line 937)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 937, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        JSONEncoder.default.__dict__.__setitem__('stypy_localization', localization)
        JSONEncoder.default.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        JSONEncoder.default.__dict__.__setitem__('stypy_type_store', module_type_store)
        JSONEncoder.default.__dict__.__setitem__('stypy_function_name', 'JSONEncoder.default')
        JSONEncoder.default.__dict__.__setitem__('stypy_param_names_list', ['o'])
        JSONEncoder.default.__dict__.__setitem__('stypy_varargs_param_name', None)
        JSONEncoder.default.__dict__.__setitem__('stypy_kwargs_param_name', None)
        JSONEncoder.default.__dict__.__setitem__('stypy_call_defaults', defaults)
        JSONEncoder.default.__dict__.__setitem__('stypy_call_varargs', varargs)
        JSONEncoder.default.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        JSONEncoder.default.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'JSONEncoder.default', ['o'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'default', localization, ['o'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'default(...)' code ##################

        
        
        # Call to isinstance(...): (line 937)
        # Processing the call arguments (line 937)
        # Getting the type of 'o' (line 937)
        o_58967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 937, 22), 'o', False)
        # Getting the type of 'FontManager' (line 937)
        FontManager_58968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 937, 25), 'FontManager', False)
        # Processing the call keyword arguments (line 937)
        kwargs_58969 = {}
        # Getting the type of 'isinstance' (line 937)
        isinstance_58966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 937, 11), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 937)
        isinstance_call_result_58970 = invoke(stypy.reporting.localization.Localization(__file__, 937, 11), isinstance_58966, *[o_58967, FontManager_58968], **kwargs_58969)
        
        # Testing the type of an if condition (line 937)
        if_condition_58971 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 937, 8), isinstance_call_result_58970)
        # Assigning a type to the variable 'if_condition_58971' (line 937)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 937, 8), 'if_condition_58971', if_condition_58971)
        # SSA begins for if statement (line 937)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to dict(...): (line 938)
        # Processing the call arguments (line 938)
        # Getting the type of 'o' (line 938)
        o_58973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 938, 24), 'o', False)
        # Obtaining the member '__dict__' of a type (line 938)
        dict___58974 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 938, 24), o_58973, '__dict__')
        # Processing the call keyword arguments (line 938)
        unicode_58975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 938, 43), 'unicode', u'FontManager')
        keyword_58976 = unicode_58975
        kwargs_58977 = {'_class': keyword_58976}
        # Getting the type of 'dict' (line 938)
        dict_58972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 938, 19), 'dict', False)
        # Calling dict(args, kwargs) (line 938)
        dict_call_result_58978 = invoke(stypy.reporting.localization.Localization(__file__, 938, 19), dict_58972, *[dict___58974], **kwargs_58977)
        
        # Assigning a type to the variable 'stypy_return_type' (line 938)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 938, 12), 'stypy_return_type', dict_call_result_58978)
        # SSA branch for the else part of an if statement (line 937)
        module_type_store.open_ssa_branch('else')
        
        
        # Call to isinstance(...): (line 939)
        # Processing the call arguments (line 939)
        # Getting the type of 'o' (line 939)
        o_58980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 939, 24), 'o', False)
        # Getting the type of 'FontEntry' (line 939)
        FontEntry_58981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 939, 27), 'FontEntry', False)
        # Processing the call keyword arguments (line 939)
        kwargs_58982 = {}
        # Getting the type of 'isinstance' (line 939)
        isinstance_58979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 939, 13), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 939)
        isinstance_call_result_58983 = invoke(stypy.reporting.localization.Localization(__file__, 939, 13), isinstance_58979, *[o_58980, FontEntry_58981], **kwargs_58982)
        
        # Testing the type of an if condition (line 939)
        if_condition_58984 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 939, 13), isinstance_call_result_58983)
        # Assigning a type to the variable 'if_condition_58984' (line 939)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 939, 13), 'if_condition_58984', if_condition_58984)
        # SSA begins for if statement (line 939)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to dict(...): (line 940)
        # Processing the call arguments (line 940)
        # Getting the type of 'o' (line 940)
        o_58986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 940, 24), 'o', False)
        # Obtaining the member '__dict__' of a type (line 940)
        dict___58987 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 940, 24), o_58986, '__dict__')
        # Processing the call keyword arguments (line 940)
        unicode_58988 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 940, 43), 'unicode', u'FontEntry')
        keyword_58989 = unicode_58988
        kwargs_58990 = {'_class': keyword_58989}
        # Getting the type of 'dict' (line 940)
        dict_58985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 940, 19), 'dict', False)
        # Calling dict(args, kwargs) (line 940)
        dict_call_result_58991 = invoke(stypy.reporting.localization.Localization(__file__, 940, 19), dict_58985, *[dict___58987], **kwargs_58990)
        
        # Assigning a type to the variable 'stypy_return_type' (line 940)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 940, 12), 'stypy_return_type', dict_call_result_58991)
        # SSA branch for the else part of an if statement (line 939)
        module_type_store.open_ssa_branch('else')
        
        # Call to default(...): (line 942)
        # Processing the call arguments (line 942)
        # Getting the type of 'o' (line 942)
        o_58998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 942, 52), 'o', False)
        # Processing the call keyword arguments (line 942)
        kwargs_58999 = {}
        
        # Call to super(...): (line 942)
        # Processing the call arguments (line 942)
        # Getting the type of 'JSONEncoder' (line 942)
        JSONEncoder_58993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 942, 25), 'JSONEncoder', False)
        # Getting the type of 'self' (line 942)
        self_58994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 942, 38), 'self', False)
        # Processing the call keyword arguments (line 942)
        kwargs_58995 = {}
        # Getting the type of 'super' (line 942)
        super_58992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 942, 19), 'super', False)
        # Calling super(args, kwargs) (line 942)
        super_call_result_58996 = invoke(stypy.reporting.localization.Localization(__file__, 942, 19), super_58992, *[JSONEncoder_58993, self_58994], **kwargs_58995)
        
        # Obtaining the member 'default' of a type (line 942)
        default_58997 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 942, 19), super_call_result_58996, 'default')
        # Calling default(args, kwargs) (line 942)
        default_call_result_59000 = invoke(stypy.reporting.localization.Localization(__file__, 942, 19), default_58997, *[o_58998], **kwargs_58999)
        
        # Assigning a type to the variable 'stypy_return_type' (line 942)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 942, 12), 'stypy_return_type', default_call_result_59000)
        # SSA join for if statement (line 939)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 937)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'default(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'default' in the type store
        # Getting the type of 'stypy_return_type' (line 936)
        stypy_return_type_59001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 936, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_59001)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'default'
        return stypy_return_type_59001


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 935, 0, False)
        # Assigning a type to the variable 'self' (line 936)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 936, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'JSONEncoder.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'JSONEncoder' (line 935)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 935, 0), 'JSONEncoder', JSONEncoder)

@norecursion
def _json_decode(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_json_decode'
    module_type_store = module_type_store.open_function_context('_json_decode', 945, 0, False)
    
    # Passed parameters checking function
    _json_decode.stypy_localization = localization
    _json_decode.stypy_type_of_self = None
    _json_decode.stypy_type_store = module_type_store
    _json_decode.stypy_function_name = '_json_decode'
    _json_decode.stypy_param_names_list = ['o']
    _json_decode.stypy_varargs_param_name = None
    _json_decode.stypy_kwargs_param_name = None
    _json_decode.stypy_call_defaults = defaults
    _json_decode.stypy_call_varargs = varargs
    _json_decode.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_json_decode', ['o'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_json_decode', localization, ['o'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_json_decode(...)' code ##################

    
    # Assigning a Call to a Name (line 946):
    
    # Assigning a Call to a Name (line 946):
    
    # Call to pop(...): (line 946)
    # Processing the call arguments (line 946)
    unicode_59004 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 946, 16), 'unicode', u'_class')
    # Getting the type of 'None' (line 946)
    None_59005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 946, 26), 'None', False)
    # Processing the call keyword arguments (line 946)
    kwargs_59006 = {}
    # Getting the type of 'o' (line 946)
    o_59002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 946, 10), 'o', False)
    # Obtaining the member 'pop' of a type (line 946)
    pop_59003 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 946, 10), o_59002, 'pop')
    # Calling pop(args, kwargs) (line 946)
    pop_call_result_59007 = invoke(stypy.reporting.localization.Localization(__file__, 946, 10), pop_59003, *[unicode_59004, None_59005], **kwargs_59006)
    
    # Assigning a type to the variable 'cls' (line 946)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 946, 4), 'cls', pop_call_result_59007)
    
    # Type idiom detected: calculating its left and rigth part (line 947)
    # Getting the type of 'cls' (line 947)
    cls_59008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 947, 7), 'cls')
    # Getting the type of 'None' (line 947)
    None_59009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 947, 14), 'None')
    
    (may_be_59010, more_types_in_union_59011) = may_be_none(cls_59008, None_59009)

    if may_be_59010:

        if more_types_in_union_59011:
            # Runtime conditional SSA (line 947)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Getting the type of 'o' (line 948)
        o_59012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 948, 15), 'o')
        # Assigning a type to the variable 'stypy_return_type' (line 948)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 948, 8), 'stypy_return_type', o_59012)

        if more_types_in_union_59011:
            # Runtime conditional SSA for else branch (line 947)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_59010) or more_types_in_union_59011):
        
        
        # Getting the type of 'cls' (line 949)
        cls_59013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 949, 9), 'cls')
        unicode_59014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 949, 16), 'unicode', u'FontManager')
        # Applying the binary operator '==' (line 949)
        result_eq_59015 = python_operator(stypy.reporting.localization.Localization(__file__, 949, 9), '==', cls_59013, unicode_59014)
        
        # Testing the type of an if condition (line 949)
        if_condition_59016 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 949, 9), result_eq_59015)
        # Assigning a type to the variable 'if_condition_59016' (line 949)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 949, 9), 'if_condition_59016', if_condition_59016)
        # SSA begins for if statement (line 949)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 950):
        
        # Assigning a Call to a Name (line 950):
        
        # Call to __new__(...): (line 950)
        # Processing the call arguments (line 950)
        # Getting the type of 'FontManager' (line 950)
        FontManager_59019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 950, 32), 'FontManager', False)
        # Processing the call keyword arguments (line 950)
        kwargs_59020 = {}
        # Getting the type of 'FontManager' (line 950)
        FontManager_59017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 950, 12), 'FontManager', False)
        # Obtaining the member '__new__' of a type (line 950)
        new___59018 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 950, 12), FontManager_59017, '__new__')
        # Calling __new__(args, kwargs) (line 950)
        new___call_result_59021 = invoke(stypy.reporting.localization.Localization(__file__, 950, 12), new___59018, *[FontManager_59019], **kwargs_59020)
        
        # Assigning a type to the variable 'r' (line 950)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 950, 8), 'r', new___call_result_59021)
        
        # Call to update(...): (line 951)
        # Processing the call arguments (line 951)
        # Getting the type of 'o' (line 951)
        o_59025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 951, 26), 'o', False)
        # Processing the call keyword arguments (line 951)
        kwargs_59026 = {}
        # Getting the type of 'r' (line 951)
        r_59022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 951, 8), 'r', False)
        # Obtaining the member '__dict__' of a type (line 951)
        dict___59023 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 951, 8), r_59022, '__dict__')
        # Obtaining the member 'update' of a type (line 951)
        update_59024 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 951, 8), dict___59023, 'update')
        # Calling update(args, kwargs) (line 951)
        update_call_result_59027 = invoke(stypy.reporting.localization.Localization(__file__, 951, 8), update_59024, *[o_59025], **kwargs_59026)
        
        # Getting the type of 'r' (line 952)
        r_59028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 952, 15), 'r')
        # Assigning a type to the variable 'stypy_return_type' (line 952)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 952, 8), 'stypy_return_type', r_59028)
        # SSA branch for the else part of an if statement (line 949)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'cls' (line 953)
        cls_59029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 953, 9), 'cls')
        unicode_59030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 953, 16), 'unicode', u'FontEntry')
        # Applying the binary operator '==' (line 953)
        result_eq_59031 = python_operator(stypy.reporting.localization.Localization(__file__, 953, 9), '==', cls_59029, unicode_59030)
        
        # Testing the type of an if condition (line 953)
        if_condition_59032 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 953, 9), result_eq_59031)
        # Assigning a type to the variable 'if_condition_59032' (line 953)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 953, 9), 'if_condition_59032', if_condition_59032)
        # SSA begins for if statement (line 953)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 954):
        
        # Assigning a Call to a Name (line 954):
        
        # Call to __new__(...): (line 954)
        # Processing the call arguments (line 954)
        # Getting the type of 'FontEntry' (line 954)
        FontEntry_59035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 954, 30), 'FontEntry', False)
        # Processing the call keyword arguments (line 954)
        kwargs_59036 = {}
        # Getting the type of 'FontEntry' (line 954)
        FontEntry_59033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 954, 12), 'FontEntry', False)
        # Obtaining the member '__new__' of a type (line 954)
        new___59034 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 954, 12), FontEntry_59033, '__new__')
        # Calling __new__(args, kwargs) (line 954)
        new___call_result_59037 = invoke(stypy.reporting.localization.Localization(__file__, 954, 12), new___59034, *[FontEntry_59035], **kwargs_59036)
        
        # Assigning a type to the variable 'r' (line 954)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 954, 8), 'r', new___call_result_59037)
        
        # Call to update(...): (line 955)
        # Processing the call arguments (line 955)
        # Getting the type of 'o' (line 955)
        o_59041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 955, 26), 'o', False)
        # Processing the call keyword arguments (line 955)
        kwargs_59042 = {}
        # Getting the type of 'r' (line 955)
        r_59038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 955, 8), 'r', False)
        # Obtaining the member '__dict__' of a type (line 955)
        dict___59039 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 955, 8), r_59038, '__dict__')
        # Obtaining the member 'update' of a type (line 955)
        update_59040 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 955, 8), dict___59039, 'update')
        # Calling update(args, kwargs) (line 955)
        update_call_result_59043 = invoke(stypy.reporting.localization.Localization(__file__, 955, 8), update_59040, *[o_59041], **kwargs_59042)
        
        # Getting the type of 'r' (line 956)
        r_59044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 956, 15), 'r')
        # Assigning a type to the variable 'stypy_return_type' (line 956)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 956, 8), 'stypy_return_type', r_59044)
        # SSA branch for the else part of an if statement (line 953)
        module_type_store.open_ssa_branch('else')
        
        # Call to ValueError(...): (line 958)
        # Processing the call arguments (line 958)
        unicode_59046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 958, 25), 'unicode', u"don't know how to deserialize _class=%s")
        # Getting the type of 'cls' (line 958)
        cls_59047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 958, 69), 'cls', False)
        # Applying the binary operator '%' (line 958)
        result_mod_59048 = python_operator(stypy.reporting.localization.Localization(__file__, 958, 25), '%', unicode_59046, cls_59047)
        
        # Processing the call keyword arguments (line 958)
        kwargs_59049 = {}
        # Getting the type of 'ValueError' (line 958)
        ValueError_59045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 958, 14), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 958)
        ValueError_call_result_59050 = invoke(stypy.reporting.localization.Localization(__file__, 958, 14), ValueError_59045, *[result_mod_59048], **kwargs_59049)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 958, 8), ValueError_call_result_59050, 'raise parameter', BaseException)
        # SSA join for if statement (line 953)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 949)
        module_type_store = module_type_store.join_ssa_context()
        

        if (may_be_59010 and more_types_in_union_59011):
            # SSA join for if statement (line 947)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # ################# End of '_json_decode(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_json_decode' in the type store
    # Getting the type of 'stypy_return_type' (line 945)
    stypy_return_type_59051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 945, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_59051)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_json_decode'
    return stypy_return_type_59051

# Assigning a type to the variable '_json_decode' (line 945)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 945, 0), '_json_decode', _json_decode)

@norecursion
def json_dump(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'json_dump'
    module_type_store = module_type_store.open_function_context('json_dump', 961, 0, False)
    
    # Passed parameters checking function
    json_dump.stypy_localization = localization
    json_dump.stypy_type_of_self = None
    json_dump.stypy_type_store = module_type_store
    json_dump.stypy_function_name = 'json_dump'
    json_dump.stypy_param_names_list = ['data', 'filename']
    json_dump.stypy_varargs_param_name = None
    json_dump.stypy_kwargs_param_name = None
    json_dump.stypy_call_defaults = defaults
    json_dump.stypy_call_varargs = varargs
    json_dump.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'json_dump', ['data', 'filename'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'json_dump', localization, ['data', 'filename'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'json_dump(...)' code ##################

    unicode_59052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 963, (-1)), 'unicode', u'Dumps a data structure as JSON in the named file.\n    Handles FontManager and its fields.')
    
    # Call to open(...): (line 965)
    # Processing the call arguments (line 965)
    # Getting the type of 'filename' (line 965)
    filename_59054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 965, 14), 'filename', False)
    unicode_59055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 965, 24), 'unicode', u'w')
    # Processing the call keyword arguments (line 965)
    kwargs_59056 = {}
    # Getting the type of 'open' (line 965)
    open_59053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 965, 9), 'open', False)
    # Calling open(args, kwargs) (line 965)
    open_call_result_59057 = invoke(stypy.reporting.localization.Localization(__file__, 965, 9), open_59053, *[filename_59054, unicode_59055], **kwargs_59056)
    
    with_59058 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 965, 9), open_call_result_59057, 'with parameter', '__enter__', '__exit__')

    if with_59058:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 965)
        enter___59059 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 965, 9), open_call_result_59057, '__enter__')
        with_enter_59060 = invoke(stypy.reporting.localization.Localization(__file__, 965, 9), enter___59059)
        # Assigning a type to the variable 'fh' (line 965)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 965, 9), 'fh', with_enter_59060)
        
        # Call to dump(...): (line 966)
        # Processing the call arguments (line 966)
        # Getting the type of 'data' (line 966)
        data_59063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 966, 18), 'data', False)
        # Getting the type of 'fh' (line 966)
        fh_59064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 966, 24), 'fh', False)
        # Processing the call keyword arguments (line 966)
        # Getting the type of 'JSONEncoder' (line 966)
        JSONEncoder_59065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 966, 32), 'JSONEncoder', False)
        keyword_59066 = JSONEncoder_59065
        int_59067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 966, 52), 'int')
        keyword_59068 = int_59067
        kwargs_59069 = {'indent': keyword_59068, 'cls': keyword_59066}
        # Getting the type of 'json' (line 966)
        json_59061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 966, 8), 'json', False)
        # Obtaining the member 'dump' of a type (line 966)
        dump_59062 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 966, 8), json_59061, 'dump')
        # Calling dump(args, kwargs) (line 966)
        dump_call_result_59070 = invoke(stypy.reporting.localization.Localization(__file__, 966, 8), dump_59062, *[data_59063, fh_59064], **kwargs_59069)
        
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 965)
        exit___59071 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 965, 9), open_call_result_59057, '__exit__')
        with_exit_59072 = invoke(stypy.reporting.localization.Localization(__file__, 965, 9), exit___59071, None, None, None)

    
    # ################# End of 'json_dump(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'json_dump' in the type store
    # Getting the type of 'stypy_return_type' (line 961)
    stypy_return_type_59073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 961, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_59073)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'json_dump'
    return stypy_return_type_59073

# Assigning a type to the variable 'json_dump' (line 961)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 961, 0), 'json_dump', json_dump)

@norecursion
def json_load(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'json_load'
    module_type_store = module_type_store.open_function_context('json_load', 969, 0, False)
    
    # Passed parameters checking function
    json_load.stypy_localization = localization
    json_load.stypy_type_of_self = None
    json_load.stypy_type_store = module_type_store
    json_load.stypy_function_name = 'json_load'
    json_load.stypy_param_names_list = ['filename']
    json_load.stypy_varargs_param_name = None
    json_load.stypy_kwargs_param_name = None
    json_load.stypy_call_defaults = defaults
    json_load.stypy_call_varargs = varargs
    json_load.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'json_load', ['filename'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'json_load', localization, ['filename'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'json_load(...)' code ##################

    unicode_59074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 971, (-1)), 'unicode', u'Loads a data structure as JSON from the named file.\n    Handles FontManager and its fields.')
    
    # Call to open(...): (line 973)
    # Processing the call arguments (line 973)
    # Getting the type of 'filename' (line 973)
    filename_59076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 973, 14), 'filename', False)
    unicode_59077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 973, 24), 'unicode', u'r')
    # Processing the call keyword arguments (line 973)
    kwargs_59078 = {}
    # Getting the type of 'open' (line 973)
    open_59075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 973, 9), 'open', False)
    # Calling open(args, kwargs) (line 973)
    open_call_result_59079 = invoke(stypy.reporting.localization.Localization(__file__, 973, 9), open_59075, *[filename_59076, unicode_59077], **kwargs_59078)
    
    with_59080 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 973, 9), open_call_result_59079, 'with parameter', '__enter__', '__exit__')

    if with_59080:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 973)
        enter___59081 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 973, 9), open_call_result_59079, '__enter__')
        with_enter_59082 = invoke(stypy.reporting.localization.Localization(__file__, 973, 9), enter___59081)
        # Assigning a type to the variable 'fh' (line 973)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 973, 9), 'fh', with_enter_59082)
        
        # Call to load(...): (line 974)
        # Processing the call arguments (line 974)
        # Getting the type of 'fh' (line 974)
        fh_59085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 974, 25), 'fh', False)
        # Processing the call keyword arguments (line 974)
        # Getting the type of '_json_decode' (line 974)
        _json_decode_59086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 974, 41), '_json_decode', False)
        keyword_59087 = _json_decode_59086
        kwargs_59088 = {'object_hook': keyword_59087}
        # Getting the type of 'json' (line 974)
        json_59083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 974, 15), 'json', False)
        # Obtaining the member 'load' of a type (line 974)
        load_59084 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 974, 15), json_59083, 'load')
        # Calling load(args, kwargs) (line 974)
        load_call_result_59089 = invoke(stypy.reporting.localization.Localization(__file__, 974, 15), load_59084, *[fh_59085], **kwargs_59088)
        
        # Assigning a type to the variable 'stypy_return_type' (line 974)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 974, 8), 'stypy_return_type', load_call_result_59089)
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 973)
        exit___59090 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 973, 9), open_call_result_59079, '__exit__')
        with_exit_59091 = invoke(stypy.reporting.localization.Localization(__file__, 973, 9), exit___59090, None, None, None)

    
    # ################# End of 'json_load(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'json_load' in the type store
    # Getting the type of 'stypy_return_type' (line 969)
    stypy_return_type_59092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 969, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_59092)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'json_load'
    return stypy_return_type_59092

# Assigning a type to the variable 'json_load' (line 969)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 969, 0), 'json_load', json_load)

@norecursion
def _normalize_font_family(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_normalize_font_family'
    module_type_store = module_type_store.open_function_context('_normalize_font_family', 977, 0, False)
    
    # Passed parameters checking function
    _normalize_font_family.stypy_localization = localization
    _normalize_font_family.stypy_type_of_self = None
    _normalize_font_family.stypy_type_store = module_type_store
    _normalize_font_family.stypy_function_name = '_normalize_font_family'
    _normalize_font_family.stypy_param_names_list = ['family']
    _normalize_font_family.stypy_varargs_param_name = None
    _normalize_font_family.stypy_kwargs_param_name = None
    _normalize_font_family.stypy_call_defaults = defaults
    _normalize_font_family.stypy_call_varargs = varargs
    _normalize_font_family.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_normalize_font_family', ['family'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_normalize_font_family', localization, ['family'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_normalize_font_family(...)' code ##################

    
    
    # Call to isinstance(...): (line 978)
    # Processing the call arguments (line 978)
    # Getting the type of 'family' (line 978)
    family_59094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 978, 18), 'family', False)
    # Getting the type of 'six' (line 978)
    six_59095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 978, 26), 'six', False)
    # Obtaining the member 'string_types' of a type (line 978)
    string_types_59096 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 978, 26), six_59095, 'string_types')
    # Processing the call keyword arguments (line 978)
    kwargs_59097 = {}
    # Getting the type of 'isinstance' (line 978)
    isinstance_59093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 978, 7), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 978)
    isinstance_call_result_59098 = invoke(stypy.reporting.localization.Localization(__file__, 978, 7), isinstance_59093, *[family_59094, string_types_59096], **kwargs_59097)
    
    # Testing the type of an if condition (line 978)
    if_condition_59099 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 978, 4), isinstance_call_result_59098)
    # Assigning a type to the variable 'if_condition_59099' (line 978)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 978, 4), 'if_condition_59099', if_condition_59099)
    # SSA begins for if statement (line 978)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a List to a Name (line 979):
    
    # Assigning a List to a Name (line 979):
    
    # Obtaining an instance of the builtin type 'list' (line 979)
    list_59100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 979, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 979)
    # Adding element type (line 979)
    
    # Call to text_type(...): (line 979)
    # Processing the call arguments (line 979)
    # Getting the type of 'family' (line 979)
    family_59103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 979, 32), 'family', False)
    # Processing the call keyword arguments (line 979)
    kwargs_59104 = {}
    # Getting the type of 'six' (line 979)
    six_59101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 979, 18), 'six', False)
    # Obtaining the member 'text_type' of a type (line 979)
    text_type_59102 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 979, 18), six_59101, 'text_type')
    # Calling text_type(args, kwargs) (line 979)
    text_type_call_result_59105 = invoke(stypy.reporting.localization.Localization(__file__, 979, 18), text_type_59102, *[family_59103], **kwargs_59104)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 979, 17), list_59100, text_type_call_result_59105)
    
    # Assigning a type to the variable 'family' (line 979)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 979, 8), 'family', list_59100)
    # SSA branch for the else part of an if statement (line 978)
    module_type_store.open_ssa_branch('else')
    
    
    # Call to isinstance(...): (line 980)
    # Processing the call arguments (line 980)
    # Getting the type of 'family' (line 980)
    family_59107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 980, 20), 'family', False)
    # Getting the type of 'Iterable' (line 980)
    Iterable_59108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 980, 28), 'Iterable', False)
    # Processing the call keyword arguments (line 980)
    kwargs_59109 = {}
    # Getting the type of 'isinstance' (line 980)
    isinstance_59106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 980, 9), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 980)
    isinstance_call_result_59110 = invoke(stypy.reporting.localization.Localization(__file__, 980, 9), isinstance_59106, *[family_59107, Iterable_59108], **kwargs_59109)
    
    # Testing the type of an if condition (line 980)
    if_condition_59111 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 980, 9), isinstance_call_result_59110)
    # Assigning a type to the variable 'if_condition_59111' (line 980)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 980, 9), 'if_condition_59111', if_condition_59111)
    # SSA begins for if statement (line 980)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a ListComp to a Name (line 981):
    
    # Assigning a ListComp to a Name (line 981):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'family' (line 981)
    family_59117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 981, 44), 'family')
    comprehension_59118 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 981, 18), family_59117)
    # Assigning a type to the variable 'f' (line 981)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 981, 18), 'f', comprehension_59118)
    
    # Call to text_type(...): (line 981)
    # Processing the call arguments (line 981)
    # Getting the type of 'f' (line 981)
    f_59114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 981, 32), 'f', False)
    # Processing the call keyword arguments (line 981)
    kwargs_59115 = {}
    # Getting the type of 'six' (line 981)
    six_59112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 981, 18), 'six', False)
    # Obtaining the member 'text_type' of a type (line 981)
    text_type_59113 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 981, 18), six_59112, 'text_type')
    # Calling text_type(args, kwargs) (line 981)
    text_type_call_result_59116 = invoke(stypy.reporting.localization.Localization(__file__, 981, 18), text_type_59113, *[f_59114], **kwargs_59115)
    
    list_59119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 981, 18), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 981, 18), list_59119, text_type_call_result_59116)
    # Assigning a type to the variable 'family' (line 981)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 981, 8), 'family', list_59119)
    # SSA join for if statement (line 980)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 978)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'family' (line 982)
    family_59120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 982, 11), 'family')
    # Assigning a type to the variable 'stypy_return_type' (line 982)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 982, 4), 'stypy_return_type', family_59120)
    
    # ################# End of '_normalize_font_family(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_normalize_font_family' in the type store
    # Getting the type of 'stypy_return_type' (line 977)
    stypy_return_type_59121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 977, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_59121)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_normalize_font_family'
    return stypy_return_type_59121

# Assigning a type to the variable '_normalize_font_family' (line 977)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 977, 0), '_normalize_font_family', _normalize_font_family)
# Declaration of the 'TempCache' class

class TempCache(object, ):
    unicode_59122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 992, (-1)), 'unicode', u'\n    A class to store temporary caches that are (a) not saved to disk\n    and (b) invalidated whenever certain font-related\n    rcParams---namely the family lookup lists---are changed or the\n    font cache is reloaded.  This avoids the expensive linear search\n    through all fonts every time a font is looked up.\n    ')
    
    # Assigning a Tuple to a Name (line 995):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 999, 4, False)
        # Assigning a type to the variable 'self' (line 1000)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1000, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TempCache.__init__', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Dict to a Attribute (line 1000):
        
        # Assigning a Dict to a Attribute (line 1000):
        
        # Obtaining an instance of the builtin type 'dict' (line 1000)
        dict_59123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1000, 29), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 1000)
        
        # Getting the type of 'self' (line 1000)
        self_59124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1000, 8), 'self')
        # Setting the type of the member '_lookup_cache' of a type (line 1000)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1000, 8), self_59124, '_lookup_cache', dict_59123)
        
        # Assigning a Call to a Attribute (line 1001):
        
        # Assigning a Call to a Attribute (line 1001):
        
        # Call to make_rcparams_key(...): (line 1001)
        # Processing the call keyword arguments (line 1001)
        kwargs_59127 = {}
        # Getting the type of 'self' (line 1001)
        self_59125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1001, 30), 'self', False)
        # Obtaining the member 'make_rcparams_key' of a type (line 1001)
        make_rcparams_key_59126 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1001, 30), self_59125, 'make_rcparams_key')
        # Calling make_rcparams_key(args, kwargs) (line 1001)
        make_rcparams_key_call_result_59128 = invoke(stypy.reporting.localization.Localization(__file__, 1001, 30), make_rcparams_key_59126, *[], **kwargs_59127)
        
        # Getting the type of 'self' (line 1001)
        self_59129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1001, 8), 'self')
        # Setting the type of the member '_last_rcParams' of a type (line 1001)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1001, 8), self_59129, '_last_rcParams', make_rcparams_key_call_result_59128)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def make_rcparams_key(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'make_rcparams_key'
        module_type_store = module_type_store.open_function_context('make_rcparams_key', 1003, 4, False)
        # Assigning a type to the variable 'self' (line 1004)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1004, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TempCache.make_rcparams_key.__dict__.__setitem__('stypy_localization', localization)
        TempCache.make_rcparams_key.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TempCache.make_rcparams_key.__dict__.__setitem__('stypy_type_store', module_type_store)
        TempCache.make_rcparams_key.__dict__.__setitem__('stypy_function_name', 'TempCache.make_rcparams_key')
        TempCache.make_rcparams_key.__dict__.__setitem__('stypy_param_names_list', [])
        TempCache.make_rcparams_key.__dict__.__setitem__('stypy_varargs_param_name', None)
        TempCache.make_rcparams_key.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TempCache.make_rcparams_key.__dict__.__setitem__('stypy_call_defaults', defaults)
        TempCache.make_rcparams_key.__dict__.__setitem__('stypy_call_varargs', varargs)
        TempCache.make_rcparams_key.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TempCache.make_rcparams_key.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TempCache.make_rcparams_key', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'make_rcparams_key', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'make_rcparams_key(...)' code ##################

        
        # Obtaining an instance of the builtin type 'list' (line 1004)
        list_59130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1004, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 1004)
        # Adding element type (line 1004)
        
        # Call to id(...): (line 1004)
        # Processing the call arguments (line 1004)
        # Getting the type of 'fontManager' (line 1004)
        fontManager_59132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1004, 19), 'fontManager', False)
        # Processing the call keyword arguments (line 1004)
        kwargs_59133 = {}
        # Getting the type of 'id' (line 1004)
        id_59131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1004, 16), 'id', False)
        # Calling id(args, kwargs) (line 1004)
        id_call_result_59134 = invoke(stypy.reporting.localization.Localization(__file__, 1004, 16), id_59131, *[fontManager_59132], **kwargs_59133)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1004, 15), list_59130, id_call_result_59134)
        
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'self' (line 1005)
        self_59139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1005, 41), 'self')
        # Obtaining the member 'invalidating_rcparams' of a type (line 1005)
        invalidating_rcparams_59140 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1005, 41), self_59139, 'invalidating_rcparams')
        comprehension_59141 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1005, 12), invalidating_rcparams_59140)
        # Assigning a type to the variable 'param' (line 1005)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1005, 12), 'param', comprehension_59141)
        
        # Obtaining the type of the subscript
        # Getting the type of 'param' (line 1005)
        param_59135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1005, 21), 'param')
        # Getting the type of 'rcParams' (line 1005)
        rcParams_59136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1005, 12), 'rcParams')
        # Obtaining the member '__getitem__' of a type (line 1005)
        getitem___59137 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1005, 12), rcParams_59136, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1005)
        subscript_call_result_59138 = invoke(stypy.reporting.localization.Localization(__file__, 1005, 12), getitem___59137, param_59135)
        
        list_59142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1005, 12), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1005, 12), list_59142, subscript_call_result_59138)
        # Applying the binary operator '+' (line 1004)
        result_add_59143 = python_operator(stypy.reporting.localization.Localization(__file__, 1004, 15), '+', list_59130, list_59142)
        
        # Assigning a type to the variable 'stypy_return_type' (line 1004)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1004, 8), 'stypy_return_type', result_add_59143)
        
        # ################# End of 'make_rcparams_key(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'make_rcparams_key' in the type store
        # Getting the type of 'stypy_return_type' (line 1003)
        stypy_return_type_59144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1003, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_59144)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'make_rcparams_key'
        return stypy_return_type_59144


    @norecursion
    def get(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get'
        module_type_store = module_type_store.open_function_context('get', 1007, 4, False)
        # Assigning a type to the variable 'self' (line 1008)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1008, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TempCache.get.__dict__.__setitem__('stypy_localization', localization)
        TempCache.get.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TempCache.get.__dict__.__setitem__('stypy_type_store', module_type_store)
        TempCache.get.__dict__.__setitem__('stypy_function_name', 'TempCache.get')
        TempCache.get.__dict__.__setitem__('stypy_param_names_list', ['prop'])
        TempCache.get.__dict__.__setitem__('stypy_varargs_param_name', None)
        TempCache.get.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TempCache.get.__dict__.__setitem__('stypy_call_defaults', defaults)
        TempCache.get.__dict__.__setitem__('stypy_call_varargs', varargs)
        TempCache.get.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TempCache.get.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TempCache.get', ['prop'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get', localization, ['prop'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get(...)' code ##################

        
        # Assigning a Call to a Name (line 1008):
        
        # Assigning a Call to a Name (line 1008):
        
        # Call to make_rcparams_key(...): (line 1008)
        # Processing the call keyword arguments (line 1008)
        kwargs_59147 = {}
        # Getting the type of 'self' (line 1008)
        self_59145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1008, 14), 'self', False)
        # Obtaining the member 'make_rcparams_key' of a type (line 1008)
        make_rcparams_key_59146 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1008, 14), self_59145, 'make_rcparams_key')
        # Calling make_rcparams_key(args, kwargs) (line 1008)
        make_rcparams_key_call_result_59148 = invoke(stypy.reporting.localization.Localization(__file__, 1008, 14), make_rcparams_key_59146, *[], **kwargs_59147)
        
        # Assigning a type to the variable 'key' (line 1008)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1008, 8), 'key', make_rcparams_key_call_result_59148)
        
        
        # Getting the type of 'key' (line 1009)
        key_59149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1009, 11), 'key')
        # Getting the type of 'self' (line 1009)
        self_59150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1009, 18), 'self')
        # Obtaining the member '_last_rcParams' of a type (line 1009)
        _last_rcParams_59151 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1009, 18), self_59150, '_last_rcParams')
        # Applying the binary operator '!=' (line 1009)
        result_ne_59152 = python_operator(stypy.reporting.localization.Localization(__file__, 1009, 11), '!=', key_59149, _last_rcParams_59151)
        
        # Testing the type of an if condition (line 1009)
        if_condition_59153 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1009, 8), result_ne_59152)
        # Assigning a type to the variable 'if_condition_59153' (line 1009)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1009, 8), 'if_condition_59153', if_condition_59153)
        # SSA begins for if statement (line 1009)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Dict to a Attribute (line 1010):
        
        # Assigning a Dict to a Attribute (line 1010):
        
        # Obtaining an instance of the builtin type 'dict' (line 1010)
        dict_59154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1010, 33), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 1010)
        
        # Getting the type of 'self' (line 1010)
        self_59155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1010, 12), 'self')
        # Setting the type of the member '_lookup_cache' of a type (line 1010)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1010, 12), self_59155, '_lookup_cache', dict_59154)
        
        # Assigning a Name to a Attribute (line 1011):
        
        # Assigning a Name to a Attribute (line 1011):
        # Getting the type of 'key' (line 1011)
        key_59156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1011, 34), 'key')
        # Getting the type of 'self' (line 1011)
        self_59157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1011, 12), 'self')
        # Setting the type of the member '_last_rcParams' of a type (line 1011)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1011, 12), self_59157, '_last_rcParams', key_59156)
        # SSA join for if statement (line 1009)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to get(...): (line 1012)
        # Processing the call arguments (line 1012)
        # Getting the type of 'prop' (line 1012)
        prop_59161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1012, 38), 'prop', False)
        # Processing the call keyword arguments (line 1012)
        kwargs_59162 = {}
        # Getting the type of 'self' (line 1012)
        self_59158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1012, 15), 'self', False)
        # Obtaining the member '_lookup_cache' of a type (line 1012)
        _lookup_cache_59159 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1012, 15), self_59158, '_lookup_cache')
        # Obtaining the member 'get' of a type (line 1012)
        get_59160 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1012, 15), _lookup_cache_59159, 'get')
        # Calling get(args, kwargs) (line 1012)
        get_call_result_59163 = invoke(stypy.reporting.localization.Localization(__file__, 1012, 15), get_59160, *[prop_59161], **kwargs_59162)
        
        # Assigning a type to the variable 'stypy_return_type' (line 1012)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1012, 8), 'stypy_return_type', get_call_result_59163)
        
        # ################# End of 'get(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get' in the type store
        # Getting the type of 'stypy_return_type' (line 1007)
        stypy_return_type_59164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1007, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_59164)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get'
        return stypy_return_type_59164


    @norecursion
    def set(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set'
        module_type_store = module_type_store.open_function_context('set', 1014, 4, False)
        # Assigning a type to the variable 'self' (line 1015)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1015, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TempCache.set.__dict__.__setitem__('stypy_localization', localization)
        TempCache.set.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TempCache.set.__dict__.__setitem__('stypy_type_store', module_type_store)
        TempCache.set.__dict__.__setitem__('stypy_function_name', 'TempCache.set')
        TempCache.set.__dict__.__setitem__('stypy_param_names_list', ['prop', 'value'])
        TempCache.set.__dict__.__setitem__('stypy_varargs_param_name', None)
        TempCache.set.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TempCache.set.__dict__.__setitem__('stypy_call_defaults', defaults)
        TempCache.set.__dict__.__setitem__('stypy_call_varargs', varargs)
        TempCache.set.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TempCache.set.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TempCache.set', ['prop', 'value'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set', localization, ['prop', 'value'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set(...)' code ##################

        
        # Assigning a Call to a Name (line 1015):
        
        # Assigning a Call to a Name (line 1015):
        
        # Call to make_rcparams_key(...): (line 1015)
        # Processing the call keyword arguments (line 1015)
        kwargs_59167 = {}
        # Getting the type of 'self' (line 1015)
        self_59165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1015, 14), 'self', False)
        # Obtaining the member 'make_rcparams_key' of a type (line 1015)
        make_rcparams_key_59166 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1015, 14), self_59165, 'make_rcparams_key')
        # Calling make_rcparams_key(args, kwargs) (line 1015)
        make_rcparams_key_call_result_59168 = invoke(stypy.reporting.localization.Localization(__file__, 1015, 14), make_rcparams_key_59166, *[], **kwargs_59167)
        
        # Assigning a type to the variable 'key' (line 1015)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1015, 8), 'key', make_rcparams_key_call_result_59168)
        
        
        # Getting the type of 'key' (line 1016)
        key_59169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1016, 11), 'key')
        # Getting the type of 'self' (line 1016)
        self_59170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1016, 18), 'self')
        # Obtaining the member '_last_rcParams' of a type (line 1016)
        _last_rcParams_59171 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1016, 18), self_59170, '_last_rcParams')
        # Applying the binary operator '!=' (line 1016)
        result_ne_59172 = python_operator(stypy.reporting.localization.Localization(__file__, 1016, 11), '!=', key_59169, _last_rcParams_59171)
        
        # Testing the type of an if condition (line 1016)
        if_condition_59173 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1016, 8), result_ne_59172)
        # Assigning a type to the variable 'if_condition_59173' (line 1016)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1016, 8), 'if_condition_59173', if_condition_59173)
        # SSA begins for if statement (line 1016)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Dict to a Attribute (line 1017):
        
        # Assigning a Dict to a Attribute (line 1017):
        
        # Obtaining an instance of the builtin type 'dict' (line 1017)
        dict_59174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1017, 33), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 1017)
        
        # Getting the type of 'self' (line 1017)
        self_59175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1017, 12), 'self')
        # Setting the type of the member '_lookup_cache' of a type (line 1017)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1017, 12), self_59175, '_lookup_cache', dict_59174)
        
        # Assigning a Name to a Attribute (line 1018):
        
        # Assigning a Name to a Attribute (line 1018):
        # Getting the type of 'key' (line 1018)
        key_59176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1018, 34), 'key')
        # Getting the type of 'self' (line 1018)
        self_59177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1018, 12), 'self')
        # Setting the type of the member '_last_rcParams' of a type (line 1018)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1018, 12), self_59177, '_last_rcParams', key_59176)
        # SSA join for if statement (line 1016)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Subscript (line 1019):
        
        # Assigning a Name to a Subscript (line 1019):
        # Getting the type of 'value' (line 1019)
        value_59178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1019, 35), 'value')
        # Getting the type of 'self' (line 1019)
        self_59179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1019, 8), 'self')
        # Obtaining the member '_lookup_cache' of a type (line 1019)
        _lookup_cache_59180 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1019, 8), self_59179, '_lookup_cache')
        # Getting the type of 'prop' (line 1019)
        prop_59181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1019, 27), 'prop')
        # Storing an element on a container (line 1019)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1019, 8), _lookup_cache_59180, (prop_59181, value_59178))
        
        # ################# End of 'set(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set' in the type store
        # Getting the type of 'stypy_return_type' (line 1014)
        stypy_return_type_59182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1014, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_59182)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set'
        return stypy_return_type_59182


# Assigning a type to the variable 'TempCache' (line 985)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 985, 0), 'TempCache', TempCache)

# Assigning a Tuple to a Name (line 995):

# Obtaining an instance of the builtin type 'tuple' (line 996)
tuple_59183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 996, 8), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 996)
# Adding element type (line 996)
unicode_59184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 996, 8), 'unicode', u'font.serif')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 996, 8), tuple_59183, unicode_59184)
# Adding element type (line 996)
unicode_59185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 996, 22), 'unicode', u'font.sans-serif')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 996, 8), tuple_59183, unicode_59185)
# Adding element type (line 996)
unicode_59186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 996, 41), 'unicode', u'font.cursive')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 996, 8), tuple_59183, unicode_59186)
# Adding element type (line 996)
unicode_59187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 996, 57), 'unicode', u'font.fantasy')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 996, 8), tuple_59183, unicode_59187)
# Adding element type (line 996)
unicode_59188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 997, 8), 'unicode', u'font.monospace')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 996, 8), tuple_59183, unicode_59188)

# Getting the type of 'TempCache'
TempCache_59189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TempCache')
# Setting the type of the member 'invalidating_rcparams' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TempCache_59189, 'invalidating_rcparams', tuple_59183)
# Declaration of the 'FontManager' class

class FontManager(object, ):
    unicode_59190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1030, (-1)), 'unicode', u'\n    On import, the :class:`FontManager` singleton instance creates a\n    list of TrueType fonts based on the font properties: name, style,\n    variant, weight, stretch, and size.  The :meth:`findfont` method\n    does a nearest neighbor search to find the font that most closely\n    matches the specification.  If no good enough match is found, a\n    default font is returned.\n    ')
    
    # Assigning a Num to a Name (line 1034):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 1036)
        None_59191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1036, 28), 'None')
        unicode_59192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1036, 41), 'unicode', u'normal')
        defaults = [None_59191, unicode_59192]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 1036, 4, False)
        # Assigning a type to the variable 'self' (line 1037)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1037, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FontManager.__init__', ['size', 'weight'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['size', 'weight'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Attribute to a Attribute (line 1037):
        
        # Assigning a Attribute to a Attribute (line 1037):
        # Getting the type of 'self' (line 1037)
        self_59193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1037, 24), 'self')
        # Obtaining the member '__version__' of a type (line 1037)
        version___59194 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1037, 24), self_59193, '__version__')
        # Getting the type of 'self' (line 1037)
        self_59195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1037, 8), 'self')
        # Setting the type of the member '_version' of a type (line 1037)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1037, 8), self_59195, '_version', version___59194)
        
        # Assigning a Name to a Attribute (line 1039):
        
        # Assigning a Name to a Attribute (line 1039):
        # Getting the type of 'weight' (line 1039)
        weight_59196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1039, 32), 'weight')
        # Getting the type of 'self' (line 1039)
        self_59197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1039, 8), 'self')
        # Setting the type of the member '__default_weight' of a type (line 1039)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1039, 8), self_59197, '__default_weight', weight_59196)
        
        # Assigning a Name to a Attribute (line 1040):
        
        # Assigning a Name to a Attribute (line 1040):
        # Getting the type of 'size' (line 1040)
        size_59198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1040, 28), 'size')
        # Getting the type of 'self' (line 1040)
        self_59199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1040, 8), 'self')
        # Setting the type of the member 'default_size' of a type (line 1040)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1040, 8), self_59199, 'default_size', size_59198)
        
        # Assigning a List to a Name (line 1042):
        
        # Assigning a List to a Name (line 1042):
        
        # Obtaining an instance of the builtin type 'list' (line 1042)
        list_59200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1042, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 1042)
        # Adding element type (line 1042)
        
        # Call to join(...): (line 1042)
        # Processing the call arguments (line 1042)
        
        # Obtaining the type of the subscript
        unicode_59204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1042, 39), 'unicode', u'datapath')
        # Getting the type of 'rcParams' (line 1042)
        rcParams_59205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1042, 30), 'rcParams', False)
        # Obtaining the member '__getitem__' of a type (line 1042)
        getitem___59206 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1042, 30), rcParams_59205, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1042)
        subscript_call_result_59207 = invoke(stypy.reporting.localization.Localization(__file__, 1042, 30), getitem___59206, unicode_59204)
        
        unicode_59208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1042, 52), 'unicode', u'fonts')
        unicode_59209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1042, 61), 'unicode', u'ttf')
        # Processing the call keyword arguments (line 1042)
        kwargs_59210 = {}
        # Getting the type of 'os' (line 1042)
        os_59201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1042, 17), 'os', False)
        # Obtaining the member 'path' of a type (line 1042)
        path_59202 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1042, 17), os_59201, 'path')
        # Obtaining the member 'join' of a type (line 1042)
        join_59203 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1042, 17), path_59202, 'join')
        # Calling join(args, kwargs) (line 1042)
        join_call_result_59211 = invoke(stypy.reporting.localization.Localization(__file__, 1042, 17), join_59203, *[subscript_call_result_59207, unicode_59208, unicode_59209], **kwargs_59210)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1042, 16), list_59200, join_call_result_59211)
        # Adding element type (line 1042)
        
        # Call to join(...): (line 1043)
        # Processing the call arguments (line 1043)
        
        # Obtaining the type of the subscript
        unicode_59215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1043, 39), 'unicode', u'datapath')
        # Getting the type of 'rcParams' (line 1043)
        rcParams_59216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1043, 30), 'rcParams', False)
        # Obtaining the member '__getitem__' of a type (line 1043)
        getitem___59217 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1043, 30), rcParams_59216, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1043)
        subscript_call_result_59218 = invoke(stypy.reporting.localization.Localization(__file__, 1043, 30), getitem___59217, unicode_59215)
        
        unicode_59219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1043, 52), 'unicode', u'fonts')
        unicode_59220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1043, 61), 'unicode', u'afm')
        # Processing the call keyword arguments (line 1043)
        kwargs_59221 = {}
        # Getting the type of 'os' (line 1043)
        os_59212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1043, 17), 'os', False)
        # Obtaining the member 'path' of a type (line 1043)
        path_59213 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1043, 17), os_59212, 'path')
        # Obtaining the member 'join' of a type (line 1043)
        join_59214 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1043, 17), path_59213, 'join')
        # Calling join(args, kwargs) (line 1043)
        join_call_result_59222 = invoke(stypy.reporting.localization.Localization(__file__, 1043, 17), join_59214, *[subscript_call_result_59218, unicode_59219, unicode_59220], **kwargs_59221)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1042, 16), list_59200, join_call_result_59222)
        # Adding element type (line 1042)
        
        # Call to join(...): (line 1044)
        # Processing the call arguments (line 1044)
        
        # Obtaining the type of the subscript
        unicode_59226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1044, 39), 'unicode', u'datapath')
        # Getting the type of 'rcParams' (line 1044)
        rcParams_59227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1044, 30), 'rcParams', False)
        # Obtaining the member '__getitem__' of a type (line 1044)
        getitem___59228 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1044, 30), rcParams_59227, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1044)
        subscript_call_result_59229 = invoke(stypy.reporting.localization.Localization(__file__, 1044, 30), getitem___59228, unicode_59226)
        
        unicode_59230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1044, 52), 'unicode', u'fonts')
        unicode_59231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1044, 61), 'unicode', u'pdfcorefonts')
        # Processing the call keyword arguments (line 1044)
        kwargs_59232 = {}
        # Getting the type of 'os' (line 1044)
        os_59223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1044, 17), 'os', False)
        # Obtaining the member 'path' of a type (line 1044)
        path_59224 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1044, 17), os_59223, 'path')
        # Obtaining the member 'join' of a type (line 1044)
        join_59225 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1044, 17), path_59224, 'join')
        # Calling join(args, kwargs) (line 1044)
        join_call_result_59233 = invoke(stypy.reporting.localization.Localization(__file__, 1044, 17), join_59225, *[subscript_call_result_59229, unicode_59230, unicode_59231], **kwargs_59232)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1042, 16), list_59200, join_call_result_59233)
        
        # Assigning a type to the variable 'paths' (line 1042)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1042, 8), 'paths', list_59200)
        
        
        # Obtaining an instance of the builtin type 'list' (line 1047)
        list_59234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1047, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 1047)
        # Adding element type (line 1047)
        unicode_59235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1047, 25), 'unicode', u'TTFPATH')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1047, 24), list_59234, unicode_59235)
        # Adding element type (line 1047)
        unicode_59236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1047, 36), 'unicode', u'AFMPATH')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1047, 24), list_59234, unicode_59236)
        
        # Testing the type of a for loop iterable (line 1047)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1047, 8), list_59234)
        # Getting the type of the for loop variable (line 1047)
        for_loop_var_59237 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1047, 8), list_59234)
        # Assigning a type to the variable 'pathname' (line 1047)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1047, 8), 'pathname', for_loop_var_59237)
        # SSA begins for a for statement (line 1047)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Getting the type of 'pathname' (line 1048)
        pathname_59238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1048, 15), 'pathname')
        # Getting the type of 'os' (line 1048)
        os_59239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1048, 27), 'os')
        # Obtaining the member 'environ' of a type (line 1048)
        environ_59240 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1048, 27), os_59239, 'environ')
        # Applying the binary operator 'in' (line 1048)
        result_contains_59241 = python_operator(stypy.reporting.localization.Localization(__file__, 1048, 15), 'in', pathname_59238, environ_59240)
        
        # Testing the type of an if condition (line 1048)
        if_condition_59242 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1048, 12), result_contains_59241)
        # Assigning a type to the variable 'if_condition_59242' (line 1048)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1048, 12), 'if_condition_59242', if_condition_59242)
        # SSA begins for if statement (line 1048)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Name (line 1049):
        
        # Assigning a Subscript to a Name (line 1049):
        
        # Obtaining the type of the subscript
        # Getting the type of 'pathname' (line 1049)
        pathname_59243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1049, 37), 'pathname')
        # Getting the type of 'os' (line 1049)
        os_59244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1049, 26), 'os')
        # Obtaining the member 'environ' of a type (line 1049)
        environ_59245 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1049, 26), os_59244, 'environ')
        # Obtaining the member '__getitem__' of a type (line 1049)
        getitem___59246 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1049, 26), environ_59245, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1049)
        subscript_call_result_59247 = invoke(stypy.reporting.localization.Localization(__file__, 1049, 26), getitem___59246, pathname_59243)
        
        # Assigning a type to the variable 'ttfpath' (line 1049)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1049, 16), 'ttfpath', subscript_call_result_59247)
        
        
        
        # Call to find(...): (line 1050)
        # Processing the call arguments (line 1050)
        unicode_59250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1050, 32), 'unicode', u';')
        # Processing the call keyword arguments (line 1050)
        kwargs_59251 = {}
        # Getting the type of 'ttfpath' (line 1050)
        ttfpath_59248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1050, 19), 'ttfpath', False)
        # Obtaining the member 'find' of a type (line 1050)
        find_59249 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1050, 19), ttfpath_59248, 'find')
        # Calling find(args, kwargs) (line 1050)
        find_call_result_59252 = invoke(stypy.reporting.localization.Localization(__file__, 1050, 19), find_59249, *[unicode_59250], **kwargs_59251)
        
        int_59253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1050, 40), 'int')
        # Applying the binary operator '>=' (line 1050)
        result_ge_59254 = python_operator(stypy.reporting.localization.Localization(__file__, 1050, 19), '>=', find_call_result_59252, int_59253)
        
        # Testing the type of an if condition (line 1050)
        if_condition_59255 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1050, 16), result_ge_59254)
        # Assigning a type to the variable 'if_condition_59255' (line 1050)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1050, 16), 'if_condition_59255', if_condition_59255)
        # SSA begins for if statement (line 1050)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to extend(...): (line 1051)
        # Processing the call arguments (line 1051)
        
        # Call to split(...): (line 1051)
        # Processing the call arguments (line 1051)
        unicode_59260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1051, 47), 'unicode', u';')
        # Processing the call keyword arguments (line 1051)
        kwargs_59261 = {}
        # Getting the type of 'ttfpath' (line 1051)
        ttfpath_59258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1051, 33), 'ttfpath', False)
        # Obtaining the member 'split' of a type (line 1051)
        split_59259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1051, 33), ttfpath_59258, 'split')
        # Calling split(args, kwargs) (line 1051)
        split_call_result_59262 = invoke(stypy.reporting.localization.Localization(__file__, 1051, 33), split_59259, *[unicode_59260], **kwargs_59261)
        
        # Processing the call keyword arguments (line 1051)
        kwargs_59263 = {}
        # Getting the type of 'paths' (line 1051)
        paths_59256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1051, 20), 'paths', False)
        # Obtaining the member 'extend' of a type (line 1051)
        extend_59257 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1051, 20), paths_59256, 'extend')
        # Calling extend(args, kwargs) (line 1051)
        extend_call_result_59264 = invoke(stypy.reporting.localization.Localization(__file__, 1051, 20), extend_59257, *[split_call_result_59262], **kwargs_59263)
        
        # SSA branch for the else part of an if statement (line 1050)
        module_type_store.open_ssa_branch('else')
        
        
        
        # Call to find(...): (line 1052)
        # Processing the call arguments (line 1052)
        unicode_59267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1052, 34), 'unicode', u':')
        # Processing the call keyword arguments (line 1052)
        kwargs_59268 = {}
        # Getting the type of 'ttfpath' (line 1052)
        ttfpath_59265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1052, 21), 'ttfpath', False)
        # Obtaining the member 'find' of a type (line 1052)
        find_59266 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1052, 21), ttfpath_59265, 'find')
        # Calling find(args, kwargs) (line 1052)
        find_call_result_59269 = invoke(stypy.reporting.localization.Localization(__file__, 1052, 21), find_59266, *[unicode_59267], **kwargs_59268)
        
        int_59270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1052, 42), 'int')
        # Applying the binary operator '>=' (line 1052)
        result_ge_59271 = python_operator(stypy.reporting.localization.Localization(__file__, 1052, 21), '>=', find_call_result_59269, int_59270)
        
        # Testing the type of an if condition (line 1052)
        if_condition_59272 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1052, 21), result_ge_59271)
        # Assigning a type to the variable 'if_condition_59272' (line 1052)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1052, 21), 'if_condition_59272', if_condition_59272)
        # SSA begins for if statement (line 1052)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to extend(...): (line 1053)
        # Processing the call arguments (line 1053)
        
        # Call to split(...): (line 1053)
        # Processing the call arguments (line 1053)
        unicode_59277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1053, 47), 'unicode', u':')
        # Processing the call keyword arguments (line 1053)
        kwargs_59278 = {}
        # Getting the type of 'ttfpath' (line 1053)
        ttfpath_59275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1053, 33), 'ttfpath', False)
        # Obtaining the member 'split' of a type (line 1053)
        split_59276 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1053, 33), ttfpath_59275, 'split')
        # Calling split(args, kwargs) (line 1053)
        split_call_result_59279 = invoke(stypy.reporting.localization.Localization(__file__, 1053, 33), split_59276, *[unicode_59277], **kwargs_59278)
        
        # Processing the call keyword arguments (line 1053)
        kwargs_59280 = {}
        # Getting the type of 'paths' (line 1053)
        paths_59273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1053, 20), 'paths', False)
        # Obtaining the member 'extend' of a type (line 1053)
        extend_59274 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1053, 20), paths_59273, 'extend')
        # Calling extend(args, kwargs) (line 1053)
        extend_call_result_59281 = invoke(stypy.reporting.localization.Localization(__file__, 1053, 20), extend_59274, *[split_call_result_59279], **kwargs_59280)
        
        # SSA branch for the else part of an if statement (line 1052)
        module_type_store.open_ssa_branch('else')
        
        # Call to append(...): (line 1055)
        # Processing the call arguments (line 1055)
        # Getting the type of 'ttfpath' (line 1055)
        ttfpath_59284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1055, 33), 'ttfpath', False)
        # Processing the call keyword arguments (line 1055)
        kwargs_59285 = {}
        # Getting the type of 'paths' (line 1055)
        paths_59282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1055, 20), 'paths', False)
        # Obtaining the member 'append' of a type (line 1055)
        append_59283 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1055, 20), paths_59282, 'append')
        # Calling append(args, kwargs) (line 1055)
        append_call_result_59286 = invoke(stypy.reporting.localization.Localization(__file__, 1055, 20), append_59283, *[ttfpath_59284], **kwargs_59285)
        
        # SSA join for if statement (line 1052)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 1050)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 1048)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to report(...): (line 1057)
        # Processing the call arguments (line 1057)
        unicode_59289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1057, 23), 'unicode', u'font search path %s')
        
        # Call to str(...): (line 1057)
        # Processing the call arguments (line 1057)
        # Getting the type of 'paths' (line 1057)
        paths_59291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1057, 50), 'paths', False)
        # Processing the call keyword arguments (line 1057)
        kwargs_59292 = {}
        # Getting the type of 'str' (line 1057)
        str_59290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1057, 46), 'str', False)
        # Calling str(args, kwargs) (line 1057)
        str_call_result_59293 = invoke(stypy.reporting.localization.Localization(__file__, 1057, 46), str_59290, *[paths_59291], **kwargs_59292)
        
        # Applying the binary operator '%' (line 1057)
        result_mod_59294 = python_operator(stypy.reporting.localization.Localization(__file__, 1057, 23), '%', unicode_59289, str_call_result_59293)
        
        # Processing the call keyword arguments (line 1057)
        kwargs_59295 = {}
        # Getting the type of 'verbose' (line 1057)
        verbose_59287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1057, 8), 'verbose', False)
        # Obtaining the member 'report' of a type (line 1057)
        report_59288 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1057, 8), verbose_59287, 'report')
        # Calling report(args, kwargs) (line 1057)
        report_call_result_59296 = invoke(stypy.reporting.localization.Localization(__file__, 1057, 8), report_59288, *[result_mod_59294], **kwargs_59295)
        
        
        # Assigning a BinOp to a Attribute (line 1060):
        
        # Assigning a BinOp to a Attribute (line 1060):
        
        # Call to findSystemFonts(...): (line 1060)
        # Processing the call arguments (line 1060)
        # Getting the type of 'paths' (line 1060)
        paths_59298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1060, 40), 'paths', False)
        # Processing the call keyword arguments (line 1060)
        kwargs_59299 = {}
        # Getting the type of 'findSystemFonts' (line 1060)
        findSystemFonts_59297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1060, 24), 'findSystemFonts', False)
        # Calling findSystemFonts(args, kwargs) (line 1060)
        findSystemFonts_call_result_59300 = invoke(stypy.reporting.localization.Localization(__file__, 1060, 24), findSystemFonts_59297, *[paths_59298], **kwargs_59299)
        
        
        # Call to findSystemFonts(...): (line 1060)
        # Processing the call keyword arguments (line 1060)
        kwargs_59302 = {}
        # Getting the type of 'findSystemFonts' (line 1060)
        findSystemFonts_59301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1060, 49), 'findSystemFonts', False)
        # Calling findSystemFonts(args, kwargs) (line 1060)
        findSystemFonts_call_result_59303 = invoke(stypy.reporting.localization.Localization(__file__, 1060, 49), findSystemFonts_59301, *[], **kwargs_59302)
        
        # Applying the binary operator '+' (line 1060)
        result_add_59304 = python_operator(stypy.reporting.localization.Localization(__file__, 1060, 24), '+', findSystemFonts_call_result_59300, findSystemFonts_call_result_59303)
        
        # Getting the type of 'self' (line 1060)
        self_59305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1060, 8), 'self')
        # Setting the type of the member 'ttffiles' of a type (line 1060)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1060, 8), self_59305, 'ttffiles', result_add_59304)
        
        # Assigning a Dict to a Attribute (line 1061):
        
        # Assigning a Dict to a Attribute (line 1061):
        
        # Obtaining an instance of the builtin type 'dict' (line 1061)
        dict_59306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1061, 29), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 1061)
        # Adding element type (key, value) (line 1061)
        unicode_59307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1062, 12), 'unicode', u'ttf')
        unicode_59308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1062, 19), 'unicode', u'DejaVu Sans')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1061, 29), dict_59306, (unicode_59307, unicode_59308))
        # Adding element type (key, value) (line 1061)
        unicode_59309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1063, 12), 'unicode', u'afm')
        unicode_59310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1063, 19), 'unicode', u'Helvetica')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1061, 29), dict_59306, (unicode_59309, unicode_59310))
        
        # Getting the type of 'self' (line 1061)
        self_59311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1061, 8), 'self')
        # Setting the type of the member 'defaultFamily' of a type (line 1061)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1061, 8), self_59311, 'defaultFamily', dict_59306)
        
        # Assigning a Dict to a Attribute (line 1064):
        
        # Assigning a Dict to a Attribute (line 1064):
        
        # Obtaining an instance of the builtin type 'dict' (line 1064)
        dict_59312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1064, 27), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 1064)
        
        # Getting the type of 'self' (line 1064)
        self_59313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1064, 8), 'self')
        # Setting the type of the member 'defaultFont' of a type (line 1064)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1064, 8), self_59313, 'defaultFont', dict_59312)
        
        # Getting the type of 'self' (line 1066)
        self_59314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1066, 21), 'self')
        # Obtaining the member 'ttffiles' of a type (line 1066)
        ttffiles_59315 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1066, 21), self_59314, 'ttffiles')
        # Testing the type of a for loop iterable (line 1066)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1066, 8), ttffiles_59315)
        # Getting the type of the for loop variable (line 1066)
        for_loop_var_59316 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1066, 8), ttffiles_59315)
        # Assigning a type to the variable 'fname' (line 1066)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1066, 8), 'fname', for_loop_var_59316)
        # SSA begins for a for statement (line 1066)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to report(...): (line 1067)
        # Processing the call arguments (line 1067)
        unicode_59319 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1067, 27), 'unicode', u'trying fontname %s')
        # Getting the type of 'fname' (line 1067)
        fname_59320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1067, 50), 'fname', False)
        # Applying the binary operator '%' (line 1067)
        result_mod_59321 = python_operator(stypy.reporting.localization.Localization(__file__, 1067, 27), '%', unicode_59319, fname_59320)
        
        unicode_59322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1067, 57), 'unicode', u'debug')
        # Processing the call keyword arguments (line 1067)
        kwargs_59323 = {}
        # Getting the type of 'verbose' (line 1067)
        verbose_59317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1067, 12), 'verbose', False)
        # Obtaining the member 'report' of a type (line 1067)
        report_59318 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1067, 12), verbose_59317, 'report')
        # Calling report(args, kwargs) (line 1067)
        report_call_result_59324 = invoke(stypy.reporting.localization.Localization(__file__, 1067, 12), report_59318, *[result_mod_59321, unicode_59322], **kwargs_59323)
        
        
        
        
        # Call to find(...): (line 1068)
        # Processing the call arguments (line 1068)
        unicode_59330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1068, 34), 'unicode', u'DejaVuSans.ttf')
        # Processing the call keyword arguments (line 1068)
        kwargs_59331 = {}
        
        # Call to lower(...): (line 1068)
        # Processing the call keyword arguments (line 1068)
        kwargs_59327 = {}
        # Getting the type of 'fname' (line 1068)
        fname_59325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1068, 15), 'fname', False)
        # Obtaining the member 'lower' of a type (line 1068)
        lower_59326 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1068, 15), fname_59325, 'lower')
        # Calling lower(args, kwargs) (line 1068)
        lower_call_result_59328 = invoke(stypy.reporting.localization.Localization(__file__, 1068, 15), lower_59326, *[], **kwargs_59327)
        
        # Obtaining the member 'find' of a type (line 1068)
        find_59329 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1068, 15), lower_call_result_59328, 'find')
        # Calling find(args, kwargs) (line 1068)
        find_call_result_59332 = invoke(stypy.reporting.localization.Localization(__file__, 1068, 15), find_59329, *[unicode_59330], **kwargs_59331)
        
        int_59333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1068, 53), 'int')
        # Applying the binary operator '>=' (line 1068)
        result_ge_59334 = python_operator(stypy.reporting.localization.Localization(__file__, 1068, 15), '>=', find_call_result_59332, int_59333)
        
        # Testing the type of an if condition (line 1068)
        if_condition_59335 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1068, 12), result_ge_59334)
        # Assigning a type to the variable 'if_condition_59335' (line 1068)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1068, 12), 'if_condition_59335', if_condition_59335)
        # SSA begins for if statement (line 1068)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Subscript (line 1069):
        
        # Assigning a Name to a Subscript (line 1069):
        # Getting the type of 'fname' (line 1069)
        fname_59336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1069, 42), 'fname')
        # Getting the type of 'self' (line 1069)
        self_59337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1069, 16), 'self')
        # Obtaining the member 'defaultFont' of a type (line 1069)
        defaultFont_59338 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1069, 16), self_59337, 'defaultFont')
        unicode_59339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1069, 33), 'unicode', u'ttf')
        # Storing an element on a container (line 1069)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1069, 16), defaultFont_59338, (unicode_59339, fname_59336))
        # SSA join for if statement (line 1068)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of a for statement (line 1066)
        module_type_store.open_ssa_branch('for loop else')
        
        # Assigning a Subscript to a Subscript (line 1073):
        
        # Assigning a Subscript to a Subscript (line 1073):
        
        # Obtaining the type of the subscript
        int_59340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1073, 52), 'int')
        # Getting the type of 'self' (line 1073)
        self_59341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1073, 38), 'self')
        # Obtaining the member 'ttffiles' of a type (line 1073)
        ttffiles_59342 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1073, 38), self_59341, 'ttffiles')
        # Obtaining the member '__getitem__' of a type (line 1073)
        getitem___59343 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1073, 38), ttffiles_59342, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1073)
        subscript_call_result_59344 = invoke(stypy.reporting.localization.Localization(__file__, 1073, 38), getitem___59343, int_59340)
        
        # Getting the type of 'self' (line 1073)
        self_59345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1073, 12), 'self')
        # Obtaining the member 'defaultFont' of a type (line 1073)
        defaultFont_59346 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1073, 12), self_59345, 'defaultFont')
        unicode_59347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1073, 29), 'unicode', u'ttf')
        # Storing an element on a container (line 1073)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1073, 12), defaultFont_59346, (unicode_59347, subscript_call_result_59344))
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Attribute (line 1075):
        
        # Assigning a Call to a Attribute (line 1075):
        
        # Call to createFontList(...): (line 1075)
        # Processing the call arguments (line 1075)
        # Getting the type of 'self' (line 1075)
        self_59349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1075, 38), 'self', False)
        # Obtaining the member 'ttffiles' of a type (line 1075)
        ttffiles_59350 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1075, 38), self_59349, 'ttffiles')
        # Processing the call keyword arguments (line 1075)
        kwargs_59351 = {}
        # Getting the type of 'createFontList' (line 1075)
        createFontList_59348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1075, 23), 'createFontList', False)
        # Calling createFontList(args, kwargs) (line 1075)
        createFontList_call_result_59352 = invoke(stypy.reporting.localization.Localization(__file__, 1075, 23), createFontList_59348, *[ttffiles_59350], **kwargs_59351)
        
        # Getting the type of 'self' (line 1075)
        self_59353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1075, 8), 'self')
        # Setting the type of the member 'ttflist' of a type (line 1075)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1075, 8), self_59353, 'ttflist', createFontList_call_result_59352)
        
        # Assigning a BinOp to a Attribute (line 1077):
        
        # Assigning a BinOp to a Attribute (line 1077):
        
        # Call to findSystemFonts(...): (line 1077)
        # Processing the call arguments (line 1077)
        # Getting the type of 'paths' (line 1077)
        paths_59355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1077, 40), 'paths', False)
        # Processing the call keyword arguments (line 1077)
        unicode_59356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1077, 55), 'unicode', u'afm')
        keyword_59357 = unicode_59356
        kwargs_59358 = {'fontext': keyword_59357}
        # Getting the type of 'findSystemFonts' (line 1077)
        findSystemFonts_59354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1077, 24), 'findSystemFonts', False)
        # Calling findSystemFonts(args, kwargs) (line 1077)
        findSystemFonts_call_result_59359 = invoke(stypy.reporting.localization.Localization(__file__, 1077, 24), findSystemFonts_59354, *[paths_59355], **kwargs_59358)
        
        
        # Call to findSystemFonts(...): (line 1078)
        # Processing the call keyword arguments (line 1078)
        unicode_59361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1078, 36), 'unicode', u'afm')
        keyword_59362 = unicode_59361
        kwargs_59363 = {'fontext': keyword_59362}
        # Getting the type of 'findSystemFonts' (line 1078)
        findSystemFonts_59360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1078, 12), 'findSystemFonts', False)
        # Calling findSystemFonts(args, kwargs) (line 1078)
        findSystemFonts_call_result_59364 = invoke(stypy.reporting.localization.Localization(__file__, 1078, 12), findSystemFonts_59360, *[], **kwargs_59363)
        
        # Applying the binary operator '+' (line 1077)
        result_add_59365 = python_operator(stypy.reporting.localization.Localization(__file__, 1077, 24), '+', findSystemFonts_call_result_59359, findSystemFonts_call_result_59364)
        
        # Getting the type of 'self' (line 1077)
        self_59366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1077, 8), 'self')
        # Setting the type of the member 'afmfiles' of a type (line 1077)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1077, 8), self_59366, 'afmfiles', result_add_59365)
        
        # Assigning a Call to a Attribute (line 1079):
        
        # Assigning a Call to a Attribute (line 1079):
        
        # Call to createFontList(...): (line 1079)
        # Processing the call arguments (line 1079)
        # Getting the type of 'self' (line 1079)
        self_59368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1079, 38), 'self', False)
        # Obtaining the member 'afmfiles' of a type (line 1079)
        afmfiles_59369 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1079, 38), self_59368, 'afmfiles')
        # Processing the call keyword arguments (line 1079)
        unicode_59370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1079, 61), 'unicode', u'afm')
        keyword_59371 = unicode_59370
        kwargs_59372 = {'fontext': keyword_59371}
        # Getting the type of 'createFontList' (line 1079)
        createFontList_59367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1079, 23), 'createFontList', False)
        # Calling createFontList(args, kwargs) (line 1079)
        createFontList_call_result_59373 = invoke(stypy.reporting.localization.Localization(__file__, 1079, 23), createFontList_59367, *[afmfiles_59369], **kwargs_59372)
        
        # Getting the type of 'self' (line 1079)
        self_59374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1079, 8), 'self')
        # Setting the type of the member 'afmlist' of a type (line 1079)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1079, 8), self_59374, 'afmlist', createFontList_call_result_59373)
        
        
        # Call to len(...): (line 1080)
        # Processing the call arguments (line 1080)
        # Getting the type of 'self' (line 1080)
        self_59376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1080, 15), 'self', False)
        # Obtaining the member 'afmfiles' of a type (line 1080)
        afmfiles_59377 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1080, 15), self_59376, 'afmfiles')
        # Processing the call keyword arguments (line 1080)
        kwargs_59378 = {}
        # Getting the type of 'len' (line 1080)
        len_59375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1080, 11), 'len', False)
        # Calling len(args, kwargs) (line 1080)
        len_call_result_59379 = invoke(stypy.reporting.localization.Localization(__file__, 1080, 11), len_59375, *[afmfiles_59377], **kwargs_59378)
        
        # Testing the type of an if condition (line 1080)
        if_condition_59380 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1080, 8), len_call_result_59379)
        # Assigning a type to the variable 'if_condition_59380' (line 1080)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1080, 8), 'if_condition_59380', if_condition_59380)
        # SSA begins for if statement (line 1080)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Subscript (line 1081):
        
        # Assigning a Subscript to a Subscript (line 1081):
        
        # Obtaining the type of the subscript
        int_59381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1081, 52), 'int')
        # Getting the type of 'self' (line 1081)
        self_59382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1081, 38), 'self')
        # Obtaining the member 'afmfiles' of a type (line 1081)
        afmfiles_59383 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1081, 38), self_59382, 'afmfiles')
        # Obtaining the member '__getitem__' of a type (line 1081)
        getitem___59384 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1081, 38), afmfiles_59383, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1081)
        subscript_call_result_59385 = invoke(stypy.reporting.localization.Localization(__file__, 1081, 38), getitem___59384, int_59381)
        
        # Getting the type of 'self' (line 1081)
        self_59386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1081, 12), 'self')
        # Obtaining the member 'defaultFont' of a type (line 1081)
        defaultFont_59387 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1081, 12), self_59386, 'defaultFont')
        unicode_59388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1081, 29), 'unicode', u'afm')
        # Storing an element on a container (line 1081)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1081, 12), defaultFont_59387, (unicode_59388, subscript_call_result_59385))
        # SSA branch for the else part of an if statement (line 1080)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Name to a Subscript (line 1083):
        
        # Assigning a Name to a Subscript (line 1083):
        # Getting the type of 'None' (line 1083)
        None_59389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1083, 38), 'None')
        # Getting the type of 'self' (line 1083)
        self_59390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1083, 12), 'self')
        # Obtaining the member 'defaultFont' of a type (line 1083)
        defaultFont_59391 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1083, 12), self_59390, 'defaultFont')
        unicode_59392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1083, 29), 'unicode', u'afm')
        # Storing an element on a container (line 1083)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1083, 12), defaultFont_59391, (unicode_59392, None_59389))
        # SSA join for if statement (line 1080)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def get_default_weight(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_default_weight'
        module_type_store = module_type_store.open_function_context('get_default_weight', 1085, 4, False)
        # Assigning a type to the variable 'self' (line 1086)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1086, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FontManager.get_default_weight.__dict__.__setitem__('stypy_localization', localization)
        FontManager.get_default_weight.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FontManager.get_default_weight.__dict__.__setitem__('stypy_type_store', module_type_store)
        FontManager.get_default_weight.__dict__.__setitem__('stypy_function_name', 'FontManager.get_default_weight')
        FontManager.get_default_weight.__dict__.__setitem__('stypy_param_names_list', [])
        FontManager.get_default_weight.__dict__.__setitem__('stypy_varargs_param_name', None)
        FontManager.get_default_weight.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FontManager.get_default_weight.__dict__.__setitem__('stypy_call_defaults', defaults)
        FontManager.get_default_weight.__dict__.__setitem__('stypy_call_varargs', varargs)
        FontManager.get_default_weight.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FontManager.get_default_weight.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FontManager.get_default_weight', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_default_weight', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_default_weight(...)' code ##################

        unicode_59393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1088, (-1)), 'unicode', u'\n        Return the default font weight.\n        ')
        # Getting the type of 'self' (line 1089)
        self_59394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1089, 15), 'self')
        # Obtaining the member '__default_weight' of a type (line 1089)
        default_weight_59395 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1089, 15), self_59394, '__default_weight')
        # Assigning a type to the variable 'stypy_return_type' (line 1089)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1089, 8), 'stypy_return_type', default_weight_59395)
        
        # ################# End of 'get_default_weight(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_default_weight' in the type store
        # Getting the type of 'stypy_return_type' (line 1085)
        stypy_return_type_59396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1085, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_59396)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_default_weight'
        return stypy_return_type_59396


    @staticmethod
    @norecursion
    def get_default_size(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_default_size'
        module_type_store = module_type_store.open_function_context('get_default_size', 1091, 4, False)
        
        # Passed parameters checking function
        FontManager.get_default_size.__dict__.__setitem__('stypy_localization', localization)
        FontManager.get_default_size.__dict__.__setitem__('stypy_type_of_self', None)
        FontManager.get_default_size.__dict__.__setitem__('stypy_type_store', module_type_store)
        FontManager.get_default_size.__dict__.__setitem__('stypy_function_name', 'get_default_size')
        FontManager.get_default_size.__dict__.__setitem__('stypy_param_names_list', [])
        FontManager.get_default_size.__dict__.__setitem__('stypy_varargs_param_name', None)
        FontManager.get_default_size.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FontManager.get_default_size.__dict__.__setitem__('stypy_call_defaults', defaults)
        FontManager.get_default_size.__dict__.__setitem__('stypy_call_varargs', varargs)
        FontManager.get_default_size.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FontManager.get_default_size.__dict__.__setitem__('stypy_declared_arg_number', 0)
        arguments = process_argument_values(localization, None, module_type_store, 'get_default_size', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_default_size', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_default_size(...)' code ##################

        unicode_59397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1095, (-1)), 'unicode', u'\n        Return the default font size.\n        ')
        
        # Obtaining the type of the subscript
        unicode_59398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1096, 24), 'unicode', u'font.size')
        # Getting the type of 'rcParams' (line 1096)
        rcParams_59399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1096, 15), 'rcParams')
        # Obtaining the member '__getitem__' of a type (line 1096)
        getitem___59400 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1096, 15), rcParams_59399, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1096)
        subscript_call_result_59401 = invoke(stypy.reporting.localization.Localization(__file__, 1096, 15), getitem___59400, unicode_59398)
        
        # Assigning a type to the variable 'stypy_return_type' (line 1096)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1096, 8), 'stypy_return_type', subscript_call_result_59401)
        
        # ################# End of 'get_default_size(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_default_size' in the type store
        # Getting the type of 'stypy_return_type' (line 1091)
        stypy_return_type_59402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1091, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_59402)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_default_size'
        return stypy_return_type_59402


    @norecursion
    def set_default_weight(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_default_weight'
        module_type_store = module_type_store.open_function_context('set_default_weight', 1098, 4, False)
        # Assigning a type to the variable 'self' (line 1099)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1099, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FontManager.set_default_weight.__dict__.__setitem__('stypy_localization', localization)
        FontManager.set_default_weight.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FontManager.set_default_weight.__dict__.__setitem__('stypy_type_store', module_type_store)
        FontManager.set_default_weight.__dict__.__setitem__('stypy_function_name', 'FontManager.set_default_weight')
        FontManager.set_default_weight.__dict__.__setitem__('stypy_param_names_list', ['weight'])
        FontManager.set_default_weight.__dict__.__setitem__('stypy_varargs_param_name', None)
        FontManager.set_default_weight.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FontManager.set_default_weight.__dict__.__setitem__('stypy_call_defaults', defaults)
        FontManager.set_default_weight.__dict__.__setitem__('stypy_call_varargs', varargs)
        FontManager.set_default_weight.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FontManager.set_default_weight.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FontManager.set_default_weight', ['weight'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_default_weight', localization, ['weight'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_default_weight(...)' code ##################

        unicode_59403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1101, (-1)), 'unicode', u"\n        Set the default font weight.  The initial value is 'normal'.\n        ")
        
        # Assigning a Name to a Attribute (line 1102):
        
        # Assigning a Name to a Attribute (line 1102):
        # Getting the type of 'weight' (line 1102)
        weight_59404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1102, 32), 'weight')
        # Getting the type of 'self' (line 1102)
        self_59405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1102, 8), 'self')
        # Setting the type of the member '__default_weight' of a type (line 1102)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1102, 8), self_59405, '__default_weight', weight_59404)
        
        # ################# End of 'set_default_weight(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_default_weight' in the type store
        # Getting the type of 'stypy_return_type' (line 1098)
        stypy_return_type_59406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1098, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_59406)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_default_weight'
        return stypy_return_type_59406


    @norecursion
    def update_fonts(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'update_fonts'
        module_type_store = module_type_store.open_function_context('update_fonts', 1104, 4, False)
        # Assigning a type to the variable 'self' (line 1105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1105, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FontManager.update_fonts.__dict__.__setitem__('stypy_localization', localization)
        FontManager.update_fonts.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FontManager.update_fonts.__dict__.__setitem__('stypy_type_store', module_type_store)
        FontManager.update_fonts.__dict__.__setitem__('stypy_function_name', 'FontManager.update_fonts')
        FontManager.update_fonts.__dict__.__setitem__('stypy_param_names_list', ['filenames'])
        FontManager.update_fonts.__dict__.__setitem__('stypy_varargs_param_name', None)
        FontManager.update_fonts.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FontManager.update_fonts.__dict__.__setitem__('stypy_call_defaults', defaults)
        FontManager.update_fonts.__dict__.__setitem__('stypy_call_varargs', varargs)
        FontManager.update_fonts.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FontManager.update_fonts.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FontManager.update_fonts', ['filenames'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'update_fonts', localization, ['filenames'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'update_fonts(...)' code ##################

        unicode_59407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1108, (-1)), 'unicode', u'\n        Update the font dictionary with new font files.\n        Currently not implemented.\n        ')
        # Getting the type of 'NotImplementedError' (line 1110)
        NotImplementedError_59408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1110, 14), 'NotImplementedError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1110, 8), NotImplementedError_59408, 'raise parameter', BaseException)
        
        # ################# End of 'update_fonts(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'update_fonts' in the type store
        # Getting the type of 'stypy_return_type' (line 1104)
        stypy_return_type_59409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1104, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_59409)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'update_fonts'
        return stypy_return_type_59409


    @norecursion
    def score_family(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'score_family'
        module_type_store = module_type_store.open_function_context('score_family', 1114, 4, False)
        # Assigning a type to the variable 'self' (line 1115)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1115, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FontManager.score_family.__dict__.__setitem__('stypy_localization', localization)
        FontManager.score_family.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FontManager.score_family.__dict__.__setitem__('stypy_type_store', module_type_store)
        FontManager.score_family.__dict__.__setitem__('stypy_function_name', 'FontManager.score_family')
        FontManager.score_family.__dict__.__setitem__('stypy_param_names_list', ['families', 'family2'])
        FontManager.score_family.__dict__.__setitem__('stypy_varargs_param_name', None)
        FontManager.score_family.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FontManager.score_family.__dict__.__setitem__('stypy_call_defaults', defaults)
        FontManager.score_family.__dict__.__setitem__('stypy_call_varargs', varargs)
        FontManager.score_family.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FontManager.score_family.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FontManager.score_family', ['families', 'family2'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'score_family', localization, ['families', 'family2'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'score_family(...)' code ##################

        unicode_59410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1124, (-1)), 'unicode', u'\n        Returns a match score between the list of font families in\n        *families* and the font family name *family2*.\n\n        An exact match at the head of the list returns 0.0.\n\n        A match further down the list will return between 0 and 1.\n\n        No match will return 1.0.\n        ')
        
        
        
        # Call to isinstance(...): (line 1125)
        # Processing the call arguments (line 1125)
        # Getting the type of 'families' (line 1125)
        families_59412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1125, 26), 'families', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 1125)
        tuple_59413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1125, 37), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 1125)
        # Adding element type (line 1125)
        # Getting the type of 'list' (line 1125)
        list_59414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1125, 37), 'list', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1125, 37), tuple_59413, list_59414)
        # Adding element type (line 1125)
        # Getting the type of 'tuple' (line 1125)
        tuple_59415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1125, 43), 'tuple', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1125, 37), tuple_59413, tuple_59415)
        
        # Processing the call keyword arguments (line 1125)
        kwargs_59416 = {}
        # Getting the type of 'isinstance' (line 1125)
        isinstance_59411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1125, 15), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 1125)
        isinstance_call_result_59417 = invoke(stypy.reporting.localization.Localization(__file__, 1125, 15), isinstance_59411, *[families_59412, tuple_59413], **kwargs_59416)
        
        # Applying the 'not' unary operator (line 1125)
        result_not__59418 = python_operator(stypy.reporting.localization.Localization(__file__, 1125, 11), 'not', isinstance_call_result_59417)
        
        # Testing the type of an if condition (line 1125)
        if_condition_59419 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1125, 8), result_not__59418)
        # Assigning a type to the variable 'if_condition_59419' (line 1125)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1125, 8), 'if_condition_59419', if_condition_59419)
        # SSA begins for if statement (line 1125)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a List to a Name (line 1126):
        
        # Assigning a List to a Name (line 1126):
        
        # Obtaining an instance of the builtin type 'list' (line 1126)
        list_59420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1126, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 1126)
        # Adding element type (line 1126)
        # Getting the type of 'families' (line 1126)
        families_59421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1126, 24), 'families')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1126, 23), list_59420, families_59421)
        
        # Assigning a type to the variable 'families' (line 1126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1126, 12), 'families', list_59420)
        # SSA branch for the else part of an if statement (line 1125)
        module_type_store.open_ssa_branch('else')
        
        
        
        # Call to len(...): (line 1127)
        # Processing the call arguments (line 1127)
        # Getting the type of 'families' (line 1127)
        families_59423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1127, 17), 'families', False)
        # Processing the call keyword arguments (line 1127)
        kwargs_59424 = {}
        # Getting the type of 'len' (line 1127)
        len_59422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1127, 13), 'len', False)
        # Calling len(args, kwargs) (line 1127)
        len_call_result_59425 = invoke(stypy.reporting.localization.Localization(__file__, 1127, 13), len_59422, *[families_59423], **kwargs_59424)
        
        int_59426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1127, 30), 'int')
        # Applying the binary operator '==' (line 1127)
        result_eq_59427 = python_operator(stypy.reporting.localization.Localization(__file__, 1127, 13), '==', len_call_result_59425, int_59426)
        
        # Testing the type of an if condition (line 1127)
        if_condition_59428 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1127, 13), result_eq_59427)
        # Assigning a type to the variable 'if_condition_59428' (line 1127)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1127, 13), 'if_condition_59428', if_condition_59428)
        # SSA begins for if statement (line 1127)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        float_59429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1128, 19), 'float')
        # Assigning a type to the variable 'stypy_return_type' (line 1128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1128, 12), 'stypy_return_type', float_59429)
        # SSA join for if statement (line 1127)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 1125)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 1129):
        
        # Assigning a Call to a Name (line 1129):
        
        # Call to lower(...): (line 1129)
        # Processing the call keyword arguments (line 1129)
        kwargs_59432 = {}
        # Getting the type of 'family2' (line 1129)
        family2_59430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1129, 18), 'family2', False)
        # Obtaining the member 'lower' of a type (line 1129)
        lower_59431 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1129, 18), family2_59430, 'lower')
        # Calling lower(args, kwargs) (line 1129)
        lower_call_result_59433 = invoke(stypy.reporting.localization.Localization(__file__, 1129, 18), lower_59431, *[], **kwargs_59432)
        
        # Assigning a type to the variable 'family2' (line 1129)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1129, 8), 'family2', lower_call_result_59433)
        
        # Assigning a BinOp to a Name (line 1130):
        
        # Assigning a BinOp to a Name (line 1130):
        int_59434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1130, 15), 'int')
        
        # Call to len(...): (line 1130)
        # Processing the call arguments (line 1130)
        # Getting the type of 'families' (line 1130)
        families_59436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1130, 23), 'families', False)
        # Processing the call keyword arguments (line 1130)
        kwargs_59437 = {}
        # Getting the type of 'len' (line 1130)
        len_59435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1130, 19), 'len', False)
        # Calling len(args, kwargs) (line 1130)
        len_call_result_59438 = invoke(stypy.reporting.localization.Localization(__file__, 1130, 19), len_59435, *[families_59436], **kwargs_59437)
        
        # Applying the binary operator 'div' (line 1130)
        result_div_59439 = python_operator(stypy.reporting.localization.Localization(__file__, 1130, 15), 'div', int_59434, len_call_result_59438)
        
        # Assigning a type to the variable 'step' (line 1130)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1130, 8), 'step', result_div_59439)
        
        
        # Call to enumerate(...): (line 1131)
        # Processing the call arguments (line 1131)
        # Getting the type of 'families' (line 1131)
        families_59441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1131, 36), 'families', False)
        # Processing the call keyword arguments (line 1131)
        kwargs_59442 = {}
        # Getting the type of 'enumerate' (line 1131)
        enumerate_59440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1131, 26), 'enumerate', False)
        # Calling enumerate(args, kwargs) (line 1131)
        enumerate_call_result_59443 = invoke(stypy.reporting.localization.Localization(__file__, 1131, 26), enumerate_59440, *[families_59441], **kwargs_59442)
        
        # Testing the type of a for loop iterable (line 1131)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1131, 8), enumerate_call_result_59443)
        # Getting the type of the for loop variable (line 1131)
        for_loop_var_59444 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1131, 8), enumerate_call_result_59443)
        # Assigning a type to the variable 'i' (line 1131)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1131, 8), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1131, 8), for_loop_var_59444))
        # Assigning a type to the variable 'family1' (line 1131)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1131, 8), 'family1', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1131, 8), for_loop_var_59444))
        # SSA begins for a for statement (line 1131)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 1132):
        
        # Assigning a Call to a Name (line 1132):
        
        # Call to lower(...): (line 1132)
        # Processing the call keyword arguments (line 1132)
        kwargs_59447 = {}
        # Getting the type of 'family1' (line 1132)
        family1_59445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1132, 22), 'family1', False)
        # Obtaining the member 'lower' of a type (line 1132)
        lower_59446 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1132, 22), family1_59445, 'lower')
        # Calling lower(args, kwargs) (line 1132)
        lower_call_result_59448 = invoke(stypy.reporting.localization.Localization(__file__, 1132, 22), lower_59446, *[], **kwargs_59447)
        
        # Assigning a type to the variable 'family1' (line 1132)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1132, 12), 'family1', lower_call_result_59448)
        
        
        # Getting the type of 'family1' (line 1133)
        family1_59449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1133, 15), 'family1')
        # Getting the type of 'font_family_aliases' (line 1133)
        font_family_aliases_59450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1133, 26), 'font_family_aliases')
        # Applying the binary operator 'in' (line 1133)
        result_contains_59451 = python_operator(stypy.reporting.localization.Localization(__file__, 1133, 15), 'in', family1_59449, font_family_aliases_59450)
        
        # Testing the type of an if condition (line 1133)
        if_condition_59452 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1133, 12), result_contains_59451)
        # Assigning a type to the variable 'if_condition_59452' (line 1133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1133, 12), 'if_condition_59452', if_condition_59452)
        # SSA begins for if statement (line 1133)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Getting the type of 'family1' (line 1134)
        family1_59453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1134, 19), 'family1')
        
        # Obtaining an instance of the builtin type 'tuple' (line 1134)
        tuple_59454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1134, 31), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 1134)
        # Adding element type (line 1134)
        unicode_59455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1134, 31), 'unicode', u'sans')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1134, 31), tuple_59454, unicode_59455)
        # Adding element type (line 1134)
        unicode_59456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1134, 39), 'unicode', u'sans serif')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1134, 31), tuple_59454, unicode_59456)
        
        # Applying the binary operator 'in' (line 1134)
        result_contains_59457 = python_operator(stypy.reporting.localization.Localization(__file__, 1134, 19), 'in', family1_59453, tuple_59454)
        
        # Testing the type of an if condition (line 1134)
        if_condition_59458 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1134, 16), result_contains_59457)
        # Assigning a type to the variable 'if_condition_59458' (line 1134)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1134, 16), 'if_condition_59458', if_condition_59458)
        # SSA begins for if statement (line 1134)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Str to a Name (line 1135):
        
        # Assigning a Str to a Name (line 1135):
        unicode_59459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1135, 30), 'unicode', u'sans-serif')
        # Assigning a type to the variable 'family1' (line 1135)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1135, 20), 'family1', unicode_59459)
        # SSA join for if statement (line 1134)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Subscript to a Name (line 1136):
        
        # Assigning a Subscript to a Name (line 1136):
        
        # Obtaining the type of the subscript
        unicode_59460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1136, 35), 'unicode', u'font.')
        # Getting the type of 'family1' (line 1136)
        family1_59461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1136, 45), 'family1')
        # Applying the binary operator '+' (line 1136)
        result_add_59462 = python_operator(stypy.reporting.localization.Localization(__file__, 1136, 35), '+', unicode_59460, family1_59461)
        
        # Getting the type of 'rcParams' (line 1136)
        rcParams_59463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1136, 26), 'rcParams')
        # Obtaining the member '__getitem__' of a type (line 1136)
        getitem___59464 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1136, 26), rcParams_59463, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1136)
        subscript_call_result_59465 = invoke(stypy.reporting.localization.Localization(__file__, 1136, 26), getitem___59464, result_add_59462)
        
        # Assigning a type to the variable 'options' (line 1136)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1136, 16), 'options', subscript_call_result_59465)
        
        # Assigning a ListComp to a Name (line 1137):
        
        # Assigning a ListComp to a Name (line 1137):
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'options' (line 1137)
        options_59470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1137, 46), 'options')
        comprehension_59471 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1137, 27), options_59470)
        # Assigning a type to the variable 'x' (line 1137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1137, 27), 'x', comprehension_59471)
        
        # Call to lower(...): (line 1137)
        # Processing the call keyword arguments (line 1137)
        kwargs_59468 = {}
        # Getting the type of 'x' (line 1137)
        x_59466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1137, 27), 'x', False)
        # Obtaining the member 'lower' of a type (line 1137)
        lower_59467 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1137, 27), x_59466, 'lower')
        # Calling lower(args, kwargs) (line 1137)
        lower_call_result_59469 = invoke(stypy.reporting.localization.Localization(__file__, 1137, 27), lower_59467, *[], **kwargs_59468)
        
        list_59472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1137, 27), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1137, 27), list_59472, lower_call_result_59469)
        # Assigning a type to the variable 'options' (line 1137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1137, 16), 'options', list_59472)
        
        
        # Getting the type of 'family2' (line 1138)
        family2_59473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1138, 19), 'family2')
        # Getting the type of 'options' (line 1138)
        options_59474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1138, 30), 'options')
        # Applying the binary operator 'in' (line 1138)
        result_contains_59475 = python_operator(stypy.reporting.localization.Localization(__file__, 1138, 19), 'in', family2_59473, options_59474)
        
        # Testing the type of an if condition (line 1138)
        if_condition_59476 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1138, 16), result_contains_59475)
        # Assigning a type to the variable 'if_condition_59476' (line 1138)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1138, 16), 'if_condition_59476', if_condition_59476)
        # SSA begins for if statement (line 1138)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 1139):
        
        # Assigning a Call to a Name (line 1139):
        
        # Call to index(...): (line 1139)
        # Processing the call arguments (line 1139)
        # Getting the type of 'family2' (line 1139)
        family2_59479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1139, 40), 'family2', False)
        # Processing the call keyword arguments (line 1139)
        kwargs_59480 = {}
        # Getting the type of 'options' (line 1139)
        options_59477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1139, 26), 'options', False)
        # Obtaining the member 'index' of a type (line 1139)
        index_59478 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1139, 26), options_59477, 'index')
        # Calling index(args, kwargs) (line 1139)
        index_call_result_59481 = invoke(stypy.reporting.localization.Localization(__file__, 1139, 26), index_59478, *[family2_59479], **kwargs_59480)
        
        # Assigning a type to the variable 'idx' (line 1139)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1139, 20), 'idx', index_call_result_59481)
        # Getting the type of 'i' (line 1140)
        i_59482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1140, 28), 'i')
        # Getting the type of 'idx' (line 1140)
        idx_59483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1140, 33), 'idx')
        
        # Call to len(...): (line 1140)
        # Processing the call arguments (line 1140)
        # Getting the type of 'options' (line 1140)
        options_59485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1140, 43), 'options', False)
        # Processing the call keyword arguments (line 1140)
        kwargs_59486 = {}
        # Getting the type of 'len' (line 1140)
        len_59484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1140, 39), 'len', False)
        # Calling len(args, kwargs) (line 1140)
        len_call_result_59487 = invoke(stypy.reporting.localization.Localization(__file__, 1140, 39), len_59484, *[options_59485], **kwargs_59486)
        
        # Applying the binary operator 'div' (line 1140)
        result_div_59488 = python_operator(stypy.reporting.localization.Localization(__file__, 1140, 33), 'div', idx_59483, len_call_result_59487)
        
        # Applying the binary operator '+' (line 1140)
        result_add_59489 = python_operator(stypy.reporting.localization.Localization(__file__, 1140, 28), '+', i_59482, result_div_59488)
        
        # Getting the type of 'step' (line 1140)
        step_59490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1140, 56), 'step')
        # Applying the binary operator '*' (line 1140)
        result_mul_59491 = python_operator(stypy.reporting.localization.Localization(__file__, 1140, 27), '*', result_add_59489, step_59490)
        
        # Assigning a type to the variable 'stypy_return_type' (line 1140)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1140, 20), 'stypy_return_type', result_mul_59491)
        # SSA join for if statement (line 1138)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 1133)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'family1' (line 1141)
        family1_59492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1141, 17), 'family1')
        # Getting the type of 'family2' (line 1141)
        family2_59493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1141, 28), 'family2')
        # Applying the binary operator '==' (line 1141)
        result_eq_59494 = python_operator(stypy.reporting.localization.Localization(__file__, 1141, 17), '==', family1_59492, family2_59493)
        
        # Testing the type of an if condition (line 1141)
        if_condition_59495 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1141, 17), result_eq_59494)
        # Assigning a type to the variable 'if_condition_59495' (line 1141)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1141, 17), 'if_condition_59495', if_condition_59495)
        # SSA begins for if statement (line 1141)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'i' (line 1144)
        i_59496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1144, 23), 'i')
        # Getting the type of 'step' (line 1144)
        step_59497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1144, 27), 'step')
        # Applying the binary operator '*' (line 1144)
        result_mul_59498 = python_operator(stypy.reporting.localization.Localization(__file__, 1144, 23), '*', i_59496, step_59497)
        
        # Assigning a type to the variable 'stypy_return_type' (line 1144)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1144, 16), 'stypy_return_type', result_mul_59498)
        # SSA join for if statement (line 1141)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 1133)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        float_59499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1145, 15), 'float')
        # Assigning a type to the variable 'stypy_return_type' (line 1145)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1145, 8), 'stypy_return_type', float_59499)
        
        # ################# End of 'score_family(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'score_family' in the type store
        # Getting the type of 'stypy_return_type' (line 1114)
        stypy_return_type_59500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1114, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_59500)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'score_family'
        return stypy_return_type_59500


    @norecursion
    def score_style(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'score_style'
        module_type_store = module_type_store.open_function_context('score_style', 1147, 4, False)
        # Assigning a type to the variable 'self' (line 1148)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1148, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FontManager.score_style.__dict__.__setitem__('stypy_localization', localization)
        FontManager.score_style.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FontManager.score_style.__dict__.__setitem__('stypy_type_store', module_type_store)
        FontManager.score_style.__dict__.__setitem__('stypy_function_name', 'FontManager.score_style')
        FontManager.score_style.__dict__.__setitem__('stypy_param_names_list', ['style1', 'style2'])
        FontManager.score_style.__dict__.__setitem__('stypy_varargs_param_name', None)
        FontManager.score_style.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FontManager.score_style.__dict__.__setitem__('stypy_call_defaults', defaults)
        FontManager.score_style.__dict__.__setitem__('stypy_call_varargs', varargs)
        FontManager.score_style.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FontManager.score_style.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FontManager.score_style', ['style1', 'style2'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'score_style', localization, ['style1', 'style2'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'score_style(...)' code ##################

        unicode_59501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1156, (-1)), 'unicode', u"\n        Returns a match score between *style1* and *style2*.\n\n        An exact match returns 0.0.\n\n        A match between 'italic' and 'oblique' returns 0.1.\n\n        No match returns 1.0.\n        ")
        
        
        # Getting the type of 'style1' (line 1157)
        style1_59502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1157, 11), 'style1')
        # Getting the type of 'style2' (line 1157)
        style2_59503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1157, 21), 'style2')
        # Applying the binary operator '==' (line 1157)
        result_eq_59504 = python_operator(stypy.reporting.localization.Localization(__file__, 1157, 11), '==', style1_59502, style2_59503)
        
        # Testing the type of an if condition (line 1157)
        if_condition_59505 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1157, 8), result_eq_59504)
        # Assigning a type to the variable 'if_condition_59505' (line 1157)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1157, 8), 'if_condition_59505', if_condition_59505)
        # SSA begins for if statement (line 1157)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        float_59506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1158, 19), 'float')
        # Assigning a type to the variable 'stypy_return_type' (line 1158)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1158, 12), 'stypy_return_type', float_59506)
        # SSA branch for the else part of an if statement (line 1157)
        module_type_store.open_ssa_branch('else')
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'style1' (line 1159)
        style1_59507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1159, 13), 'style1')
        
        # Obtaining an instance of the builtin type 'tuple' (line 1159)
        tuple_59508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1159, 24), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 1159)
        # Adding element type (line 1159)
        unicode_59509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1159, 24), 'unicode', u'italic')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1159, 24), tuple_59508, unicode_59509)
        # Adding element type (line 1159)
        unicode_59510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1159, 34), 'unicode', u'oblique')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1159, 24), tuple_59508, unicode_59510)
        
        # Applying the binary operator 'in' (line 1159)
        result_contains_59511 = python_operator(stypy.reporting.localization.Localization(__file__, 1159, 13), 'in', style1_59507, tuple_59508)
        
        
        # Getting the type of 'style2' (line 1160)
        style2_59512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1160, 16), 'style2')
        
        # Obtaining an instance of the builtin type 'tuple' (line 1160)
        tuple_59513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1160, 27), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 1160)
        # Adding element type (line 1160)
        unicode_59514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1160, 27), 'unicode', u'italic')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1160, 27), tuple_59513, unicode_59514)
        # Adding element type (line 1160)
        unicode_59515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1160, 37), 'unicode', u'oblique')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1160, 27), tuple_59513, unicode_59515)
        
        # Applying the binary operator 'in' (line 1160)
        result_contains_59516 = python_operator(stypy.reporting.localization.Localization(__file__, 1160, 16), 'in', style2_59512, tuple_59513)
        
        # Applying the binary operator 'and' (line 1159)
        result_and_keyword_59517 = python_operator(stypy.reporting.localization.Localization(__file__, 1159, 13), 'and', result_contains_59511, result_contains_59516)
        
        # Testing the type of an if condition (line 1159)
        if_condition_59518 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1159, 13), result_and_keyword_59517)
        # Assigning a type to the variable 'if_condition_59518' (line 1159)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1159, 13), 'if_condition_59518', if_condition_59518)
        # SSA begins for if statement (line 1159)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        float_59519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1161, 19), 'float')
        # Assigning a type to the variable 'stypy_return_type' (line 1161)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1161, 12), 'stypy_return_type', float_59519)
        # SSA join for if statement (line 1159)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 1157)
        module_type_store = module_type_store.join_ssa_context()
        
        float_59520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1162, 15), 'float')
        # Assigning a type to the variable 'stypy_return_type' (line 1162)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1162, 8), 'stypy_return_type', float_59520)
        
        # ################# End of 'score_style(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'score_style' in the type store
        # Getting the type of 'stypy_return_type' (line 1147)
        stypy_return_type_59521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1147, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_59521)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'score_style'
        return stypy_return_type_59521


    @norecursion
    def score_variant(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'score_variant'
        module_type_store = module_type_store.open_function_context('score_variant', 1164, 4, False)
        # Assigning a type to the variable 'self' (line 1165)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1165, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FontManager.score_variant.__dict__.__setitem__('stypy_localization', localization)
        FontManager.score_variant.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FontManager.score_variant.__dict__.__setitem__('stypy_type_store', module_type_store)
        FontManager.score_variant.__dict__.__setitem__('stypy_function_name', 'FontManager.score_variant')
        FontManager.score_variant.__dict__.__setitem__('stypy_param_names_list', ['variant1', 'variant2'])
        FontManager.score_variant.__dict__.__setitem__('stypy_varargs_param_name', None)
        FontManager.score_variant.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FontManager.score_variant.__dict__.__setitem__('stypy_call_defaults', defaults)
        FontManager.score_variant.__dict__.__setitem__('stypy_call_varargs', varargs)
        FontManager.score_variant.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FontManager.score_variant.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FontManager.score_variant', ['variant1', 'variant2'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'score_variant', localization, ['variant1', 'variant2'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'score_variant(...)' code ##################

        unicode_59522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1169, (-1)), 'unicode', u'\n        Returns a match score between *variant1* and *variant2*.\n\n        An exact match returns 0.0, otherwise 1.0.\n        ')
        
        
        # Getting the type of 'variant1' (line 1170)
        variant1_59523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1170, 11), 'variant1')
        # Getting the type of 'variant2' (line 1170)
        variant2_59524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1170, 23), 'variant2')
        # Applying the binary operator '==' (line 1170)
        result_eq_59525 = python_operator(stypy.reporting.localization.Localization(__file__, 1170, 11), '==', variant1_59523, variant2_59524)
        
        # Testing the type of an if condition (line 1170)
        if_condition_59526 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1170, 8), result_eq_59525)
        # Assigning a type to the variable 'if_condition_59526' (line 1170)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1170, 8), 'if_condition_59526', if_condition_59526)
        # SSA begins for if statement (line 1170)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        float_59527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1171, 19), 'float')
        # Assigning a type to the variable 'stypy_return_type' (line 1171)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1171, 12), 'stypy_return_type', float_59527)
        # SSA branch for the else part of an if statement (line 1170)
        module_type_store.open_ssa_branch('else')
        float_59528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1173, 19), 'float')
        # Assigning a type to the variable 'stypy_return_type' (line 1173)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1173, 12), 'stypy_return_type', float_59528)
        # SSA join for if statement (line 1170)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'score_variant(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'score_variant' in the type store
        # Getting the type of 'stypy_return_type' (line 1164)
        stypy_return_type_59529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1164, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_59529)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'score_variant'
        return stypy_return_type_59529


    @norecursion
    def score_stretch(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'score_stretch'
        module_type_store = module_type_store.open_function_context('score_stretch', 1175, 4, False)
        # Assigning a type to the variable 'self' (line 1176)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1176, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FontManager.score_stretch.__dict__.__setitem__('stypy_localization', localization)
        FontManager.score_stretch.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FontManager.score_stretch.__dict__.__setitem__('stypy_type_store', module_type_store)
        FontManager.score_stretch.__dict__.__setitem__('stypy_function_name', 'FontManager.score_stretch')
        FontManager.score_stretch.__dict__.__setitem__('stypy_param_names_list', ['stretch1', 'stretch2'])
        FontManager.score_stretch.__dict__.__setitem__('stypy_varargs_param_name', None)
        FontManager.score_stretch.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FontManager.score_stretch.__dict__.__setitem__('stypy_call_defaults', defaults)
        FontManager.score_stretch.__dict__.__setitem__('stypy_call_varargs', varargs)
        FontManager.score_stretch.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FontManager.score_stretch.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FontManager.score_stretch', ['stretch1', 'stretch2'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'score_stretch', localization, ['stretch1', 'stretch2'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'score_stretch(...)' code ##################

        unicode_59530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1182, (-1)), 'unicode', u'\n        Returns a match score between *stretch1* and *stretch2*.\n\n        The result is the absolute value of the difference between the\n        CSS numeric values of *stretch1* and *stretch2*, normalized\n        between 0.0 and 1.0.\n        ')
        
        
        # SSA begins for try-except statement (line 1183)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Name (line 1184):
        
        # Assigning a Call to a Name (line 1184):
        
        # Call to int(...): (line 1184)
        # Processing the call arguments (line 1184)
        # Getting the type of 'stretch1' (line 1184)
        stretch1_59532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1184, 30), 'stretch1', False)
        # Processing the call keyword arguments (line 1184)
        kwargs_59533 = {}
        # Getting the type of 'int' (line 1184)
        int_59531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1184, 26), 'int', False)
        # Calling int(args, kwargs) (line 1184)
        int_call_result_59534 = invoke(stypy.reporting.localization.Localization(__file__, 1184, 26), int_59531, *[stretch1_59532], **kwargs_59533)
        
        # Assigning a type to the variable 'stretchval1' (line 1184)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1184, 12), 'stretchval1', int_call_result_59534)
        # SSA branch for the except part of a try statement (line 1183)
        # SSA branch for the except 'ValueError' branch of a try statement (line 1183)
        module_type_store.open_ssa_branch('except')
        
        # Assigning a Call to a Name (line 1186):
        
        # Assigning a Call to a Name (line 1186):
        
        # Call to get(...): (line 1186)
        # Processing the call arguments (line 1186)
        # Getting the type of 'stretch1' (line 1186)
        stretch1_59537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1186, 43), 'stretch1', False)
        int_59538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1186, 53), 'int')
        # Processing the call keyword arguments (line 1186)
        kwargs_59539 = {}
        # Getting the type of 'stretch_dict' (line 1186)
        stretch_dict_59535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1186, 26), 'stretch_dict', False)
        # Obtaining the member 'get' of a type (line 1186)
        get_59536 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1186, 26), stretch_dict_59535, 'get')
        # Calling get(args, kwargs) (line 1186)
        get_call_result_59540 = invoke(stypy.reporting.localization.Localization(__file__, 1186, 26), get_59536, *[stretch1_59537, int_59538], **kwargs_59539)
        
        # Assigning a type to the variable 'stretchval1' (line 1186)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1186, 12), 'stretchval1', get_call_result_59540)
        # SSA join for try-except statement (line 1183)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # SSA begins for try-except statement (line 1187)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Name (line 1188):
        
        # Assigning a Call to a Name (line 1188):
        
        # Call to int(...): (line 1188)
        # Processing the call arguments (line 1188)
        # Getting the type of 'stretch2' (line 1188)
        stretch2_59542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1188, 30), 'stretch2', False)
        # Processing the call keyword arguments (line 1188)
        kwargs_59543 = {}
        # Getting the type of 'int' (line 1188)
        int_59541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1188, 26), 'int', False)
        # Calling int(args, kwargs) (line 1188)
        int_call_result_59544 = invoke(stypy.reporting.localization.Localization(__file__, 1188, 26), int_59541, *[stretch2_59542], **kwargs_59543)
        
        # Assigning a type to the variable 'stretchval2' (line 1188)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1188, 12), 'stretchval2', int_call_result_59544)
        # SSA branch for the except part of a try statement (line 1187)
        # SSA branch for the except 'ValueError' branch of a try statement (line 1187)
        module_type_store.open_ssa_branch('except')
        
        # Assigning a Call to a Name (line 1190):
        
        # Assigning a Call to a Name (line 1190):
        
        # Call to get(...): (line 1190)
        # Processing the call arguments (line 1190)
        # Getting the type of 'stretch2' (line 1190)
        stretch2_59547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1190, 43), 'stretch2', False)
        int_59548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1190, 53), 'int')
        # Processing the call keyword arguments (line 1190)
        kwargs_59549 = {}
        # Getting the type of 'stretch_dict' (line 1190)
        stretch_dict_59545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1190, 26), 'stretch_dict', False)
        # Obtaining the member 'get' of a type (line 1190)
        get_59546 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1190, 26), stretch_dict_59545, 'get')
        # Calling get(args, kwargs) (line 1190)
        get_call_result_59550 = invoke(stypy.reporting.localization.Localization(__file__, 1190, 26), get_59546, *[stretch2_59547, int_59548], **kwargs_59549)
        
        # Assigning a type to the variable 'stretchval2' (line 1190)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1190, 12), 'stretchval2', get_call_result_59550)
        # SSA join for try-except statement (line 1187)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to abs(...): (line 1191)
        # Processing the call arguments (line 1191)
        # Getting the type of 'stretchval1' (line 1191)
        stretchval1_59552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1191, 19), 'stretchval1', False)
        # Getting the type of 'stretchval2' (line 1191)
        stretchval2_59553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1191, 33), 'stretchval2', False)
        # Applying the binary operator '-' (line 1191)
        result_sub_59554 = python_operator(stypy.reporting.localization.Localization(__file__, 1191, 19), '-', stretchval1_59552, stretchval2_59553)
        
        # Processing the call keyword arguments (line 1191)
        kwargs_59555 = {}
        # Getting the type of 'abs' (line 1191)
        abs_59551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1191, 15), 'abs', False)
        # Calling abs(args, kwargs) (line 1191)
        abs_call_result_59556 = invoke(stypy.reporting.localization.Localization(__file__, 1191, 15), abs_59551, *[result_sub_59554], **kwargs_59555)
        
        float_59557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1191, 48), 'float')
        # Applying the binary operator 'div' (line 1191)
        result_div_59558 = python_operator(stypy.reporting.localization.Localization(__file__, 1191, 15), 'div', abs_call_result_59556, float_59557)
        
        # Assigning a type to the variable 'stypy_return_type' (line 1191)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1191, 8), 'stypy_return_type', result_div_59558)
        
        # ################# End of 'score_stretch(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'score_stretch' in the type store
        # Getting the type of 'stypy_return_type' (line 1175)
        stypy_return_type_59559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1175, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_59559)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'score_stretch'
        return stypy_return_type_59559


    @norecursion
    def score_weight(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'score_weight'
        module_type_store = module_type_store.open_function_context('score_weight', 1193, 4, False)
        # Assigning a type to the variable 'self' (line 1194)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1194, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FontManager.score_weight.__dict__.__setitem__('stypy_localization', localization)
        FontManager.score_weight.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FontManager.score_weight.__dict__.__setitem__('stypy_type_store', module_type_store)
        FontManager.score_weight.__dict__.__setitem__('stypy_function_name', 'FontManager.score_weight')
        FontManager.score_weight.__dict__.__setitem__('stypy_param_names_list', ['weight1', 'weight2'])
        FontManager.score_weight.__dict__.__setitem__('stypy_varargs_param_name', None)
        FontManager.score_weight.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FontManager.score_weight.__dict__.__setitem__('stypy_call_defaults', defaults)
        FontManager.score_weight.__dict__.__setitem__('stypy_call_varargs', varargs)
        FontManager.score_weight.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FontManager.score_weight.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FontManager.score_weight', ['weight1', 'weight2'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'score_weight', localization, ['weight1', 'weight2'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'score_weight(...)' code ##################

        unicode_59560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1203, (-1)), 'unicode', u'\n        Returns a match score between *weight1* and *weight2*.\n\n        The result is 0.0 if both weight1 and weight 2 are given as strings\n        and have the same value.\n\n        Otherwise, the result is the absolute value of the difference between the\n        CSS numeric values of *weight1* and *weight2*, normalized\n        between 0.05 and 1.0.\n        ')
        
        
        # Evaluating a boolean operation
        
        # Call to isinstance(...): (line 1206)
        # Processing the call arguments (line 1206)
        # Getting the type of 'weight1' (line 1206)
        weight1_59562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1206, 23), 'weight1', False)
        # Getting the type of 'six' (line 1206)
        six_59563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1206, 32), 'six', False)
        # Obtaining the member 'string_types' of a type (line 1206)
        string_types_59564 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1206, 32), six_59563, 'string_types')
        # Processing the call keyword arguments (line 1206)
        kwargs_59565 = {}
        # Getting the type of 'isinstance' (line 1206)
        isinstance_59561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1206, 12), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 1206)
        isinstance_call_result_59566 = invoke(stypy.reporting.localization.Localization(__file__, 1206, 12), isinstance_59561, *[weight1_59562, string_types_59564], **kwargs_59565)
        
        
        # Call to isinstance(...): (line 1207)
        # Processing the call arguments (line 1207)
        # Getting the type of 'weight2' (line 1207)
        weight2_59568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1207, 27), 'weight2', False)
        # Getting the type of 'six' (line 1207)
        six_59569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1207, 36), 'six', False)
        # Obtaining the member 'string_types' of a type (line 1207)
        string_types_59570 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1207, 36), six_59569, 'string_types')
        # Processing the call keyword arguments (line 1207)
        kwargs_59571 = {}
        # Getting the type of 'isinstance' (line 1207)
        isinstance_59567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1207, 16), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 1207)
        isinstance_call_result_59572 = invoke(stypy.reporting.localization.Localization(__file__, 1207, 16), isinstance_59567, *[weight2_59568, string_types_59570], **kwargs_59571)
        
        # Applying the binary operator 'and' (line 1206)
        result_and_keyword_59573 = python_operator(stypy.reporting.localization.Localization(__file__, 1206, 12), 'and', isinstance_call_result_59566, isinstance_call_result_59572)
        
        # Getting the type of 'weight1' (line 1208)
        weight1_59574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1208, 16), 'weight1')
        # Getting the type of 'weight2' (line 1208)
        weight2_59575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1208, 27), 'weight2')
        # Applying the binary operator '==' (line 1208)
        result_eq_59576 = python_operator(stypy.reporting.localization.Localization(__file__, 1208, 16), '==', weight1_59574, weight2_59575)
        
        # Applying the binary operator 'and' (line 1206)
        result_and_keyword_59577 = python_operator(stypy.reporting.localization.Localization(__file__, 1206, 12), 'and', result_and_keyword_59573, result_eq_59576)
        
        # Testing the type of an if condition (line 1206)
        if_condition_59578 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1206, 8), result_and_keyword_59577)
        # Assigning a type to the variable 'if_condition_59578' (line 1206)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1206, 8), 'if_condition_59578', if_condition_59578)
        # SSA begins for if statement (line 1206)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        float_59579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1209, 19), 'float')
        # Assigning a type to the variable 'stypy_return_type' (line 1209)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1209, 12), 'stypy_return_type', float_59579)
        # SSA join for if statement (line 1206)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # SSA begins for try-except statement (line 1210)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Name (line 1211):
        
        # Assigning a Call to a Name (line 1211):
        
        # Call to int(...): (line 1211)
        # Processing the call arguments (line 1211)
        # Getting the type of 'weight1' (line 1211)
        weight1_59581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1211, 29), 'weight1', False)
        # Processing the call keyword arguments (line 1211)
        kwargs_59582 = {}
        # Getting the type of 'int' (line 1211)
        int_59580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1211, 25), 'int', False)
        # Calling int(args, kwargs) (line 1211)
        int_call_result_59583 = invoke(stypy.reporting.localization.Localization(__file__, 1211, 25), int_59580, *[weight1_59581], **kwargs_59582)
        
        # Assigning a type to the variable 'weightval1' (line 1211)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1211, 12), 'weightval1', int_call_result_59583)
        # SSA branch for the except part of a try statement (line 1210)
        # SSA branch for the except 'ValueError' branch of a try statement (line 1210)
        module_type_store.open_ssa_branch('except')
        
        # Assigning a Call to a Name (line 1213):
        
        # Assigning a Call to a Name (line 1213):
        
        # Call to get(...): (line 1213)
        # Processing the call arguments (line 1213)
        # Getting the type of 'weight1' (line 1213)
        weight1_59586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1213, 41), 'weight1', False)
        int_59587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1213, 50), 'int')
        # Processing the call keyword arguments (line 1213)
        kwargs_59588 = {}
        # Getting the type of 'weight_dict' (line 1213)
        weight_dict_59584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1213, 25), 'weight_dict', False)
        # Obtaining the member 'get' of a type (line 1213)
        get_59585 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1213, 25), weight_dict_59584, 'get')
        # Calling get(args, kwargs) (line 1213)
        get_call_result_59589 = invoke(stypy.reporting.localization.Localization(__file__, 1213, 25), get_59585, *[weight1_59586, int_59587], **kwargs_59588)
        
        # Assigning a type to the variable 'weightval1' (line 1213)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1213, 12), 'weightval1', get_call_result_59589)
        # SSA join for try-except statement (line 1210)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # SSA begins for try-except statement (line 1214)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Name (line 1215):
        
        # Assigning a Call to a Name (line 1215):
        
        # Call to int(...): (line 1215)
        # Processing the call arguments (line 1215)
        # Getting the type of 'weight2' (line 1215)
        weight2_59591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1215, 29), 'weight2', False)
        # Processing the call keyword arguments (line 1215)
        kwargs_59592 = {}
        # Getting the type of 'int' (line 1215)
        int_59590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1215, 25), 'int', False)
        # Calling int(args, kwargs) (line 1215)
        int_call_result_59593 = invoke(stypy.reporting.localization.Localization(__file__, 1215, 25), int_59590, *[weight2_59591], **kwargs_59592)
        
        # Assigning a type to the variable 'weightval2' (line 1215)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1215, 12), 'weightval2', int_call_result_59593)
        # SSA branch for the except part of a try statement (line 1214)
        # SSA branch for the except 'ValueError' branch of a try statement (line 1214)
        module_type_store.open_ssa_branch('except')
        
        # Assigning a Call to a Name (line 1217):
        
        # Assigning a Call to a Name (line 1217):
        
        # Call to get(...): (line 1217)
        # Processing the call arguments (line 1217)
        # Getting the type of 'weight2' (line 1217)
        weight2_59596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1217, 41), 'weight2', False)
        int_59597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1217, 50), 'int')
        # Processing the call keyword arguments (line 1217)
        kwargs_59598 = {}
        # Getting the type of 'weight_dict' (line 1217)
        weight_dict_59594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1217, 25), 'weight_dict', False)
        # Obtaining the member 'get' of a type (line 1217)
        get_59595 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1217, 25), weight_dict_59594, 'get')
        # Calling get(args, kwargs) (line 1217)
        get_call_result_59599 = invoke(stypy.reporting.localization.Localization(__file__, 1217, 25), get_59595, *[weight2_59596, int_59597], **kwargs_59598)
        
        # Assigning a type to the variable 'weightval2' (line 1217)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1217, 12), 'weightval2', get_call_result_59599)
        # SSA join for try-except statement (line 1214)
        module_type_store = module_type_store.join_ssa_context()
        
        float_59600 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1218, 15), 'float')
        
        # Call to abs(...): (line 1218)
        # Processing the call arguments (line 1218)
        # Getting the type of 'weightval1' (line 1218)
        weightval1_59602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1218, 25), 'weightval1', False)
        # Getting the type of 'weightval2' (line 1218)
        weightval2_59603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1218, 38), 'weightval2', False)
        # Applying the binary operator '-' (line 1218)
        result_sub_59604 = python_operator(stypy.reporting.localization.Localization(__file__, 1218, 25), '-', weightval1_59602, weightval2_59603)
        
        # Processing the call keyword arguments (line 1218)
        kwargs_59605 = {}
        # Getting the type of 'abs' (line 1218)
        abs_59601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1218, 21), 'abs', False)
        # Calling abs(args, kwargs) (line 1218)
        abs_call_result_59606 = invoke(stypy.reporting.localization.Localization(__file__, 1218, 21), abs_59601, *[result_sub_59604], **kwargs_59605)
        
        float_59607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1218, 52), 'float')
        # Applying the binary operator 'div' (line 1218)
        result_div_59608 = python_operator(stypy.reporting.localization.Localization(__file__, 1218, 21), 'div', abs_call_result_59606, float_59607)
        
        # Applying the binary operator '*' (line 1218)
        result_mul_59609 = python_operator(stypy.reporting.localization.Localization(__file__, 1218, 15), '*', float_59600, result_div_59608)
        
        float_59610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1218, 62), 'float')
        # Applying the binary operator '+' (line 1218)
        result_add_59611 = python_operator(stypy.reporting.localization.Localization(__file__, 1218, 15), '+', result_mul_59609, float_59610)
        
        # Assigning a type to the variable 'stypy_return_type' (line 1218)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1218, 8), 'stypy_return_type', result_add_59611)
        
        # ################# End of 'score_weight(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'score_weight' in the type store
        # Getting the type of 'stypy_return_type' (line 1193)
        stypy_return_type_59612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1193, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_59612)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'score_weight'
        return stypy_return_type_59612


    @norecursion
    def score_size(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'score_size'
        module_type_store = module_type_store.open_function_context('score_size', 1220, 4, False)
        # Assigning a type to the variable 'self' (line 1221)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1221, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FontManager.score_size.__dict__.__setitem__('stypy_localization', localization)
        FontManager.score_size.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FontManager.score_size.__dict__.__setitem__('stypy_type_store', module_type_store)
        FontManager.score_size.__dict__.__setitem__('stypy_function_name', 'FontManager.score_size')
        FontManager.score_size.__dict__.__setitem__('stypy_param_names_list', ['size1', 'size2'])
        FontManager.score_size.__dict__.__setitem__('stypy_varargs_param_name', None)
        FontManager.score_size.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FontManager.score_size.__dict__.__setitem__('stypy_call_defaults', defaults)
        FontManager.score_size.__dict__.__setitem__('stypy_call_varargs', varargs)
        FontManager.score_size.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FontManager.score_size.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FontManager.score_size', ['size1', 'size2'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'score_size', localization, ['size1', 'size2'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'score_size(...)' code ##################

        unicode_59613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1230, (-1)), 'unicode', u"\n        Returns a match score between *size1* and *size2*.\n\n        If *size2* (the size specified in the font file) is 'scalable', this\n        function always returns 0.0, since any font size can be generated.\n\n        Otherwise, the result is the absolute distance between *size1* and\n        *size2*, normalized so that the usual range of font sizes (6pt -\n        72pt) will lie between 0.0 and 1.0.\n        ")
        
        
        # Getting the type of 'size2' (line 1231)
        size2_59614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1231, 11), 'size2')
        unicode_59615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1231, 20), 'unicode', u'scalable')
        # Applying the binary operator '==' (line 1231)
        result_eq_59616 = python_operator(stypy.reporting.localization.Localization(__file__, 1231, 11), '==', size2_59614, unicode_59615)
        
        # Testing the type of an if condition (line 1231)
        if_condition_59617 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1231, 8), result_eq_59616)
        # Assigning a type to the variable 'if_condition_59617' (line 1231)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1231, 8), 'if_condition_59617', if_condition_59617)
        # SSA begins for if statement (line 1231)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        float_59618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1232, 19), 'float')
        # Assigning a type to the variable 'stypy_return_type' (line 1232)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1232, 12), 'stypy_return_type', float_59618)
        # SSA join for if statement (line 1231)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # SSA begins for try-except statement (line 1234)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Name (line 1235):
        
        # Assigning a Call to a Name (line 1235):
        
        # Call to float(...): (line 1235)
        # Processing the call arguments (line 1235)
        # Getting the type of 'size1' (line 1235)
        size1_59620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1235, 29), 'size1', False)
        # Processing the call keyword arguments (line 1235)
        kwargs_59621 = {}
        # Getting the type of 'float' (line 1235)
        float_59619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1235, 23), 'float', False)
        # Calling float(args, kwargs) (line 1235)
        float_call_result_59622 = invoke(stypy.reporting.localization.Localization(__file__, 1235, 23), float_59619, *[size1_59620], **kwargs_59621)
        
        # Assigning a type to the variable 'sizeval1' (line 1235)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1235, 12), 'sizeval1', float_call_result_59622)
        # SSA branch for the except part of a try statement (line 1234)
        # SSA branch for the except 'ValueError' branch of a try statement (line 1234)
        module_type_store.open_ssa_branch('except')
        
        # Assigning a BinOp to a Name (line 1237):
        
        # Assigning a BinOp to a Name (line 1237):
        # Getting the type of 'self' (line 1237)
        self_59623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1237, 23), 'self')
        # Obtaining the member 'default_size' of a type (line 1237)
        default_size_59624 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1237, 23), self_59623, 'default_size')
        
        # Obtaining the type of the subscript
        # Getting the type of 'size1' (line 1237)
        size1_59625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1237, 57), 'size1')
        # Getting the type of 'font_scalings' (line 1237)
        font_scalings_59626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1237, 43), 'font_scalings')
        # Obtaining the member '__getitem__' of a type (line 1237)
        getitem___59627 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1237, 43), font_scalings_59626, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1237)
        subscript_call_result_59628 = invoke(stypy.reporting.localization.Localization(__file__, 1237, 43), getitem___59627, size1_59625)
        
        # Applying the binary operator '*' (line 1237)
        result_mul_59629 = python_operator(stypy.reporting.localization.Localization(__file__, 1237, 23), '*', default_size_59624, subscript_call_result_59628)
        
        # Assigning a type to the variable 'sizeval1' (line 1237)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1237, 12), 'sizeval1', result_mul_59629)
        # SSA join for try-except statement (line 1234)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # SSA begins for try-except statement (line 1238)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Name (line 1239):
        
        # Assigning a Call to a Name (line 1239):
        
        # Call to float(...): (line 1239)
        # Processing the call arguments (line 1239)
        # Getting the type of 'size2' (line 1239)
        size2_59631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1239, 29), 'size2', False)
        # Processing the call keyword arguments (line 1239)
        kwargs_59632 = {}
        # Getting the type of 'float' (line 1239)
        float_59630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1239, 23), 'float', False)
        # Calling float(args, kwargs) (line 1239)
        float_call_result_59633 = invoke(stypy.reporting.localization.Localization(__file__, 1239, 23), float_59630, *[size2_59631], **kwargs_59632)
        
        # Assigning a type to the variable 'sizeval2' (line 1239)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1239, 12), 'sizeval2', float_call_result_59633)
        # SSA branch for the except part of a try statement (line 1238)
        # SSA branch for the except 'ValueError' branch of a try statement (line 1238)
        module_type_store.open_ssa_branch('except')
        float_59634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1241, 19), 'float')
        # Assigning a type to the variable 'stypy_return_type' (line 1241)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1241, 12), 'stypy_return_type', float_59634)
        # SSA join for try-except statement (line 1238)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to abs(...): (line 1242)
        # Processing the call arguments (line 1242)
        # Getting the type of 'sizeval1' (line 1242)
        sizeval1_59636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1242, 19), 'sizeval1', False)
        # Getting the type of 'sizeval2' (line 1242)
        sizeval2_59637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1242, 30), 'sizeval2', False)
        # Applying the binary operator '-' (line 1242)
        result_sub_59638 = python_operator(stypy.reporting.localization.Localization(__file__, 1242, 19), '-', sizeval1_59636, sizeval2_59637)
        
        # Processing the call keyword arguments (line 1242)
        kwargs_59639 = {}
        # Getting the type of 'abs' (line 1242)
        abs_59635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1242, 15), 'abs', False)
        # Calling abs(args, kwargs) (line 1242)
        abs_call_result_59640 = invoke(stypy.reporting.localization.Localization(__file__, 1242, 15), abs_59635, *[result_sub_59638], **kwargs_59639)
        
        float_59641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1242, 42), 'float')
        # Applying the binary operator 'div' (line 1242)
        result_div_59642 = python_operator(stypy.reporting.localization.Localization(__file__, 1242, 15), 'div', abs_call_result_59640, float_59641)
        
        # Assigning a type to the variable 'stypy_return_type' (line 1242)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1242, 8), 'stypy_return_type', result_div_59642)
        
        # ################# End of 'score_size(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'score_size' in the type store
        # Getting the type of 'stypy_return_type' (line 1220)
        stypy_return_type_59643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1220, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_59643)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'score_size'
        return stypy_return_type_59643


    @norecursion
    def findfont(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        unicode_59644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1244, 37), 'unicode', u'ttf')
        # Getting the type of 'None' (line 1244)
        None_59645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1244, 54), 'None')
        # Getting the type of 'True' (line 1245)
        True_59646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1245, 37), 'True')
        # Getting the type of 'True' (line 1245)
        True_59647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1245, 62), 'True')
        defaults = [unicode_59644, None_59645, True_59646, True_59647]
        # Create a new context for function 'findfont'
        module_type_store = module_type_store.open_function_context('findfont', 1244, 4, False)
        # Assigning a type to the variable 'self' (line 1245)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1245, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FontManager.findfont.__dict__.__setitem__('stypy_localization', localization)
        FontManager.findfont.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FontManager.findfont.__dict__.__setitem__('stypy_type_store', module_type_store)
        FontManager.findfont.__dict__.__setitem__('stypy_function_name', 'FontManager.findfont')
        FontManager.findfont.__dict__.__setitem__('stypy_param_names_list', ['prop', 'fontext', 'directory', 'fallback_to_default', 'rebuild_if_missing'])
        FontManager.findfont.__dict__.__setitem__('stypy_varargs_param_name', None)
        FontManager.findfont.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FontManager.findfont.__dict__.__setitem__('stypy_call_defaults', defaults)
        FontManager.findfont.__dict__.__setitem__('stypy_call_varargs', varargs)
        FontManager.findfont.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FontManager.findfont.__dict__.__setitem__('stypy_declared_arg_number', 6)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FontManager.findfont', ['prop', 'fontext', 'directory', 'fallback_to_default', 'rebuild_if_missing'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'findfont', localization, ['prop', 'fontext', 'directory', 'fallback_to_default', 'rebuild_if_missing'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'findfont(...)' code ##################

        unicode_59648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1269, (-1)), 'unicode', u'\n        Search the font list for the font that most closely matches\n        the :class:`FontProperties` *prop*.\n\n        :meth:`findfont` performs a nearest neighbor search.  Each\n        font is given a similarity score to the target font\n        properties.  The first font with the highest score is\n        returned.  If no matches below a certain threshold are found,\n        the default font (usually DejaVu Sans) is returned.\n\n        `directory`, is specified, will only return fonts from the\n        given directory (or subdirectory of that directory).\n\n        The result is cached, so subsequent lookups don\'t have to\n        perform the O(n) nearest neighbor search.\n\n        If `fallback_to_default` is True, will fallback to the default\n        font family (usually "DejaVu Sans" or "Helvetica") if\n        the first lookup hard-fails.\n\n        See the `W3C Cascading Style Sheet, Level 1\n        <http://www.w3.org/TR/1998/REC-CSS2-19980512/>`_ documentation\n        for a description of the font finding algorithm.\n        ')
        
        
        
        # Call to isinstance(...): (line 1270)
        # Processing the call arguments (line 1270)
        # Getting the type of 'prop' (line 1270)
        prop_59650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1270, 26), 'prop', False)
        # Getting the type of 'FontProperties' (line 1270)
        FontProperties_59651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1270, 32), 'FontProperties', False)
        # Processing the call keyword arguments (line 1270)
        kwargs_59652 = {}
        # Getting the type of 'isinstance' (line 1270)
        isinstance_59649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1270, 15), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 1270)
        isinstance_call_result_59653 = invoke(stypy.reporting.localization.Localization(__file__, 1270, 15), isinstance_59649, *[prop_59650, FontProperties_59651], **kwargs_59652)
        
        # Applying the 'not' unary operator (line 1270)
        result_not__59654 = python_operator(stypy.reporting.localization.Localization(__file__, 1270, 11), 'not', isinstance_call_result_59653)
        
        # Testing the type of an if condition (line 1270)
        if_condition_59655 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1270, 8), result_not__59654)
        # Assigning a type to the variable 'if_condition_59655' (line 1270)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1270, 8), 'if_condition_59655', if_condition_59655)
        # SSA begins for if statement (line 1270)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 1271):
        
        # Assigning a Call to a Name (line 1271):
        
        # Call to FontProperties(...): (line 1271)
        # Processing the call arguments (line 1271)
        # Getting the type of 'prop' (line 1271)
        prop_59657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1271, 34), 'prop', False)
        # Processing the call keyword arguments (line 1271)
        kwargs_59658 = {}
        # Getting the type of 'FontProperties' (line 1271)
        FontProperties_59656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1271, 19), 'FontProperties', False)
        # Calling FontProperties(args, kwargs) (line 1271)
        FontProperties_call_result_59659 = invoke(stypy.reporting.localization.Localization(__file__, 1271, 19), FontProperties_59656, *[prop_59657], **kwargs_59658)
        
        # Assigning a type to the variable 'prop' (line 1271)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1271, 12), 'prop', FontProperties_call_result_59659)
        # SSA join for if statement (line 1270)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 1272):
        
        # Assigning a Call to a Name (line 1272):
        
        # Call to get_file(...): (line 1272)
        # Processing the call keyword arguments (line 1272)
        kwargs_59662 = {}
        # Getting the type of 'prop' (line 1272)
        prop_59660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1272, 16), 'prop', False)
        # Obtaining the member 'get_file' of a type (line 1272)
        get_file_59661 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1272, 16), prop_59660, 'get_file')
        # Calling get_file(args, kwargs) (line 1272)
        get_file_call_result_59663 = invoke(stypy.reporting.localization.Localization(__file__, 1272, 16), get_file_59661, *[], **kwargs_59662)
        
        # Assigning a type to the variable 'fname' (line 1272)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1272, 8), 'fname', get_file_call_result_59663)
        
        # Type idiom detected: calculating its left and rigth part (line 1273)
        # Getting the type of 'fname' (line 1273)
        fname_59664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1273, 8), 'fname')
        # Getting the type of 'None' (line 1273)
        None_59665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1273, 24), 'None')
        
        (may_be_59666, more_types_in_union_59667) = may_not_be_none(fname_59664, None_59665)

        if may_be_59666:

            if more_types_in_union_59667:
                # Runtime conditional SSA (line 1273)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to report(...): (line 1274)
            # Processing the call arguments (line 1274)
            unicode_59670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1274, 27), 'unicode', u'findfont returning %s')
            # Getting the type of 'fname' (line 1274)
            fname_59671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1274, 51), 'fname', False)
            # Applying the binary operator '%' (line 1274)
            result_mod_59672 = python_operator(stypy.reporting.localization.Localization(__file__, 1274, 27), '%', unicode_59670, fname_59671)
            
            unicode_59673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1274, 58), 'unicode', u'debug')
            # Processing the call keyword arguments (line 1274)
            kwargs_59674 = {}
            # Getting the type of 'verbose' (line 1274)
            verbose_59668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1274, 12), 'verbose', False)
            # Obtaining the member 'report' of a type (line 1274)
            report_59669 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1274, 12), verbose_59668, 'report')
            # Calling report(args, kwargs) (line 1274)
            report_call_result_59675 = invoke(stypy.reporting.localization.Localization(__file__, 1274, 12), report_59669, *[result_mod_59672, unicode_59673], **kwargs_59674)
            
            # Getting the type of 'fname' (line 1275)
            fname_59676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1275, 19), 'fname')
            # Assigning a type to the variable 'stypy_return_type' (line 1275)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1275, 12), 'stypy_return_type', fname_59676)

            if more_types_in_union_59667:
                # SSA join for if statement (line 1273)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        # Getting the type of 'fontext' (line 1277)
        fontext_59677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1277, 11), 'fontext')
        unicode_59678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1277, 22), 'unicode', u'afm')
        # Applying the binary operator '==' (line 1277)
        result_eq_59679 = python_operator(stypy.reporting.localization.Localization(__file__, 1277, 11), '==', fontext_59677, unicode_59678)
        
        # Testing the type of an if condition (line 1277)
        if_condition_59680 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1277, 8), result_eq_59679)
        # Assigning a type to the variable 'if_condition_59680' (line 1277)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1277, 8), 'if_condition_59680', if_condition_59680)
        # SSA begins for if statement (line 1277)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Name (line 1278):
        
        # Assigning a Attribute to a Name (line 1278):
        # Getting the type of 'self' (line 1278)
        self_59681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1278, 23), 'self')
        # Obtaining the member 'afmlist' of a type (line 1278)
        afmlist_59682 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1278, 23), self_59681, 'afmlist')
        # Assigning a type to the variable 'fontlist' (line 1278)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1278, 12), 'fontlist', afmlist_59682)
        # SSA branch for the else part of an if statement (line 1277)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Attribute to a Name (line 1280):
        
        # Assigning a Attribute to a Name (line 1280):
        # Getting the type of 'self' (line 1280)
        self_59683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1280, 23), 'self')
        # Obtaining the member 'ttflist' of a type (line 1280)
        ttflist_59684 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1280, 23), self_59683, 'ttflist')
        # Assigning a type to the variable 'fontlist' (line 1280)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1280, 12), 'fontlist', ttflist_59684)
        # SSA join for if statement (line 1277)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Type idiom detected: calculating its left and rigth part (line 1282)
        # Getting the type of 'directory' (line 1282)
        directory_59685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1282, 11), 'directory')
        # Getting the type of 'None' (line 1282)
        None_59686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1282, 24), 'None')
        
        (may_be_59687, more_types_in_union_59688) = may_be_none(directory_59685, None_59686)

        if may_be_59687:

            if more_types_in_union_59688:
                # Runtime conditional SSA (line 1282)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 1283):
            
            # Assigning a Call to a Name (line 1283):
            
            # Call to get(...): (line 1283)
            # Processing the call arguments (line 1283)
            # Getting the type of 'prop' (line 1283)
            prop_59694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1283, 48), 'prop', False)
            # Processing the call keyword arguments (line 1283)
            kwargs_59695 = {}
            
            # Obtaining the type of the subscript
            # Getting the type of 'fontext' (line 1283)
            fontext_59689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1283, 35), 'fontext', False)
            # Getting the type of '_lookup_cache' (line 1283)
            _lookup_cache_59690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1283, 21), '_lookup_cache', False)
            # Obtaining the member '__getitem__' of a type (line 1283)
            getitem___59691 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1283, 21), _lookup_cache_59690, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 1283)
            subscript_call_result_59692 = invoke(stypy.reporting.localization.Localization(__file__, 1283, 21), getitem___59691, fontext_59689)
            
            # Obtaining the member 'get' of a type (line 1283)
            get_59693 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1283, 21), subscript_call_result_59692, 'get')
            # Calling get(args, kwargs) (line 1283)
            get_call_result_59696 = invoke(stypy.reporting.localization.Localization(__file__, 1283, 21), get_59693, *[prop_59694], **kwargs_59695)
            
            # Assigning a type to the variable 'cached' (line 1283)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1283, 12), 'cached', get_call_result_59696)
            
            # Type idiom detected: calculating its left and rigth part (line 1284)
            # Getting the type of 'cached' (line 1284)
            cached_59697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1284, 12), 'cached')
            # Getting the type of 'None' (line 1284)
            None_59698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1284, 29), 'None')
            
            (may_be_59699, more_types_in_union_59700) = may_not_be_none(cached_59697, None_59698)

            if may_be_59699:

                if more_types_in_union_59700:
                    # Runtime conditional SSA (line 1284)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                # Getting the type of 'cached' (line 1285)
                cached_59701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1285, 23), 'cached')
                # Assigning a type to the variable 'stypy_return_type' (line 1285)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1285, 16), 'stypy_return_type', cached_59701)

                if more_types_in_union_59700:
                    # SSA join for if statement (line 1284)
                    module_type_store = module_type_store.join_ssa_context()


            

            if more_types_in_union_59688:
                # Runtime conditional SSA for else branch (line 1282)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_59687) or more_types_in_union_59688):
            
            # Assigning a Call to a Name (line 1287):
            
            # Assigning a Call to a Name (line 1287):
            
            # Call to normcase(...): (line 1287)
            # Processing the call arguments (line 1287)
            # Getting the type of 'directory' (line 1287)
            directory_59705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1287, 41), 'directory', False)
            # Processing the call keyword arguments (line 1287)
            kwargs_59706 = {}
            # Getting the type of 'os' (line 1287)
            os_59702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1287, 24), 'os', False)
            # Obtaining the member 'path' of a type (line 1287)
            path_59703 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1287, 24), os_59702, 'path')
            # Obtaining the member 'normcase' of a type (line 1287)
            normcase_59704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1287, 24), path_59703, 'normcase')
            # Calling normcase(args, kwargs) (line 1287)
            normcase_call_result_59707 = invoke(stypy.reporting.localization.Localization(__file__, 1287, 24), normcase_59704, *[directory_59705], **kwargs_59706)
            
            # Assigning a type to the variable 'directory' (line 1287)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1287, 12), 'directory', normcase_call_result_59707)

            if (may_be_59687 and more_types_in_union_59688):
                # SSA join for if statement (line 1282)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Num to a Name (line 1289):
        
        # Assigning a Num to a Name (line 1289):
        float_59708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1289, 21), 'float')
        # Assigning a type to the variable 'best_score' (line 1289)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1289, 8), 'best_score', float_59708)
        
        # Assigning a Name to a Name (line 1290):
        
        # Assigning a Name to a Name (line 1290):
        # Getting the type of 'None' (line 1290)
        None_59709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1290, 20), 'None')
        # Assigning a type to the variable 'best_font' (line 1290)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1290, 8), 'best_font', None_59709)
        
        # Getting the type of 'fontlist' (line 1292)
        fontlist_59710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1292, 20), 'fontlist')
        # Testing the type of a for loop iterable (line 1292)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1292, 8), fontlist_59710)
        # Getting the type of the for loop variable (line 1292)
        for_loop_var_59711 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1292, 8), fontlist_59710)
        # Assigning a type to the variable 'font' (line 1292)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1292, 8), 'font', for_loop_var_59711)
        # SSA begins for a for statement (line 1292)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'directory' (line 1293)
        directory_59712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1293, 16), 'directory')
        # Getting the type of 'None' (line 1293)
        None_59713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1293, 33), 'None')
        # Applying the binary operator 'isnot' (line 1293)
        result_is_not_59714 = python_operator(stypy.reporting.localization.Localization(__file__, 1293, 16), 'isnot', directory_59712, None_59713)
        
        
        
        # Call to commonprefix(...): (line 1294)
        # Processing the call arguments (line 1294)
        
        # Obtaining an instance of the builtin type 'list' (line 1294)
        list_59718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1294, 41), 'list')
        # Adding type elements to the builtin type 'list' instance (line 1294)
        # Adding element type (line 1294)
        
        # Call to normcase(...): (line 1294)
        # Processing the call arguments (line 1294)
        # Getting the type of 'font' (line 1294)
        font_59722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1294, 59), 'font', False)
        # Obtaining the member 'fname' of a type (line 1294)
        fname_59723 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1294, 59), font_59722, 'fname')
        # Processing the call keyword arguments (line 1294)
        kwargs_59724 = {}
        # Getting the type of 'os' (line 1294)
        os_59719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1294, 42), 'os', False)
        # Obtaining the member 'path' of a type (line 1294)
        path_59720 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1294, 42), os_59719, 'path')
        # Obtaining the member 'normcase' of a type (line 1294)
        normcase_59721 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1294, 42), path_59720, 'normcase')
        # Calling normcase(args, kwargs) (line 1294)
        normcase_call_result_59725 = invoke(stypy.reporting.localization.Localization(__file__, 1294, 42), normcase_59721, *[fname_59723], **kwargs_59724)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1294, 41), list_59718, normcase_call_result_59725)
        # Adding element type (line 1294)
        # Getting the type of 'directory' (line 1295)
        directory_59726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1295, 42), 'directory', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1294, 41), list_59718, directory_59726)
        
        # Processing the call keyword arguments (line 1294)
        kwargs_59727 = {}
        # Getting the type of 'os' (line 1294)
        os_59715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1294, 20), 'os', False)
        # Obtaining the member 'path' of a type (line 1294)
        path_59716 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1294, 20), os_59715, 'path')
        # Obtaining the member 'commonprefix' of a type (line 1294)
        commonprefix_59717 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1294, 20), path_59716, 'commonprefix')
        # Calling commonprefix(args, kwargs) (line 1294)
        commonprefix_call_result_59728 = invoke(stypy.reporting.localization.Localization(__file__, 1294, 20), commonprefix_59717, *[list_59718], **kwargs_59727)
        
        # Getting the type of 'directory' (line 1295)
        directory_59729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1295, 57), 'directory')
        # Applying the binary operator '!=' (line 1294)
        result_ne_59730 = python_operator(stypy.reporting.localization.Localization(__file__, 1294, 20), '!=', commonprefix_call_result_59728, directory_59729)
        
        # Applying the binary operator 'and' (line 1293)
        result_and_keyword_59731 = python_operator(stypy.reporting.localization.Localization(__file__, 1293, 16), 'and', result_is_not_59714, result_ne_59730)
        
        # Testing the type of an if condition (line 1293)
        if_condition_59732 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1293, 12), result_and_keyword_59731)
        # Assigning a type to the variable 'if_condition_59732' (line 1293)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1293, 12), 'if_condition_59732', if_condition_59732)
        # SSA begins for if statement (line 1293)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # SSA join for if statement (line 1293)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 1299):
        
        # Assigning a BinOp to a Name (line 1299):
        
        # Call to score_family(...): (line 1300)
        # Processing the call arguments (line 1300)
        
        # Call to get_family(...): (line 1300)
        # Processing the call keyword arguments (line 1300)
        kwargs_59737 = {}
        # Getting the type of 'prop' (line 1300)
        prop_59735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1300, 34), 'prop', False)
        # Obtaining the member 'get_family' of a type (line 1300)
        get_family_59736 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1300, 34), prop_59735, 'get_family')
        # Calling get_family(args, kwargs) (line 1300)
        get_family_call_result_59738 = invoke(stypy.reporting.localization.Localization(__file__, 1300, 34), get_family_59736, *[], **kwargs_59737)
        
        # Getting the type of 'font' (line 1300)
        font_59739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1300, 53), 'font', False)
        # Obtaining the member 'name' of a type (line 1300)
        name_59740 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1300, 53), font_59739, 'name')
        # Processing the call keyword arguments (line 1300)
        kwargs_59741 = {}
        # Getting the type of 'self' (line 1300)
        self_59733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1300, 16), 'self', False)
        # Obtaining the member 'score_family' of a type (line 1300)
        score_family_59734 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1300, 16), self_59733, 'score_family')
        # Calling score_family(args, kwargs) (line 1300)
        score_family_call_result_59742 = invoke(stypy.reporting.localization.Localization(__file__, 1300, 16), score_family_59734, *[get_family_call_result_59738, name_59740], **kwargs_59741)
        
        float_59743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1300, 66), 'float')
        # Applying the binary operator '*' (line 1300)
        result_mul_59744 = python_operator(stypy.reporting.localization.Localization(__file__, 1300, 16), '*', score_family_call_result_59742, float_59743)
        
        
        # Call to score_style(...): (line 1301)
        # Processing the call arguments (line 1301)
        
        # Call to get_style(...): (line 1301)
        # Processing the call keyword arguments (line 1301)
        kwargs_59749 = {}
        # Getting the type of 'prop' (line 1301)
        prop_59747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1301, 33), 'prop', False)
        # Obtaining the member 'get_style' of a type (line 1301)
        get_style_59748 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1301, 33), prop_59747, 'get_style')
        # Calling get_style(args, kwargs) (line 1301)
        get_style_call_result_59750 = invoke(stypy.reporting.localization.Localization(__file__, 1301, 33), get_style_59748, *[], **kwargs_59749)
        
        # Getting the type of 'font' (line 1301)
        font_59751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1301, 51), 'font', False)
        # Obtaining the member 'style' of a type (line 1301)
        style_59752 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1301, 51), font_59751, 'style')
        # Processing the call keyword arguments (line 1301)
        kwargs_59753 = {}
        # Getting the type of 'self' (line 1301)
        self_59745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1301, 16), 'self', False)
        # Obtaining the member 'score_style' of a type (line 1301)
        score_style_59746 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1301, 16), self_59745, 'score_style')
        # Calling score_style(args, kwargs) (line 1301)
        score_style_call_result_59754 = invoke(stypy.reporting.localization.Localization(__file__, 1301, 16), score_style_59746, *[get_style_call_result_59750, style_59752], **kwargs_59753)
        
        # Applying the binary operator '+' (line 1300)
        result_add_59755 = python_operator(stypy.reporting.localization.Localization(__file__, 1300, 16), '+', result_mul_59744, score_style_call_result_59754)
        
        
        # Call to score_variant(...): (line 1302)
        # Processing the call arguments (line 1302)
        
        # Call to get_variant(...): (line 1302)
        # Processing the call keyword arguments (line 1302)
        kwargs_59760 = {}
        # Getting the type of 'prop' (line 1302)
        prop_59758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1302, 35), 'prop', False)
        # Obtaining the member 'get_variant' of a type (line 1302)
        get_variant_59759 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1302, 35), prop_59758, 'get_variant')
        # Calling get_variant(args, kwargs) (line 1302)
        get_variant_call_result_59761 = invoke(stypy.reporting.localization.Localization(__file__, 1302, 35), get_variant_59759, *[], **kwargs_59760)
        
        # Getting the type of 'font' (line 1302)
        font_59762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1302, 55), 'font', False)
        # Obtaining the member 'variant' of a type (line 1302)
        variant_59763 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1302, 55), font_59762, 'variant')
        # Processing the call keyword arguments (line 1302)
        kwargs_59764 = {}
        # Getting the type of 'self' (line 1302)
        self_59756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1302, 16), 'self', False)
        # Obtaining the member 'score_variant' of a type (line 1302)
        score_variant_59757 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1302, 16), self_59756, 'score_variant')
        # Calling score_variant(args, kwargs) (line 1302)
        score_variant_call_result_59765 = invoke(stypy.reporting.localization.Localization(__file__, 1302, 16), score_variant_59757, *[get_variant_call_result_59761, variant_59763], **kwargs_59764)
        
        # Applying the binary operator '+' (line 1301)
        result_add_59766 = python_operator(stypy.reporting.localization.Localization(__file__, 1301, 63), '+', result_add_59755, score_variant_call_result_59765)
        
        
        # Call to score_weight(...): (line 1303)
        # Processing the call arguments (line 1303)
        
        # Call to get_weight(...): (line 1303)
        # Processing the call keyword arguments (line 1303)
        kwargs_59771 = {}
        # Getting the type of 'prop' (line 1303)
        prop_59769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1303, 34), 'prop', False)
        # Obtaining the member 'get_weight' of a type (line 1303)
        get_weight_59770 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1303, 34), prop_59769, 'get_weight')
        # Calling get_weight(args, kwargs) (line 1303)
        get_weight_call_result_59772 = invoke(stypy.reporting.localization.Localization(__file__, 1303, 34), get_weight_59770, *[], **kwargs_59771)
        
        # Getting the type of 'font' (line 1303)
        font_59773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1303, 53), 'font', False)
        # Obtaining the member 'weight' of a type (line 1303)
        weight_59774 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1303, 53), font_59773, 'weight')
        # Processing the call keyword arguments (line 1303)
        kwargs_59775 = {}
        # Getting the type of 'self' (line 1303)
        self_59767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1303, 16), 'self', False)
        # Obtaining the member 'score_weight' of a type (line 1303)
        score_weight_59768 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1303, 16), self_59767, 'score_weight')
        # Calling score_weight(args, kwargs) (line 1303)
        score_weight_call_result_59776 = invoke(stypy.reporting.localization.Localization(__file__, 1303, 16), score_weight_59768, *[get_weight_call_result_59772, weight_59774], **kwargs_59775)
        
        # Applying the binary operator '+' (line 1302)
        result_add_59777 = python_operator(stypy.reporting.localization.Localization(__file__, 1302, 69), '+', result_add_59766, score_weight_call_result_59776)
        
        
        # Call to score_stretch(...): (line 1304)
        # Processing the call arguments (line 1304)
        
        # Call to get_stretch(...): (line 1304)
        # Processing the call keyword arguments (line 1304)
        kwargs_59782 = {}
        # Getting the type of 'prop' (line 1304)
        prop_59780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1304, 35), 'prop', False)
        # Obtaining the member 'get_stretch' of a type (line 1304)
        get_stretch_59781 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1304, 35), prop_59780, 'get_stretch')
        # Calling get_stretch(args, kwargs) (line 1304)
        get_stretch_call_result_59783 = invoke(stypy.reporting.localization.Localization(__file__, 1304, 35), get_stretch_59781, *[], **kwargs_59782)
        
        # Getting the type of 'font' (line 1304)
        font_59784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1304, 55), 'font', False)
        # Obtaining the member 'stretch' of a type (line 1304)
        stretch_59785 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1304, 55), font_59784, 'stretch')
        # Processing the call keyword arguments (line 1304)
        kwargs_59786 = {}
        # Getting the type of 'self' (line 1304)
        self_59778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1304, 16), 'self', False)
        # Obtaining the member 'score_stretch' of a type (line 1304)
        score_stretch_59779 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1304, 16), self_59778, 'score_stretch')
        # Calling score_stretch(args, kwargs) (line 1304)
        score_stretch_call_result_59787 = invoke(stypy.reporting.localization.Localization(__file__, 1304, 16), score_stretch_59779, *[get_stretch_call_result_59783, stretch_59785], **kwargs_59786)
        
        # Applying the binary operator '+' (line 1303)
        result_add_59788 = python_operator(stypy.reporting.localization.Localization(__file__, 1303, 66), '+', result_add_59777, score_stretch_call_result_59787)
        
        
        # Call to score_size(...): (line 1305)
        # Processing the call arguments (line 1305)
        
        # Call to get_size(...): (line 1305)
        # Processing the call keyword arguments (line 1305)
        kwargs_59793 = {}
        # Getting the type of 'prop' (line 1305)
        prop_59791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1305, 32), 'prop', False)
        # Obtaining the member 'get_size' of a type (line 1305)
        get_size_59792 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1305, 32), prop_59791, 'get_size')
        # Calling get_size(args, kwargs) (line 1305)
        get_size_call_result_59794 = invoke(stypy.reporting.localization.Localization(__file__, 1305, 32), get_size_59792, *[], **kwargs_59793)
        
        # Getting the type of 'font' (line 1305)
        font_59795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1305, 49), 'font', False)
        # Obtaining the member 'size' of a type (line 1305)
        size_59796 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1305, 49), font_59795, 'size')
        # Processing the call keyword arguments (line 1305)
        kwargs_59797 = {}
        # Getting the type of 'self' (line 1305)
        self_59789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1305, 16), 'self', False)
        # Obtaining the member 'score_size' of a type (line 1305)
        score_size_59790 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1305, 16), self_59789, 'score_size')
        # Calling score_size(args, kwargs) (line 1305)
        score_size_call_result_59798 = invoke(stypy.reporting.localization.Localization(__file__, 1305, 16), score_size_59790, *[get_size_call_result_59794, size_59796], **kwargs_59797)
        
        # Applying the binary operator '+' (line 1304)
        result_add_59799 = python_operator(stypy.reporting.localization.Localization(__file__, 1304, 69), '+', result_add_59788, score_size_call_result_59798)
        
        # Assigning a type to the variable 'score' (line 1299)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1299, 12), 'score', result_add_59799)
        
        
        # Getting the type of 'score' (line 1306)
        score_59800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1306, 15), 'score')
        # Getting the type of 'best_score' (line 1306)
        best_score_59801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1306, 23), 'best_score')
        # Applying the binary operator '<' (line 1306)
        result_lt_59802 = python_operator(stypy.reporting.localization.Localization(__file__, 1306, 15), '<', score_59800, best_score_59801)
        
        # Testing the type of an if condition (line 1306)
        if_condition_59803 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1306, 12), result_lt_59802)
        # Assigning a type to the variable 'if_condition_59803' (line 1306)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1306, 12), 'if_condition_59803', if_condition_59803)
        # SSA begins for if statement (line 1306)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 1307):
        
        # Assigning a Name to a Name (line 1307):
        # Getting the type of 'score' (line 1307)
        score_59804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1307, 29), 'score')
        # Assigning a type to the variable 'best_score' (line 1307)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1307, 16), 'best_score', score_59804)
        
        # Assigning a Name to a Name (line 1308):
        
        # Assigning a Name to a Name (line 1308):
        # Getting the type of 'font' (line 1308)
        font_59805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1308, 28), 'font')
        # Assigning a type to the variable 'best_font' (line 1308)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1308, 16), 'best_font', font_59805)
        # SSA join for if statement (line 1306)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'score' (line 1309)
        score_59806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1309, 15), 'score')
        int_59807 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1309, 24), 'int')
        # Applying the binary operator '==' (line 1309)
        result_eq_59808 = python_operator(stypy.reporting.localization.Localization(__file__, 1309, 15), '==', score_59806, int_59807)
        
        # Testing the type of an if condition (line 1309)
        if_condition_59809 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1309, 12), result_eq_59808)
        # Assigning a type to the variable 'if_condition_59809' (line 1309)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1309, 12), 'if_condition_59809', if_condition_59809)
        # SSA begins for if statement (line 1309)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # SSA join for if statement (line 1309)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'best_font' (line 1312)
        best_font_59810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1312, 11), 'best_font')
        # Getting the type of 'None' (line 1312)
        None_59811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1312, 24), 'None')
        # Applying the binary operator 'is' (line 1312)
        result_is__59812 = python_operator(stypy.reporting.localization.Localization(__file__, 1312, 11), 'is', best_font_59810, None_59811)
        
        
        # Getting the type of 'best_score' (line 1312)
        best_score_59813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1312, 32), 'best_score')
        float_59814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1312, 46), 'float')
        # Applying the binary operator '>=' (line 1312)
        result_ge_59815 = python_operator(stypy.reporting.localization.Localization(__file__, 1312, 32), '>=', best_score_59813, float_59814)
        
        # Applying the binary operator 'or' (line 1312)
        result_or_keyword_59816 = python_operator(stypy.reporting.localization.Localization(__file__, 1312, 11), 'or', result_is__59812, result_ge_59815)
        
        # Testing the type of an if condition (line 1312)
        if_condition_59817 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1312, 8), result_or_keyword_59816)
        # Assigning a type to the variable 'if_condition_59817' (line 1312)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1312, 8), 'if_condition_59817', if_condition_59817)
        # SSA begins for if statement (line 1312)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'fallback_to_default' (line 1313)
        fallback_to_default_59818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1313, 15), 'fallback_to_default')
        # Testing the type of an if condition (line 1313)
        if_condition_59819 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1313, 12), fallback_to_default_59818)
        # Assigning a type to the variable 'if_condition_59819' (line 1313)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1313, 12), 'if_condition_59819', if_condition_59819)
        # SSA begins for if statement (line 1313)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to warn(...): (line 1314)
        # Processing the call arguments (line 1314)
        unicode_59822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1315, 20), 'unicode', u'findfont: Font family %s not found. Falling back to %s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 1316)
        tuple_59823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1316, 21), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 1316)
        # Adding element type (line 1316)
        
        # Call to get_family(...): (line 1316)
        # Processing the call keyword arguments (line 1316)
        kwargs_59826 = {}
        # Getting the type of 'prop' (line 1316)
        prop_59824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1316, 21), 'prop', False)
        # Obtaining the member 'get_family' of a type (line 1316)
        get_family_59825 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1316, 21), prop_59824, 'get_family')
        # Calling get_family(args, kwargs) (line 1316)
        get_family_call_result_59827 = invoke(stypy.reporting.localization.Localization(__file__, 1316, 21), get_family_59825, *[], **kwargs_59826)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1316, 21), tuple_59823, get_family_call_result_59827)
        # Adding element type (line 1316)
        
        # Obtaining the type of the subscript
        # Getting the type of 'fontext' (line 1316)
        fontext_59828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1316, 59), 'fontext', False)
        # Getting the type of 'self' (line 1316)
        self_59829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1316, 40), 'self', False)
        # Obtaining the member 'defaultFamily' of a type (line 1316)
        defaultFamily_59830 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1316, 40), self_59829, 'defaultFamily')
        # Obtaining the member '__getitem__' of a type (line 1316)
        getitem___59831 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1316, 40), defaultFamily_59830, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1316)
        subscript_call_result_59832 = invoke(stypy.reporting.localization.Localization(__file__, 1316, 40), getitem___59831, fontext_59828)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1316, 21), tuple_59823, subscript_call_result_59832)
        
        # Applying the binary operator '%' (line 1315)
        result_mod_59833 = python_operator(stypy.reporting.localization.Localization(__file__, 1315, 20), '%', unicode_59822, tuple_59823)
        
        # Processing the call keyword arguments (line 1314)
        kwargs_59834 = {}
        # Getting the type of 'warnings' (line 1314)
        warnings_59820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1314, 16), 'warnings', False)
        # Obtaining the member 'warn' of a type (line 1314)
        warn_59821 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1314, 16), warnings_59820, 'warn')
        # Calling warn(args, kwargs) (line 1314)
        warn_call_result_59835 = invoke(stypy.reporting.localization.Localization(__file__, 1314, 16), warn_59821, *[result_mod_59833], **kwargs_59834)
        
        
        # Assigning a Call to a Name (line 1317):
        
        # Assigning a Call to a Name (line 1317):
        
        # Call to copy(...): (line 1317)
        # Processing the call keyword arguments (line 1317)
        kwargs_59838 = {}
        # Getting the type of 'prop' (line 1317)
        prop_59836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1317, 31), 'prop', False)
        # Obtaining the member 'copy' of a type (line 1317)
        copy_59837 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1317, 31), prop_59836, 'copy')
        # Calling copy(args, kwargs) (line 1317)
        copy_call_result_59839 = invoke(stypy.reporting.localization.Localization(__file__, 1317, 31), copy_59837, *[], **kwargs_59838)
        
        # Assigning a type to the variable 'default_prop' (line 1317)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1317, 16), 'default_prop', copy_call_result_59839)
        
        # Call to set_family(...): (line 1318)
        # Processing the call arguments (line 1318)
        
        # Obtaining the type of the subscript
        # Getting the type of 'fontext' (line 1318)
        fontext_59842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1318, 59), 'fontext', False)
        # Getting the type of 'self' (line 1318)
        self_59843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1318, 40), 'self', False)
        # Obtaining the member 'defaultFamily' of a type (line 1318)
        defaultFamily_59844 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1318, 40), self_59843, 'defaultFamily')
        # Obtaining the member '__getitem__' of a type (line 1318)
        getitem___59845 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1318, 40), defaultFamily_59844, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1318)
        subscript_call_result_59846 = invoke(stypy.reporting.localization.Localization(__file__, 1318, 40), getitem___59845, fontext_59842)
        
        # Processing the call keyword arguments (line 1318)
        kwargs_59847 = {}
        # Getting the type of 'default_prop' (line 1318)
        default_prop_59840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1318, 16), 'default_prop', False)
        # Obtaining the member 'set_family' of a type (line 1318)
        set_family_59841 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1318, 16), default_prop_59840, 'set_family')
        # Calling set_family(args, kwargs) (line 1318)
        set_family_call_result_59848 = invoke(stypy.reporting.localization.Localization(__file__, 1318, 16), set_family_59841, *[subscript_call_result_59846], **kwargs_59847)
        
        
        # Call to findfont(...): (line 1319)
        # Processing the call arguments (line 1319)
        # Getting the type of 'default_prop' (line 1319)
        default_prop_59851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1319, 37), 'default_prop', False)
        # Getting the type of 'fontext' (line 1319)
        fontext_59852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1319, 51), 'fontext', False)
        # Getting the type of 'directory' (line 1319)
        directory_59853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1319, 60), 'directory', False)
        # Getting the type of 'False' (line 1319)
        False_59854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1319, 71), 'False', False)
        # Processing the call keyword arguments (line 1319)
        kwargs_59855 = {}
        # Getting the type of 'self' (line 1319)
        self_59849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1319, 23), 'self', False)
        # Obtaining the member 'findfont' of a type (line 1319)
        findfont_59850 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1319, 23), self_59849, 'findfont')
        # Calling findfont(args, kwargs) (line 1319)
        findfont_call_result_59856 = invoke(stypy.reporting.localization.Localization(__file__, 1319, 23), findfont_59850, *[default_prop_59851, fontext_59852, directory_59853, False_59854], **kwargs_59855)
        
        # Assigning a type to the variable 'stypy_return_type' (line 1319)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1319, 16), 'stypy_return_type', findfont_call_result_59856)
        # SSA branch for the else part of an if statement (line 1313)
        module_type_store.open_ssa_branch('else')
        
        # Call to warn(...): (line 1323)
        # Processing the call arguments (line 1323)
        unicode_59859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1324, 20), 'unicode', u'findfont: Could not match %s. Returning %s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 1325)
        tuple_59860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1325, 21), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 1325)
        # Adding element type (line 1325)
        # Getting the type of 'prop' (line 1325)
        prop_59861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1325, 21), 'prop', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1325, 21), tuple_59860, prop_59861)
        # Adding element type (line 1325)
        
        # Obtaining the type of the subscript
        # Getting the type of 'fontext' (line 1325)
        fontext_59862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1325, 44), 'fontext', False)
        # Getting the type of 'self' (line 1325)
        self_59863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1325, 27), 'self', False)
        # Obtaining the member 'defaultFont' of a type (line 1325)
        defaultFont_59864 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1325, 27), self_59863, 'defaultFont')
        # Obtaining the member '__getitem__' of a type (line 1325)
        getitem___59865 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1325, 27), defaultFont_59864, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1325)
        subscript_call_result_59866 = invoke(stypy.reporting.localization.Localization(__file__, 1325, 27), getitem___59865, fontext_59862)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1325, 21), tuple_59860, subscript_call_result_59866)
        
        # Applying the binary operator '%' (line 1324)
        result_mod_59867 = python_operator(stypy.reporting.localization.Localization(__file__, 1324, 20), '%', unicode_59859, tuple_59860)
        
        # Getting the type of 'UserWarning' (line 1326)
        UserWarning_59868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1326, 20), 'UserWarning', False)
        # Processing the call keyword arguments (line 1323)
        kwargs_59869 = {}
        # Getting the type of 'warnings' (line 1323)
        warnings_59857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1323, 16), 'warnings', False)
        # Obtaining the member 'warn' of a type (line 1323)
        warn_59858 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1323, 16), warnings_59857, 'warn')
        # Calling warn(args, kwargs) (line 1323)
        warn_call_result_59870 = invoke(stypy.reporting.localization.Localization(__file__, 1323, 16), warn_59858, *[result_mod_59867, UserWarning_59868], **kwargs_59869)
        
        
        # Assigning a Subscript to a Name (line 1327):
        
        # Assigning a Subscript to a Name (line 1327):
        
        # Obtaining the type of the subscript
        # Getting the type of 'fontext' (line 1327)
        fontext_59871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1327, 42), 'fontext')
        # Getting the type of 'self' (line 1327)
        self_59872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1327, 25), 'self')
        # Obtaining the member 'defaultFont' of a type (line 1327)
        defaultFont_59873 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1327, 25), self_59872, 'defaultFont')
        # Obtaining the member '__getitem__' of a type (line 1327)
        getitem___59874 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1327, 25), defaultFont_59873, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1327)
        subscript_call_result_59875 = invoke(stypy.reporting.localization.Localization(__file__, 1327, 25), getitem___59874, fontext_59871)
        
        # Assigning a type to the variable 'result' (line 1327)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1327, 16), 'result', subscript_call_result_59875)
        # SSA join for if statement (line 1313)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 1312)
        module_type_store.open_ssa_branch('else')
        
        # Call to report(...): (line 1329)
        # Processing the call arguments (line 1329)
        unicode_59878 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1330, 16), 'unicode', u'findfont: Matching %s to %s (%s) with score of %f')
        
        # Obtaining an instance of the builtin type 'tuple' (line 1331)
        tuple_59879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1331, 17), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 1331)
        # Adding element type (line 1331)
        # Getting the type of 'prop' (line 1331)
        prop_59880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1331, 17), 'prop', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1331, 17), tuple_59879, prop_59880)
        # Adding element type (line 1331)
        # Getting the type of 'best_font' (line 1331)
        best_font_59881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1331, 23), 'best_font', False)
        # Obtaining the member 'name' of a type (line 1331)
        name_59882 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1331, 23), best_font_59881, 'name')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1331, 17), tuple_59879, name_59882)
        # Adding element type (line 1331)
        
        # Call to repr(...): (line 1331)
        # Processing the call arguments (line 1331)
        # Getting the type of 'best_font' (line 1331)
        best_font_59884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1331, 44), 'best_font', False)
        # Obtaining the member 'fname' of a type (line 1331)
        fname_59885 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1331, 44), best_font_59884, 'fname')
        # Processing the call keyword arguments (line 1331)
        kwargs_59886 = {}
        # Getting the type of 'repr' (line 1331)
        repr_59883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1331, 39), 'repr', False)
        # Calling repr(args, kwargs) (line 1331)
        repr_call_result_59887 = invoke(stypy.reporting.localization.Localization(__file__, 1331, 39), repr_59883, *[fname_59885], **kwargs_59886)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1331, 17), tuple_59879, repr_call_result_59887)
        # Adding element type (line 1331)
        # Getting the type of 'best_score' (line 1331)
        best_score_59888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1331, 62), 'best_score', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1331, 17), tuple_59879, best_score_59888)
        
        # Applying the binary operator '%' (line 1330)
        result_mod_59889 = python_operator(stypy.reporting.localization.Localization(__file__, 1330, 16), '%', unicode_59878, tuple_59879)
        
        # Processing the call keyword arguments (line 1329)
        kwargs_59890 = {}
        # Getting the type of 'verbose' (line 1329)
        verbose_59876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1329, 12), 'verbose', False)
        # Obtaining the member 'report' of a type (line 1329)
        report_59877 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1329, 12), verbose_59876, 'report')
        # Calling report(args, kwargs) (line 1329)
        report_call_result_59891 = invoke(stypy.reporting.localization.Localization(__file__, 1329, 12), report_59877, *[result_mod_59889], **kwargs_59890)
        
        
        # Assigning a Attribute to a Name (line 1332):
        
        # Assigning a Attribute to a Name (line 1332):
        # Getting the type of 'best_font' (line 1332)
        best_font_59892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1332, 21), 'best_font')
        # Obtaining the member 'fname' of a type (line 1332)
        fname_59893 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1332, 21), best_font_59892, 'fname')
        # Assigning a type to the variable 'result' (line 1332)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1332, 12), 'result', fname_59893)
        # SSA join for if statement (line 1312)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Call to isfile(...): (line 1334)
        # Processing the call arguments (line 1334)
        # Getting the type of 'result' (line 1334)
        result_59897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1334, 30), 'result', False)
        # Processing the call keyword arguments (line 1334)
        kwargs_59898 = {}
        # Getting the type of 'os' (line 1334)
        os_59894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1334, 15), 'os', False)
        # Obtaining the member 'path' of a type (line 1334)
        path_59895 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1334, 15), os_59894, 'path')
        # Obtaining the member 'isfile' of a type (line 1334)
        isfile_59896 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1334, 15), path_59895, 'isfile')
        # Calling isfile(args, kwargs) (line 1334)
        isfile_call_result_59899 = invoke(stypy.reporting.localization.Localization(__file__, 1334, 15), isfile_59896, *[result_59897], **kwargs_59898)
        
        # Applying the 'not' unary operator (line 1334)
        result_not__59900 = python_operator(stypy.reporting.localization.Localization(__file__, 1334, 11), 'not', isfile_call_result_59899)
        
        # Testing the type of an if condition (line 1334)
        if_condition_59901 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1334, 8), result_not__59900)
        # Assigning a type to the variable 'if_condition_59901' (line 1334)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1334, 8), 'if_condition_59901', if_condition_59901)
        # SSA begins for if statement (line 1334)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'rebuild_if_missing' (line 1335)
        rebuild_if_missing_59902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1335, 15), 'rebuild_if_missing')
        # Testing the type of an if condition (line 1335)
        if_condition_59903 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1335, 12), rebuild_if_missing_59902)
        # Assigning a type to the variable 'if_condition_59903' (line 1335)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1335, 12), 'if_condition_59903', if_condition_59903)
        # SSA begins for if statement (line 1335)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to report(...): (line 1336)
        # Processing the call arguments (line 1336)
        unicode_59906 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1337, 20), 'unicode', u'findfont: Found a missing font file.  Rebuilding cache.')
        # Processing the call keyword arguments (line 1336)
        kwargs_59907 = {}
        # Getting the type of 'verbose' (line 1336)
        verbose_59904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1336, 16), 'verbose', False)
        # Obtaining the member 'report' of a type (line 1336)
        report_59905 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1336, 16), verbose_59904, 'report')
        # Calling report(args, kwargs) (line 1336)
        report_call_result_59908 = invoke(stypy.reporting.localization.Localization(__file__, 1336, 16), report_59905, *[unicode_59906], **kwargs_59907)
        
        
        # Call to _rebuild(...): (line 1338)
        # Processing the call keyword arguments (line 1338)
        kwargs_59910 = {}
        # Getting the type of '_rebuild' (line 1338)
        _rebuild_59909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1338, 16), '_rebuild', False)
        # Calling _rebuild(args, kwargs) (line 1338)
        _rebuild_call_result_59911 = invoke(stypy.reporting.localization.Localization(__file__, 1338, 16), _rebuild_59909, *[], **kwargs_59910)
        
        
        # Call to findfont(...): (line 1339)
        # Processing the call arguments (line 1339)
        # Getting the type of 'prop' (line 1340)
        prop_59914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1340, 20), 'prop', False)
        # Getting the type of 'fontext' (line 1340)
        fontext_59915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1340, 26), 'fontext', False)
        # Getting the type of 'directory' (line 1340)
        directory_59916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1340, 35), 'directory', False)
        # Getting the type of 'True' (line 1340)
        True_59917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1340, 46), 'True', False)
        # Getting the type of 'False' (line 1340)
        False_59918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1340, 52), 'False', False)
        # Processing the call keyword arguments (line 1339)
        kwargs_59919 = {}
        # Getting the type of 'fontManager' (line 1339)
        fontManager_59912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1339, 23), 'fontManager', False)
        # Obtaining the member 'findfont' of a type (line 1339)
        findfont_59913 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1339, 23), fontManager_59912, 'findfont')
        # Calling findfont(args, kwargs) (line 1339)
        findfont_call_result_59920 = invoke(stypy.reporting.localization.Localization(__file__, 1339, 23), findfont_59913, *[prop_59914, fontext_59915, directory_59916, True_59917, False_59918], **kwargs_59919)
        
        # Assigning a type to the variable 'stypy_return_type' (line 1339)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1339, 16), 'stypy_return_type', findfont_call_result_59920)
        # SSA branch for the else part of an if statement (line 1335)
        module_type_store.open_ssa_branch('else')
        
        # Call to ValueError(...): (line 1342)
        # Processing the call arguments (line 1342)
        unicode_59922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1342, 33), 'unicode', u'No valid font could be found')
        # Processing the call keyword arguments (line 1342)
        kwargs_59923 = {}
        # Getting the type of 'ValueError' (line 1342)
        ValueError_59921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1342, 22), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 1342)
        ValueError_call_result_59924 = invoke(stypy.reporting.localization.Localization(__file__, 1342, 22), ValueError_59921, *[unicode_59922], **kwargs_59923)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1342, 16), ValueError_call_result_59924, 'raise parameter', BaseException)
        # SSA join for if statement (line 1335)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 1334)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Type idiom detected: calculating its left and rigth part (line 1344)
        # Getting the type of 'directory' (line 1344)
        directory_59925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1344, 11), 'directory')
        # Getting the type of 'None' (line 1344)
        None_59926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1344, 24), 'None')
        
        (may_be_59927, more_types_in_union_59928) = may_be_none(directory_59925, None_59926)

        if may_be_59927:

            if more_types_in_union_59928:
                # Runtime conditional SSA (line 1344)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to set(...): (line 1345)
            # Processing the call arguments (line 1345)
            # Getting the type of 'prop' (line 1345)
            prop_59934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1345, 39), 'prop', False)
            # Getting the type of 'result' (line 1345)
            result_59935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1345, 45), 'result', False)
            # Processing the call keyword arguments (line 1345)
            kwargs_59936 = {}
            
            # Obtaining the type of the subscript
            # Getting the type of 'fontext' (line 1345)
            fontext_59929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1345, 26), 'fontext', False)
            # Getting the type of '_lookup_cache' (line 1345)
            _lookup_cache_59930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1345, 12), '_lookup_cache', False)
            # Obtaining the member '__getitem__' of a type (line 1345)
            getitem___59931 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1345, 12), _lookup_cache_59930, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 1345)
            subscript_call_result_59932 = invoke(stypy.reporting.localization.Localization(__file__, 1345, 12), getitem___59931, fontext_59929)
            
            # Obtaining the member 'set' of a type (line 1345)
            set_59933 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1345, 12), subscript_call_result_59932, 'set')
            # Calling set(args, kwargs) (line 1345)
            set_call_result_59937 = invoke(stypy.reporting.localization.Localization(__file__, 1345, 12), set_59933, *[prop_59934, result_59935], **kwargs_59936)
            

            if more_types_in_union_59928:
                # SSA join for if statement (line 1344)
                module_type_store = module_type_store.join_ssa_context()


        
        # Getting the type of 'result' (line 1346)
        result_59938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1346, 15), 'result')
        # Assigning a type to the variable 'stypy_return_type' (line 1346)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1346, 8), 'stypy_return_type', result_59938)
        
        # ################# End of 'findfont(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'findfont' in the type store
        # Getting the type of 'stypy_return_type' (line 1244)
        stypy_return_type_59939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1244, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_59939)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'findfont'
        return stypy_return_type_59939


# Assigning a type to the variable 'FontManager' (line 1022)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1022, 0), 'FontManager', FontManager)

# Assigning a Num to a Name (line 1034):
int_59940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1034, 18), 'int')
# Getting the type of 'FontManager'
FontManager_59941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FontManager')
# Setting the type of the member '__version__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FontManager_59941, '__version__', int_59940)

# Assigning a Dict to a Name (line 1348):

# Assigning a Dict to a Name (line 1348):

# Obtaining an instance of the builtin type 'dict' (line 1348)
dict_59942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1348, 30), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 1348)

# Assigning a type to the variable '_is_opentype_cff_font_cache' (line 1348)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1348, 0), '_is_opentype_cff_font_cache', dict_59942)

@norecursion
def is_opentype_cff_font(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'is_opentype_cff_font'
    module_type_store = module_type_store.open_function_context('is_opentype_cff_font', 1349, 0, False)
    
    # Passed parameters checking function
    is_opentype_cff_font.stypy_localization = localization
    is_opentype_cff_font.stypy_type_of_self = None
    is_opentype_cff_font.stypy_type_store = module_type_store
    is_opentype_cff_font.stypy_function_name = 'is_opentype_cff_font'
    is_opentype_cff_font.stypy_param_names_list = ['filename']
    is_opentype_cff_font.stypy_varargs_param_name = None
    is_opentype_cff_font.stypy_kwargs_param_name = None
    is_opentype_cff_font.stypy_call_defaults = defaults
    is_opentype_cff_font.stypy_call_varargs = varargs
    is_opentype_cff_font.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'is_opentype_cff_font', ['filename'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'is_opentype_cff_font', localization, ['filename'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'is_opentype_cff_font(...)' code ##################

    unicode_59943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1354, (-1)), 'unicode', u'\n    Returns True if the given font is a Postscript Compact Font Format\n    Font embedded in an OpenType wrapper.  Used by the PostScript and\n    PDF backends that can not subset these fonts.\n    ')
    
    
    
    # Call to lower(...): (line 1355)
    # Processing the call keyword arguments (line 1355)
    kwargs_59954 = {}
    
    # Obtaining the type of the subscript
    int_59944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1355, 34), 'int')
    
    # Call to splitext(...): (line 1355)
    # Processing the call arguments (line 1355)
    # Getting the type of 'filename' (line 1355)
    filename_59948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1355, 24), 'filename', False)
    # Processing the call keyword arguments (line 1355)
    kwargs_59949 = {}
    # Getting the type of 'os' (line 1355)
    os_59945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1355, 7), 'os', False)
    # Obtaining the member 'path' of a type (line 1355)
    path_59946 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1355, 7), os_59945, 'path')
    # Obtaining the member 'splitext' of a type (line 1355)
    splitext_59947 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1355, 7), path_59946, 'splitext')
    # Calling splitext(args, kwargs) (line 1355)
    splitext_call_result_59950 = invoke(stypy.reporting.localization.Localization(__file__, 1355, 7), splitext_59947, *[filename_59948], **kwargs_59949)
    
    # Obtaining the member '__getitem__' of a type (line 1355)
    getitem___59951 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1355, 7), splitext_call_result_59950, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1355)
    subscript_call_result_59952 = invoke(stypy.reporting.localization.Localization(__file__, 1355, 7), getitem___59951, int_59944)
    
    # Obtaining the member 'lower' of a type (line 1355)
    lower_59953 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1355, 7), subscript_call_result_59952, 'lower')
    # Calling lower(args, kwargs) (line 1355)
    lower_call_result_59955 = invoke(stypy.reporting.localization.Localization(__file__, 1355, 7), lower_59953, *[], **kwargs_59954)
    
    unicode_59956 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1355, 48), 'unicode', u'.otf')
    # Applying the binary operator '==' (line 1355)
    result_eq_59957 = python_operator(stypy.reporting.localization.Localization(__file__, 1355, 7), '==', lower_call_result_59955, unicode_59956)
    
    # Testing the type of an if condition (line 1355)
    if_condition_59958 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1355, 4), result_eq_59957)
    # Assigning a type to the variable 'if_condition_59958' (line 1355)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1355, 4), 'if_condition_59958', if_condition_59958)
    # SSA begins for if statement (line 1355)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 1356):
    
    # Assigning a Call to a Name (line 1356):
    
    # Call to get(...): (line 1356)
    # Processing the call arguments (line 1356)
    # Getting the type of 'filename' (line 1356)
    filename_59961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1356, 49), 'filename', False)
    # Processing the call keyword arguments (line 1356)
    kwargs_59962 = {}
    # Getting the type of '_is_opentype_cff_font_cache' (line 1356)
    _is_opentype_cff_font_cache_59959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1356, 17), '_is_opentype_cff_font_cache', False)
    # Obtaining the member 'get' of a type (line 1356)
    get_59960 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1356, 17), _is_opentype_cff_font_cache_59959, 'get')
    # Calling get(args, kwargs) (line 1356)
    get_call_result_59963 = invoke(stypy.reporting.localization.Localization(__file__, 1356, 17), get_59960, *[filename_59961], **kwargs_59962)
    
    # Assigning a type to the variable 'result' (line 1356)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1356, 8), 'result', get_call_result_59963)
    
    # Type idiom detected: calculating its left and rigth part (line 1357)
    # Getting the type of 'result' (line 1357)
    result_59964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1357, 11), 'result')
    # Getting the type of 'None' (line 1357)
    None_59965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1357, 21), 'None')
    
    (may_be_59966, more_types_in_union_59967) = may_be_none(result_59964, None_59965)

    if may_be_59966:

        if more_types_in_union_59967:
            # Runtime conditional SSA (line 1357)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Call to open(...): (line 1358)
        # Processing the call arguments (line 1358)
        # Getting the type of 'filename' (line 1358)
        filename_59969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1358, 22), 'filename', False)
        unicode_59970 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1358, 32), 'unicode', u'rb')
        # Processing the call keyword arguments (line 1358)
        kwargs_59971 = {}
        # Getting the type of 'open' (line 1358)
        open_59968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1358, 17), 'open', False)
        # Calling open(args, kwargs) (line 1358)
        open_call_result_59972 = invoke(stypy.reporting.localization.Localization(__file__, 1358, 17), open_59968, *[filename_59969, unicode_59970], **kwargs_59971)
        
        with_59973 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 1358, 17), open_call_result_59972, 'with parameter', '__enter__', '__exit__')

        if with_59973:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 1358)
            enter___59974 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1358, 17), open_call_result_59972, '__enter__')
            with_enter_59975 = invoke(stypy.reporting.localization.Localization(__file__, 1358, 17), enter___59974)
            # Assigning a type to the variable 'fd' (line 1358)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1358, 17), 'fd', with_enter_59975)
            
            # Assigning a Call to a Name (line 1359):
            
            # Assigning a Call to a Name (line 1359):
            
            # Call to read(...): (line 1359)
            # Processing the call arguments (line 1359)
            int_59978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1359, 30), 'int')
            # Processing the call keyword arguments (line 1359)
            kwargs_59979 = {}
            # Getting the type of 'fd' (line 1359)
            fd_59976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1359, 22), 'fd', False)
            # Obtaining the member 'read' of a type (line 1359)
            read_59977 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1359, 22), fd_59976, 'read')
            # Calling read(args, kwargs) (line 1359)
            read_call_result_59980 = invoke(stypy.reporting.localization.Localization(__file__, 1359, 22), read_59977, *[int_59978], **kwargs_59979)
            
            # Assigning a type to the variable 'tag' (line 1359)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1359, 16), 'tag', read_call_result_59980)
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 1358)
            exit___59981 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1358, 17), open_call_result_59972, '__exit__')
            with_exit_59982 = invoke(stypy.reporting.localization.Localization(__file__, 1358, 17), exit___59981, None, None, None)

        
        # Assigning a Compare to a Name (line 1360):
        
        # Assigning a Compare to a Name (line 1360):
        
        # Getting the type of 'tag' (line 1360)
        tag_59983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1360, 22), 'tag')
        str_59984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1360, 29), 'str', 'OTTO')
        # Applying the binary operator '==' (line 1360)
        result_eq_59985 = python_operator(stypy.reporting.localization.Localization(__file__, 1360, 22), '==', tag_59983, str_59984)
        
        # Assigning a type to the variable 'result' (line 1360)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1360, 12), 'result', result_eq_59985)
        
        # Assigning a Name to a Subscript (line 1361):
        
        # Assigning a Name to a Subscript (line 1361):
        # Getting the type of 'result' (line 1361)
        result_59986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1361, 52), 'result')
        # Getting the type of '_is_opentype_cff_font_cache' (line 1361)
        _is_opentype_cff_font_cache_59987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1361, 12), '_is_opentype_cff_font_cache')
        # Getting the type of 'filename' (line 1361)
        filename_59988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1361, 40), 'filename')
        # Storing an element on a container (line 1361)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1361, 12), _is_opentype_cff_font_cache_59987, (filename_59988, result_59986))

        if more_types_in_union_59967:
            # SSA join for if statement (line 1357)
            module_type_store = module_type_store.join_ssa_context()


    
    # Getting the type of 'result' (line 1362)
    result_59989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1362, 15), 'result')
    # Assigning a type to the variable 'stypy_return_type' (line 1362)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1362, 8), 'stypy_return_type', result_59989)
    # SSA join for if statement (line 1355)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'False' (line 1363)
    False_59990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1363, 11), 'False')
    # Assigning a type to the variable 'stypy_return_type' (line 1363)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1363, 4), 'stypy_return_type', False_59990)
    
    # ################# End of 'is_opentype_cff_font(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'is_opentype_cff_font' in the type store
    # Getting the type of 'stypy_return_type' (line 1349)
    stypy_return_type_59991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1349, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_59991)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'is_opentype_cff_font'
    return stypy_return_type_59991

# Assigning a type to the variable 'is_opentype_cff_font' (line 1349)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1349, 0), 'is_opentype_cff_font', is_opentype_cff_font)

# Assigning a Name to a Name (line 1365):

# Assigning a Name to a Name (line 1365):
# Getting the type of 'None' (line 1365)
None_59992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1365, 14), 'None')
# Assigning a type to the variable 'fontManager' (line 1365)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1365, 0), 'fontManager', None_59992)

# Assigning a Name to a Name (line 1366):

# Assigning a Name to a Name (line 1366):
# Getting the type of 'None' (line 1366)
None_59993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1366, 11), 'None')
# Assigning a type to the variable '_fmcache' (line 1366)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1366, 0), '_fmcache', None_59993)

# Assigning a Call to a Name (line 1369):

# Assigning a Call to a Name (line 1369):

# Call to (...): (line 1369)
# Processing the call arguments (line 1369)
# Getting the type of 'ft2font' (line 1369)
ft2font_59998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1369, 25), 'ft2font', False)
# Obtaining the member 'FT2Font' of a type (line 1369)
FT2Font_59999 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1369, 25), ft2font_59998, 'FT2Font')
# Processing the call keyword arguments (line 1369)
kwargs_60000 = {}

# Call to lru_cache(...): (line 1369)
# Processing the call arguments (line 1369)
int_59995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1369, 21), 'int')
# Processing the call keyword arguments (line 1369)
kwargs_59996 = {}
# Getting the type of 'lru_cache' (line 1369)
lru_cache_59994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1369, 11), 'lru_cache', False)
# Calling lru_cache(args, kwargs) (line 1369)
lru_cache_call_result_59997 = invoke(stypy.reporting.localization.Localization(__file__, 1369, 11), lru_cache_59994, *[int_59995], **kwargs_59996)

# Calling (args, kwargs) (line 1369)
_call_result_60001 = invoke(stypy.reporting.localization.Localization(__file__, 1369, 11), lru_cache_call_result_59997, *[FT2Font_59999], **kwargs_60000)

# Assigning a type to the variable 'get_font' (line 1369)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1369, 0), 'get_font', _call_result_60001)


# Evaluating a boolean operation
# Getting the type of 'USE_FONTCONFIG' (line 1373)
USE_FONTCONFIG_60002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1373, 3), 'USE_FONTCONFIG')

# Getting the type of 'sys' (line 1373)
sys_60003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1373, 22), 'sys')
# Obtaining the member 'platform' of a type (line 1373)
platform_60004 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1373, 22), sys_60003, 'platform')
unicode_60005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1373, 38), 'unicode', u'win32')
# Applying the binary operator '!=' (line 1373)
result_ne_60006 = python_operator(stypy.reporting.localization.Localization(__file__, 1373, 22), '!=', platform_60004, unicode_60005)

# Applying the binary operator 'and' (line 1373)
result_and_keyword_60007 = python_operator(stypy.reporting.localization.Localization(__file__, 1373, 3), 'and', USE_FONTCONFIG_60002, result_ne_60006)

# Testing the type of an if condition (line 1373)
if_condition_60008 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1373, 0), result_and_keyword_60007)
# Assigning a type to the variable 'if_condition_60008' (line 1373)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1373, 0), 'if_condition_60008', if_condition_60008)
# SSA begins for if statement (line 1373)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1374, 4))

# 'import re' statement (line 1374)
import re

import_module(stypy.reporting.localization.Localization(__file__, 1374, 4), 're', re, module_type_store)


@norecursion
def fc_match(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'fc_match'
    module_type_store = module_type_store.open_function_context('fc_match', 1376, 4, False)
    
    # Passed parameters checking function
    fc_match.stypy_localization = localization
    fc_match.stypy_type_of_self = None
    fc_match.stypy_type_store = module_type_store
    fc_match.stypy_function_name = 'fc_match'
    fc_match.stypy_param_names_list = ['pattern', 'fontext']
    fc_match.stypy_varargs_param_name = None
    fc_match.stypy_kwargs_param_name = None
    fc_match.stypy_call_defaults = defaults
    fc_match.stypy_call_varargs = varargs
    fc_match.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'fc_match', ['pattern', 'fontext'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'fc_match', localization, ['pattern', 'fontext'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'fc_match(...)' code ##################

    
    # Assigning a Call to a Name (line 1377):
    
    # Assigning a Call to a Name (line 1377):
    
    # Call to get_fontext_synonyms(...): (line 1377)
    # Processing the call arguments (line 1377)
    # Getting the type of 'fontext' (line 1377)
    fontext_60010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1377, 40), 'fontext', False)
    # Processing the call keyword arguments (line 1377)
    kwargs_60011 = {}
    # Getting the type of 'get_fontext_synonyms' (line 1377)
    get_fontext_synonyms_60009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1377, 19), 'get_fontext_synonyms', False)
    # Calling get_fontext_synonyms(args, kwargs) (line 1377)
    get_fontext_synonyms_call_result_60012 = invoke(stypy.reporting.localization.Localization(__file__, 1377, 19), get_fontext_synonyms_60009, *[fontext_60010], **kwargs_60011)
    
    # Assigning a type to the variable 'fontexts' (line 1377)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1377, 8), 'fontexts', get_fontext_synonyms_call_result_60012)
    
    # Assigning a BinOp to a Name (line 1378):
    
    # Assigning a BinOp to a Name (line 1378):
    unicode_60013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1378, 14), 'unicode', u'.')
    # Getting the type of 'fontext' (line 1378)
    fontext_60014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1378, 20), 'fontext')
    # Applying the binary operator '+' (line 1378)
    result_add_60015 = python_operator(stypy.reporting.localization.Localization(__file__, 1378, 14), '+', unicode_60013, fontext_60014)
    
    # Assigning a type to the variable 'ext' (line 1378)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1378, 8), 'ext', result_add_60015)
    
    
    # SSA begins for try-except statement (line 1379)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Name (line 1380):
    
    # Assigning a Call to a Name (line 1380):
    
    # Call to Popen(...): (line 1380)
    # Processing the call arguments (line 1380)
    
    # Obtaining an instance of the builtin type 'list' (line 1381)
    list_60018 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1381, 16), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1381)
    # Adding element type (line 1381)
    unicode_60019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1381, 17), 'unicode', u'fc-match')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1381, 16), list_60018, unicode_60019)
    # Adding element type (line 1381)
    unicode_60020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1381, 29), 'unicode', u'-s')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1381, 16), list_60018, unicode_60020)
    # Adding element type (line 1381)
    unicode_60021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1381, 35), 'unicode', u'--format=%{file}\\n')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1381, 16), list_60018, unicode_60021)
    # Adding element type (line 1381)
    # Getting the type of 'pattern' (line 1381)
    pattern_60022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1381, 58), 'pattern', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1381, 16), list_60018, pattern_60022)
    
    # Processing the call keyword arguments (line 1380)
    # Getting the type of 'subprocess' (line 1382)
    subprocess_60023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1382, 23), 'subprocess', False)
    # Obtaining the member 'PIPE' of a type (line 1382)
    PIPE_60024 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1382, 23), subprocess_60023, 'PIPE')
    keyword_60025 = PIPE_60024
    # Getting the type of 'subprocess' (line 1383)
    subprocess_60026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1383, 23), 'subprocess', False)
    # Obtaining the member 'PIPE' of a type (line 1383)
    PIPE_60027 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1383, 23), subprocess_60026, 'PIPE')
    keyword_60028 = PIPE_60027
    kwargs_60029 = {'stderr': keyword_60028, 'stdout': keyword_60025}
    # Getting the type of 'subprocess' (line 1380)
    subprocess_60016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1380, 19), 'subprocess', False)
    # Obtaining the member 'Popen' of a type (line 1380)
    Popen_60017 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1380, 19), subprocess_60016, 'Popen')
    # Calling Popen(args, kwargs) (line 1380)
    Popen_call_result_60030 = invoke(stypy.reporting.localization.Localization(__file__, 1380, 19), Popen_60017, *[list_60018], **kwargs_60029)
    
    # Assigning a type to the variable 'pipe' (line 1380)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1380, 12), 'pipe', Popen_call_result_60030)
    
    # Assigning a Subscript to a Name (line 1384):
    
    # Assigning a Subscript to a Name (line 1384):
    
    # Obtaining the type of the subscript
    int_60031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1384, 40), 'int')
    
    # Call to communicate(...): (line 1384)
    # Processing the call keyword arguments (line 1384)
    kwargs_60034 = {}
    # Getting the type of 'pipe' (line 1384)
    pipe_60032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1384, 21), 'pipe', False)
    # Obtaining the member 'communicate' of a type (line 1384)
    communicate_60033 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1384, 21), pipe_60032, 'communicate')
    # Calling communicate(args, kwargs) (line 1384)
    communicate_call_result_60035 = invoke(stypy.reporting.localization.Localization(__file__, 1384, 21), communicate_60033, *[], **kwargs_60034)
    
    # Obtaining the member '__getitem__' of a type (line 1384)
    getitem___60036 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1384, 21), communicate_call_result_60035, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1384)
    subscript_call_result_60037 = invoke(stypy.reporting.localization.Localization(__file__, 1384, 21), getitem___60036, int_60031)
    
    # Assigning a type to the variable 'output' (line 1384)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1384, 12), 'output', subscript_call_result_60037)
    # SSA branch for the except part of a try statement (line 1379)
    # SSA branch for the except 'Tuple' branch of a try statement (line 1379)
    module_type_store.open_ssa_branch('except')
    # Getting the type of 'None' (line 1386)
    None_60038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1386, 19), 'None')
    # Assigning a type to the variable 'stypy_return_type' (line 1386)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1386, 12), 'stypy_return_type', None_60038)
    # SSA join for try-except statement (line 1379)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'pipe' (line 1391)
    pipe_60039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1391, 11), 'pipe')
    # Obtaining the member 'returncode' of a type (line 1391)
    returncode_60040 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1391, 11), pipe_60039, 'returncode')
    int_60041 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1391, 30), 'int')
    # Applying the binary operator '==' (line 1391)
    result_eq_60042 = python_operator(stypy.reporting.localization.Localization(__file__, 1391, 11), '==', returncode_60040, int_60041)
    
    # Testing the type of an if condition (line 1391)
    if_condition_60043 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1391, 8), result_eq_60042)
    # Assigning a type to the variable 'if_condition_60043' (line 1391)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1391, 8), 'if_condition_60043', if_condition_60043)
    # SSA begins for if statement (line 1391)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Call to split(...): (line 1392)
    # Processing the call arguments (line 1392)
    str_60046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1392, 38), 'str', '\n')
    # Processing the call keyword arguments (line 1392)
    kwargs_60047 = {}
    # Getting the type of 'output' (line 1392)
    output_60044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1392, 25), 'output', False)
    # Obtaining the member 'split' of a type (line 1392)
    split_60045 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1392, 25), output_60044, 'split')
    # Calling split(args, kwargs) (line 1392)
    split_call_result_60048 = invoke(stypy.reporting.localization.Localization(__file__, 1392, 25), split_60045, *[str_60046], **kwargs_60047)
    
    # Testing the type of a for loop iterable (line 1392)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1392, 12), split_call_result_60048)
    # Getting the type of the for loop variable (line 1392)
    for_loop_var_60049 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1392, 12), split_call_result_60048)
    # Assigning a type to the variable 'fname' (line 1392)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1392, 12), 'fname', for_loop_var_60049)
    # SSA begins for a for statement (line 1392)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # SSA begins for try-except statement (line 1393)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Name (line 1394):
    
    # Assigning a Call to a Name (line 1394):
    
    # Call to text_type(...): (line 1394)
    # Processing the call arguments (line 1394)
    # Getting the type of 'fname' (line 1394)
    fname_60052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1394, 42), 'fname', False)
    
    # Call to getfilesystemencoding(...): (line 1394)
    # Processing the call keyword arguments (line 1394)
    kwargs_60055 = {}
    # Getting the type of 'sys' (line 1394)
    sys_60053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1394, 49), 'sys', False)
    # Obtaining the member 'getfilesystemencoding' of a type (line 1394)
    getfilesystemencoding_60054 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1394, 49), sys_60053, 'getfilesystemencoding')
    # Calling getfilesystemencoding(args, kwargs) (line 1394)
    getfilesystemencoding_call_result_60056 = invoke(stypy.reporting.localization.Localization(__file__, 1394, 49), getfilesystemencoding_60054, *[], **kwargs_60055)
    
    # Processing the call keyword arguments (line 1394)
    kwargs_60057 = {}
    # Getting the type of 'six' (line 1394)
    six_60050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1394, 28), 'six', False)
    # Obtaining the member 'text_type' of a type (line 1394)
    text_type_60051 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1394, 28), six_60050, 'text_type')
    # Calling text_type(args, kwargs) (line 1394)
    text_type_call_result_60058 = invoke(stypy.reporting.localization.Localization(__file__, 1394, 28), text_type_60051, *[fname_60052, getfilesystemencoding_call_result_60056], **kwargs_60057)
    
    # Assigning a type to the variable 'fname' (line 1394)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1394, 20), 'fname', text_type_call_result_60058)
    # SSA branch for the except part of a try statement (line 1393)
    # SSA branch for the except 'UnicodeDecodeError' branch of a try statement (line 1393)
    module_type_store.open_ssa_branch('except')
    # SSA join for try-except statement (line 1393)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Obtaining the type of the subscript
    int_60059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1397, 46), 'int')
    slice_60060 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1397, 19), int_60059, None, None)
    
    # Obtaining the type of the subscript
    int_60061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1397, 43), 'int')
    
    # Call to splitext(...): (line 1397)
    # Processing the call arguments (line 1397)
    # Getting the type of 'fname' (line 1397)
    fname_60065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1397, 36), 'fname', False)
    # Processing the call keyword arguments (line 1397)
    kwargs_60066 = {}
    # Getting the type of 'os' (line 1397)
    os_60062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1397, 19), 'os', False)
    # Obtaining the member 'path' of a type (line 1397)
    path_60063 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1397, 19), os_60062, 'path')
    # Obtaining the member 'splitext' of a type (line 1397)
    splitext_60064 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1397, 19), path_60063, 'splitext')
    # Calling splitext(args, kwargs) (line 1397)
    splitext_call_result_60067 = invoke(stypy.reporting.localization.Localization(__file__, 1397, 19), splitext_60064, *[fname_60065], **kwargs_60066)
    
    # Obtaining the member '__getitem__' of a type (line 1397)
    getitem___60068 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1397, 19), splitext_call_result_60067, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1397)
    subscript_call_result_60069 = invoke(stypy.reporting.localization.Localization(__file__, 1397, 19), getitem___60068, int_60061)
    
    # Obtaining the member '__getitem__' of a type (line 1397)
    getitem___60070 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1397, 19), subscript_call_result_60069, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1397)
    subscript_call_result_60071 = invoke(stypy.reporting.localization.Localization(__file__, 1397, 19), getitem___60070, slice_60060)
    
    # Getting the type of 'fontexts' (line 1397)
    fontexts_60072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1397, 53), 'fontexts')
    # Applying the binary operator 'in' (line 1397)
    result_contains_60073 = python_operator(stypy.reporting.localization.Localization(__file__, 1397, 19), 'in', subscript_call_result_60071, fontexts_60072)
    
    # Testing the type of an if condition (line 1397)
    if_condition_60074 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1397, 16), result_contains_60073)
    # Assigning a type to the variable 'if_condition_60074' (line 1397)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1397, 16), 'if_condition_60074', if_condition_60074)
    # SSA begins for if statement (line 1397)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'fname' (line 1398)
    fname_60075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1398, 27), 'fname')
    # Assigning a type to the variable 'stypy_return_type' (line 1398)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1398, 20), 'stypy_return_type', fname_60075)
    # SSA join for if statement (line 1397)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 1391)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'None' (line 1399)
    None_60076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1399, 15), 'None')
    # Assigning a type to the variable 'stypy_return_type' (line 1399)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1399, 8), 'stypy_return_type', None_60076)
    
    # ################# End of 'fc_match(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'fc_match' in the type store
    # Getting the type of 'stypy_return_type' (line 1376)
    stypy_return_type_60077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1376, 4), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_60077)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'fc_match'
    return stypy_return_type_60077

# Assigning a type to the variable 'fc_match' (line 1376)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1376, 4), 'fc_match', fc_match)

# Assigning a Dict to a Name (line 1401):

# Assigning a Dict to a Name (line 1401):

# Obtaining an instance of the builtin type 'dict' (line 1401)
dict_60078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1401, 22), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 1401)

# Assigning a type to the variable '_fc_match_cache' (line 1401)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1401, 4), '_fc_match_cache', dict_60078)

@norecursion
def findfont(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    unicode_60079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1403, 31), 'unicode', u'ttf')
    defaults = [unicode_60079]
    # Create a new context for function 'findfont'
    module_type_store = module_type_store.open_function_context('findfont', 1403, 4, False)
    
    # Passed parameters checking function
    findfont.stypy_localization = localization
    findfont.stypy_type_of_self = None
    findfont.stypy_type_store = module_type_store
    findfont.stypy_function_name = 'findfont'
    findfont.stypy_param_names_list = ['prop', 'fontext']
    findfont.stypy_varargs_param_name = None
    findfont.stypy_kwargs_param_name = None
    findfont.stypy_call_defaults = defaults
    findfont.stypy_call_varargs = varargs
    findfont.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'findfont', ['prop', 'fontext'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'findfont', localization, ['prop', 'fontext'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'findfont(...)' code ##################

    
    
    
    # Call to isinstance(...): (line 1404)
    # Processing the call arguments (line 1404)
    # Getting the type of 'prop' (line 1404)
    prop_60081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1404, 26), 'prop', False)
    # Getting the type of 'six' (line 1404)
    six_60082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1404, 32), 'six', False)
    # Obtaining the member 'string_types' of a type (line 1404)
    string_types_60083 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1404, 32), six_60082, 'string_types')
    # Processing the call keyword arguments (line 1404)
    kwargs_60084 = {}
    # Getting the type of 'isinstance' (line 1404)
    isinstance_60080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1404, 15), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 1404)
    isinstance_call_result_60085 = invoke(stypy.reporting.localization.Localization(__file__, 1404, 15), isinstance_60080, *[prop_60081, string_types_60083], **kwargs_60084)
    
    # Applying the 'not' unary operator (line 1404)
    result_not__60086 = python_operator(stypy.reporting.localization.Localization(__file__, 1404, 11), 'not', isinstance_call_result_60085)
    
    # Testing the type of an if condition (line 1404)
    if_condition_60087 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1404, 8), result_not__60086)
    # Assigning a type to the variable 'if_condition_60087' (line 1404)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1404, 8), 'if_condition_60087', if_condition_60087)
    # SSA begins for if statement (line 1404)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 1405):
    
    # Assigning a Call to a Name (line 1405):
    
    # Call to get_fontconfig_pattern(...): (line 1405)
    # Processing the call keyword arguments (line 1405)
    kwargs_60090 = {}
    # Getting the type of 'prop' (line 1405)
    prop_60088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1405, 19), 'prop', False)
    # Obtaining the member 'get_fontconfig_pattern' of a type (line 1405)
    get_fontconfig_pattern_60089 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1405, 19), prop_60088, 'get_fontconfig_pattern')
    # Calling get_fontconfig_pattern(args, kwargs) (line 1405)
    get_fontconfig_pattern_call_result_60091 = invoke(stypy.reporting.localization.Localization(__file__, 1405, 19), get_fontconfig_pattern_60089, *[], **kwargs_60090)
    
    # Assigning a type to the variable 'prop' (line 1405)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1405, 12), 'prop', get_fontconfig_pattern_call_result_60091)
    # SSA join for if statement (line 1404)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 1406):
    
    # Assigning a Call to a Name (line 1406):
    
    # Call to get(...): (line 1406)
    # Processing the call arguments (line 1406)
    # Getting the type of 'prop' (line 1406)
    prop_60094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1406, 37), 'prop', False)
    # Processing the call keyword arguments (line 1406)
    kwargs_60095 = {}
    # Getting the type of '_fc_match_cache' (line 1406)
    _fc_match_cache_60092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1406, 17), '_fc_match_cache', False)
    # Obtaining the member 'get' of a type (line 1406)
    get_60093 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1406, 17), _fc_match_cache_60092, 'get')
    # Calling get(args, kwargs) (line 1406)
    get_call_result_60096 = invoke(stypy.reporting.localization.Localization(__file__, 1406, 17), get_60093, *[prop_60094], **kwargs_60095)
    
    # Assigning a type to the variable 'cached' (line 1406)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1406, 8), 'cached', get_call_result_60096)
    
    # Type idiom detected: calculating its left and rigth part (line 1407)
    # Getting the type of 'cached' (line 1407)
    cached_60097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1407, 8), 'cached')
    # Getting the type of 'None' (line 1407)
    None_60098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1407, 25), 'None')
    
    (may_be_60099, more_types_in_union_60100) = may_not_be_none(cached_60097, None_60098)

    if may_be_60099:

        if more_types_in_union_60100:
            # Runtime conditional SSA (line 1407)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Getting the type of 'cached' (line 1408)
        cached_60101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1408, 19), 'cached')
        # Assigning a type to the variable 'stypy_return_type' (line 1408)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1408, 12), 'stypy_return_type', cached_60101)

        if more_types_in_union_60100:
            # SSA join for if statement (line 1407)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 1410):
    
    # Assigning a Call to a Name (line 1410):
    
    # Call to fc_match(...): (line 1410)
    # Processing the call arguments (line 1410)
    # Getting the type of 'prop' (line 1410)
    prop_60103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1410, 26), 'prop', False)
    # Getting the type of 'fontext' (line 1410)
    fontext_60104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1410, 32), 'fontext', False)
    # Processing the call keyword arguments (line 1410)
    kwargs_60105 = {}
    # Getting the type of 'fc_match' (line 1410)
    fc_match_60102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1410, 17), 'fc_match', False)
    # Calling fc_match(args, kwargs) (line 1410)
    fc_match_call_result_60106 = invoke(stypy.reporting.localization.Localization(__file__, 1410, 17), fc_match_60102, *[prop_60103, fontext_60104], **kwargs_60105)
    
    # Assigning a type to the variable 'result' (line 1410)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1410, 8), 'result', fc_match_call_result_60106)
    
    # Type idiom detected: calculating its left and rigth part (line 1411)
    # Getting the type of 'result' (line 1411)
    result_60107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1411, 11), 'result')
    # Getting the type of 'None' (line 1411)
    None_60108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1411, 21), 'None')
    
    (may_be_60109, more_types_in_union_60110) = may_be_none(result_60107, None_60108)

    if may_be_60109:

        if more_types_in_union_60110:
            # Runtime conditional SSA (line 1411)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 1412):
        
        # Assigning a Call to a Name (line 1412):
        
        # Call to fc_match(...): (line 1412)
        # Processing the call arguments (line 1412)
        unicode_60112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1412, 30), 'unicode', u':')
        # Getting the type of 'fontext' (line 1412)
        fontext_60113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1412, 35), 'fontext', False)
        # Processing the call keyword arguments (line 1412)
        kwargs_60114 = {}
        # Getting the type of 'fc_match' (line 1412)
        fc_match_60111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1412, 21), 'fc_match', False)
        # Calling fc_match(args, kwargs) (line 1412)
        fc_match_call_result_60115 = invoke(stypy.reporting.localization.Localization(__file__, 1412, 21), fc_match_60111, *[unicode_60112, fontext_60113], **kwargs_60114)
        
        # Assigning a type to the variable 'result' (line 1412)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1412, 12), 'result', fc_match_call_result_60115)

        if more_types_in_union_60110:
            # SSA join for if statement (line 1411)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Name to a Subscript (line 1414):
    
    # Assigning a Name to a Subscript (line 1414):
    # Getting the type of 'result' (line 1414)
    result_60116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1414, 32), 'result')
    # Getting the type of '_fc_match_cache' (line 1414)
    _fc_match_cache_60117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1414, 8), '_fc_match_cache')
    # Getting the type of 'prop' (line 1414)
    prop_60118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1414, 24), 'prop')
    # Storing an element on a container (line 1414)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1414, 8), _fc_match_cache_60117, (prop_60118, result_60116))
    # Getting the type of 'result' (line 1415)
    result_60119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1415, 15), 'result')
    # Assigning a type to the variable 'stypy_return_type' (line 1415)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1415, 8), 'stypy_return_type', result_60119)
    
    # ################# End of 'findfont(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'findfont' in the type store
    # Getting the type of 'stypy_return_type' (line 1403)
    stypy_return_type_60120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1403, 4), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_60120)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'findfont'
    return stypy_return_type_60120

# Assigning a type to the variable 'findfont' (line 1403)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1403, 4), 'findfont', findfont)
# SSA branch for the else part of an if statement (line 1373)
module_type_store.open_ssa_branch('else')

# Assigning a Name to a Name (line 1418):

# Assigning a Name to a Name (line 1418):
# Getting the type of 'None' (line 1418)
None_60121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1418, 15), 'None')
# Assigning a type to the variable '_fmcache' (line 1418)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1418, 4), '_fmcache', None_60121)

# Assigning a Call to a Name (line 1420):

# Assigning a Call to a Name (line 1420):

# Call to get_cachedir(...): (line 1420)
# Processing the call keyword arguments (line 1420)
kwargs_60123 = {}
# Getting the type of 'get_cachedir' (line 1420)
get_cachedir_60122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1420, 15), 'get_cachedir', False)
# Calling get_cachedir(args, kwargs) (line 1420)
get_cachedir_call_result_60124 = invoke(stypy.reporting.localization.Localization(__file__, 1420, 15), get_cachedir_60122, *[], **kwargs_60123)

# Assigning a type to the variable 'cachedir' (line 1420)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1420, 4), 'cachedir', get_cachedir_call_result_60124)

# Type idiom detected: calculating its left and rigth part (line 1421)
# Getting the type of 'cachedir' (line 1421)
cachedir_60125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1421, 4), 'cachedir')
# Getting the type of 'None' (line 1421)
None_60126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1421, 23), 'None')

(may_be_60127, more_types_in_union_60128) = may_not_be_none(cachedir_60125, None_60126)

if may_be_60127:

    if more_types_in_union_60128:
        # Runtime conditional SSA (line 1421)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
    else:
        module_type_store = module_type_store

    
    # Assigning a Call to a Name (line 1422):
    
    # Assigning a Call to a Name (line 1422):
    
    # Call to join(...): (line 1422)
    # Processing the call arguments (line 1422)
    # Getting the type of 'cachedir' (line 1422)
    cachedir_60132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1422, 32), 'cachedir', False)
    unicode_60133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1422, 42), 'unicode', u'fontList.json')
    # Processing the call keyword arguments (line 1422)
    kwargs_60134 = {}
    # Getting the type of 'os' (line 1422)
    os_60129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1422, 19), 'os', False)
    # Obtaining the member 'path' of a type (line 1422)
    path_60130 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1422, 19), os_60129, 'path')
    # Obtaining the member 'join' of a type (line 1422)
    join_60131 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1422, 19), path_60130, 'join')
    # Calling join(args, kwargs) (line 1422)
    join_call_result_60135 = invoke(stypy.reporting.localization.Localization(__file__, 1422, 19), join_60131, *[cachedir_60132, unicode_60133], **kwargs_60134)
    
    # Assigning a type to the variable '_fmcache' (line 1422)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1422, 8), '_fmcache', join_call_result_60135)

    if more_types_in_union_60128:
        # SSA join for if statement (line 1421)
        module_type_store = module_type_store.join_ssa_context()




# Assigning a Name to a Name (line 1424):

# Assigning a Name to a Name (line 1424):
# Getting the type of 'None' (line 1424)
None_60136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1424, 18), 'None')
# Assigning a type to the variable 'fontManager' (line 1424)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1424, 4), 'fontManager', None_60136)

# Assigning a Dict to a Name (line 1426):

# Assigning a Dict to a Name (line 1426):

# Obtaining an instance of the builtin type 'dict' (line 1426)
dict_60137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1426, 20), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 1426)
# Adding element type (key, value) (line 1426)
unicode_60138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1427, 8), 'unicode', u'ttf')

# Call to TempCache(...): (line 1427)
# Processing the call keyword arguments (line 1427)
kwargs_60140 = {}
# Getting the type of 'TempCache' (line 1427)
TempCache_60139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1427, 15), 'TempCache', False)
# Calling TempCache(args, kwargs) (line 1427)
TempCache_call_result_60141 = invoke(stypy.reporting.localization.Localization(__file__, 1427, 15), TempCache_60139, *[], **kwargs_60140)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1426, 20), dict_60137, (unicode_60138, TempCache_call_result_60141))
# Adding element type (key, value) (line 1426)
unicode_60142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1428, 8), 'unicode', u'afm')

# Call to TempCache(...): (line 1428)
# Processing the call keyword arguments (line 1428)
kwargs_60144 = {}
# Getting the type of 'TempCache' (line 1428)
TempCache_60143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1428, 15), 'TempCache', False)
# Calling TempCache(args, kwargs) (line 1428)
TempCache_call_result_60145 = invoke(stypy.reporting.localization.Localization(__file__, 1428, 15), TempCache_60143, *[], **kwargs_60144)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1426, 20), dict_60137, (unicode_60142, TempCache_call_result_60145))

# Assigning a type to the variable '_lookup_cache' (line 1426)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1426, 4), '_lookup_cache', dict_60137)

@norecursion
def _rebuild(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_rebuild'
    module_type_store = module_type_store.open_function_context('_rebuild', 1431, 4, False)
    
    # Passed parameters checking function
    _rebuild.stypy_localization = localization
    _rebuild.stypy_type_of_self = None
    _rebuild.stypy_type_store = module_type_store
    _rebuild.stypy_function_name = '_rebuild'
    _rebuild.stypy_param_names_list = []
    _rebuild.stypy_varargs_param_name = None
    _rebuild.stypy_kwargs_param_name = None
    _rebuild.stypy_call_defaults = defaults
    _rebuild.stypy_call_varargs = varargs
    _rebuild.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_rebuild', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_rebuild', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_rebuild(...)' code ##################

    # Marking variables as global (line 1432)
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 1432, 8), 'fontManager')
    
    # Assigning a Call to a Name (line 1434):
    
    # Assigning a Call to a Name (line 1434):
    
    # Call to FontManager(...): (line 1434)
    # Processing the call keyword arguments (line 1434)
    kwargs_60147 = {}
    # Getting the type of 'FontManager' (line 1434)
    FontManager_60146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1434, 22), 'FontManager', False)
    # Calling FontManager(args, kwargs) (line 1434)
    FontManager_call_result_60148 = invoke(stypy.reporting.localization.Localization(__file__, 1434, 22), FontManager_60146, *[], **kwargs_60147)
    
    # Assigning a type to the variable 'fontManager' (line 1434)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1434, 8), 'fontManager', FontManager_call_result_60148)
    
    # Getting the type of '_fmcache' (line 1436)
    _fmcache_60149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1436, 11), '_fmcache')
    # Testing the type of an if condition (line 1436)
    if_condition_60150 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1436, 8), _fmcache_60149)
    # Assigning a type to the variable 'if_condition_60150' (line 1436)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1436, 8), 'if_condition_60150', if_condition_60150)
    # SSA begins for if statement (line 1436)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to Locked(...): (line 1437)
    # Processing the call arguments (line 1437)
    # Getting the type of 'cachedir' (line 1437)
    cachedir_60153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1437, 30), 'cachedir', False)
    # Processing the call keyword arguments (line 1437)
    kwargs_60154 = {}
    # Getting the type of 'cbook' (line 1437)
    cbook_60151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1437, 17), 'cbook', False)
    # Obtaining the member 'Locked' of a type (line 1437)
    Locked_60152 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1437, 17), cbook_60151, 'Locked')
    # Calling Locked(args, kwargs) (line 1437)
    Locked_call_result_60155 = invoke(stypy.reporting.localization.Localization(__file__, 1437, 17), Locked_60152, *[cachedir_60153], **kwargs_60154)
    
    with_60156 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 1437, 17), Locked_call_result_60155, 'with parameter', '__enter__', '__exit__')

    if with_60156:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 1437)
        enter___60157 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1437, 17), Locked_call_result_60155, '__enter__')
        with_enter_60158 = invoke(stypy.reporting.localization.Localization(__file__, 1437, 17), enter___60157)
        
        # Call to json_dump(...): (line 1438)
        # Processing the call arguments (line 1438)
        # Getting the type of 'fontManager' (line 1438)
        fontManager_60160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1438, 26), 'fontManager', False)
        # Getting the type of '_fmcache' (line 1438)
        _fmcache_60161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1438, 39), '_fmcache', False)
        # Processing the call keyword arguments (line 1438)
        kwargs_60162 = {}
        # Getting the type of 'json_dump' (line 1438)
        json_dump_60159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1438, 16), 'json_dump', False)
        # Calling json_dump(args, kwargs) (line 1438)
        json_dump_call_result_60163 = invoke(stypy.reporting.localization.Localization(__file__, 1438, 16), json_dump_60159, *[fontManager_60160, _fmcache_60161], **kwargs_60162)
        
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 1437)
        exit___60164 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1437, 17), Locked_call_result_60155, '__exit__')
        with_exit_60165 = invoke(stypy.reporting.localization.Localization(__file__, 1437, 17), exit___60164, None, None, None)

    # SSA join for if statement (line 1436)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to report(...): (line 1440)
    # Processing the call arguments (line 1440)
    unicode_60168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1440, 23), 'unicode', u'generated new fontManager')
    # Processing the call keyword arguments (line 1440)
    kwargs_60169 = {}
    # Getting the type of 'verbose' (line 1440)
    verbose_60166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1440, 8), 'verbose', False)
    # Obtaining the member 'report' of a type (line 1440)
    report_60167 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1440, 8), verbose_60166, 'report')
    # Calling report(args, kwargs) (line 1440)
    report_call_result_60170 = invoke(stypy.reporting.localization.Localization(__file__, 1440, 8), report_60167, *[unicode_60168], **kwargs_60169)
    
    
    # ################# End of '_rebuild(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_rebuild' in the type store
    # Getting the type of 'stypy_return_type' (line 1431)
    stypy_return_type_60171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1431, 4), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_60171)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_rebuild'
    return stypy_return_type_60171

# Assigning a type to the variable '_rebuild' (line 1431)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1431, 4), '_rebuild', _rebuild)

# Getting the type of '_fmcache' (line 1442)
_fmcache_60172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1442, 7), '_fmcache')
# Testing the type of an if condition (line 1442)
if_condition_60173 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1442, 4), _fmcache_60172)
# Assigning a type to the variable 'if_condition_60173' (line 1442)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1442, 4), 'if_condition_60173', if_condition_60173)
# SSA begins for if statement (line 1442)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')


# SSA begins for try-except statement (line 1443)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')

# Assigning a Call to a Name (line 1444):

# Assigning a Call to a Name (line 1444):

# Call to json_load(...): (line 1444)
# Processing the call arguments (line 1444)
# Getting the type of '_fmcache' (line 1444)
_fmcache_60175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1444, 36), '_fmcache', False)
# Processing the call keyword arguments (line 1444)
kwargs_60176 = {}
# Getting the type of 'json_load' (line 1444)
json_load_60174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1444, 26), 'json_load', False)
# Calling json_load(args, kwargs) (line 1444)
json_load_call_result_60177 = invoke(stypy.reporting.localization.Localization(__file__, 1444, 26), json_load_60174, *[_fmcache_60175], **kwargs_60176)

# Assigning a type to the variable 'fontManager' (line 1444)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1444, 12), 'fontManager', json_load_call_result_60177)


# Evaluating a boolean operation


# Call to hasattr(...): (line 1445)
# Processing the call arguments (line 1445)
# Getting the type of 'fontManager' (line 1445)
fontManager_60179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1445, 28), 'fontManager', False)
unicode_60180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1445, 41), 'unicode', u'_version')
# Processing the call keyword arguments (line 1445)
kwargs_60181 = {}
# Getting the type of 'hasattr' (line 1445)
hasattr_60178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1445, 20), 'hasattr', False)
# Calling hasattr(args, kwargs) (line 1445)
hasattr_call_result_60182 = invoke(stypy.reporting.localization.Localization(__file__, 1445, 20), hasattr_60178, *[fontManager_60179, unicode_60180], **kwargs_60181)

# Applying the 'not' unary operator (line 1445)
result_not__60183 = python_operator(stypy.reporting.localization.Localization(__file__, 1445, 16), 'not', hasattr_call_result_60182)


# Getting the type of 'fontManager' (line 1446)
fontManager_60184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1446, 16), 'fontManager')
# Obtaining the member '_version' of a type (line 1446)
_version_60185 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1446, 16), fontManager_60184, '_version')
# Getting the type of 'FontManager' (line 1446)
FontManager_60186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1446, 40), 'FontManager')
# Obtaining the member '__version__' of a type (line 1446)
version___60187 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1446, 40), FontManager_60186, '__version__')
# Applying the binary operator '!=' (line 1446)
result_ne_60188 = python_operator(stypy.reporting.localization.Localization(__file__, 1446, 16), '!=', _version_60185, version___60187)

# Applying the binary operator 'or' (line 1445)
result_or_keyword_60189 = python_operator(stypy.reporting.localization.Localization(__file__, 1445, 16), 'or', result_not__60183, result_ne_60188)

# Testing the type of an if condition (line 1445)
if_condition_60190 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1445, 12), result_or_keyword_60189)
# Assigning a type to the variable 'if_condition_60190' (line 1445)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1445, 12), 'if_condition_60190', if_condition_60190)
# SSA begins for if statement (line 1445)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Call to _rebuild(...): (line 1447)
# Processing the call keyword arguments (line 1447)
kwargs_60192 = {}
# Getting the type of '_rebuild' (line 1447)
_rebuild_60191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1447, 16), '_rebuild', False)
# Calling _rebuild(args, kwargs) (line 1447)
_rebuild_call_result_60193 = invoke(stypy.reporting.localization.Localization(__file__, 1447, 16), _rebuild_60191, *[], **kwargs_60192)

# SSA branch for the else part of an if statement (line 1445)
module_type_store.open_ssa_branch('else')

# Assigning a Name to a Attribute (line 1449):

# Assigning a Name to a Attribute (line 1449):
# Getting the type of 'None' (line 1449)
None_60194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1449, 43), 'None')
# Getting the type of 'fontManager' (line 1449)
fontManager_60195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1449, 16), 'fontManager')
# Setting the type of the member 'default_size' of a type (line 1449)
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1449, 16), fontManager_60195, 'default_size', None_60194)

# Call to report(...): (line 1450)
# Processing the call arguments (line 1450)
unicode_60198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1450, 31), 'unicode', u'Using fontManager instance from %s')
# Getting the type of '_fmcache' (line 1450)
_fmcache_60199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1450, 70), '_fmcache', False)
# Applying the binary operator '%' (line 1450)
result_mod_60200 = python_operator(stypy.reporting.localization.Localization(__file__, 1450, 31), '%', unicode_60198, _fmcache_60199)

# Processing the call keyword arguments (line 1450)
kwargs_60201 = {}
# Getting the type of 'verbose' (line 1450)
verbose_60196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1450, 16), 'verbose', False)
# Obtaining the member 'report' of a type (line 1450)
report_60197 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1450, 16), verbose_60196, 'report')
# Calling report(args, kwargs) (line 1450)
report_call_result_60202 = invoke(stypy.reporting.localization.Localization(__file__, 1450, 16), report_60197, *[result_mod_60200], **kwargs_60201)

# SSA join for if statement (line 1445)
module_type_store = module_type_store.join_ssa_context()

# SSA branch for the except part of a try statement (line 1443)
# SSA branch for the except 'Attribute' branch of a try statement (line 1443)
module_type_store.open_ssa_branch('except')
# SSA branch for the except '<any exception>' branch of a try statement (line 1443)
module_type_store.open_ssa_branch('except')

# Call to _rebuild(...): (line 1454)
# Processing the call keyword arguments (line 1454)
kwargs_60204 = {}
# Getting the type of '_rebuild' (line 1454)
_rebuild_60203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1454, 12), '_rebuild', False)
# Calling _rebuild(args, kwargs) (line 1454)
_rebuild_call_result_60205 = invoke(stypy.reporting.localization.Localization(__file__, 1454, 12), _rebuild_60203, *[], **kwargs_60204)

# SSA join for try-except statement (line 1443)
module_type_store = module_type_store.join_ssa_context()

# SSA branch for the else part of an if statement (line 1442)
module_type_store.open_ssa_branch('else')

# Call to _rebuild(...): (line 1456)
# Processing the call keyword arguments (line 1456)
kwargs_60207 = {}
# Getting the type of '_rebuild' (line 1456)
_rebuild_60206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1456, 8), '_rebuild', False)
# Calling _rebuild(args, kwargs) (line 1456)
_rebuild_call_result_60208 = invoke(stypy.reporting.localization.Localization(__file__, 1456, 8), _rebuild_60206, *[], **kwargs_60207)

# SSA join for if statement (line 1442)
module_type_store = module_type_store.join_ssa_context()


@norecursion
def findfont(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'findfont'
    module_type_store = module_type_store.open_function_context('findfont', 1458, 4, False)
    
    # Passed parameters checking function
    findfont.stypy_localization = localization
    findfont.stypy_type_of_self = None
    findfont.stypy_type_store = module_type_store
    findfont.stypy_function_name = 'findfont'
    findfont.stypy_param_names_list = ['prop']
    findfont.stypy_varargs_param_name = None
    findfont.stypy_kwargs_param_name = 'kw'
    findfont.stypy_call_defaults = defaults
    findfont.stypy_call_varargs = varargs
    findfont.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'findfont', ['prop'], None, 'kw', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'findfont', localization, ['prop'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'findfont(...)' code ##################

    # Marking variables as global (line 1459)
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 1459, 8), 'fontManager')
    
    # Assigning a Call to a Name (line 1460):
    
    # Assigning a Call to a Name (line 1460):
    
    # Call to findfont(...): (line 1460)
    # Processing the call arguments (line 1460)
    # Getting the type of 'prop' (line 1460)
    prop_60211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1460, 36), 'prop', False)
    # Processing the call keyword arguments (line 1460)
    # Getting the type of 'kw' (line 1460)
    kw_60212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1460, 44), 'kw', False)
    kwargs_60213 = {'kw_60212': kw_60212}
    # Getting the type of 'fontManager' (line 1460)
    fontManager_60209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1460, 15), 'fontManager', False)
    # Obtaining the member 'findfont' of a type (line 1460)
    findfont_60210 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1460, 15), fontManager_60209, 'findfont')
    # Calling findfont(args, kwargs) (line 1460)
    findfont_call_result_60214 = invoke(stypy.reporting.localization.Localization(__file__, 1460, 15), findfont_60210, *[prop_60211], **kwargs_60213)
    
    # Assigning a type to the variable 'font' (line 1460)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1460, 8), 'font', findfont_call_result_60214)
    # Getting the type of 'font' (line 1461)
    font_60215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1461, 15), 'font')
    # Assigning a type to the variable 'stypy_return_type' (line 1461)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1461, 8), 'stypy_return_type', font_60215)
    
    # ################# End of 'findfont(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'findfont' in the type store
    # Getting the type of 'stypy_return_type' (line 1458)
    stypy_return_type_60216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1458, 4), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_60216)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'findfont'
    return stypy_return_type_60216

# Assigning a type to the variable 'findfont' (line 1458)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1458, 4), 'findfont', findfont)
# SSA join for if statement (line 1373)
module_type_store = module_type_store.join_ssa_context()


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
