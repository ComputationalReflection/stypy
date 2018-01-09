
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: This module supports embedded TeX expressions in matplotlib via dvipng
3: and dvips for the raster and postscript backends.  The tex and
4: dvipng/dvips information is cached in ~/.matplotlib/tex.cache for reuse between
5: sessions
6: 
7: Requirements:
8: 
9: * latex
10: * \\*Agg backends: dvipng>=1.6
11: * PS backend: psfrag, dvips, and Ghostscript>=8.60
12: 
13: Backends:
14: 
15: * \\*Agg
16: * PS
17: * PDF
18: 
19: For raster output, you can get RGBA numpy arrays from TeX expressions
20: as follows::
21: 
22:   texmanager = TexManager()
23:   s = ('\\TeX\\ is Number '
24:        '$\\displaystyle\\sum_{n=1}^\\infty\\frac{-e^{i\\pi}}{2^n}$!')
25:   Z = texmanager.get_rgba(s, fontsize=12, dpi=80, rgb=(1,0,0))
26: 
27: To enable tex rendering of all text in your matplotlib figure, set
28: text.usetex in your matplotlibrc file or include these two lines in
29: your script::
30: 
31:   from matplotlib import rc
32:   rc('text', usetex=True)
33: 
34: '''
35: 
36: from __future__ import (absolute_import, division, print_function,
37:                         unicode_literals)
38: 
39: import six
40: 
41: import copy
42: import glob
43: import os
44: import shutil
45: import sys
46: import warnings
47: 
48: from hashlib import md5
49: 
50: import distutils.version
51: import numpy as np
52: import matplotlib as mpl
53: from matplotlib import rcParams
54: from matplotlib._png import read_png
55: from matplotlib.cbook import mkdirs, Locked
56: from matplotlib.compat.subprocess import subprocess, Popen, PIPE, STDOUT
57: import matplotlib.dviread as dviread
58: import re
59: 
60: DEBUG = False
61: 
62: if sys.platform.startswith('win'):
63:     cmd_split = '&'
64: else:
65:     cmd_split = ';'
66: 
67: 
68: @mpl.cbook.deprecated("2.1")
69: def dvipng_hack_alpha():
70:     try:
71:         p = Popen([str('dvipng'), '-version'], stdin=PIPE, stdout=PIPE,
72:                   stderr=STDOUT, close_fds=(sys.platform != 'win32'))
73:         stdout, stderr = p.communicate()
74:     except OSError:
75:         mpl.verbose.report('No dvipng was found', 'helpful')
76:         return False
77:     lines = stdout.decode(sys.getdefaultencoding()).split('\n')
78:     for line in lines:
79:         if line.startswith('dvipng '):
80:             version = line.split()[-1]
81:             mpl.verbose.report('Found dvipng version %s' % version,
82:                                'helpful')
83:             version = distutils.version.LooseVersion(version)
84:             return version < distutils.version.LooseVersion('1.6')
85:     mpl.verbose.report('Unexpected response from dvipng -version', 'helpful')
86:     return False
87: 
88: 
89: class TexManager(object):
90:     '''
91:     Convert strings to dvi files using TeX, caching the results to a
92:     working dir
93:     '''
94: 
95:     oldpath = mpl.get_home()
96:     if oldpath is None:
97:         oldpath = mpl.get_data_path()
98:     oldcache = os.path.join(oldpath, '.tex.cache')
99: 
100:     cachedir = mpl.get_cachedir()
101:     if cachedir is not None:
102:         texcache = os.path.join(cachedir, 'tex.cache')
103:     else:
104:         # Should only happen in a restricted environment (such as Google App
105:         # Engine). Deal with this gracefully by not creating a cache directory.
106:         texcache = None
107: 
108:     if os.path.exists(oldcache):
109:         if texcache is not None:
110:             try:
111:                 shutil.move(oldcache, texcache)
112:             except IOError as e:
113:                 warnings.warn('File could not be renamed: %s' % e)
114:             else:
115:                 warnings.warn('''\
116: Found a TeX cache dir in the deprecated location "%s".
117:     Moving it to the new default location "%s".''' % (oldcache, texcache))
118:         else:
119:             warnings.warn('''\
120: Could not rename old TeX cache dir "%s": a suitable configuration
121:     directory could not be found.''' % oldcache)
122: 
123:     if texcache is not None:
124:         mkdirs(texcache)
125: 
126:     # mappable cache of
127:     rgba_arrayd = {}
128:     grey_arrayd = {}
129:     postscriptd = {}
130:     pscnt = 0
131: 
132:     serif = ('cmr', '')
133:     sans_serif = ('cmss', '')
134:     monospace = ('cmtt', '')
135:     cursive = ('pzc', '\\usepackage{chancery}')
136:     font_family = 'serif'
137:     font_families = ('serif', 'sans-serif', 'cursive', 'monospace')
138: 
139:     font_info = {'new century schoolbook': ('pnc',
140:                                             r'\renewcommand{\rmdefault}{pnc}'),
141:                  'bookman': ('pbk', r'\renewcommand{\rmdefault}{pbk}'),
142:                  'times': ('ptm', '\\usepackage{mathptmx}'),
143:                  'palatino': ('ppl', '\\usepackage{mathpazo}'),
144:                  'zapf chancery': ('pzc', '\\usepackage{chancery}'),
145:                  'cursive': ('pzc', '\\usepackage{chancery}'),
146:                  'charter': ('pch', '\\usepackage{charter}'),
147:                  'serif': ('cmr', ''),
148:                  'sans-serif': ('cmss', ''),
149:                  'helvetica': ('phv', '\\usepackage{helvet}'),
150:                  'avant garde': ('pag', '\\usepackage{avant}'),
151:                  'courier': ('pcr', '\\usepackage{courier}'),
152:                  'monospace': ('cmtt', ''),
153:                  'computer modern roman': ('cmr', ''),
154:                  'computer modern sans serif': ('cmss', ''),
155:                  'computer modern typewriter': ('cmtt', '')}
156: 
157:     _rc_cache = None
158:     _rc_cache_keys = (('text.latex.preamble', ) +
159:                       tuple(['font.' + n for n in ('family', ) +
160:                              font_families]))
161: 
162:     def __init__(self):
163: 
164:         if self.texcache is None:
165:             raise RuntimeError(
166:                 ('Cannot create TexManager, as there is no cache directory '
167:                  'available'))
168: 
169:         mkdirs(self.texcache)
170:         ff = rcParams['font.family']
171:         if len(ff) == 1 and ff[0].lower() in self.font_families:
172:             self.font_family = ff[0].lower()
173:         elif isinstance(ff, six.string_types) and ff.lower() in self.font_families:
174:             self.font_family = ff.lower()
175:         else:
176:             mpl.verbose.report(
177:                 'font.family must be one of (%s) when text.usetex is True. '
178:                 'serif will be used by default.' %
179:                    ', '.join(self.font_families),
180:                 'helpful')
181:             self.font_family = 'serif'
182: 
183:         fontconfig = [self.font_family]
184:         for font_family, font_family_attr in [(ff, ff.replace('-', '_'))
185:                                               for ff in self.font_families]:
186:             for font in rcParams['font.' + font_family]:
187:                 if font.lower() in self.font_info:
188:                     setattr(self, font_family_attr,
189:                             self.font_info[font.lower()])
190:                     if DEBUG:
191:                         print('family: %s, font: %s, info: %s' %
192:                               (font_family, font,
193:                                self.font_info[font.lower()]))
194:                     break
195:                 else:
196:                     if DEBUG:
197:                         print('$s font is not compatible with usetex')
198:             else:
199:                 mpl.verbose.report('No LaTeX-compatible font found for the '
200:                                    '%s font family in rcParams. Using '
201:                                    'default.' % font_family, 'helpful')
202:                 setattr(self, font_family_attr, self.font_info[font_family])
203:             fontconfig.append(getattr(self, font_family_attr)[0])
204:         # Add a hash of the latex preamble to self._fontconfig so that the
205:         # correct png is selected for strings rendered with same font and dpi
206:         # even if the latex preamble changes within the session
207:         preamble_bytes = six.text_type(self.get_custom_preamble()).encode('utf-8')
208:         fontconfig.append(md5(preamble_bytes).hexdigest())
209:         self._fontconfig = ''.join(fontconfig)
210: 
211:         # The following packages and commands need to be included in the latex
212:         # file's preamble:
213:         cmd = [self.serif[1], self.sans_serif[1], self.monospace[1]]
214:         if self.font_family == 'cursive':
215:             cmd.append(self.cursive[1])
216:         while '\\usepackage{type1cm}' in cmd:
217:             cmd.remove('\\usepackage{type1cm}')
218:         cmd = '\n'.join(cmd)
219:         self._font_preamble = '\n'.join(['\\usepackage{type1cm}', cmd,
220:                                          '\\usepackage{textcomp}'])
221: 
222:     def get_basefile(self, tex, fontsize, dpi=None):
223:         '''
224:         returns a filename based on a hash of the string, fontsize, and dpi
225:         '''
226:         s = ''.join([tex, self.get_font_config(), '%f' % fontsize,
227:                      self.get_custom_preamble(), str(dpi or '')])
228:         # make sure hash is consistent for all strings, regardless of encoding:
229:         bytes = six.text_type(s).encode('utf-8')
230:         return os.path.join(self.texcache, md5(bytes).hexdigest())
231: 
232:     def get_font_config(self):
233:         '''Reinitializes self if relevant rcParams on have changed.'''
234:         if self._rc_cache is None:
235:             self._rc_cache = dict.fromkeys(self._rc_cache_keys)
236:         changed = [par for par in self._rc_cache_keys
237:                    if rcParams[par] != self._rc_cache[par]]
238:         if changed:
239:             if DEBUG:
240:                 print('DEBUG following keys changed:', changed)
241:             for k in changed:
242:                 if DEBUG:
243:                     print('DEBUG %-20s: %-10s -> %-10s' %
244:                           (k, self._rc_cache[k], rcParams[k]))
245:                 # deepcopy may not be necessary, but feels more future-proof
246:                 self._rc_cache[k] = copy.deepcopy(rcParams[k])
247:             if DEBUG:
248:                 print('DEBUG RE-INIT\nold fontconfig:', self._fontconfig)
249:             self.__init__()
250:         if DEBUG:
251:             print('DEBUG fontconfig:', self._fontconfig)
252:         return self._fontconfig
253: 
254:     def get_font_preamble(self):
255:         '''
256:         returns a string containing font configuration for the tex preamble
257:         '''
258:         return self._font_preamble
259: 
260:     def get_custom_preamble(self):
261:         '''returns a string containing user additions to the tex preamble'''
262:         return '\n'.join(rcParams['text.latex.preamble'])
263: 
264:     def make_tex(self, tex, fontsize):
265:         '''
266:         Generate a tex file to render the tex string at a specific font size
267: 
268:         returns the file name
269:         '''
270:         basefile = self.get_basefile(tex, fontsize)
271:         texfile = '%s.tex' % basefile
272:         custom_preamble = self.get_custom_preamble()
273:         fontcmd = {'sans-serif': r'{\sffamily %s}',
274:                    'monospace': r'{\ttfamily %s}'}.get(self.font_family,
275:                                                        r'{\rmfamily %s}')
276:         tex = fontcmd % tex
277: 
278:         if rcParams['text.latex.unicode']:
279:             unicode_preamble = '''\\usepackage{ucs}
280: \\usepackage[utf8x]{inputenc}'''
281:         else:
282:             unicode_preamble = ''
283: 
284:         s = '''\\documentclass{article}
285: %s
286: %s
287: %s
288: \\usepackage[papersize={72in,72in},body={70in,70in},margin={1in,1in}]{geometry}
289: \\pagestyle{empty}
290: \\begin{document}
291: \\fontsize{%f}{%f}%s
292: \\end{document}
293: ''' % (self._font_preamble, unicode_preamble, custom_preamble,
294:        fontsize, fontsize * 1.25, tex)
295:         with open(texfile, 'wb') as fh:
296:             if rcParams['text.latex.unicode']:
297:                 fh.write(s.encode('utf8'))
298:             else:
299:                 try:
300:                     fh.write(s.encode('ascii'))
301:                 except UnicodeEncodeError as err:
302:                     mpl.verbose.report("You are using unicode and latex, but "
303:                                        "have not enabled the matplotlib "
304:                                        "'text.latex.unicode' rcParam.",
305:                                        'helpful')
306:                     raise
307: 
308:         return texfile
309: 
310:     _re_vbox = re.compile(
311:         r"MatplotlibBox:\(([\d.]+)pt\+([\d.]+)pt\)x([\d.]+)pt")
312: 
313:     def make_tex_preview(self, tex, fontsize):
314:         '''
315:         Generate a tex file to render the tex string at a specific
316:         font size. It uses the preview.sty to determine the dimension
317:         (width, height, descent) of the output.
318: 
319:         returns the file name
320:         '''
321:         basefile = self.get_basefile(tex, fontsize)
322:         texfile = '%s.tex' % basefile
323:         custom_preamble = self.get_custom_preamble()
324:         fontcmd = {'sans-serif': r'{\sffamily %s}',
325:                    'monospace': r'{\ttfamily %s}'}.get(self.font_family,
326:                                                        r'{\rmfamily %s}')
327:         tex = fontcmd % tex
328: 
329:         if rcParams['text.latex.unicode']:
330:             unicode_preamble = '''\\usepackage{ucs}
331: \\usepackage[utf8x]{inputenc}'''
332:         else:
333:             unicode_preamble = ''
334: 
335:         # newbox, setbox, immediate, etc. are used to find the box
336:         # extent of the rendered text.
337: 
338:         s = '''\\documentclass{article}
339: %s
340: %s
341: %s
342: \\usepackage[active,showbox,tightpage]{preview}
343: \\usepackage[papersize={72in,72in},body={70in,70in},margin={1in,1in}]{geometry}
344: 
345: %% we override the default showbox as it is treated as an error and makes
346: %% the exit status not zero
347: \\def\\showbox#1{\\immediate\\write16{MatplotlibBox:(\\the\\ht#1+\\the\\dp#1)x\\the\\wd#1}}
348: 
349: \\begin{document}
350: \\begin{preview}
351: {\\fontsize{%f}{%f}%s}
352: \\end{preview}
353: \\end{document}
354: ''' % (self._font_preamble, unicode_preamble, custom_preamble,
355:        fontsize, fontsize * 1.25, tex)
356:         with open(texfile, 'wb') as fh:
357:             if rcParams['text.latex.unicode']:
358:                 fh.write(s.encode('utf8'))
359:             else:
360:                 try:
361:                     fh.write(s.encode('ascii'))
362:                 except UnicodeEncodeError as err:
363:                     mpl.verbose.report("You are using unicode and latex, but "
364:                                        "have not enabled the matplotlib "
365:                                        "'text.latex.unicode' rcParam.",
366:                                        'helpful')
367:                     raise
368: 
369:         return texfile
370: 
371:     def make_dvi(self, tex, fontsize):
372:         '''
373:         generates a dvi file containing latex's layout of tex string
374: 
375:         returns the file name
376:         '''
377: 
378:         if rcParams['text.latex.preview']:
379:             return self.make_dvi_preview(tex, fontsize)
380: 
381:         basefile = self.get_basefile(tex, fontsize)
382:         dvifile = '%s.dvi' % basefile
383: 
384:         if DEBUG or not os.path.exists(dvifile):
385:             texfile = self.make_tex(tex, fontsize)
386:             command = [str("latex"), "-interaction=nonstopmode",
387:                        os.path.basename(texfile)]
388:             mpl.verbose.report(command, 'debug')
389:             with Locked(self.texcache):
390:                 try:
391:                     report = subprocess.check_output(command,
392:                                                      cwd=self.texcache,
393:                                                      stderr=subprocess.STDOUT)
394:                 except subprocess.CalledProcessError as exc:
395:                     raise RuntimeError(
396:                         ('LaTeX was not able to process the following '
397:                          'string:\n%s\n\n'
398:                          'Here is the full report generated by LaTeX:\n%s '
399:                          '\n\n' % (repr(tex.encode('unicode_escape')),
400:                                    exc.output.decode("utf-8"))))
401:                 mpl.verbose.report(report, 'debug')
402:             for fname in glob.glob(basefile + '*'):
403:                 if fname.endswith('dvi'):
404:                     pass
405:                 elif fname.endswith('tex'):
406:                     pass
407:                 else:
408:                     try:
409:                         os.remove(fname)
410:                     except OSError:
411:                         pass
412: 
413:         return dvifile
414: 
415:     def make_dvi_preview(self, tex, fontsize):
416:         '''
417:         generates a dvi file containing latex's layout of tex
418:         string. It calls make_tex_preview() method and store the size
419:         information (width, height, descent) in a separte file.
420: 
421:         returns the file name
422:         '''
423:         basefile = self.get_basefile(tex, fontsize)
424:         dvifile = '%s.dvi' % basefile
425:         baselinefile = '%s.baseline' % basefile
426: 
427:         if (DEBUG or not os.path.exists(dvifile) or
428:                 not os.path.exists(baselinefile)):
429:             texfile = self.make_tex_preview(tex, fontsize)
430:             command = [str("latex"), "-interaction=nonstopmode",
431:                        os.path.basename(texfile)]
432:             mpl.verbose.report(command, 'debug')
433:             try:
434:                 report = subprocess.check_output(command,
435:                                                  cwd=self.texcache,
436:                                                  stderr=subprocess.STDOUT)
437:             except subprocess.CalledProcessError as exc:
438:                 raise RuntimeError(
439:                     ('LaTeX was not able to process the following '
440:                      'string:\n%s\n\n'
441:                      'Here is the full report generated by LaTeX:\n%s '
442:                      '\n\n' % (repr(tex.encode('unicode_escape')),
443:                                exc.output.decode("utf-8"))))
444:             mpl.verbose.report(report, 'debug')
445: 
446:             # find the box extent information in the latex output
447:             # file and store them in ".baseline" file
448:             m = TexManager._re_vbox.search(report.decode("utf-8"))
449:             with open(basefile + '.baseline', "w") as fh:
450:                 fh.write(" ".join(m.groups()))
451: 
452:             for fname in glob.glob(basefile + '*'):
453:                 if fname.endswith('dvi'):
454:                     pass
455:                 elif fname.endswith('tex'):
456:                     pass
457:                 elif fname.endswith('baseline'):
458:                     pass
459:                 else:
460:                     try:
461:                         os.remove(fname)
462:                     except OSError:
463:                         pass
464: 
465:         return dvifile
466: 
467:     def make_png(self, tex, fontsize, dpi):
468:         '''
469:         generates a png file containing latex's rendering of tex string
470: 
471:         returns the filename
472:         '''
473:         basefile = self.get_basefile(tex, fontsize, dpi)
474:         pngfile = '%s.png' % basefile
475: 
476:         # see get_rgba for a discussion of the background
477:         if DEBUG or not os.path.exists(pngfile):
478:             dvifile = self.make_dvi(tex, fontsize)
479:             command = [str("dvipng"), "-bg", "Transparent", "-D", str(dpi),
480:                        "-T", "tight", "-o", os.path.basename(pngfile),
481:                        os.path.basename(dvifile)]
482:             mpl.verbose.report(command, 'debug')
483:             try:
484:                 report = subprocess.check_output(command,
485:                                                  cwd=self.texcache,
486:                                                  stderr=subprocess.STDOUT)
487:             except subprocess.CalledProcessError as exc:
488:                 raise RuntimeError(
489:                     ('dvipng was not able to process the following '
490:                      'string:\n%s\n\n'
491:                      'Here is the full report generated by dvipng:\n%s '
492:                      '\n\n' % (repr(tex.encode('unicode_escape')),
493:                                exc.output.decode("utf-8"))))
494:             mpl.verbose.report(report, 'debug')
495: 
496:         return pngfile
497: 
498:     def make_ps(self, tex, fontsize):
499:         '''
500:         generates a postscript file containing latex's rendering of tex string
501: 
502:         returns the file name
503:         '''
504:         basefile = self.get_basefile(tex, fontsize)
505:         psfile = '%s.epsf' % basefile
506: 
507:         if DEBUG or not os.path.exists(psfile):
508:             dvifile = self.make_dvi(tex, fontsize)
509:             command = [str("dvips"), "-q", "-E", "-o",
510:                        os.path.basename(psfile),
511:                        os.path.basename(dvifile)]
512:             mpl.verbose.report(command, 'debug')
513:             try:
514:                 report = subprocess.check_output(command,
515:                                                  cwd=self.texcache,
516:                                                  stderr=subprocess.STDOUT)
517:             except subprocess.CalledProcessError as exc:
518:                 raise RuntimeError(
519:                     ('dvips was not able to process the following '
520:                      'string:\n%s\n\n'
521:                      'Here is the full report generated by dvips:\n%s '
522:                      '\n\n' % (repr(tex.encode('unicode_escape')),
523:                                exc.output.decode("utf-8"))))
524:             mpl.verbose.report(report, 'debug')
525: 
526:         return psfile
527: 
528:     def get_ps_bbox(self, tex, fontsize):
529:         '''
530:         returns a list containing the postscript bounding box for latex's
531:         rendering of the tex string
532:         '''
533:         psfile = self.make_ps(tex, fontsize)
534:         with open(psfile) as ps:
535:             for line in ps:
536:                 if line.startswith('%%BoundingBox:'):
537:                     return [int(val) for val in line.split()[1:]]
538:         raise RuntimeError('Could not parse %s' % psfile)
539: 
540:     def get_grey(self, tex, fontsize=None, dpi=None):
541:         '''returns the alpha channel'''
542:         key = tex, self.get_font_config(), fontsize, dpi
543:         alpha = self.grey_arrayd.get(key)
544:         if alpha is None:
545:             pngfile = self.make_png(tex, fontsize, dpi)
546:             X = read_png(os.path.join(self.texcache, pngfile))
547:             self.grey_arrayd[key] = alpha = X[:, :, -1]
548:         return alpha
549: 
550:     def get_rgba(self, tex, fontsize=None, dpi=None, rgb=(0, 0, 0)):
551:         '''
552:         Returns latex's rendering of the tex string as an rgba array
553:         '''
554:         if not fontsize:
555:             fontsize = rcParams['font.size']
556:         if not dpi:
557:             dpi = rcParams['savefig.dpi']
558:         r, g, b = rgb
559:         key = tex, self.get_font_config(), fontsize, dpi, tuple(rgb)
560:         Z = self.rgba_arrayd.get(key)
561: 
562:         if Z is None:
563:             alpha = self.get_grey(tex, fontsize, dpi)
564: 
565:             Z = np.zeros((alpha.shape[0], alpha.shape[1], 4), float)
566: 
567:             Z[:, :, 0] = r
568:             Z[:, :, 1] = g
569:             Z[:, :, 2] = b
570:             Z[:, :, 3] = alpha
571:             self.rgba_arrayd[key] = Z
572: 
573:         return Z
574: 
575:     def get_text_width_height_descent(self, tex, fontsize, renderer=None):
576:         '''
577:         return width, heigth and descent of the text.
578:         '''
579:         if tex.strip() == '':
580:             return 0, 0, 0
581: 
582:         if renderer:
583:             dpi_fraction = renderer.points_to_pixels(1.)
584:         else:
585:             dpi_fraction = 1.
586: 
587:         if rcParams['text.latex.preview']:
588:             # use preview.sty
589:             basefile = self.get_basefile(tex, fontsize)
590:             baselinefile = '%s.baseline' % basefile
591: 
592:             if DEBUG or not os.path.exists(baselinefile):
593:                 dvifile = self.make_dvi_preview(tex, fontsize)
594: 
595:             with open(baselinefile) as fh:
596:                 l = fh.read().split()
597:             height, depth, width = [float(l1) * dpi_fraction for l1 in l]
598:             return width, height + depth, depth
599: 
600:         else:
601:             # use dviread. It sometimes returns a wrong descent.
602:             dvifile = self.make_dvi(tex, fontsize)
603:             with dviread.Dvi(dvifile, 72 * dpi_fraction) as dvi:
604:                 page = next(iter(dvi))
605:             # A total height (including the descent) needs to be returned.
606:             return page.width, page.height + page.descent, page.descent
607: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

unicode_137206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, (-1)), 'unicode', u"\nThis module supports embedded TeX expressions in matplotlib via dvipng\nand dvips for the raster and postscript backends.  The tex and\ndvipng/dvips information is cached in ~/.matplotlib/tex.cache for reuse between\nsessions\n\nRequirements:\n\n* latex\n* \\*Agg backends: dvipng>=1.6\n* PS backend: psfrag, dvips, and Ghostscript>=8.60\n\nBackends:\n\n* \\*Agg\n* PS\n* PDF\n\nFor raster output, you can get RGBA numpy arrays from TeX expressions\nas follows::\n\n  texmanager = TexManager()\n  s = ('\\TeX\\ is Number '\n       '$\\displaystyle\\sum_{n=1}^\\infty\\frac{-e^{i\\pi}}{2^n}$!')\n  Z = texmanager.get_rgba(s, fontsize=12, dpi=80, rgb=(1,0,0))\n\nTo enable tex rendering of all text in your matplotlib figure, set\ntext.usetex in your matplotlibrc file or include these two lines in\nyour script::\n\n  from matplotlib import rc\n  rc('text', usetex=True)\n\n")
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 39, 0))

# 'import six' statement (line 39)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_137207 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 39, 0), 'six')

if (type(import_137207) is not StypyTypeError):

    if (import_137207 != 'pyd_module'):
        __import__(import_137207)
        sys_modules_137208 = sys.modules[import_137207]
        import_module(stypy.reporting.localization.Localization(__file__, 39, 0), 'six', sys_modules_137208.module_type_store, module_type_store)
    else:
        import six

        import_module(stypy.reporting.localization.Localization(__file__, 39, 0), 'six', six, module_type_store)

else:
    # Assigning a type to the variable 'six' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 0), 'six', import_137207)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 41, 0))

# 'import copy' statement (line 41)
import copy

import_module(stypy.reporting.localization.Localization(__file__, 41, 0), 'copy', copy, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 42, 0))

# 'import glob' statement (line 42)
import glob

import_module(stypy.reporting.localization.Localization(__file__, 42, 0), 'glob', glob, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 43, 0))

# 'import os' statement (line 43)
import os

import_module(stypy.reporting.localization.Localization(__file__, 43, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 44, 0))

# 'import shutil' statement (line 44)
import shutil

import_module(stypy.reporting.localization.Localization(__file__, 44, 0), 'shutil', shutil, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 45, 0))

# 'import sys' statement (line 45)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 45, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 46, 0))

# 'import warnings' statement (line 46)
import warnings

import_module(stypy.reporting.localization.Localization(__file__, 46, 0), 'warnings', warnings, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 48, 0))

# 'from hashlib import md5' statement (line 48)
try:
    from hashlib import md5

except:
    md5 = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 48, 0), 'hashlib', None, module_type_store, ['md5'], [md5])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 50, 0))

# 'import distutils.version' statement (line 50)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_137209 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 50, 0), 'distutils.version')

if (type(import_137209) is not StypyTypeError):

    if (import_137209 != 'pyd_module'):
        __import__(import_137209)
        sys_modules_137210 = sys.modules[import_137209]
        import_module(stypy.reporting.localization.Localization(__file__, 50, 0), 'distutils.version', sys_modules_137210.module_type_store, module_type_store)
    else:
        import distutils.version

        import_module(stypy.reporting.localization.Localization(__file__, 50, 0), 'distutils.version', distutils.version, module_type_store)

else:
    # Assigning a type to the variable 'distutils.version' (line 50)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 0), 'distutils.version', import_137209)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 51, 0))

# 'import numpy' statement (line 51)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_137211 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 51, 0), 'numpy')

if (type(import_137211) is not StypyTypeError):

    if (import_137211 != 'pyd_module'):
        __import__(import_137211)
        sys_modules_137212 = sys.modules[import_137211]
        import_module(stypy.reporting.localization.Localization(__file__, 51, 0), 'np', sys_modules_137212.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 51, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 51)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 0), 'numpy', import_137211)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 52, 0))

# 'import matplotlib' statement (line 52)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_137213 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 52, 0), 'matplotlib')

if (type(import_137213) is not StypyTypeError):

    if (import_137213 != 'pyd_module'):
        __import__(import_137213)
        sys_modules_137214 = sys.modules[import_137213]
        import_module(stypy.reporting.localization.Localization(__file__, 52, 0), 'mpl', sys_modules_137214.module_type_store, module_type_store)
    else:
        import matplotlib as mpl

        import_module(stypy.reporting.localization.Localization(__file__, 52, 0), 'mpl', matplotlib, module_type_store)

else:
    # Assigning a type to the variable 'matplotlib' (line 52)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 0), 'matplotlib', import_137213)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 53, 0))

# 'from matplotlib import rcParams' statement (line 53)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_137215 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 53, 0), 'matplotlib')

if (type(import_137215) is not StypyTypeError):

    if (import_137215 != 'pyd_module'):
        __import__(import_137215)
        sys_modules_137216 = sys.modules[import_137215]
        import_from_module(stypy.reporting.localization.Localization(__file__, 53, 0), 'matplotlib', sys_modules_137216.module_type_store, module_type_store, ['rcParams'])
        nest_module(stypy.reporting.localization.Localization(__file__, 53, 0), __file__, sys_modules_137216, sys_modules_137216.module_type_store, module_type_store)
    else:
        from matplotlib import rcParams

        import_from_module(stypy.reporting.localization.Localization(__file__, 53, 0), 'matplotlib', None, module_type_store, ['rcParams'], [rcParams])

else:
    # Assigning a type to the variable 'matplotlib' (line 53)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 0), 'matplotlib', import_137215)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 54, 0))

# 'from matplotlib._png import read_png' statement (line 54)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_137217 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 54, 0), 'matplotlib._png')

if (type(import_137217) is not StypyTypeError):

    if (import_137217 != 'pyd_module'):
        __import__(import_137217)
        sys_modules_137218 = sys.modules[import_137217]
        import_from_module(stypy.reporting.localization.Localization(__file__, 54, 0), 'matplotlib._png', sys_modules_137218.module_type_store, module_type_store, ['read_png'])
        nest_module(stypy.reporting.localization.Localization(__file__, 54, 0), __file__, sys_modules_137218, sys_modules_137218.module_type_store, module_type_store)
    else:
        from matplotlib._png import read_png

        import_from_module(stypy.reporting.localization.Localization(__file__, 54, 0), 'matplotlib._png', None, module_type_store, ['read_png'], [read_png])

else:
    # Assigning a type to the variable 'matplotlib._png' (line 54)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 0), 'matplotlib._png', import_137217)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 55, 0))

# 'from matplotlib.cbook import mkdirs, Locked' statement (line 55)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_137219 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 55, 0), 'matplotlib.cbook')

if (type(import_137219) is not StypyTypeError):

    if (import_137219 != 'pyd_module'):
        __import__(import_137219)
        sys_modules_137220 = sys.modules[import_137219]
        import_from_module(stypy.reporting.localization.Localization(__file__, 55, 0), 'matplotlib.cbook', sys_modules_137220.module_type_store, module_type_store, ['mkdirs', 'Locked'])
        nest_module(stypy.reporting.localization.Localization(__file__, 55, 0), __file__, sys_modules_137220, sys_modules_137220.module_type_store, module_type_store)
    else:
        from matplotlib.cbook import mkdirs, Locked

        import_from_module(stypy.reporting.localization.Localization(__file__, 55, 0), 'matplotlib.cbook', None, module_type_store, ['mkdirs', 'Locked'], [mkdirs, Locked])

else:
    # Assigning a type to the variable 'matplotlib.cbook' (line 55)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 0), 'matplotlib.cbook', import_137219)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 56, 0))

# 'from matplotlib.compat.subprocess import subprocess, Popen, PIPE, STDOUT' statement (line 56)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_137221 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 56, 0), 'matplotlib.compat.subprocess')

if (type(import_137221) is not StypyTypeError):

    if (import_137221 != 'pyd_module'):
        __import__(import_137221)
        sys_modules_137222 = sys.modules[import_137221]
        import_from_module(stypy.reporting.localization.Localization(__file__, 56, 0), 'matplotlib.compat.subprocess', sys_modules_137222.module_type_store, module_type_store, ['subprocess', 'Popen', 'PIPE', 'STDOUT'])
        nest_module(stypy.reporting.localization.Localization(__file__, 56, 0), __file__, sys_modules_137222, sys_modules_137222.module_type_store, module_type_store)
    else:
        from matplotlib.compat.subprocess import subprocess, Popen, PIPE, STDOUT

        import_from_module(stypy.reporting.localization.Localization(__file__, 56, 0), 'matplotlib.compat.subprocess', None, module_type_store, ['subprocess', 'Popen', 'PIPE', 'STDOUT'], [subprocess, Popen, PIPE, STDOUT])

else:
    # Assigning a type to the variable 'matplotlib.compat.subprocess' (line 56)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 0), 'matplotlib.compat.subprocess', import_137221)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 57, 0))

# 'import matplotlib.dviread' statement (line 57)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_137223 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 57, 0), 'matplotlib.dviread')

if (type(import_137223) is not StypyTypeError):

    if (import_137223 != 'pyd_module'):
        __import__(import_137223)
        sys_modules_137224 = sys.modules[import_137223]
        import_module(stypy.reporting.localization.Localization(__file__, 57, 0), 'dviread', sys_modules_137224.module_type_store, module_type_store)
    else:
        import matplotlib.dviread as dviread

        import_module(stypy.reporting.localization.Localization(__file__, 57, 0), 'dviread', matplotlib.dviread, module_type_store)

else:
    # Assigning a type to the variable 'matplotlib.dviread' (line 57)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 0), 'matplotlib.dviread', import_137223)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 58, 0))

# 'import re' statement (line 58)
import re

import_module(stypy.reporting.localization.Localization(__file__, 58, 0), 're', re, module_type_store)


# Assigning a Name to a Name (line 60):

# Assigning a Name to a Name (line 60):
# Getting the type of 'False' (line 60)
False_137225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'False')
# Assigning a type to the variable 'DEBUG' (line 60)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 0), 'DEBUG', False_137225)


# Call to startswith(...): (line 62)
# Processing the call arguments (line 62)
unicode_137229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 27), 'unicode', u'win')
# Processing the call keyword arguments (line 62)
kwargs_137230 = {}
# Getting the type of 'sys' (line 62)
sys_137226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 3), 'sys', False)
# Obtaining the member 'platform' of a type (line 62)
platform_137227 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 3), sys_137226, 'platform')
# Obtaining the member 'startswith' of a type (line 62)
startswith_137228 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 3), platform_137227, 'startswith')
# Calling startswith(args, kwargs) (line 62)
startswith_call_result_137231 = invoke(stypy.reporting.localization.Localization(__file__, 62, 3), startswith_137228, *[unicode_137229], **kwargs_137230)

# Testing the type of an if condition (line 62)
if_condition_137232 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 62, 0), startswith_call_result_137231)
# Assigning a type to the variable 'if_condition_137232' (line 62)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 0), 'if_condition_137232', if_condition_137232)
# SSA begins for if statement (line 62)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Assigning a Str to a Name (line 63):

# Assigning a Str to a Name (line 63):
unicode_137233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 16), 'unicode', u'&')
# Assigning a type to the variable 'cmd_split' (line 63)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'cmd_split', unicode_137233)
# SSA branch for the else part of an if statement (line 62)
module_type_store.open_ssa_branch('else')

# Assigning a Str to a Name (line 65):

# Assigning a Str to a Name (line 65):
unicode_137234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 16), 'unicode', u';')
# Assigning a type to the variable 'cmd_split' (line 65)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'cmd_split', unicode_137234)
# SSA join for if statement (line 62)
module_type_store = module_type_store.join_ssa_context()


@norecursion
def dvipng_hack_alpha(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'dvipng_hack_alpha'
    module_type_store = module_type_store.open_function_context('dvipng_hack_alpha', 68, 0, False)
    
    # Passed parameters checking function
    dvipng_hack_alpha.stypy_localization = localization
    dvipng_hack_alpha.stypy_type_of_self = None
    dvipng_hack_alpha.stypy_type_store = module_type_store
    dvipng_hack_alpha.stypy_function_name = 'dvipng_hack_alpha'
    dvipng_hack_alpha.stypy_param_names_list = []
    dvipng_hack_alpha.stypy_varargs_param_name = None
    dvipng_hack_alpha.stypy_kwargs_param_name = None
    dvipng_hack_alpha.stypy_call_defaults = defaults
    dvipng_hack_alpha.stypy_call_varargs = varargs
    dvipng_hack_alpha.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'dvipng_hack_alpha', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'dvipng_hack_alpha', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'dvipng_hack_alpha(...)' code ##################

    
    
    # SSA begins for try-except statement (line 70)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Name (line 71):
    
    # Assigning a Call to a Name (line 71):
    
    # Call to Popen(...): (line 71)
    # Processing the call arguments (line 71)
    
    # Obtaining an instance of the builtin type 'list' (line 71)
    list_137236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 71)
    # Adding element type (line 71)
    
    # Call to str(...): (line 71)
    # Processing the call arguments (line 71)
    unicode_137238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 23), 'unicode', u'dvipng')
    # Processing the call keyword arguments (line 71)
    kwargs_137239 = {}
    # Getting the type of 'str' (line 71)
    str_137237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 19), 'str', False)
    # Calling str(args, kwargs) (line 71)
    str_call_result_137240 = invoke(stypy.reporting.localization.Localization(__file__, 71, 19), str_137237, *[unicode_137238], **kwargs_137239)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 18), list_137236, str_call_result_137240)
    # Adding element type (line 71)
    unicode_137241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 34), 'unicode', u'-version')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 18), list_137236, unicode_137241)
    
    # Processing the call keyword arguments (line 71)
    # Getting the type of 'PIPE' (line 71)
    PIPE_137242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 53), 'PIPE', False)
    keyword_137243 = PIPE_137242
    # Getting the type of 'PIPE' (line 71)
    PIPE_137244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 66), 'PIPE', False)
    keyword_137245 = PIPE_137244
    # Getting the type of 'STDOUT' (line 72)
    STDOUT_137246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 25), 'STDOUT', False)
    keyword_137247 = STDOUT_137246
    
    # Getting the type of 'sys' (line 72)
    sys_137248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 44), 'sys', False)
    # Obtaining the member 'platform' of a type (line 72)
    platform_137249 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 44), sys_137248, 'platform')
    unicode_137250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 60), 'unicode', u'win32')
    # Applying the binary operator '!=' (line 72)
    result_ne_137251 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 44), '!=', platform_137249, unicode_137250)
    
    keyword_137252 = result_ne_137251
    kwargs_137253 = {'close_fds': keyword_137252, 'stdin': keyword_137243, 'stderr': keyword_137247, 'stdout': keyword_137245}
    # Getting the type of 'Popen' (line 71)
    Popen_137235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 12), 'Popen', False)
    # Calling Popen(args, kwargs) (line 71)
    Popen_call_result_137254 = invoke(stypy.reporting.localization.Localization(__file__, 71, 12), Popen_137235, *[list_137236], **kwargs_137253)
    
    # Assigning a type to the variable 'p' (line 71)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'p', Popen_call_result_137254)
    
    # Assigning a Call to a Tuple (line 73):
    
    # Assigning a Call to a Name:
    
    # Call to communicate(...): (line 73)
    # Processing the call keyword arguments (line 73)
    kwargs_137257 = {}
    # Getting the type of 'p' (line 73)
    p_137255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 25), 'p', False)
    # Obtaining the member 'communicate' of a type (line 73)
    communicate_137256 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 25), p_137255, 'communicate')
    # Calling communicate(args, kwargs) (line 73)
    communicate_call_result_137258 = invoke(stypy.reporting.localization.Localization(__file__, 73, 25), communicate_137256, *[], **kwargs_137257)
    
    # Assigning a type to the variable 'call_assignment_137197' (line 73)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'call_assignment_137197', communicate_call_result_137258)
    
    # Assigning a Call to a Name (line 73):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_137261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 8), 'int')
    # Processing the call keyword arguments
    kwargs_137262 = {}
    # Getting the type of 'call_assignment_137197' (line 73)
    call_assignment_137197_137259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'call_assignment_137197', False)
    # Obtaining the member '__getitem__' of a type (line 73)
    getitem___137260 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 8), call_assignment_137197_137259, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_137263 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___137260, *[int_137261], **kwargs_137262)
    
    # Assigning a type to the variable 'call_assignment_137198' (line 73)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'call_assignment_137198', getitem___call_result_137263)
    
    # Assigning a Name to a Name (line 73):
    # Getting the type of 'call_assignment_137198' (line 73)
    call_assignment_137198_137264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'call_assignment_137198')
    # Assigning a type to the variable 'stdout' (line 73)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'stdout', call_assignment_137198_137264)
    
    # Assigning a Call to a Name (line 73):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_137267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 8), 'int')
    # Processing the call keyword arguments
    kwargs_137268 = {}
    # Getting the type of 'call_assignment_137197' (line 73)
    call_assignment_137197_137265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'call_assignment_137197', False)
    # Obtaining the member '__getitem__' of a type (line 73)
    getitem___137266 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 8), call_assignment_137197_137265, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_137269 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___137266, *[int_137267], **kwargs_137268)
    
    # Assigning a type to the variable 'call_assignment_137199' (line 73)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'call_assignment_137199', getitem___call_result_137269)
    
    # Assigning a Name to a Name (line 73):
    # Getting the type of 'call_assignment_137199' (line 73)
    call_assignment_137199_137270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'call_assignment_137199')
    # Assigning a type to the variable 'stderr' (line 73)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 16), 'stderr', call_assignment_137199_137270)
    # SSA branch for the except part of a try statement (line 70)
    # SSA branch for the except 'OSError' branch of a try statement (line 70)
    module_type_store.open_ssa_branch('except')
    
    # Call to report(...): (line 75)
    # Processing the call arguments (line 75)
    unicode_137274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 27), 'unicode', u'No dvipng was found')
    unicode_137275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 50), 'unicode', u'helpful')
    # Processing the call keyword arguments (line 75)
    kwargs_137276 = {}
    # Getting the type of 'mpl' (line 75)
    mpl_137271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'mpl', False)
    # Obtaining the member 'verbose' of a type (line 75)
    verbose_137272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 8), mpl_137271, 'verbose')
    # Obtaining the member 'report' of a type (line 75)
    report_137273 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 8), verbose_137272, 'report')
    # Calling report(args, kwargs) (line 75)
    report_call_result_137277 = invoke(stypy.reporting.localization.Localization(__file__, 75, 8), report_137273, *[unicode_137274, unicode_137275], **kwargs_137276)
    
    # Getting the type of 'False' (line 76)
    False_137278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 15), 'False')
    # Assigning a type to the variable 'stypy_return_type' (line 76)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'stypy_return_type', False_137278)
    # SSA join for try-except statement (line 70)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 77):
    
    # Assigning a Call to a Name (line 77):
    
    # Call to split(...): (line 77)
    # Processing the call arguments (line 77)
    unicode_137288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 58), 'unicode', u'\n')
    # Processing the call keyword arguments (line 77)
    kwargs_137289 = {}
    
    # Call to decode(...): (line 77)
    # Processing the call arguments (line 77)
    
    # Call to getdefaultencoding(...): (line 77)
    # Processing the call keyword arguments (line 77)
    kwargs_137283 = {}
    # Getting the type of 'sys' (line 77)
    sys_137281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 26), 'sys', False)
    # Obtaining the member 'getdefaultencoding' of a type (line 77)
    getdefaultencoding_137282 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 26), sys_137281, 'getdefaultencoding')
    # Calling getdefaultencoding(args, kwargs) (line 77)
    getdefaultencoding_call_result_137284 = invoke(stypy.reporting.localization.Localization(__file__, 77, 26), getdefaultencoding_137282, *[], **kwargs_137283)
    
    # Processing the call keyword arguments (line 77)
    kwargs_137285 = {}
    # Getting the type of 'stdout' (line 77)
    stdout_137279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 12), 'stdout', False)
    # Obtaining the member 'decode' of a type (line 77)
    decode_137280 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 12), stdout_137279, 'decode')
    # Calling decode(args, kwargs) (line 77)
    decode_call_result_137286 = invoke(stypy.reporting.localization.Localization(__file__, 77, 12), decode_137280, *[getdefaultencoding_call_result_137284], **kwargs_137285)
    
    # Obtaining the member 'split' of a type (line 77)
    split_137287 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 12), decode_call_result_137286, 'split')
    # Calling split(args, kwargs) (line 77)
    split_call_result_137290 = invoke(stypy.reporting.localization.Localization(__file__, 77, 12), split_137287, *[unicode_137288], **kwargs_137289)
    
    # Assigning a type to the variable 'lines' (line 77)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 4), 'lines', split_call_result_137290)
    
    # Getting the type of 'lines' (line 78)
    lines_137291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 16), 'lines')
    # Testing the type of a for loop iterable (line 78)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 78, 4), lines_137291)
    # Getting the type of the for loop variable (line 78)
    for_loop_var_137292 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 78, 4), lines_137291)
    # Assigning a type to the variable 'line' (line 78)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'line', for_loop_var_137292)
    # SSA begins for a for statement (line 78)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Call to startswith(...): (line 79)
    # Processing the call arguments (line 79)
    unicode_137295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 27), 'unicode', u'dvipng ')
    # Processing the call keyword arguments (line 79)
    kwargs_137296 = {}
    # Getting the type of 'line' (line 79)
    line_137293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 11), 'line', False)
    # Obtaining the member 'startswith' of a type (line 79)
    startswith_137294 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 11), line_137293, 'startswith')
    # Calling startswith(args, kwargs) (line 79)
    startswith_call_result_137297 = invoke(stypy.reporting.localization.Localization(__file__, 79, 11), startswith_137294, *[unicode_137295], **kwargs_137296)
    
    # Testing the type of an if condition (line 79)
    if_condition_137298 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 79, 8), startswith_call_result_137297)
    # Assigning a type to the variable 'if_condition_137298' (line 79)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'if_condition_137298', if_condition_137298)
    # SSA begins for if statement (line 79)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 80):
    
    # Assigning a Subscript to a Name (line 80):
    
    # Obtaining the type of the subscript
    int_137299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 35), 'int')
    
    # Call to split(...): (line 80)
    # Processing the call keyword arguments (line 80)
    kwargs_137302 = {}
    # Getting the type of 'line' (line 80)
    line_137300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 22), 'line', False)
    # Obtaining the member 'split' of a type (line 80)
    split_137301 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 22), line_137300, 'split')
    # Calling split(args, kwargs) (line 80)
    split_call_result_137303 = invoke(stypy.reporting.localization.Localization(__file__, 80, 22), split_137301, *[], **kwargs_137302)
    
    # Obtaining the member '__getitem__' of a type (line 80)
    getitem___137304 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 22), split_call_result_137303, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 80)
    subscript_call_result_137305 = invoke(stypy.reporting.localization.Localization(__file__, 80, 22), getitem___137304, int_137299)
    
    # Assigning a type to the variable 'version' (line 80)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 12), 'version', subscript_call_result_137305)
    
    # Call to report(...): (line 81)
    # Processing the call arguments (line 81)
    unicode_137309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 31), 'unicode', u'Found dvipng version %s')
    # Getting the type of 'version' (line 81)
    version_137310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 59), 'version', False)
    # Applying the binary operator '%' (line 81)
    result_mod_137311 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 31), '%', unicode_137309, version_137310)
    
    unicode_137312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 31), 'unicode', u'helpful')
    # Processing the call keyword arguments (line 81)
    kwargs_137313 = {}
    # Getting the type of 'mpl' (line 81)
    mpl_137306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 12), 'mpl', False)
    # Obtaining the member 'verbose' of a type (line 81)
    verbose_137307 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 12), mpl_137306, 'verbose')
    # Obtaining the member 'report' of a type (line 81)
    report_137308 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 12), verbose_137307, 'report')
    # Calling report(args, kwargs) (line 81)
    report_call_result_137314 = invoke(stypy.reporting.localization.Localization(__file__, 81, 12), report_137308, *[result_mod_137311, unicode_137312], **kwargs_137313)
    
    
    # Assigning a Call to a Name (line 83):
    
    # Assigning a Call to a Name (line 83):
    
    # Call to LooseVersion(...): (line 83)
    # Processing the call arguments (line 83)
    # Getting the type of 'version' (line 83)
    version_137318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 53), 'version', False)
    # Processing the call keyword arguments (line 83)
    kwargs_137319 = {}
    # Getting the type of 'distutils' (line 83)
    distutils_137315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 22), 'distutils', False)
    # Obtaining the member 'version' of a type (line 83)
    version_137316 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 22), distutils_137315, 'version')
    # Obtaining the member 'LooseVersion' of a type (line 83)
    LooseVersion_137317 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 22), version_137316, 'LooseVersion')
    # Calling LooseVersion(args, kwargs) (line 83)
    LooseVersion_call_result_137320 = invoke(stypy.reporting.localization.Localization(__file__, 83, 22), LooseVersion_137317, *[version_137318], **kwargs_137319)
    
    # Assigning a type to the variable 'version' (line 83)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 12), 'version', LooseVersion_call_result_137320)
    
    # Getting the type of 'version' (line 84)
    version_137321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 19), 'version')
    
    # Call to LooseVersion(...): (line 84)
    # Processing the call arguments (line 84)
    unicode_137325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 60), 'unicode', u'1.6')
    # Processing the call keyword arguments (line 84)
    kwargs_137326 = {}
    # Getting the type of 'distutils' (line 84)
    distutils_137322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 29), 'distutils', False)
    # Obtaining the member 'version' of a type (line 84)
    version_137323 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 29), distutils_137322, 'version')
    # Obtaining the member 'LooseVersion' of a type (line 84)
    LooseVersion_137324 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 29), version_137323, 'LooseVersion')
    # Calling LooseVersion(args, kwargs) (line 84)
    LooseVersion_call_result_137327 = invoke(stypy.reporting.localization.Localization(__file__, 84, 29), LooseVersion_137324, *[unicode_137325], **kwargs_137326)
    
    # Applying the binary operator '<' (line 84)
    result_lt_137328 = python_operator(stypy.reporting.localization.Localization(__file__, 84, 19), '<', version_137321, LooseVersion_call_result_137327)
    
    # Assigning a type to the variable 'stypy_return_type' (line 84)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 12), 'stypy_return_type', result_lt_137328)
    # SSA join for if statement (line 79)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to report(...): (line 85)
    # Processing the call arguments (line 85)
    unicode_137332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 23), 'unicode', u'Unexpected response from dvipng -version')
    unicode_137333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 67), 'unicode', u'helpful')
    # Processing the call keyword arguments (line 85)
    kwargs_137334 = {}
    # Getting the type of 'mpl' (line 85)
    mpl_137329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'mpl', False)
    # Obtaining the member 'verbose' of a type (line 85)
    verbose_137330 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 4), mpl_137329, 'verbose')
    # Obtaining the member 'report' of a type (line 85)
    report_137331 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 4), verbose_137330, 'report')
    # Calling report(args, kwargs) (line 85)
    report_call_result_137335 = invoke(stypy.reporting.localization.Localization(__file__, 85, 4), report_137331, *[unicode_137332, unicode_137333], **kwargs_137334)
    
    # Getting the type of 'False' (line 86)
    False_137336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 11), 'False')
    # Assigning a type to the variable 'stypy_return_type' (line 86)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'stypy_return_type', False_137336)
    
    # ################# End of 'dvipng_hack_alpha(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'dvipng_hack_alpha' in the type store
    # Getting the type of 'stypy_return_type' (line 68)
    stypy_return_type_137337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_137337)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'dvipng_hack_alpha'
    return stypy_return_type_137337

# Assigning a type to the variable 'dvipng_hack_alpha' (line 68)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 0), 'dvipng_hack_alpha', dvipng_hack_alpha)
# Declaration of the 'TexManager' class

class TexManager(object, ):
    unicode_137338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, (-1)), 'unicode', u'\n    Convert strings to dvi files using TeX, caching the results to a\n    working dir\n    ')
    
    # Assigning a Dict to a Name (line 127):
    
    # Assigning a Dict to a Name (line 128):
    
    # Assigning a Dict to a Name (line 129):
    
    # Assigning a Num to a Name (line 130):
    
    # Assigning a Tuple to a Name (line 132):
    
    # Assigning a Tuple to a Name (line 133):
    
    # Assigning a Tuple to a Name (line 134):
    
    # Assigning a Tuple to a Name (line 135):
    
    # Assigning a Str to a Name (line 136):
    
    # Assigning a Tuple to a Name (line 137):
    
    # Assigning a Dict to a Name (line 139):
    
    # Assigning a Name to a Name (line 157):
    
    # Assigning a BinOp to a Name (line 158):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 162, 4, False)
        # Assigning a type to the variable 'self' (line 163)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TexManager.__init__', [], None, None, defaults, varargs, kwargs)

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

        
        # Type idiom detected: calculating its left and rigth part (line 164)
        # Getting the type of 'self' (line 164)
        self_137339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 11), 'self')
        # Obtaining the member 'texcache' of a type (line 164)
        texcache_137340 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 11), self_137339, 'texcache')
        # Getting the type of 'None' (line 164)
        None_137341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 28), 'None')
        
        (may_be_137342, more_types_in_union_137343) = may_be_none(texcache_137340, None_137341)

        if may_be_137342:

            if more_types_in_union_137343:
                # Runtime conditional SSA (line 164)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to RuntimeError(...): (line 165)
            # Processing the call arguments (line 165)
            unicode_137345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 17), 'unicode', u'Cannot create TexManager, as there is no cache directory available')
            # Processing the call keyword arguments (line 165)
            kwargs_137346 = {}
            # Getting the type of 'RuntimeError' (line 165)
            RuntimeError_137344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 18), 'RuntimeError', False)
            # Calling RuntimeError(args, kwargs) (line 165)
            RuntimeError_call_result_137347 = invoke(stypy.reporting.localization.Localization(__file__, 165, 18), RuntimeError_137344, *[unicode_137345], **kwargs_137346)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 165, 12), RuntimeError_call_result_137347, 'raise parameter', BaseException)

            if more_types_in_union_137343:
                # SSA join for if statement (line 164)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Call to mkdirs(...): (line 169)
        # Processing the call arguments (line 169)
        # Getting the type of 'self' (line 169)
        self_137349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 15), 'self', False)
        # Obtaining the member 'texcache' of a type (line 169)
        texcache_137350 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 15), self_137349, 'texcache')
        # Processing the call keyword arguments (line 169)
        kwargs_137351 = {}
        # Getting the type of 'mkdirs' (line 169)
        mkdirs_137348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'mkdirs', False)
        # Calling mkdirs(args, kwargs) (line 169)
        mkdirs_call_result_137352 = invoke(stypy.reporting.localization.Localization(__file__, 169, 8), mkdirs_137348, *[texcache_137350], **kwargs_137351)
        
        
        # Assigning a Subscript to a Name (line 170):
        
        # Assigning a Subscript to a Name (line 170):
        
        # Obtaining the type of the subscript
        unicode_137353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 22), 'unicode', u'font.family')
        # Getting the type of 'rcParams' (line 170)
        rcParams_137354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 13), 'rcParams')
        # Obtaining the member '__getitem__' of a type (line 170)
        getitem___137355 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 13), rcParams_137354, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 170)
        subscript_call_result_137356 = invoke(stypy.reporting.localization.Localization(__file__, 170, 13), getitem___137355, unicode_137353)
        
        # Assigning a type to the variable 'ff' (line 170)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 8), 'ff', subscript_call_result_137356)
        
        
        # Evaluating a boolean operation
        
        
        # Call to len(...): (line 171)
        # Processing the call arguments (line 171)
        # Getting the type of 'ff' (line 171)
        ff_137358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 15), 'ff', False)
        # Processing the call keyword arguments (line 171)
        kwargs_137359 = {}
        # Getting the type of 'len' (line 171)
        len_137357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 11), 'len', False)
        # Calling len(args, kwargs) (line 171)
        len_call_result_137360 = invoke(stypy.reporting.localization.Localization(__file__, 171, 11), len_137357, *[ff_137358], **kwargs_137359)
        
        int_137361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 22), 'int')
        # Applying the binary operator '==' (line 171)
        result_eq_137362 = python_operator(stypy.reporting.localization.Localization(__file__, 171, 11), '==', len_call_result_137360, int_137361)
        
        
        
        # Call to lower(...): (line 171)
        # Processing the call keyword arguments (line 171)
        kwargs_137368 = {}
        
        # Obtaining the type of the subscript
        int_137363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 31), 'int')
        # Getting the type of 'ff' (line 171)
        ff_137364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 28), 'ff', False)
        # Obtaining the member '__getitem__' of a type (line 171)
        getitem___137365 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 28), ff_137364, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 171)
        subscript_call_result_137366 = invoke(stypy.reporting.localization.Localization(__file__, 171, 28), getitem___137365, int_137363)
        
        # Obtaining the member 'lower' of a type (line 171)
        lower_137367 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 28), subscript_call_result_137366, 'lower')
        # Calling lower(args, kwargs) (line 171)
        lower_call_result_137369 = invoke(stypy.reporting.localization.Localization(__file__, 171, 28), lower_137367, *[], **kwargs_137368)
        
        # Getting the type of 'self' (line 171)
        self_137370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 45), 'self')
        # Obtaining the member 'font_families' of a type (line 171)
        font_families_137371 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 45), self_137370, 'font_families')
        # Applying the binary operator 'in' (line 171)
        result_contains_137372 = python_operator(stypy.reporting.localization.Localization(__file__, 171, 28), 'in', lower_call_result_137369, font_families_137371)
        
        # Applying the binary operator 'and' (line 171)
        result_and_keyword_137373 = python_operator(stypy.reporting.localization.Localization(__file__, 171, 11), 'and', result_eq_137362, result_contains_137372)
        
        # Testing the type of an if condition (line 171)
        if_condition_137374 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 171, 8), result_and_keyword_137373)
        # Assigning a type to the variable 'if_condition_137374' (line 171)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 8), 'if_condition_137374', if_condition_137374)
        # SSA begins for if statement (line 171)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Attribute (line 172):
        
        # Assigning a Call to a Attribute (line 172):
        
        # Call to lower(...): (line 172)
        # Processing the call keyword arguments (line 172)
        kwargs_137380 = {}
        
        # Obtaining the type of the subscript
        int_137375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 34), 'int')
        # Getting the type of 'ff' (line 172)
        ff_137376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 31), 'ff', False)
        # Obtaining the member '__getitem__' of a type (line 172)
        getitem___137377 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 31), ff_137376, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 172)
        subscript_call_result_137378 = invoke(stypy.reporting.localization.Localization(__file__, 172, 31), getitem___137377, int_137375)
        
        # Obtaining the member 'lower' of a type (line 172)
        lower_137379 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 31), subscript_call_result_137378, 'lower')
        # Calling lower(args, kwargs) (line 172)
        lower_call_result_137381 = invoke(stypy.reporting.localization.Localization(__file__, 172, 31), lower_137379, *[], **kwargs_137380)
        
        # Getting the type of 'self' (line 172)
        self_137382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 12), 'self')
        # Setting the type of the member 'font_family' of a type (line 172)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 12), self_137382, 'font_family', lower_call_result_137381)
        # SSA branch for the else part of an if statement (line 171)
        module_type_store.open_ssa_branch('else')
        
        
        # Evaluating a boolean operation
        
        # Call to isinstance(...): (line 173)
        # Processing the call arguments (line 173)
        # Getting the type of 'ff' (line 173)
        ff_137384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 24), 'ff', False)
        # Getting the type of 'six' (line 173)
        six_137385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 28), 'six', False)
        # Obtaining the member 'string_types' of a type (line 173)
        string_types_137386 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 28), six_137385, 'string_types')
        # Processing the call keyword arguments (line 173)
        kwargs_137387 = {}
        # Getting the type of 'isinstance' (line 173)
        isinstance_137383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 13), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 173)
        isinstance_call_result_137388 = invoke(stypy.reporting.localization.Localization(__file__, 173, 13), isinstance_137383, *[ff_137384, string_types_137386], **kwargs_137387)
        
        
        
        # Call to lower(...): (line 173)
        # Processing the call keyword arguments (line 173)
        kwargs_137391 = {}
        # Getting the type of 'ff' (line 173)
        ff_137389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 50), 'ff', False)
        # Obtaining the member 'lower' of a type (line 173)
        lower_137390 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 50), ff_137389, 'lower')
        # Calling lower(args, kwargs) (line 173)
        lower_call_result_137392 = invoke(stypy.reporting.localization.Localization(__file__, 173, 50), lower_137390, *[], **kwargs_137391)
        
        # Getting the type of 'self' (line 173)
        self_137393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 64), 'self')
        # Obtaining the member 'font_families' of a type (line 173)
        font_families_137394 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 64), self_137393, 'font_families')
        # Applying the binary operator 'in' (line 173)
        result_contains_137395 = python_operator(stypy.reporting.localization.Localization(__file__, 173, 50), 'in', lower_call_result_137392, font_families_137394)
        
        # Applying the binary operator 'and' (line 173)
        result_and_keyword_137396 = python_operator(stypy.reporting.localization.Localization(__file__, 173, 13), 'and', isinstance_call_result_137388, result_contains_137395)
        
        # Testing the type of an if condition (line 173)
        if_condition_137397 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 173, 13), result_and_keyword_137396)
        # Assigning a type to the variable 'if_condition_137397' (line 173)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 13), 'if_condition_137397', if_condition_137397)
        # SSA begins for if statement (line 173)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Attribute (line 174):
        
        # Assigning a Call to a Attribute (line 174):
        
        # Call to lower(...): (line 174)
        # Processing the call keyword arguments (line 174)
        kwargs_137400 = {}
        # Getting the type of 'ff' (line 174)
        ff_137398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 31), 'ff', False)
        # Obtaining the member 'lower' of a type (line 174)
        lower_137399 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 31), ff_137398, 'lower')
        # Calling lower(args, kwargs) (line 174)
        lower_call_result_137401 = invoke(stypy.reporting.localization.Localization(__file__, 174, 31), lower_137399, *[], **kwargs_137400)
        
        # Getting the type of 'self' (line 174)
        self_137402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 12), 'self')
        # Setting the type of the member 'font_family' of a type (line 174)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 12), self_137402, 'font_family', lower_call_result_137401)
        # SSA branch for the else part of an if statement (line 173)
        module_type_store.open_ssa_branch('else')
        
        # Call to report(...): (line 176)
        # Processing the call arguments (line 176)
        unicode_137406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 16), 'unicode', u'font.family must be one of (%s) when text.usetex is True. serif will be used by default.')
        
        # Call to join(...): (line 179)
        # Processing the call arguments (line 179)
        # Getting the type of 'self' (line 179)
        self_137409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 29), 'self', False)
        # Obtaining the member 'font_families' of a type (line 179)
        font_families_137410 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 29), self_137409, 'font_families')
        # Processing the call keyword arguments (line 179)
        kwargs_137411 = {}
        unicode_137407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 19), 'unicode', u', ')
        # Obtaining the member 'join' of a type (line 179)
        join_137408 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 19), unicode_137407, 'join')
        # Calling join(args, kwargs) (line 179)
        join_call_result_137412 = invoke(stypy.reporting.localization.Localization(__file__, 179, 19), join_137408, *[font_families_137410], **kwargs_137411)
        
        # Applying the binary operator '%' (line 177)
        result_mod_137413 = python_operator(stypy.reporting.localization.Localization(__file__, 177, 16), '%', unicode_137406, join_call_result_137412)
        
        unicode_137414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 16), 'unicode', u'helpful')
        # Processing the call keyword arguments (line 176)
        kwargs_137415 = {}
        # Getting the type of 'mpl' (line 176)
        mpl_137403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 12), 'mpl', False)
        # Obtaining the member 'verbose' of a type (line 176)
        verbose_137404 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 12), mpl_137403, 'verbose')
        # Obtaining the member 'report' of a type (line 176)
        report_137405 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 12), verbose_137404, 'report')
        # Calling report(args, kwargs) (line 176)
        report_call_result_137416 = invoke(stypy.reporting.localization.Localization(__file__, 176, 12), report_137405, *[result_mod_137413, unicode_137414], **kwargs_137415)
        
        
        # Assigning a Str to a Attribute (line 181):
        
        # Assigning a Str to a Attribute (line 181):
        unicode_137417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 31), 'unicode', u'serif')
        # Getting the type of 'self' (line 181)
        self_137418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 12), 'self')
        # Setting the type of the member 'font_family' of a type (line 181)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 12), self_137418, 'font_family', unicode_137417)
        # SSA join for if statement (line 173)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 171)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a List to a Name (line 183):
        
        # Assigning a List to a Name (line 183):
        
        # Obtaining an instance of the builtin type 'list' (line 183)
        list_137419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 183)
        # Adding element type (line 183)
        # Getting the type of 'self' (line 183)
        self_137420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 22), 'self')
        # Obtaining the member 'font_family' of a type (line 183)
        font_family_137421 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 22), self_137420, 'font_family')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 21), list_137419, font_family_137421)
        
        # Assigning a type to the variable 'fontconfig' (line 183)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 8), 'fontconfig', list_137419)
        
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'self' (line 185)
        self_137430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 56), 'self')
        # Obtaining the member 'font_families' of a type (line 185)
        font_families_137431 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 56), self_137430, 'font_families')
        comprehension_137432 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 184, 46), font_families_137431)
        # Assigning a type to the variable 'ff' (line 184)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 46), 'ff', comprehension_137432)
        
        # Obtaining an instance of the builtin type 'tuple' (line 184)
        tuple_137422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 47), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 184)
        # Adding element type (line 184)
        # Getting the type of 'ff' (line 184)
        ff_137423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 47), 'ff')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 184, 47), tuple_137422, ff_137423)
        # Adding element type (line 184)
        
        # Call to replace(...): (line 184)
        # Processing the call arguments (line 184)
        unicode_137426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 62), 'unicode', u'-')
        unicode_137427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 67), 'unicode', u'_')
        # Processing the call keyword arguments (line 184)
        kwargs_137428 = {}
        # Getting the type of 'ff' (line 184)
        ff_137424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 51), 'ff', False)
        # Obtaining the member 'replace' of a type (line 184)
        replace_137425 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 51), ff_137424, 'replace')
        # Calling replace(args, kwargs) (line 184)
        replace_call_result_137429 = invoke(stypy.reporting.localization.Localization(__file__, 184, 51), replace_137425, *[unicode_137426, unicode_137427], **kwargs_137428)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 184, 47), tuple_137422, replace_call_result_137429)
        
        list_137433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 46), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 184, 46), list_137433, tuple_137422)
        # Testing the type of a for loop iterable (line 184)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 184, 8), list_137433)
        # Getting the type of the for loop variable (line 184)
        for_loop_var_137434 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 184, 8), list_137433)
        # Assigning a type to the variable 'font_family' (line 184)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 'font_family', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 184, 8), for_loop_var_137434))
        # Assigning a type to the variable 'font_family_attr' (line 184)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 'font_family_attr', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 184, 8), for_loop_var_137434))
        # SSA begins for a for statement (line 184)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Obtaining the type of the subscript
        unicode_137435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 33), 'unicode', u'font.')
        # Getting the type of 'font_family' (line 186)
        font_family_137436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 43), 'font_family')
        # Applying the binary operator '+' (line 186)
        result_add_137437 = python_operator(stypy.reporting.localization.Localization(__file__, 186, 33), '+', unicode_137435, font_family_137436)
        
        # Getting the type of 'rcParams' (line 186)
        rcParams_137438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 24), 'rcParams')
        # Obtaining the member '__getitem__' of a type (line 186)
        getitem___137439 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 24), rcParams_137438, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 186)
        subscript_call_result_137440 = invoke(stypy.reporting.localization.Localization(__file__, 186, 24), getitem___137439, result_add_137437)
        
        # Testing the type of a for loop iterable (line 186)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 186, 12), subscript_call_result_137440)
        # Getting the type of the for loop variable (line 186)
        for_loop_var_137441 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 186, 12), subscript_call_result_137440)
        # Assigning a type to the variable 'font' (line 186)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 12), 'font', for_loop_var_137441)
        # SSA begins for a for statement (line 186)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        
        # Call to lower(...): (line 187)
        # Processing the call keyword arguments (line 187)
        kwargs_137444 = {}
        # Getting the type of 'font' (line 187)
        font_137442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 19), 'font', False)
        # Obtaining the member 'lower' of a type (line 187)
        lower_137443 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 19), font_137442, 'lower')
        # Calling lower(args, kwargs) (line 187)
        lower_call_result_137445 = invoke(stypy.reporting.localization.Localization(__file__, 187, 19), lower_137443, *[], **kwargs_137444)
        
        # Getting the type of 'self' (line 187)
        self_137446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 35), 'self')
        # Obtaining the member 'font_info' of a type (line 187)
        font_info_137447 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 35), self_137446, 'font_info')
        # Applying the binary operator 'in' (line 187)
        result_contains_137448 = python_operator(stypy.reporting.localization.Localization(__file__, 187, 19), 'in', lower_call_result_137445, font_info_137447)
        
        # Testing the type of an if condition (line 187)
        if_condition_137449 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 187, 16), result_contains_137448)
        # Assigning a type to the variable 'if_condition_137449' (line 187)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 16), 'if_condition_137449', if_condition_137449)
        # SSA begins for if statement (line 187)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to setattr(...): (line 188)
        # Processing the call arguments (line 188)
        # Getting the type of 'self' (line 188)
        self_137451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 28), 'self', False)
        # Getting the type of 'font_family_attr' (line 188)
        font_family_attr_137452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 34), 'font_family_attr', False)
        
        # Obtaining the type of the subscript
        
        # Call to lower(...): (line 189)
        # Processing the call keyword arguments (line 189)
        kwargs_137455 = {}
        # Getting the type of 'font' (line 189)
        font_137453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 43), 'font', False)
        # Obtaining the member 'lower' of a type (line 189)
        lower_137454 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 43), font_137453, 'lower')
        # Calling lower(args, kwargs) (line 189)
        lower_call_result_137456 = invoke(stypy.reporting.localization.Localization(__file__, 189, 43), lower_137454, *[], **kwargs_137455)
        
        # Getting the type of 'self' (line 189)
        self_137457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 28), 'self', False)
        # Obtaining the member 'font_info' of a type (line 189)
        font_info_137458 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 28), self_137457, 'font_info')
        # Obtaining the member '__getitem__' of a type (line 189)
        getitem___137459 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 28), font_info_137458, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 189)
        subscript_call_result_137460 = invoke(stypy.reporting.localization.Localization(__file__, 189, 28), getitem___137459, lower_call_result_137456)
        
        # Processing the call keyword arguments (line 188)
        kwargs_137461 = {}
        # Getting the type of 'setattr' (line 188)
        setattr_137450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 20), 'setattr', False)
        # Calling setattr(args, kwargs) (line 188)
        setattr_call_result_137462 = invoke(stypy.reporting.localization.Localization(__file__, 188, 20), setattr_137450, *[self_137451, font_family_attr_137452, subscript_call_result_137460], **kwargs_137461)
        
        
        # Getting the type of 'DEBUG' (line 190)
        DEBUG_137463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 23), 'DEBUG')
        # Testing the type of an if condition (line 190)
        if_condition_137464 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 190, 20), DEBUG_137463)
        # Assigning a type to the variable 'if_condition_137464' (line 190)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 20), 'if_condition_137464', if_condition_137464)
        # SSA begins for if statement (line 190)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to print(...): (line 191)
        # Processing the call arguments (line 191)
        unicode_137466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 30), 'unicode', u'family: %s, font: %s, info: %s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 192)
        tuple_137467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 31), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 192)
        # Adding element type (line 192)
        # Getting the type of 'font_family' (line 192)
        font_family_137468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 31), 'font_family', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 192, 31), tuple_137467, font_family_137468)
        # Adding element type (line 192)
        # Getting the type of 'font' (line 192)
        font_137469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 44), 'font', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 192, 31), tuple_137467, font_137469)
        # Adding element type (line 192)
        
        # Obtaining the type of the subscript
        
        # Call to lower(...): (line 193)
        # Processing the call keyword arguments (line 193)
        kwargs_137472 = {}
        # Getting the type of 'font' (line 193)
        font_137470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 46), 'font', False)
        # Obtaining the member 'lower' of a type (line 193)
        lower_137471 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 46), font_137470, 'lower')
        # Calling lower(args, kwargs) (line 193)
        lower_call_result_137473 = invoke(stypy.reporting.localization.Localization(__file__, 193, 46), lower_137471, *[], **kwargs_137472)
        
        # Getting the type of 'self' (line 193)
        self_137474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 31), 'self', False)
        # Obtaining the member 'font_info' of a type (line 193)
        font_info_137475 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 31), self_137474, 'font_info')
        # Obtaining the member '__getitem__' of a type (line 193)
        getitem___137476 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 31), font_info_137475, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 193)
        subscript_call_result_137477 = invoke(stypy.reporting.localization.Localization(__file__, 193, 31), getitem___137476, lower_call_result_137473)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 192, 31), tuple_137467, subscript_call_result_137477)
        
        # Applying the binary operator '%' (line 191)
        result_mod_137478 = python_operator(stypy.reporting.localization.Localization(__file__, 191, 30), '%', unicode_137466, tuple_137467)
        
        # Processing the call keyword arguments (line 191)
        kwargs_137479 = {}
        # Getting the type of 'print' (line 191)
        print_137465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 24), 'print', False)
        # Calling print(args, kwargs) (line 191)
        print_call_result_137480 = invoke(stypy.reporting.localization.Localization(__file__, 191, 24), print_137465, *[result_mod_137478], **kwargs_137479)
        
        # SSA join for if statement (line 190)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 187)
        module_type_store.open_ssa_branch('else')
        
        # Getting the type of 'DEBUG' (line 196)
        DEBUG_137481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 23), 'DEBUG')
        # Testing the type of an if condition (line 196)
        if_condition_137482 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 196, 20), DEBUG_137481)
        # Assigning a type to the variable 'if_condition_137482' (line 196)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 20), 'if_condition_137482', if_condition_137482)
        # SSA begins for if statement (line 196)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to print(...): (line 197)
        # Processing the call arguments (line 197)
        unicode_137484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 30), 'unicode', u'$s font is not compatible with usetex')
        # Processing the call keyword arguments (line 197)
        kwargs_137485 = {}
        # Getting the type of 'print' (line 197)
        print_137483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 24), 'print', False)
        # Calling print(args, kwargs) (line 197)
        print_call_result_137486 = invoke(stypy.reporting.localization.Localization(__file__, 197, 24), print_137483, *[unicode_137484], **kwargs_137485)
        
        # SSA join for if statement (line 196)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 187)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of a for statement (line 186)
        module_type_store.open_ssa_branch('for loop else')
        
        # Call to report(...): (line 199)
        # Processing the call arguments (line 199)
        unicode_137490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 35), 'unicode', u'No LaTeX-compatible font found for the %s font family in rcParams. Using default.')
        # Getting the type of 'font_family' (line 201)
        font_family_137491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 48), 'font_family', False)
        # Applying the binary operator '%' (line 199)
        result_mod_137492 = python_operator(stypy.reporting.localization.Localization(__file__, 199, 35), '%', unicode_137490, font_family_137491)
        
        unicode_137493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 61), 'unicode', u'helpful')
        # Processing the call keyword arguments (line 199)
        kwargs_137494 = {}
        # Getting the type of 'mpl' (line 199)
        mpl_137487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 16), 'mpl', False)
        # Obtaining the member 'verbose' of a type (line 199)
        verbose_137488 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 16), mpl_137487, 'verbose')
        # Obtaining the member 'report' of a type (line 199)
        report_137489 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 16), verbose_137488, 'report')
        # Calling report(args, kwargs) (line 199)
        report_call_result_137495 = invoke(stypy.reporting.localization.Localization(__file__, 199, 16), report_137489, *[result_mod_137492, unicode_137493], **kwargs_137494)
        
        
        # Call to setattr(...): (line 202)
        # Processing the call arguments (line 202)
        # Getting the type of 'self' (line 202)
        self_137497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 24), 'self', False)
        # Getting the type of 'font_family_attr' (line 202)
        font_family_attr_137498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 30), 'font_family_attr', False)
        
        # Obtaining the type of the subscript
        # Getting the type of 'font_family' (line 202)
        font_family_137499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 63), 'font_family', False)
        # Getting the type of 'self' (line 202)
        self_137500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 48), 'self', False)
        # Obtaining the member 'font_info' of a type (line 202)
        font_info_137501 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 48), self_137500, 'font_info')
        # Obtaining the member '__getitem__' of a type (line 202)
        getitem___137502 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 48), font_info_137501, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 202)
        subscript_call_result_137503 = invoke(stypy.reporting.localization.Localization(__file__, 202, 48), getitem___137502, font_family_137499)
        
        # Processing the call keyword arguments (line 202)
        kwargs_137504 = {}
        # Getting the type of 'setattr' (line 202)
        setattr_137496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 16), 'setattr', False)
        # Calling setattr(args, kwargs) (line 202)
        setattr_call_result_137505 = invoke(stypy.reporting.localization.Localization(__file__, 202, 16), setattr_137496, *[self_137497, font_family_attr_137498, subscript_call_result_137503], **kwargs_137504)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to append(...): (line 203)
        # Processing the call arguments (line 203)
        
        # Obtaining the type of the subscript
        int_137508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 62), 'int')
        
        # Call to getattr(...): (line 203)
        # Processing the call arguments (line 203)
        # Getting the type of 'self' (line 203)
        self_137510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 38), 'self', False)
        # Getting the type of 'font_family_attr' (line 203)
        font_family_attr_137511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 44), 'font_family_attr', False)
        # Processing the call keyword arguments (line 203)
        kwargs_137512 = {}
        # Getting the type of 'getattr' (line 203)
        getattr_137509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 30), 'getattr', False)
        # Calling getattr(args, kwargs) (line 203)
        getattr_call_result_137513 = invoke(stypy.reporting.localization.Localization(__file__, 203, 30), getattr_137509, *[self_137510, font_family_attr_137511], **kwargs_137512)
        
        # Obtaining the member '__getitem__' of a type (line 203)
        getitem___137514 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 30), getattr_call_result_137513, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 203)
        subscript_call_result_137515 = invoke(stypy.reporting.localization.Localization(__file__, 203, 30), getitem___137514, int_137508)
        
        # Processing the call keyword arguments (line 203)
        kwargs_137516 = {}
        # Getting the type of 'fontconfig' (line 203)
        fontconfig_137506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 12), 'fontconfig', False)
        # Obtaining the member 'append' of a type (line 203)
        append_137507 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 12), fontconfig_137506, 'append')
        # Calling append(args, kwargs) (line 203)
        append_call_result_137517 = invoke(stypy.reporting.localization.Localization(__file__, 203, 12), append_137507, *[subscript_call_result_137515], **kwargs_137516)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 207):
        
        # Assigning a Call to a Name (line 207):
        
        # Call to encode(...): (line 207)
        # Processing the call arguments (line 207)
        unicode_137527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 74), 'unicode', u'utf-8')
        # Processing the call keyword arguments (line 207)
        kwargs_137528 = {}
        
        # Call to text_type(...): (line 207)
        # Processing the call arguments (line 207)
        
        # Call to get_custom_preamble(...): (line 207)
        # Processing the call keyword arguments (line 207)
        kwargs_137522 = {}
        # Getting the type of 'self' (line 207)
        self_137520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 39), 'self', False)
        # Obtaining the member 'get_custom_preamble' of a type (line 207)
        get_custom_preamble_137521 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 39), self_137520, 'get_custom_preamble')
        # Calling get_custom_preamble(args, kwargs) (line 207)
        get_custom_preamble_call_result_137523 = invoke(stypy.reporting.localization.Localization(__file__, 207, 39), get_custom_preamble_137521, *[], **kwargs_137522)
        
        # Processing the call keyword arguments (line 207)
        kwargs_137524 = {}
        # Getting the type of 'six' (line 207)
        six_137518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 25), 'six', False)
        # Obtaining the member 'text_type' of a type (line 207)
        text_type_137519 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 25), six_137518, 'text_type')
        # Calling text_type(args, kwargs) (line 207)
        text_type_call_result_137525 = invoke(stypy.reporting.localization.Localization(__file__, 207, 25), text_type_137519, *[get_custom_preamble_call_result_137523], **kwargs_137524)
        
        # Obtaining the member 'encode' of a type (line 207)
        encode_137526 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 25), text_type_call_result_137525, 'encode')
        # Calling encode(args, kwargs) (line 207)
        encode_call_result_137529 = invoke(stypy.reporting.localization.Localization(__file__, 207, 25), encode_137526, *[unicode_137527], **kwargs_137528)
        
        # Assigning a type to the variable 'preamble_bytes' (line 207)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 8), 'preamble_bytes', encode_call_result_137529)
        
        # Call to append(...): (line 208)
        # Processing the call arguments (line 208)
        
        # Call to hexdigest(...): (line 208)
        # Processing the call keyword arguments (line 208)
        kwargs_137537 = {}
        
        # Call to md5(...): (line 208)
        # Processing the call arguments (line 208)
        # Getting the type of 'preamble_bytes' (line 208)
        preamble_bytes_137533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 30), 'preamble_bytes', False)
        # Processing the call keyword arguments (line 208)
        kwargs_137534 = {}
        # Getting the type of 'md5' (line 208)
        md5_137532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 26), 'md5', False)
        # Calling md5(args, kwargs) (line 208)
        md5_call_result_137535 = invoke(stypy.reporting.localization.Localization(__file__, 208, 26), md5_137532, *[preamble_bytes_137533], **kwargs_137534)
        
        # Obtaining the member 'hexdigest' of a type (line 208)
        hexdigest_137536 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 26), md5_call_result_137535, 'hexdigest')
        # Calling hexdigest(args, kwargs) (line 208)
        hexdigest_call_result_137538 = invoke(stypy.reporting.localization.Localization(__file__, 208, 26), hexdigest_137536, *[], **kwargs_137537)
        
        # Processing the call keyword arguments (line 208)
        kwargs_137539 = {}
        # Getting the type of 'fontconfig' (line 208)
        fontconfig_137530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 8), 'fontconfig', False)
        # Obtaining the member 'append' of a type (line 208)
        append_137531 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 8), fontconfig_137530, 'append')
        # Calling append(args, kwargs) (line 208)
        append_call_result_137540 = invoke(stypy.reporting.localization.Localization(__file__, 208, 8), append_137531, *[hexdigest_call_result_137538], **kwargs_137539)
        
        
        # Assigning a Call to a Attribute (line 209):
        
        # Assigning a Call to a Attribute (line 209):
        
        # Call to join(...): (line 209)
        # Processing the call arguments (line 209)
        # Getting the type of 'fontconfig' (line 209)
        fontconfig_137543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 35), 'fontconfig', False)
        # Processing the call keyword arguments (line 209)
        kwargs_137544 = {}
        unicode_137541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 27), 'unicode', u'')
        # Obtaining the member 'join' of a type (line 209)
        join_137542 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 27), unicode_137541, 'join')
        # Calling join(args, kwargs) (line 209)
        join_call_result_137545 = invoke(stypy.reporting.localization.Localization(__file__, 209, 27), join_137542, *[fontconfig_137543], **kwargs_137544)
        
        # Getting the type of 'self' (line 209)
        self_137546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 8), 'self')
        # Setting the type of the member '_fontconfig' of a type (line 209)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 8), self_137546, '_fontconfig', join_call_result_137545)
        
        # Assigning a List to a Name (line 213):
        
        # Assigning a List to a Name (line 213):
        
        # Obtaining an instance of the builtin type 'list' (line 213)
        list_137547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 14), 'list')
        # Adding type elements to the builtin type 'list' instance (line 213)
        # Adding element type (line 213)
        
        # Obtaining the type of the subscript
        int_137548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 26), 'int')
        # Getting the type of 'self' (line 213)
        self_137549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 15), 'self')
        # Obtaining the member 'serif' of a type (line 213)
        serif_137550 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 15), self_137549, 'serif')
        # Obtaining the member '__getitem__' of a type (line 213)
        getitem___137551 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 15), serif_137550, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 213)
        subscript_call_result_137552 = invoke(stypy.reporting.localization.Localization(__file__, 213, 15), getitem___137551, int_137548)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 213, 14), list_137547, subscript_call_result_137552)
        # Adding element type (line 213)
        
        # Obtaining the type of the subscript
        int_137553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 46), 'int')
        # Getting the type of 'self' (line 213)
        self_137554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 30), 'self')
        # Obtaining the member 'sans_serif' of a type (line 213)
        sans_serif_137555 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 30), self_137554, 'sans_serif')
        # Obtaining the member '__getitem__' of a type (line 213)
        getitem___137556 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 30), sans_serif_137555, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 213)
        subscript_call_result_137557 = invoke(stypy.reporting.localization.Localization(__file__, 213, 30), getitem___137556, int_137553)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 213, 14), list_137547, subscript_call_result_137557)
        # Adding element type (line 213)
        
        # Obtaining the type of the subscript
        int_137558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 65), 'int')
        # Getting the type of 'self' (line 213)
        self_137559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 50), 'self')
        # Obtaining the member 'monospace' of a type (line 213)
        monospace_137560 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 50), self_137559, 'monospace')
        # Obtaining the member '__getitem__' of a type (line 213)
        getitem___137561 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 50), monospace_137560, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 213)
        subscript_call_result_137562 = invoke(stypy.reporting.localization.Localization(__file__, 213, 50), getitem___137561, int_137558)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 213, 14), list_137547, subscript_call_result_137562)
        
        # Assigning a type to the variable 'cmd' (line 213)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 8), 'cmd', list_137547)
        
        
        # Getting the type of 'self' (line 214)
        self_137563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 11), 'self')
        # Obtaining the member 'font_family' of a type (line 214)
        font_family_137564 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 11), self_137563, 'font_family')
        unicode_137565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 31), 'unicode', u'cursive')
        # Applying the binary operator '==' (line 214)
        result_eq_137566 = python_operator(stypy.reporting.localization.Localization(__file__, 214, 11), '==', font_family_137564, unicode_137565)
        
        # Testing the type of an if condition (line 214)
        if_condition_137567 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 214, 8), result_eq_137566)
        # Assigning a type to the variable 'if_condition_137567' (line 214)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 8), 'if_condition_137567', if_condition_137567)
        # SSA begins for if statement (line 214)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 215)
        # Processing the call arguments (line 215)
        
        # Obtaining the type of the subscript
        int_137570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 36), 'int')
        # Getting the type of 'self' (line 215)
        self_137571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 23), 'self', False)
        # Obtaining the member 'cursive' of a type (line 215)
        cursive_137572 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 23), self_137571, 'cursive')
        # Obtaining the member '__getitem__' of a type (line 215)
        getitem___137573 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 23), cursive_137572, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 215)
        subscript_call_result_137574 = invoke(stypy.reporting.localization.Localization(__file__, 215, 23), getitem___137573, int_137570)
        
        # Processing the call keyword arguments (line 215)
        kwargs_137575 = {}
        # Getting the type of 'cmd' (line 215)
        cmd_137568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 12), 'cmd', False)
        # Obtaining the member 'append' of a type (line 215)
        append_137569 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 12), cmd_137568, 'append')
        # Calling append(args, kwargs) (line 215)
        append_call_result_137576 = invoke(stypy.reporting.localization.Localization(__file__, 215, 12), append_137569, *[subscript_call_result_137574], **kwargs_137575)
        
        # SSA join for if statement (line 214)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        unicode_137577 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 14), 'unicode', u'\\usepackage{type1cm}')
        # Getting the type of 'cmd' (line 216)
        cmd_137578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 41), 'cmd')
        # Applying the binary operator 'in' (line 216)
        result_contains_137579 = python_operator(stypy.reporting.localization.Localization(__file__, 216, 14), 'in', unicode_137577, cmd_137578)
        
        # Testing the type of an if condition (line 216)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 216, 8), result_contains_137579)
        # SSA begins for while statement (line 216)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Call to remove(...): (line 217)
        # Processing the call arguments (line 217)
        unicode_137582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 23), 'unicode', u'\\usepackage{type1cm}')
        # Processing the call keyword arguments (line 217)
        kwargs_137583 = {}
        # Getting the type of 'cmd' (line 217)
        cmd_137580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 12), 'cmd', False)
        # Obtaining the member 'remove' of a type (line 217)
        remove_137581 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 12), cmd_137580, 'remove')
        # Calling remove(args, kwargs) (line 217)
        remove_call_result_137584 = invoke(stypy.reporting.localization.Localization(__file__, 217, 12), remove_137581, *[unicode_137582], **kwargs_137583)
        
        # SSA join for while statement (line 216)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 218):
        
        # Assigning a Call to a Name (line 218):
        
        # Call to join(...): (line 218)
        # Processing the call arguments (line 218)
        # Getting the type of 'cmd' (line 218)
        cmd_137587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 24), 'cmd', False)
        # Processing the call keyword arguments (line 218)
        kwargs_137588 = {}
        unicode_137585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 14), 'unicode', u'\n')
        # Obtaining the member 'join' of a type (line 218)
        join_137586 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 14), unicode_137585, 'join')
        # Calling join(args, kwargs) (line 218)
        join_call_result_137589 = invoke(stypy.reporting.localization.Localization(__file__, 218, 14), join_137586, *[cmd_137587], **kwargs_137588)
        
        # Assigning a type to the variable 'cmd' (line 218)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 8), 'cmd', join_call_result_137589)
        
        # Assigning a Call to a Attribute (line 219):
        
        # Assigning a Call to a Attribute (line 219):
        
        # Call to join(...): (line 219)
        # Processing the call arguments (line 219)
        
        # Obtaining an instance of the builtin type 'list' (line 219)
        list_137592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 40), 'list')
        # Adding type elements to the builtin type 'list' instance (line 219)
        # Adding element type (line 219)
        unicode_137593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 41), 'unicode', u'\\usepackage{type1cm}')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 219, 40), list_137592, unicode_137593)
        # Adding element type (line 219)
        # Getting the type of 'cmd' (line 219)
        cmd_137594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 66), 'cmd', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 219, 40), list_137592, cmd_137594)
        # Adding element type (line 219)
        unicode_137595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 41), 'unicode', u'\\usepackage{textcomp}')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 219, 40), list_137592, unicode_137595)
        
        # Processing the call keyword arguments (line 219)
        kwargs_137596 = {}
        unicode_137590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 30), 'unicode', u'\n')
        # Obtaining the member 'join' of a type (line 219)
        join_137591 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 30), unicode_137590, 'join')
        # Calling join(args, kwargs) (line 219)
        join_call_result_137597 = invoke(stypy.reporting.localization.Localization(__file__, 219, 30), join_137591, *[list_137592], **kwargs_137596)
        
        # Getting the type of 'self' (line 219)
        self_137598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 8), 'self')
        # Setting the type of the member '_font_preamble' of a type (line 219)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 8), self_137598, '_font_preamble', join_call_result_137597)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def get_basefile(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 222)
        None_137599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 46), 'None')
        defaults = [None_137599]
        # Create a new context for function 'get_basefile'
        module_type_store = module_type_store.open_function_context('get_basefile', 222, 4, False)
        # Assigning a type to the variable 'self' (line 223)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TexManager.get_basefile.__dict__.__setitem__('stypy_localization', localization)
        TexManager.get_basefile.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TexManager.get_basefile.__dict__.__setitem__('stypy_type_store', module_type_store)
        TexManager.get_basefile.__dict__.__setitem__('stypy_function_name', 'TexManager.get_basefile')
        TexManager.get_basefile.__dict__.__setitem__('stypy_param_names_list', ['tex', 'fontsize', 'dpi'])
        TexManager.get_basefile.__dict__.__setitem__('stypy_varargs_param_name', None)
        TexManager.get_basefile.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TexManager.get_basefile.__dict__.__setitem__('stypy_call_defaults', defaults)
        TexManager.get_basefile.__dict__.__setitem__('stypy_call_varargs', varargs)
        TexManager.get_basefile.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TexManager.get_basefile.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TexManager.get_basefile', ['tex', 'fontsize', 'dpi'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_basefile', localization, ['tex', 'fontsize', 'dpi'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_basefile(...)' code ##################

        unicode_137600 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, (-1)), 'unicode', u'\n        returns a filename based on a hash of the string, fontsize, and dpi\n        ')
        
        # Assigning a Call to a Name (line 226):
        
        # Assigning a Call to a Name (line 226):
        
        # Call to join(...): (line 226)
        # Processing the call arguments (line 226)
        
        # Obtaining an instance of the builtin type 'list' (line 226)
        list_137603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 226)
        # Adding element type (line 226)
        # Getting the type of 'tex' (line 226)
        tex_137604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 21), 'tex', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 226, 20), list_137603, tex_137604)
        # Adding element type (line 226)
        
        # Call to get_font_config(...): (line 226)
        # Processing the call keyword arguments (line 226)
        kwargs_137607 = {}
        # Getting the type of 'self' (line 226)
        self_137605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 26), 'self', False)
        # Obtaining the member 'get_font_config' of a type (line 226)
        get_font_config_137606 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 26), self_137605, 'get_font_config')
        # Calling get_font_config(args, kwargs) (line 226)
        get_font_config_call_result_137608 = invoke(stypy.reporting.localization.Localization(__file__, 226, 26), get_font_config_137606, *[], **kwargs_137607)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 226, 20), list_137603, get_font_config_call_result_137608)
        # Adding element type (line 226)
        unicode_137609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 50), 'unicode', u'%f')
        # Getting the type of 'fontsize' (line 226)
        fontsize_137610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 57), 'fontsize', False)
        # Applying the binary operator '%' (line 226)
        result_mod_137611 = python_operator(stypy.reporting.localization.Localization(__file__, 226, 50), '%', unicode_137609, fontsize_137610)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 226, 20), list_137603, result_mod_137611)
        # Adding element type (line 226)
        
        # Call to get_custom_preamble(...): (line 227)
        # Processing the call keyword arguments (line 227)
        kwargs_137614 = {}
        # Getting the type of 'self' (line 227)
        self_137612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 21), 'self', False)
        # Obtaining the member 'get_custom_preamble' of a type (line 227)
        get_custom_preamble_137613 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 21), self_137612, 'get_custom_preamble')
        # Calling get_custom_preamble(args, kwargs) (line 227)
        get_custom_preamble_call_result_137615 = invoke(stypy.reporting.localization.Localization(__file__, 227, 21), get_custom_preamble_137613, *[], **kwargs_137614)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 226, 20), list_137603, get_custom_preamble_call_result_137615)
        # Adding element type (line 226)
        
        # Call to str(...): (line 227)
        # Processing the call arguments (line 227)
        
        # Evaluating a boolean operation
        # Getting the type of 'dpi' (line 227)
        dpi_137617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 53), 'dpi', False)
        unicode_137618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 60), 'unicode', u'')
        # Applying the binary operator 'or' (line 227)
        result_or_keyword_137619 = python_operator(stypy.reporting.localization.Localization(__file__, 227, 53), 'or', dpi_137617, unicode_137618)
        
        # Processing the call keyword arguments (line 227)
        kwargs_137620 = {}
        # Getting the type of 'str' (line 227)
        str_137616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 49), 'str', False)
        # Calling str(args, kwargs) (line 227)
        str_call_result_137621 = invoke(stypy.reporting.localization.Localization(__file__, 227, 49), str_137616, *[result_or_keyword_137619], **kwargs_137620)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 226, 20), list_137603, str_call_result_137621)
        
        # Processing the call keyword arguments (line 226)
        kwargs_137622 = {}
        unicode_137601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 12), 'unicode', u'')
        # Obtaining the member 'join' of a type (line 226)
        join_137602 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 12), unicode_137601, 'join')
        # Calling join(args, kwargs) (line 226)
        join_call_result_137623 = invoke(stypy.reporting.localization.Localization(__file__, 226, 12), join_137602, *[list_137603], **kwargs_137622)
        
        # Assigning a type to the variable 's' (line 226)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 's', join_call_result_137623)
        
        # Assigning a Call to a Name (line 229):
        
        # Assigning a Call to a Name (line 229):
        
        # Call to encode(...): (line 229)
        # Processing the call arguments (line 229)
        unicode_137630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 40), 'unicode', u'utf-8')
        # Processing the call keyword arguments (line 229)
        kwargs_137631 = {}
        
        # Call to text_type(...): (line 229)
        # Processing the call arguments (line 229)
        # Getting the type of 's' (line 229)
        s_137626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 30), 's', False)
        # Processing the call keyword arguments (line 229)
        kwargs_137627 = {}
        # Getting the type of 'six' (line 229)
        six_137624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 16), 'six', False)
        # Obtaining the member 'text_type' of a type (line 229)
        text_type_137625 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 16), six_137624, 'text_type')
        # Calling text_type(args, kwargs) (line 229)
        text_type_call_result_137628 = invoke(stypy.reporting.localization.Localization(__file__, 229, 16), text_type_137625, *[s_137626], **kwargs_137627)
        
        # Obtaining the member 'encode' of a type (line 229)
        encode_137629 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 16), text_type_call_result_137628, 'encode')
        # Calling encode(args, kwargs) (line 229)
        encode_call_result_137632 = invoke(stypy.reporting.localization.Localization(__file__, 229, 16), encode_137629, *[unicode_137630], **kwargs_137631)
        
        # Assigning a type to the variable 'bytes' (line 229)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 8), 'bytes', encode_call_result_137632)
        
        # Call to join(...): (line 230)
        # Processing the call arguments (line 230)
        # Getting the type of 'self' (line 230)
        self_137636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 28), 'self', False)
        # Obtaining the member 'texcache' of a type (line 230)
        texcache_137637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 28), self_137636, 'texcache')
        
        # Call to hexdigest(...): (line 230)
        # Processing the call keyword arguments (line 230)
        kwargs_137643 = {}
        
        # Call to md5(...): (line 230)
        # Processing the call arguments (line 230)
        # Getting the type of 'bytes' (line 230)
        bytes_137639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 47), 'bytes', False)
        # Processing the call keyword arguments (line 230)
        kwargs_137640 = {}
        # Getting the type of 'md5' (line 230)
        md5_137638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 43), 'md5', False)
        # Calling md5(args, kwargs) (line 230)
        md5_call_result_137641 = invoke(stypy.reporting.localization.Localization(__file__, 230, 43), md5_137638, *[bytes_137639], **kwargs_137640)
        
        # Obtaining the member 'hexdigest' of a type (line 230)
        hexdigest_137642 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 43), md5_call_result_137641, 'hexdigest')
        # Calling hexdigest(args, kwargs) (line 230)
        hexdigest_call_result_137644 = invoke(stypy.reporting.localization.Localization(__file__, 230, 43), hexdigest_137642, *[], **kwargs_137643)
        
        # Processing the call keyword arguments (line 230)
        kwargs_137645 = {}
        # Getting the type of 'os' (line 230)
        os_137633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 15), 'os', False)
        # Obtaining the member 'path' of a type (line 230)
        path_137634 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 15), os_137633, 'path')
        # Obtaining the member 'join' of a type (line 230)
        join_137635 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 15), path_137634, 'join')
        # Calling join(args, kwargs) (line 230)
        join_call_result_137646 = invoke(stypy.reporting.localization.Localization(__file__, 230, 15), join_137635, *[texcache_137637, hexdigest_call_result_137644], **kwargs_137645)
        
        # Assigning a type to the variable 'stypy_return_type' (line 230)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 8), 'stypy_return_type', join_call_result_137646)
        
        # ################# End of 'get_basefile(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_basefile' in the type store
        # Getting the type of 'stypy_return_type' (line 222)
        stypy_return_type_137647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_137647)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_basefile'
        return stypy_return_type_137647


    @norecursion
    def get_font_config(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_font_config'
        module_type_store = module_type_store.open_function_context('get_font_config', 232, 4, False)
        # Assigning a type to the variable 'self' (line 233)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TexManager.get_font_config.__dict__.__setitem__('stypy_localization', localization)
        TexManager.get_font_config.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TexManager.get_font_config.__dict__.__setitem__('stypy_type_store', module_type_store)
        TexManager.get_font_config.__dict__.__setitem__('stypy_function_name', 'TexManager.get_font_config')
        TexManager.get_font_config.__dict__.__setitem__('stypy_param_names_list', [])
        TexManager.get_font_config.__dict__.__setitem__('stypy_varargs_param_name', None)
        TexManager.get_font_config.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TexManager.get_font_config.__dict__.__setitem__('stypy_call_defaults', defaults)
        TexManager.get_font_config.__dict__.__setitem__('stypy_call_varargs', varargs)
        TexManager.get_font_config.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TexManager.get_font_config.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TexManager.get_font_config', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_font_config', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_font_config(...)' code ##################

        unicode_137648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 8), 'unicode', u'Reinitializes self if relevant rcParams on have changed.')
        
        # Type idiom detected: calculating its left and rigth part (line 234)
        # Getting the type of 'self' (line 234)
        self_137649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 11), 'self')
        # Obtaining the member '_rc_cache' of a type (line 234)
        _rc_cache_137650 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 234, 11), self_137649, '_rc_cache')
        # Getting the type of 'None' (line 234)
        None_137651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 29), 'None')
        
        (may_be_137652, more_types_in_union_137653) = may_be_none(_rc_cache_137650, None_137651)

        if may_be_137652:

            if more_types_in_union_137653:
                # Runtime conditional SSA (line 234)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Attribute (line 235):
            
            # Assigning a Call to a Attribute (line 235):
            
            # Call to fromkeys(...): (line 235)
            # Processing the call arguments (line 235)
            # Getting the type of 'self' (line 235)
            self_137656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 43), 'self', False)
            # Obtaining the member '_rc_cache_keys' of a type (line 235)
            _rc_cache_keys_137657 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 43), self_137656, '_rc_cache_keys')
            # Processing the call keyword arguments (line 235)
            kwargs_137658 = {}
            # Getting the type of 'dict' (line 235)
            dict_137654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 29), 'dict', False)
            # Obtaining the member 'fromkeys' of a type (line 235)
            fromkeys_137655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 29), dict_137654, 'fromkeys')
            # Calling fromkeys(args, kwargs) (line 235)
            fromkeys_call_result_137659 = invoke(stypy.reporting.localization.Localization(__file__, 235, 29), fromkeys_137655, *[_rc_cache_keys_137657], **kwargs_137658)
            
            # Getting the type of 'self' (line 235)
            self_137660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 12), 'self')
            # Setting the type of the member '_rc_cache' of a type (line 235)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 12), self_137660, '_rc_cache', fromkeys_call_result_137659)

            if more_types_in_union_137653:
                # SSA join for if statement (line 234)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a ListComp to a Name (line 236):
        
        # Assigning a ListComp to a Name (line 236):
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'self' (line 236)
        self_137672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 34), 'self')
        # Obtaining the member '_rc_cache_keys' of a type (line 236)
        _rc_cache_keys_137673 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 34), self_137672, '_rc_cache_keys')
        comprehension_137674 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 236, 19), _rc_cache_keys_137673)
        # Assigning a type to the variable 'par' (line 236)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 19), 'par', comprehension_137674)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'par' (line 237)
        par_137662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 31), 'par')
        # Getting the type of 'rcParams' (line 237)
        rcParams_137663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 22), 'rcParams')
        # Obtaining the member '__getitem__' of a type (line 237)
        getitem___137664 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 22), rcParams_137663, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 237)
        subscript_call_result_137665 = invoke(stypy.reporting.localization.Localization(__file__, 237, 22), getitem___137664, par_137662)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'par' (line 237)
        par_137666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 54), 'par')
        # Getting the type of 'self' (line 237)
        self_137667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 39), 'self')
        # Obtaining the member '_rc_cache' of a type (line 237)
        _rc_cache_137668 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 39), self_137667, '_rc_cache')
        # Obtaining the member '__getitem__' of a type (line 237)
        getitem___137669 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 39), _rc_cache_137668, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 237)
        subscript_call_result_137670 = invoke(stypy.reporting.localization.Localization(__file__, 237, 39), getitem___137669, par_137666)
        
        # Applying the binary operator '!=' (line 237)
        result_ne_137671 = python_operator(stypy.reporting.localization.Localization(__file__, 237, 22), '!=', subscript_call_result_137665, subscript_call_result_137670)
        
        # Getting the type of 'par' (line 236)
        par_137661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 19), 'par')
        list_137675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 19), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 236, 19), list_137675, par_137661)
        # Assigning a type to the variable 'changed' (line 236)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 8), 'changed', list_137675)
        
        # Getting the type of 'changed' (line 238)
        changed_137676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 11), 'changed')
        # Testing the type of an if condition (line 238)
        if_condition_137677 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 238, 8), changed_137676)
        # Assigning a type to the variable 'if_condition_137677' (line 238)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 8), 'if_condition_137677', if_condition_137677)
        # SSA begins for if statement (line 238)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'DEBUG' (line 239)
        DEBUG_137678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 15), 'DEBUG')
        # Testing the type of an if condition (line 239)
        if_condition_137679 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 239, 12), DEBUG_137678)
        # Assigning a type to the variable 'if_condition_137679' (line 239)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 12), 'if_condition_137679', if_condition_137679)
        # SSA begins for if statement (line 239)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to print(...): (line 240)
        # Processing the call arguments (line 240)
        unicode_137681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 22), 'unicode', u'DEBUG following keys changed:')
        # Getting the type of 'changed' (line 240)
        changed_137682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 55), 'changed', False)
        # Processing the call keyword arguments (line 240)
        kwargs_137683 = {}
        # Getting the type of 'print' (line 240)
        print_137680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 16), 'print', False)
        # Calling print(args, kwargs) (line 240)
        print_call_result_137684 = invoke(stypy.reporting.localization.Localization(__file__, 240, 16), print_137680, *[unicode_137681, changed_137682], **kwargs_137683)
        
        # SSA join for if statement (line 239)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'changed' (line 241)
        changed_137685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 21), 'changed')
        # Testing the type of a for loop iterable (line 241)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 241, 12), changed_137685)
        # Getting the type of the for loop variable (line 241)
        for_loop_var_137686 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 241, 12), changed_137685)
        # Assigning a type to the variable 'k' (line 241)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 12), 'k', for_loop_var_137686)
        # SSA begins for a for statement (line 241)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'DEBUG' (line 242)
        DEBUG_137687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 19), 'DEBUG')
        # Testing the type of an if condition (line 242)
        if_condition_137688 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 242, 16), DEBUG_137687)
        # Assigning a type to the variable 'if_condition_137688' (line 242)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 16), 'if_condition_137688', if_condition_137688)
        # SSA begins for if statement (line 242)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to print(...): (line 243)
        # Processing the call arguments (line 243)
        unicode_137690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 26), 'unicode', u'DEBUG %-20s: %-10s -> %-10s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 244)
        tuple_137691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 27), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 244)
        # Adding element type (line 244)
        # Getting the type of 'k' (line 244)
        k_137692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 27), 'k', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 244, 27), tuple_137691, k_137692)
        # Adding element type (line 244)
        
        # Obtaining the type of the subscript
        # Getting the type of 'k' (line 244)
        k_137693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 45), 'k', False)
        # Getting the type of 'self' (line 244)
        self_137694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 30), 'self', False)
        # Obtaining the member '_rc_cache' of a type (line 244)
        _rc_cache_137695 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 30), self_137694, '_rc_cache')
        # Obtaining the member '__getitem__' of a type (line 244)
        getitem___137696 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 30), _rc_cache_137695, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 244)
        subscript_call_result_137697 = invoke(stypy.reporting.localization.Localization(__file__, 244, 30), getitem___137696, k_137693)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 244, 27), tuple_137691, subscript_call_result_137697)
        # Adding element type (line 244)
        
        # Obtaining the type of the subscript
        # Getting the type of 'k' (line 244)
        k_137698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 58), 'k', False)
        # Getting the type of 'rcParams' (line 244)
        rcParams_137699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 49), 'rcParams', False)
        # Obtaining the member '__getitem__' of a type (line 244)
        getitem___137700 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 49), rcParams_137699, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 244)
        subscript_call_result_137701 = invoke(stypy.reporting.localization.Localization(__file__, 244, 49), getitem___137700, k_137698)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 244, 27), tuple_137691, subscript_call_result_137701)
        
        # Applying the binary operator '%' (line 243)
        result_mod_137702 = python_operator(stypy.reporting.localization.Localization(__file__, 243, 26), '%', unicode_137690, tuple_137691)
        
        # Processing the call keyword arguments (line 243)
        kwargs_137703 = {}
        # Getting the type of 'print' (line 243)
        print_137689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 20), 'print', False)
        # Calling print(args, kwargs) (line 243)
        print_call_result_137704 = invoke(stypy.reporting.localization.Localization(__file__, 243, 20), print_137689, *[result_mod_137702], **kwargs_137703)
        
        # SSA join for if statement (line 242)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Subscript (line 246):
        
        # Assigning a Call to a Subscript (line 246):
        
        # Call to deepcopy(...): (line 246)
        # Processing the call arguments (line 246)
        
        # Obtaining the type of the subscript
        # Getting the type of 'k' (line 246)
        k_137707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 59), 'k', False)
        # Getting the type of 'rcParams' (line 246)
        rcParams_137708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 50), 'rcParams', False)
        # Obtaining the member '__getitem__' of a type (line 246)
        getitem___137709 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 50), rcParams_137708, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 246)
        subscript_call_result_137710 = invoke(stypy.reporting.localization.Localization(__file__, 246, 50), getitem___137709, k_137707)
        
        # Processing the call keyword arguments (line 246)
        kwargs_137711 = {}
        # Getting the type of 'copy' (line 246)
        copy_137705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 36), 'copy', False)
        # Obtaining the member 'deepcopy' of a type (line 246)
        deepcopy_137706 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 36), copy_137705, 'deepcopy')
        # Calling deepcopy(args, kwargs) (line 246)
        deepcopy_call_result_137712 = invoke(stypy.reporting.localization.Localization(__file__, 246, 36), deepcopy_137706, *[subscript_call_result_137710], **kwargs_137711)
        
        # Getting the type of 'self' (line 246)
        self_137713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 16), 'self')
        # Obtaining the member '_rc_cache' of a type (line 246)
        _rc_cache_137714 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 16), self_137713, '_rc_cache')
        # Getting the type of 'k' (line 246)
        k_137715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 31), 'k')
        # Storing an element on a container (line 246)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 246, 16), _rc_cache_137714, (k_137715, deepcopy_call_result_137712))
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'DEBUG' (line 247)
        DEBUG_137716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 15), 'DEBUG')
        # Testing the type of an if condition (line 247)
        if_condition_137717 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 247, 12), DEBUG_137716)
        # Assigning a type to the variable 'if_condition_137717' (line 247)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 12), 'if_condition_137717', if_condition_137717)
        # SSA begins for if statement (line 247)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to print(...): (line 248)
        # Processing the call arguments (line 248)
        unicode_137719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 22), 'unicode', u'DEBUG RE-INIT\nold fontconfig:')
        # Getting the type of 'self' (line 248)
        self_137720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 56), 'self', False)
        # Obtaining the member '_fontconfig' of a type (line 248)
        _fontconfig_137721 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 56), self_137720, '_fontconfig')
        # Processing the call keyword arguments (line 248)
        kwargs_137722 = {}
        # Getting the type of 'print' (line 248)
        print_137718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 16), 'print', False)
        # Calling print(args, kwargs) (line 248)
        print_call_result_137723 = invoke(stypy.reporting.localization.Localization(__file__, 248, 16), print_137718, *[unicode_137719, _fontconfig_137721], **kwargs_137722)
        
        # SSA join for if statement (line 247)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to __init__(...): (line 249)
        # Processing the call keyword arguments (line 249)
        kwargs_137726 = {}
        # Getting the type of 'self' (line 249)
        self_137724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 12), 'self', False)
        # Obtaining the member '__init__' of a type (line 249)
        init___137725 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 12), self_137724, '__init__')
        # Calling __init__(args, kwargs) (line 249)
        init___call_result_137727 = invoke(stypy.reporting.localization.Localization(__file__, 249, 12), init___137725, *[], **kwargs_137726)
        
        # SSA join for if statement (line 238)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'DEBUG' (line 250)
        DEBUG_137728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 11), 'DEBUG')
        # Testing the type of an if condition (line 250)
        if_condition_137729 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 250, 8), DEBUG_137728)
        # Assigning a type to the variable 'if_condition_137729' (line 250)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 8), 'if_condition_137729', if_condition_137729)
        # SSA begins for if statement (line 250)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to print(...): (line 251)
        # Processing the call arguments (line 251)
        unicode_137731 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 18), 'unicode', u'DEBUG fontconfig:')
        # Getting the type of 'self' (line 251)
        self_137732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 39), 'self', False)
        # Obtaining the member '_fontconfig' of a type (line 251)
        _fontconfig_137733 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 39), self_137732, '_fontconfig')
        # Processing the call keyword arguments (line 251)
        kwargs_137734 = {}
        # Getting the type of 'print' (line 251)
        print_137730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 12), 'print', False)
        # Calling print(args, kwargs) (line 251)
        print_call_result_137735 = invoke(stypy.reporting.localization.Localization(__file__, 251, 12), print_137730, *[unicode_137731, _fontconfig_137733], **kwargs_137734)
        
        # SSA join for if statement (line 250)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'self' (line 252)
        self_137736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 15), 'self')
        # Obtaining the member '_fontconfig' of a type (line 252)
        _fontconfig_137737 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 15), self_137736, '_fontconfig')
        # Assigning a type to the variable 'stypy_return_type' (line 252)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 8), 'stypy_return_type', _fontconfig_137737)
        
        # ################# End of 'get_font_config(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_font_config' in the type store
        # Getting the type of 'stypy_return_type' (line 232)
        stypy_return_type_137738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_137738)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_font_config'
        return stypy_return_type_137738


    @norecursion
    def get_font_preamble(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_font_preamble'
        module_type_store = module_type_store.open_function_context('get_font_preamble', 254, 4, False)
        # Assigning a type to the variable 'self' (line 255)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TexManager.get_font_preamble.__dict__.__setitem__('stypy_localization', localization)
        TexManager.get_font_preamble.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TexManager.get_font_preamble.__dict__.__setitem__('stypy_type_store', module_type_store)
        TexManager.get_font_preamble.__dict__.__setitem__('stypy_function_name', 'TexManager.get_font_preamble')
        TexManager.get_font_preamble.__dict__.__setitem__('stypy_param_names_list', [])
        TexManager.get_font_preamble.__dict__.__setitem__('stypy_varargs_param_name', None)
        TexManager.get_font_preamble.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TexManager.get_font_preamble.__dict__.__setitem__('stypy_call_defaults', defaults)
        TexManager.get_font_preamble.__dict__.__setitem__('stypy_call_varargs', varargs)
        TexManager.get_font_preamble.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TexManager.get_font_preamble.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TexManager.get_font_preamble', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_font_preamble', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_font_preamble(...)' code ##################

        unicode_137739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, (-1)), 'unicode', u'\n        returns a string containing font configuration for the tex preamble\n        ')
        # Getting the type of 'self' (line 258)
        self_137740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 15), 'self')
        # Obtaining the member '_font_preamble' of a type (line 258)
        _font_preamble_137741 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 15), self_137740, '_font_preamble')
        # Assigning a type to the variable 'stypy_return_type' (line 258)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 8), 'stypy_return_type', _font_preamble_137741)
        
        # ################# End of 'get_font_preamble(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_font_preamble' in the type store
        # Getting the type of 'stypy_return_type' (line 254)
        stypy_return_type_137742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_137742)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_font_preamble'
        return stypy_return_type_137742


    @norecursion
    def get_custom_preamble(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_custom_preamble'
        module_type_store = module_type_store.open_function_context('get_custom_preamble', 260, 4, False)
        # Assigning a type to the variable 'self' (line 261)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TexManager.get_custom_preamble.__dict__.__setitem__('stypy_localization', localization)
        TexManager.get_custom_preamble.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TexManager.get_custom_preamble.__dict__.__setitem__('stypy_type_store', module_type_store)
        TexManager.get_custom_preamble.__dict__.__setitem__('stypy_function_name', 'TexManager.get_custom_preamble')
        TexManager.get_custom_preamble.__dict__.__setitem__('stypy_param_names_list', [])
        TexManager.get_custom_preamble.__dict__.__setitem__('stypy_varargs_param_name', None)
        TexManager.get_custom_preamble.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TexManager.get_custom_preamble.__dict__.__setitem__('stypy_call_defaults', defaults)
        TexManager.get_custom_preamble.__dict__.__setitem__('stypy_call_varargs', varargs)
        TexManager.get_custom_preamble.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TexManager.get_custom_preamble.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TexManager.get_custom_preamble', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_custom_preamble', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_custom_preamble(...)' code ##################

        unicode_137743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 8), 'unicode', u'returns a string containing user additions to the tex preamble')
        
        # Call to join(...): (line 262)
        # Processing the call arguments (line 262)
        
        # Obtaining the type of the subscript
        unicode_137746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 34), 'unicode', u'text.latex.preamble')
        # Getting the type of 'rcParams' (line 262)
        rcParams_137747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 25), 'rcParams', False)
        # Obtaining the member '__getitem__' of a type (line 262)
        getitem___137748 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 25), rcParams_137747, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 262)
        subscript_call_result_137749 = invoke(stypy.reporting.localization.Localization(__file__, 262, 25), getitem___137748, unicode_137746)
        
        # Processing the call keyword arguments (line 262)
        kwargs_137750 = {}
        unicode_137744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 15), 'unicode', u'\n')
        # Obtaining the member 'join' of a type (line 262)
        join_137745 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 15), unicode_137744, 'join')
        # Calling join(args, kwargs) (line 262)
        join_call_result_137751 = invoke(stypy.reporting.localization.Localization(__file__, 262, 15), join_137745, *[subscript_call_result_137749], **kwargs_137750)
        
        # Assigning a type to the variable 'stypy_return_type' (line 262)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 8), 'stypy_return_type', join_call_result_137751)
        
        # ################# End of 'get_custom_preamble(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_custom_preamble' in the type store
        # Getting the type of 'stypy_return_type' (line 260)
        stypy_return_type_137752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_137752)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_custom_preamble'
        return stypy_return_type_137752


    @norecursion
    def make_tex(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'make_tex'
        module_type_store = module_type_store.open_function_context('make_tex', 264, 4, False)
        # Assigning a type to the variable 'self' (line 265)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TexManager.make_tex.__dict__.__setitem__('stypy_localization', localization)
        TexManager.make_tex.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TexManager.make_tex.__dict__.__setitem__('stypy_type_store', module_type_store)
        TexManager.make_tex.__dict__.__setitem__('stypy_function_name', 'TexManager.make_tex')
        TexManager.make_tex.__dict__.__setitem__('stypy_param_names_list', ['tex', 'fontsize'])
        TexManager.make_tex.__dict__.__setitem__('stypy_varargs_param_name', None)
        TexManager.make_tex.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TexManager.make_tex.__dict__.__setitem__('stypy_call_defaults', defaults)
        TexManager.make_tex.__dict__.__setitem__('stypy_call_varargs', varargs)
        TexManager.make_tex.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TexManager.make_tex.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TexManager.make_tex', ['tex', 'fontsize'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'make_tex', localization, ['tex', 'fontsize'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'make_tex(...)' code ##################

        unicode_137753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, (-1)), 'unicode', u'\n        Generate a tex file to render the tex string at a specific font size\n\n        returns the file name\n        ')
        
        # Assigning a Call to a Name (line 270):
        
        # Assigning a Call to a Name (line 270):
        
        # Call to get_basefile(...): (line 270)
        # Processing the call arguments (line 270)
        # Getting the type of 'tex' (line 270)
        tex_137756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 37), 'tex', False)
        # Getting the type of 'fontsize' (line 270)
        fontsize_137757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 42), 'fontsize', False)
        # Processing the call keyword arguments (line 270)
        kwargs_137758 = {}
        # Getting the type of 'self' (line 270)
        self_137754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 19), 'self', False)
        # Obtaining the member 'get_basefile' of a type (line 270)
        get_basefile_137755 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 19), self_137754, 'get_basefile')
        # Calling get_basefile(args, kwargs) (line 270)
        get_basefile_call_result_137759 = invoke(stypy.reporting.localization.Localization(__file__, 270, 19), get_basefile_137755, *[tex_137756, fontsize_137757], **kwargs_137758)
        
        # Assigning a type to the variable 'basefile' (line 270)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 270, 8), 'basefile', get_basefile_call_result_137759)
        
        # Assigning a BinOp to a Name (line 271):
        
        # Assigning a BinOp to a Name (line 271):
        unicode_137760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 18), 'unicode', u'%s.tex')
        # Getting the type of 'basefile' (line 271)
        basefile_137761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 29), 'basefile')
        # Applying the binary operator '%' (line 271)
        result_mod_137762 = python_operator(stypy.reporting.localization.Localization(__file__, 271, 18), '%', unicode_137760, basefile_137761)
        
        # Assigning a type to the variable 'texfile' (line 271)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 8), 'texfile', result_mod_137762)
        
        # Assigning a Call to a Name (line 272):
        
        # Assigning a Call to a Name (line 272):
        
        # Call to get_custom_preamble(...): (line 272)
        # Processing the call keyword arguments (line 272)
        kwargs_137765 = {}
        # Getting the type of 'self' (line 272)
        self_137763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 26), 'self', False)
        # Obtaining the member 'get_custom_preamble' of a type (line 272)
        get_custom_preamble_137764 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 26), self_137763, 'get_custom_preamble')
        # Calling get_custom_preamble(args, kwargs) (line 272)
        get_custom_preamble_call_result_137766 = invoke(stypy.reporting.localization.Localization(__file__, 272, 26), get_custom_preamble_137764, *[], **kwargs_137765)
        
        # Assigning a type to the variable 'custom_preamble' (line 272)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 8), 'custom_preamble', get_custom_preamble_call_result_137766)
        
        # Assigning a Call to a Name (line 273):
        
        # Assigning a Call to a Name (line 273):
        
        # Call to get(...): (line 273)
        # Processing the call arguments (line 273)
        # Getting the type of 'self' (line 274)
        self_137773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 55), 'self', False)
        # Obtaining the member 'font_family' of a type (line 274)
        font_family_137774 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 55), self_137773, 'font_family')
        unicode_137775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 55), 'unicode', u'{\\rmfamily %s}')
        # Processing the call keyword arguments (line 273)
        kwargs_137776 = {}
        
        # Obtaining an instance of the builtin type 'dict' (line 273)
        dict_137767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 18), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 273)
        # Adding element type (key, value) (line 273)
        unicode_137768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 19), 'unicode', u'sans-serif')
        unicode_137769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 33), 'unicode', u'{\\sffamily %s}')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 273, 18), dict_137767, (unicode_137768, unicode_137769))
        # Adding element type (key, value) (line 273)
        unicode_137770 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 19), 'unicode', u'monospace')
        unicode_137771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 32), 'unicode', u'{\\ttfamily %s}')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 273, 18), dict_137767, (unicode_137770, unicode_137771))
        
        # Obtaining the member 'get' of a type (line 273)
        get_137772 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 18), dict_137767, 'get')
        # Calling get(args, kwargs) (line 273)
        get_call_result_137777 = invoke(stypy.reporting.localization.Localization(__file__, 273, 18), get_137772, *[font_family_137774, unicode_137775], **kwargs_137776)
        
        # Assigning a type to the variable 'fontcmd' (line 273)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 8), 'fontcmd', get_call_result_137777)
        
        # Assigning a BinOp to a Name (line 276):
        
        # Assigning a BinOp to a Name (line 276):
        # Getting the type of 'fontcmd' (line 276)
        fontcmd_137778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 14), 'fontcmd')
        # Getting the type of 'tex' (line 276)
        tex_137779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 24), 'tex')
        # Applying the binary operator '%' (line 276)
        result_mod_137780 = python_operator(stypy.reporting.localization.Localization(__file__, 276, 14), '%', fontcmd_137778, tex_137779)
        
        # Assigning a type to the variable 'tex' (line 276)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 8), 'tex', result_mod_137780)
        
        
        # Obtaining the type of the subscript
        unicode_137781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 20), 'unicode', u'text.latex.unicode')
        # Getting the type of 'rcParams' (line 278)
        rcParams_137782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 11), 'rcParams')
        # Obtaining the member '__getitem__' of a type (line 278)
        getitem___137783 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 11), rcParams_137782, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 278)
        subscript_call_result_137784 = invoke(stypy.reporting.localization.Localization(__file__, 278, 11), getitem___137783, unicode_137781)
        
        # Testing the type of an if condition (line 278)
        if_condition_137785 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 278, 8), subscript_call_result_137784)
        # Assigning a type to the variable 'if_condition_137785' (line 278)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 8), 'if_condition_137785', if_condition_137785)
        # SSA begins for if statement (line 278)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Str to a Name (line 279):
        
        # Assigning a Str to a Name (line 279):
        unicode_137786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, (-1)), 'unicode', u'\\usepackage{ucs}\n\\usepackage[utf8x]{inputenc}')
        # Assigning a type to the variable 'unicode_preamble' (line 279)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 12), 'unicode_preamble', unicode_137786)
        # SSA branch for the else part of an if statement (line 278)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Str to a Name (line 282):
        
        # Assigning a Str to a Name (line 282):
        unicode_137787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 31), 'unicode', u'')
        # Assigning a type to the variable 'unicode_preamble' (line 282)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 12), 'unicode_preamble', unicode_137787)
        # SSA join for if statement (line 278)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 284):
        
        # Assigning a BinOp to a Name (line 284):
        unicode_137788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, (-1)), 'unicode', u'\\documentclass{article}\n%s\n%s\n%s\n\\usepackage[papersize={72in,72in},body={70in,70in},margin={1in,1in}]{geometry}\n\\pagestyle{empty}\n\\begin{document}\n\\fontsize{%f}{%f}%s\n\\end{document}\n')
        
        # Obtaining an instance of the builtin type 'tuple' (line 293)
        tuple_137789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 7), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 293)
        # Adding element type (line 293)
        # Getting the type of 'self' (line 293)
        self_137790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 7), 'self')
        # Obtaining the member '_font_preamble' of a type (line 293)
        _font_preamble_137791 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 7), self_137790, '_font_preamble')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 293, 7), tuple_137789, _font_preamble_137791)
        # Adding element type (line 293)
        # Getting the type of 'unicode_preamble' (line 293)
        unicode_preamble_137792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 28), 'unicode_preamble')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 293, 7), tuple_137789, unicode_preamble_137792)
        # Adding element type (line 293)
        # Getting the type of 'custom_preamble' (line 293)
        custom_preamble_137793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 46), 'custom_preamble')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 293, 7), tuple_137789, custom_preamble_137793)
        # Adding element type (line 293)
        # Getting the type of 'fontsize' (line 294)
        fontsize_137794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 7), 'fontsize')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 293, 7), tuple_137789, fontsize_137794)
        # Adding element type (line 293)
        # Getting the type of 'fontsize' (line 294)
        fontsize_137795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 17), 'fontsize')
        float_137796 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 28), 'float')
        # Applying the binary operator '*' (line 294)
        result_mul_137797 = python_operator(stypy.reporting.localization.Localization(__file__, 294, 17), '*', fontsize_137795, float_137796)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 293, 7), tuple_137789, result_mul_137797)
        # Adding element type (line 293)
        # Getting the type of 'tex' (line 294)
        tex_137798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 34), 'tex')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 293, 7), tuple_137789, tex_137798)
        
        # Applying the binary operator '%' (line 293)
        result_mod_137799 = python_operator(stypy.reporting.localization.Localization(__file__, 293, (-1)), '%', unicode_137788, tuple_137789)
        
        # Assigning a type to the variable 's' (line 284)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 8), 's', result_mod_137799)
        
        # Call to open(...): (line 295)
        # Processing the call arguments (line 295)
        # Getting the type of 'texfile' (line 295)
        texfile_137801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 18), 'texfile', False)
        unicode_137802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 27), 'unicode', u'wb')
        # Processing the call keyword arguments (line 295)
        kwargs_137803 = {}
        # Getting the type of 'open' (line 295)
        open_137800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 13), 'open', False)
        # Calling open(args, kwargs) (line 295)
        open_call_result_137804 = invoke(stypy.reporting.localization.Localization(__file__, 295, 13), open_137800, *[texfile_137801, unicode_137802], **kwargs_137803)
        
        with_137805 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 295, 13), open_call_result_137804, 'with parameter', '__enter__', '__exit__')

        if with_137805:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 295)
            enter___137806 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 13), open_call_result_137804, '__enter__')
            with_enter_137807 = invoke(stypy.reporting.localization.Localization(__file__, 295, 13), enter___137806)
            # Assigning a type to the variable 'fh' (line 295)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 13), 'fh', with_enter_137807)
            
            
            # Obtaining the type of the subscript
            unicode_137808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 24), 'unicode', u'text.latex.unicode')
            # Getting the type of 'rcParams' (line 296)
            rcParams_137809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 15), 'rcParams')
            # Obtaining the member '__getitem__' of a type (line 296)
            getitem___137810 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 15), rcParams_137809, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 296)
            subscript_call_result_137811 = invoke(stypy.reporting.localization.Localization(__file__, 296, 15), getitem___137810, unicode_137808)
            
            # Testing the type of an if condition (line 296)
            if_condition_137812 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 296, 12), subscript_call_result_137811)
            # Assigning a type to the variable 'if_condition_137812' (line 296)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 296, 12), 'if_condition_137812', if_condition_137812)
            # SSA begins for if statement (line 296)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to write(...): (line 297)
            # Processing the call arguments (line 297)
            
            # Call to encode(...): (line 297)
            # Processing the call arguments (line 297)
            unicode_137817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, 34), 'unicode', u'utf8')
            # Processing the call keyword arguments (line 297)
            kwargs_137818 = {}
            # Getting the type of 's' (line 297)
            s_137815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 25), 's', False)
            # Obtaining the member 'encode' of a type (line 297)
            encode_137816 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 297, 25), s_137815, 'encode')
            # Calling encode(args, kwargs) (line 297)
            encode_call_result_137819 = invoke(stypy.reporting.localization.Localization(__file__, 297, 25), encode_137816, *[unicode_137817], **kwargs_137818)
            
            # Processing the call keyword arguments (line 297)
            kwargs_137820 = {}
            # Getting the type of 'fh' (line 297)
            fh_137813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 16), 'fh', False)
            # Obtaining the member 'write' of a type (line 297)
            write_137814 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 297, 16), fh_137813, 'write')
            # Calling write(args, kwargs) (line 297)
            write_call_result_137821 = invoke(stypy.reporting.localization.Localization(__file__, 297, 16), write_137814, *[encode_call_result_137819], **kwargs_137820)
            
            # SSA branch for the else part of an if statement (line 296)
            module_type_store.open_ssa_branch('else')
            
            
            # SSA begins for try-except statement (line 299)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
            
            # Call to write(...): (line 300)
            # Processing the call arguments (line 300)
            
            # Call to encode(...): (line 300)
            # Processing the call arguments (line 300)
            unicode_137826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, 38), 'unicode', u'ascii')
            # Processing the call keyword arguments (line 300)
            kwargs_137827 = {}
            # Getting the type of 's' (line 300)
            s_137824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 29), 's', False)
            # Obtaining the member 'encode' of a type (line 300)
            encode_137825 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 29), s_137824, 'encode')
            # Calling encode(args, kwargs) (line 300)
            encode_call_result_137828 = invoke(stypy.reporting.localization.Localization(__file__, 300, 29), encode_137825, *[unicode_137826], **kwargs_137827)
            
            # Processing the call keyword arguments (line 300)
            kwargs_137829 = {}
            # Getting the type of 'fh' (line 300)
            fh_137822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 20), 'fh', False)
            # Obtaining the member 'write' of a type (line 300)
            write_137823 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 20), fh_137822, 'write')
            # Calling write(args, kwargs) (line 300)
            write_call_result_137830 = invoke(stypy.reporting.localization.Localization(__file__, 300, 20), write_137823, *[encode_call_result_137828], **kwargs_137829)
            
            # SSA branch for the except part of a try statement (line 299)
            # SSA branch for the except 'UnicodeEncodeError' branch of a try statement (line 299)
            # Storing handler type
            module_type_store.open_ssa_branch('except')
            # Getting the type of 'UnicodeEncodeError' (line 301)
            UnicodeEncodeError_137831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 23), 'UnicodeEncodeError')
            # Assigning a type to the variable 'err' (line 301)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 16), 'err', UnicodeEncodeError_137831)
            
            # Call to report(...): (line 302)
            # Processing the call arguments (line 302)
            unicode_137835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 39), 'unicode', u"You are using unicode and latex, but have not enabled the matplotlib 'text.latex.unicode' rcParam.")
            unicode_137836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 39), 'unicode', u'helpful')
            # Processing the call keyword arguments (line 302)
            kwargs_137837 = {}
            # Getting the type of 'mpl' (line 302)
            mpl_137832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 20), 'mpl', False)
            # Obtaining the member 'verbose' of a type (line 302)
            verbose_137833 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 20), mpl_137832, 'verbose')
            # Obtaining the member 'report' of a type (line 302)
            report_137834 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 20), verbose_137833, 'report')
            # Calling report(args, kwargs) (line 302)
            report_call_result_137838 = invoke(stypy.reporting.localization.Localization(__file__, 302, 20), report_137834, *[unicode_137835, unicode_137836], **kwargs_137837)
            
            # SSA join for try-except statement (line 299)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 296)
            module_type_store = module_type_store.join_ssa_context()
            
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 295)
            exit___137839 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 13), open_call_result_137804, '__exit__')
            with_exit_137840 = invoke(stypy.reporting.localization.Localization(__file__, 295, 13), exit___137839, None, None, None)

        # Getting the type of 'texfile' (line 308)
        texfile_137841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 15), 'texfile')
        # Assigning a type to the variable 'stypy_return_type' (line 308)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 308, 8), 'stypy_return_type', texfile_137841)
        
        # ################# End of 'make_tex(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'make_tex' in the type store
        # Getting the type of 'stypy_return_type' (line 264)
        stypy_return_type_137842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_137842)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'make_tex'
        return stypy_return_type_137842

    
    # Assigning a Call to a Name (line 310):

    @norecursion
    def make_tex_preview(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'make_tex_preview'
        module_type_store = module_type_store.open_function_context('make_tex_preview', 313, 4, False)
        # Assigning a type to the variable 'self' (line 314)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 314, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TexManager.make_tex_preview.__dict__.__setitem__('stypy_localization', localization)
        TexManager.make_tex_preview.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TexManager.make_tex_preview.__dict__.__setitem__('stypy_type_store', module_type_store)
        TexManager.make_tex_preview.__dict__.__setitem__('stypy_function_name', 'TexManager.make_tex_preview')
        TexManager.make_tex_preview.__dict__.__setitem__('stypy_param_names_list', ['tex', 'fontsize'])
        TexManager.make_tex_preview.__dict__.__setitem__('stypy_varargs_param_name', None)
        TexManager.make_tex_preview.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TexManager.make_tex_preview.__dict__.__setitem__('stypy_call_defaults', defaults)
        TexManager.make_tex_preview.__dict__.__setitem__('stypy_call_varargs', varargs)
        TexManager.make_tex_preview.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TexManager.make_tex_preview.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TexManager.make_tex_preview', ['tex', 'fontsize'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'make_tex_preview', localization, ['tex', 'fontsize'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'make_tex_preview(...)' code ##################

        unicode_137843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, (-1)), 'unicode', u'\n        Generate a tex file to render the tex string at a specific\n        font size. It uses the preview.sty to determine the dimension\n        (width, height, descent) of the output.\n\n        returns the file name\n        ')
        
        # Assigning a Call to a Name (line 321):
        
        # Assigning a Call to a Name (line 321):
        
        # Call to get_basefile(...): (line 321)
        # Processing the call arguments (line 321)
        # Getting the type of 'tex' (line 321)
        tex_137846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 37), 'tex', False)
        # Getting the type of 'fontsize' (line 321)
        fontsize_137847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 42), 'fontsize', False)
        # Processing the call keyword arguments (line 321)
        kwargs_137848 = {}
        # Getting the type of 'self' (line 321)
        self_137844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 19), 'self', False)
        # Obtaining the member 'get_basefile' of a type (line 321)
        get_basefile_137845 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 321, 19), self_137844, 'get_basefile')
        # Calling get_basefile(args, kwargs) (line 321)
        get_basefile_call_result_137849 = invoke(stypy.reporting.localization.Localization(__file__, 321, 19), get_basefile_137845, *[tex_137846, fontsize_137847], **kwargs_137848)
        
        # Assigning a type to the variable 'basefile' (line 321)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 321, 8), 'basefile', get_basefile_call_result_137849)
        
        # Assigning a BinOp to a Name (line 322):
        
        # Assigning a BinOp to a Name (line 322):
        unicode_137850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 18), 'unicode', u'%s.tex')
        # Getting the type of 'basefile' (line 322)
        basefile_137851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 29), 'basefile')
        # Applying the binary operator '%' (line 322)
        result_mod_137852 = python_operator(stypy.reporting.localization.Localization(__file__, 322, 18), '%', unicode_137850, basefile_137851)
        
        # Assigning a type to the variable 'texfile' (line 322)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 322, 8), 'texfile', result_mod_137852)
        
        # Assigning a Call to a Name (line 323):
        
        # Assigning a Call to a Name (line 323):
        
        # Call to get_custom_preamble(...): (line 323)
        # Processing the call keyword arguments (line 323)
        kwargs_137855 = {}
        # Getting the type of 'self' (line 323)
        self_137853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 26), 'self', False)
        # Obtaining the member 'get_custom_preamble' of a type (line 323)
        get_custom_preamble_137854 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 323, 26), self_137853, 'get_custom_preamble')
        # Calling get_custom_preamble(args, kwargs) (line 323)
        get_custom_preamble_call_result_137856 = invoke(stypy.reporting.localization.Localization(__file__, 323, 26), get_custom_preamble_137854, *[], **kwargs_137855)
        
        # Assigning a type to the variable 'custom_preamble' (line 323)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 8), 'custom_preamble', get_custom_preamble_call_result_137856)
        
        # Assigning a Call to a Name (line 324):
        
        # Assigning a Call to a Name (line 324):
        
        # Call to get(...): (line 324)
        # Processing the call arguments (line 324)
        # Getting the type of 'self' (line 325)
        self_137863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 55), 'self', False)
        # Obtaining the member 'font_family' of a type (line 325)
        font_family_137864 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 325, 55), self_137863, 'font_family')
        unicode_137865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, 55), 'unicode', u'{\\rmfamily %s}')
        # Processing the call keyword arguments (line 324)
        kwargs_137866 = {}
        
        # Obtaining an instance of the builtin type 'dict' (line 324)
        dict_137857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 18), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 324)
        # Adding element type (key, value) (line 324)
        unicode_137858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 19), 'unicode', u'sans-serif')
        unicode_137859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 33), 'unicode', u'{\\sffamily %s}')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 324, 18), dict_137857, (unicode_137858, unicode_137859))
        # Adding element type (key, value) (line 324)
        unicode_137860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 19), 'unicode', u'monospace')
        unicode_137861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 32), 'unicode', u'{\\ttfamily %s}')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 324, 18), dict_137857, (unicode_137860, unicode_137861))
        
        # Obtaining the member 'get' of a type (line 324)
        get_137862 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 324, 18), dict_137857, 'get')
        # Calling get(args, kwargs) (line 324)
        get_call_result_137867 = invoke(stypy.reporting.localization.Localization(__file__, 324, 18), get_137862, *[font_family_137864, unicode_137865], **kwargs_137866)
        
        # Assigning a type to the variable 'fontcmd' (line 324)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 324, 8), 'fontcmd', get_call_result_137867)
        
        # Assigning a BinOp to a Name (line 327):
        
        # Assigning a BinOp to a Name (line 327):
        # Getting the type of 'fontcmd' (line 327)
        fontcmd_137868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 14), 'fontcmd')
        # Getting the type of 'tex' (line 327)
        tex_137869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 24), 'tex')
        # Applying the binary operator '%' (line 327)
        result_mod_137870 = python_operator(stypy.reporting.localization.Localization(__file__, 327, 14), '%', fontcmd_137868, tex_137869)
        
        # Assigning a type to the variable 'tex' (line 327)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 327, 8), 'tex', result_mod_137870)
        
        
        # Obtaining the type of the subscript
        unicode_137871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 20), 'unicode', u'text.latex.unicode')
        # Getting the type of 'rcParams' (line 329)
        rcParams_137872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 11), 'rcParams')
        # Obtaining the member '__getitem__' of a type (line 329)
        getitem___137873 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 11), rcParams_137872, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 329)
        subscript_call_result_137874 = invoke(stypy.reporting.localization.Localization(__file__, 329, 11), getitem___137873, unicode_137871)
        
        # Testing the type of an if condition (line 329)
        if_condition_137875 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 329, 8), subscript_call_result_137874)
        # Assigning a type to the variable 'if_condition_137875' (line 329)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 8), 'if_condition_137875', if_condition_137875)
        # SSA begins for if statement (line 329)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Str to a Name (line 330):
        
        # Assigning a Str to a Name (line 330):
        unicode_137876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, (-1)), 'unicode', u'\\usepackage{ucs}\n\\usepackage[utf8x]{inputenc}')
        # Assigning a type to the variable 'unicode_preamble' (line 330)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 12), 'unicode_preamble', unicode_137876)
        # SSA branch for the else part of an if statement (line 329)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Str to a Name (line 333):
        
        # Assigning a Str to a Name (line 333):
        unicode_137877 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 31), 'unicode', u'')
        # Assigning a type to the variable 'unicode_preamble' (line 333)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 333, 12), 'unicode_preamble', unicode_137877)
        # SSA join for if statement (line 329)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 338):
        
        # Assigning a BinOp to a Name (line 338):
        unicode_137878 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, (-1)), 'unicode', u'\\documentclass{article}\n%s\n%s\n%s\n\\usepackage[active,showbox,tightpage]{preview}\n\\usepackage[papersize={72in,72in},body={70in,70in},margin={1in,1in}]{geometry}\n\n%% we override the default showbox as it is treated as an error and makes\n%% the exit status not zero\n\\def\\showbox#1{\\immediate\\write16{MatplotlibBox:(\\the\\ht#1+\\the\\dp#1)x\\the\\wd#1}}\n\n\\begin{document}\n\\begin{preview}\n{\\fontsize{%f}{%f}%s}\n\\end{preview}\n\\end{document}\n')
        
        # Obtaining an instance of the builtin type 'tuple' (line 354)
        tuple_137879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 7), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 354)
        # Adding element type (line 354)
        # Getting the type of 'self' (line 354)
        self_137880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 7), 'self')
        # Obtaining the member '_font_preamble' of a type (line 354)
        _font_preamble_137881 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 354, 7), self_137880, '_font_preamble')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 354, 7), tuple_137879, _font_preamble_137881)
        # Adding element type (line 354)
        # Getting the type of 'unicode_preamble' (line 354)
        unicode_preamble_137882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 28), 'unicode_preamble')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 354, 7), tuple_137879, unicode_preamble_137882)
        # Adding element type (line 354)
        # Getting the type of 'custom_preamble' (line 354)
        custom_preamble_137883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 46), 'custom_preamble')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 354, 7), tuple_137879, custom_preamble_137883)
        # Adding element type (line 354)
        # Getting the type of 'fontsize' (line 355)
        fontsize_137884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 7), 'fontsize')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 354, 7), tuple_137879, fontsize_137884)
        # Adding element type (line 354)
        # Getting the type of 'fontsize' (line 355)
        fontsize_137885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 17), 'fontsize')
        float_137886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 28), 'float')
        # Applying the binary operator '*' (line 355)
        result_mul_137887 = python_operator(stypy.reporting.localization.Localization(__file__, 355, 17), '*', fontsize_137885, float_137886)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 354, 7), tuple_137879, result_mul_137887)
        # Adding element type (line 354)
        # Getting the type of 'tex' (line 355)
        tex_137888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 34), 'tex')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 354, 7), tuple_137879, tex_137888)
        
        # Applying the binary operator '%' (line 354)
        result_mod_137889 = python_operator(stypy.reporting.localization.Localization(__file__, 354, (-1)), '%', unicode_137878, tuple_137879)
        
        # Assigning a type to the variable 's' (line 338)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 338, 8), 's', result_mod_137889)
        
        # Call to open(...): (line 356)
        # Processing the call arguments (line 356)
        # Getting the type of 'texfile' (line 356)
        texfile_137891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 18), 'texfile', False)
        unicode_137892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 356, 27), 'unicode', u'wb')
        # Processing the call keyword arguments (line 356)
        kwargs_137893 = {}
        # Getting the type of 'open' (line 356)
        open_137890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 13), 'open', False)
        # Calling open(args, kwargs) (line 356)
        open_call_result_137894 = invoke(stypy.reporting.localization.Localization(__file__, 356, 13), open_137890, *[texfile_137891, unicode_137892], **kwargs_137893)
        
        with_137895 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 356, 13), open_call_result_137894, 'with parameter', '__enter__', '__exit__')

        if with_137895:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 356)
            enter___137896 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 356, 13), open_call_result_137894, '__enter__')
            with_enter_137897 = invoke(stypy.reporting.localization.Localization(__file__, 356, 13), enter___137896)
            # Assigning a type to the variable 'fh' (line 356)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 356, 13), 'fh', with_enter_137897)
            
            
            # Obtaining the type of the subscript
            unicode_137898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 357, 24), 'unicode', u'text.latex.unicode')
            # Getting the type of 'rcParams' (line 357)
            rcParams_137899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 15), 'rcParams')
            # Obtaining the member '__getitem__' of a type (line 357)
            getitem___137900 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 357, 15), rcParams_137899, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 357)
            subscript_call_result_137901 = invoke(stypy.reporting.localization.Localization(__file__, 357, 15), getitem___137900, unicode_137898)
            
            # Testing the type of an if condition (line 357)
            if_condition_137902 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 357, 12), subscript_call_result_137901)
            # Assigning a type to the variable 'if_condition_137902' (line 357)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 357, 12), 'if_condition_137902', if_condition_137902)
            # SSA begins for if statement (line 357)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to write(...): (line 358)
            # Processing the call arguments (line 358)
            
            # Call to encode(...): (line 358)
            # Processing the call arguments (line 358)
            unicode_137907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 358, 34), 'unicode', u'utf8')
            # Processing the call keyword arguments (line 358)
            kwargs_137908 = {}
            # Getting the type of 's' (line 358)
            s_137905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 25), 's', False)
            # Obtaining the member 'encode' of a type (line 358)
            encode_137906 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 25), s_137905, 'encode')
            # Calling encode(args, kwargs) (line 358)
            encode_call_result_137909 = invoke(stypy.reporting.localization.Localization(__file__, 358, 25), encode_137906, *[unicode_137907], **kwargs_137908)
            
            # Processing the call keyword arguments (line 358)
            kwargs_137910 = {}
            # Getting the type of 'fh' (line 358)
            fh_137903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 16), 'fh', False)
            # Obtaining the member 'write' of a type (line 358)
            write_137904 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 16), fh_137903, 'write')
            # Calling write(args, kwargs) (line 358)
            write_call_result_137911 = invoke(stypy.reporting.localization.Localization(__file__, 358, 16), write_137904, *[encode_call_result_137909], **kwargs_137910)
            
            # SSA branch for the else part of an if statement (line 357)
            module_type_store.open_ssa_branch('else')
            
            
            # SSA begins for try-except statement (line 360)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
            
            # Call to write(...): (line 361)
            # Processing the call arguments (line 361)
            
            # Call to encode(...): (line 361)
            # Processing the call arguments (line 361)
            unicode_137916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 38), 'unicode', u'ascii')
            # Processing the call keyword arguments (line 361)
            kwargs_137917 = {}
            # Getting the type of 's' (line 361)
            s_137914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 29), 's', False)
            # Obtaining the member 'encode' of a type (line 361)
            encode_137915 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 361, 29), s_137914, 'encode')
            # Calling encode(args, kwargs) (line 361)
            encode_call_result_137918 = invoke(stypy.reporting.localization.Localization(__file__, 361, 29), encode_137915, *[unicode_137916], **kwargs_137917)
            
            # Processing the call keyword arguments (line 361)
            kwargs_137919 = {}
            # Getting the type of 'fh' (line 361)
            fh_137912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 20), 'fh', False)
            # Obtaining the member 'write' of a type (line 361)
            write_137913 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 361, 20), fh_137912, 'write')
            # Calling write(args, kwargs) (line 361)
            write_call_result_137920 = invoke(stypy.reporting.localization.Localization(__file__, 361, 20), write_137913, *[encode_call_result_137918], **kwargs_137919)
            
            # SSA branch for the except part of a try statement (line 360)
            # SSA branch for the except 'UnicodeEncodeError' branch of a try statement (line 360)
            # Storing handler type
            module_type_store.open_ssa_branch('except')
            # Getting the type of 'UnicodeEncodeError' (line 362)
            UnicodeEncodeError_137921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 23), 'UnicodeEncodeError')
            # Assigning a type to the variable 'err' (line 362)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 16), 'err', UnicodeEncodeError_137921)
            
            # Call to report(...): (line 363)
            # Processing the call arguments (line 363)
            unicode_137925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 363, 39), 'unicode', u"You are using unicode and latex, but have not enabled the matplotlib 'text.latex.unicode' rcParam.")
            unicode_137926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 39), 'unicode', u'helpful')
            # Processing the call keyword arguments (line 363)
            kwargs_137927 = {}
            # Getting the type of 'mpl' (line 363)
            mpl_137922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 20), 'mpl', False)
            # Obtaining the member 'verbose' of a type (line 363)
            verbose_137923 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 363, 20), mpl_137922, 'verbose')
            # Obtaining the member 'report' of a type (line 363)
            report_137924 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 363, 20), verbose_137923, 'report')
            # Calling report(args, kwargs) (line 363)
            report_call_result_137928 = invoke(stypy.reporting.localization.Localization(__file__, 363, 20), report_137924, *[unicode_137925, unicode_137926], **kwargs_137927)
            
            # SSA join for try-except statement (line 360)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 357)
            module_type_store = module_type_store.join_ssa_context()
            
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 356)
            exit___137929 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 356, 13), open_call_result_137894, '__exit__')
            with_exit_137930 = invoke(stypy.reporting.localization.Localization(__file__, 356, 13), exit___137929, None, None, None)

        # Getting the type of 'texfile' (line 369)
        texfile_137931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 15), 'texfile')
        # Assigning a type to the variable 'stypy_return_type' (line 369)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 369, 8), 'stypy_return_type', texfile_137931)
        
        # ################# End of 'make_tex_preview(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'make_tex_preview' in the type store
        # Getting the type of 'stypy_return_type' (line 313)
        stypy_return_type_137932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_137932)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'make_tex_preview'
        return stypy_return_type_137932


    @norecursion
    def make_dvi(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'make_dvi'
        module_type_store = module_type_store.open_function_context('make_dvi', 371, 4, False)
        # Assigning a type to the variable 'self' (line 372)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 372, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TexManager.make_dvi.__dict__.__setitem__('stypy_localization', localization)
        TexManager.make_dvi.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TexManager.make_dvi.__dict__.__setitem__('stypy_type_store', module_type_store)
        TexManager.make_dvi.__dict__.__setitem__('stypy_function_name', 'TexManager.make_dvi')
        TexManager.make_dvi.__dict__.__setitem__('stypy_param_names_list', ['tex', 'fontsize'])
        TexManager.make_dvi.__dict__.__setitem__('stypy_varargs_param_name', None)
        TexManager.make_dvi.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TexManager.make_dvi.__dict__.__setitem__('stypy_call_defaults', defaults)
        TexManager.make_dvi.__dict__.__setitem__('stypy_call_varargs', varargs)
        TexManager.make_dvi.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TexManager.make_dvi.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TexManager.make_dvi', ['tex', 'fontsize'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'make_dvi', localization, ['tex', 'fontsize'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'make_dvi(...)' code ##################

        unicode_137933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 376, (-1)), 'unicode', u"\n        generates a dvi file containing latex's layout of tex string\n\n        returns the file name\n        ")
        
        
        # Obtaining the type of the subscript
        unicode_137934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 378, 20), 'unicode', u'text.latex.preview')
        # Getting the type of 'rcParams' (line 378)
        rcParams_137935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 11), 'rcParams')
        # Obtaining the member '__getitem__' of a type (line 378)
        getitem___137936 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 378, 11), rcParams_137935, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 378)
        subscript_call_result_137937 = invoke(stypy.reporting.localization.Localization(__file__, 378, 11), getitem___137936, unicode_137934)
        
        # Testing the type of an if condition (line 378)
        if_condition_137938 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 378, 8), subscript_call_result_137937)
        # Assigning a type to the variable 'if_condition_137938' (line 378)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 378, 8), 'if_condition_137938', if_condition_137938)
        # SSA begins for if statement (line 378)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to make_dvi_preview(...): (line 379)
        # Processing the call arguments (line 379)
        # Getting the type of 'tex' (line 379)
        tex_137941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 41), 'tex', False)
        # Getting the type of 'fontsize' (line 379)
        fontsize_137942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 46), 'fontsize', False)
        # Processing the call keyword arguments (line 379)
        kwargs_137943 = {}
        # Getting the type of 'self' (line 379)
        self_137939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 19), 'self', False)
        # Obtaining the member 'make_dvi_preview' of a type (line 379)
        make_dvi_preview_137940 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 379, 19), self_137939, 'make_dvi_preview')
        # Calling make_dvi_preview(args, kwargs) (line 379)
        make_dvi_preview_call_result_137944 = invoke(stypy.reporting.localization.Localization(__file__, 379, 19), make_dvi_preview_137940, *[tex_137941, fontsize_137942], **kwargs_137943)
        
        # Assigning a type to the variable 'stypy_return_type' (line 379)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 379, 12), 'stypy_return_type', make_dvi_preview_call_result_137944)
        # SSA join for if statement (line 378)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 381):
        
        # Assigning a Call to a Name (line 381):
        
        # Call to get_basefile(...): (line 381)
        # Processing the call arguments (line 381)
        # Getting the type of 'tex' (line 381)
        tex_137947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 37), 'tex', False)
        # Getting the type of 'fontsize' (line 381)
        fontsize_137948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 42), 'fontsize', False)
        # Processing the call keyword arguments (line 381)
        kwargs_137949 = {}
        # Getting the type of 'self' (line 381)
        self_137945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 19), 'self', False)
        # Obtaining the member 'get_basefile' of a type (line 381)
        get_basefile_137946 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 381, 19), self_137945, 'get_basefile')
        # Calling get_basefile(args, kwargs) (line 381)
        get_basefile_call_result_137950 = invoke(stypy.reporting.localization.Localization(__file__, 381, 19), get_basefile_137946, *[tex_137947, fontsize_137948], **kwargs_137949)
        
        # Assigning a type to the variable 'basefile' (line 381)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 381, 8), 'basefile', get_basefile_call_result_137950)
        
        # Assigning a BinOp to a Name (line 382):
        
        # Assigning a BinOp to a Name (line 382):
        unicode_137951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 382, 18), 'unicode', u'%s.dvi')
        # Getting the type of 'basefile' (line 382)
        basefile_137952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 29), 'basefile')
        # Applying the binary operator '%' (line 382)
        result_mod_137953 = python_operator(stypy.reporting.localization.Localization(__file__, 382, 18), '%', unicode_137951, basefile_137952)
        
        # Assigning a type to the variable 'dvifile' (line 382)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 382, 8), 'dvifile', result_mod_137953)
        
        
        # Evaluating a boolean operation
        # Getting the type of 'DEBUG' (line 384)
        DEBUG_137954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 11), 'DEBUG')
        
        
        # Call to exists(...): (line 384)
        # Processing the call arguments (line 384)
        # Getting the type of 'dvifile' (line 384)
        dvifile_137958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 39), 'dvifile', False)
        # Processing the call keyword arguments (line 384)
        kwargs_137959 = {}
        # Getting the type of 'os' (line 384)
        os_137955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 24), 'os', False)
        # Obtaining the member 'path' of a type (line 384)
        path_137956 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 384, 24), os_137955, 'path')
        # Obtaining the member 'exists' of a type (line 384)
        exists_137957 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 384, 24), path_137956, 'exists')
        # Calling exists(args, kwargs) (line 384)
        exists_call_result_137960 = invoke(stypy.reporting.localization.Localization(__file__, 384, 24), exists_137957, *[dvifile_137958], **kwargs_137959)
        
        # Applying the 'not' unary operator (line 384)
        result_not__137961 = python_operator(stypy.reporting.localization.Localization(__file__, 384, 20), 'not', exists_call_result_137960)
        
        # Applying the binary operator 'or' (line 384)
        result_or_keyword_137962 = python_operator(stypy.reporting.localization.Localization(__file__, 384, 11), 'or', DEBUG_137954, result_not__137961)
        
        # Testing the type of an if condition (line 384)
        if_condition_137963 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 384, 8), result_or_keyword_137962)
        # Assigning a type to the variable 'if_condition_137963' (line 384)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 384, 8), 'if_condition_137963', if_condition_137963)
        # SSA begins for if statement (line 384)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 385):
        
        # Assigning a Call to a Name (line 385):
        
        # Call to make_tex(...): (line 385)
        # Processing the call arguments (line 385)
        # Getting the type of 'tex' (line 385)
        tex_137966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 36), 'tex', False)
        # Getting the type of 'fontsize' (line 385)
        fontsize_137967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 41), 'fontsize', False)
        # Processing the call keyword arguments (line 385)
        kwargs_137968 = {}
        # Getting the type of 'self' (line 385)
        self_137964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 22), 'self', False)
        # Obtaining the member 'make_tex' of a type (line 385)
        make_tex_137965 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 385, 22), self_137964, 'make_tex')
        # Calling make_tex(args, kwargs) (line 385)
        make_tex_call_result_137969 = invoke(stypy.reporting.localization.Localization(__file__, 385, 22), make_tex_137965, *[tex_137966, fontsize_137967], **kwargs_137968)
        
        # Assigning a type to the variable 'texfile' (line 385)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 385, 12), 'texfile', make_tex_call_result_137969)
        
        # Assigning a List to a Name (line 386):
        
        # Assigning a List to a Name (line 386):
        
        # Obtaining an instance of the builtin type 'list' (line 386)
        list_137970 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 386)
        # Adding element type (line 386)
        
        # Call to str(...): (line 386)
        # Processing the call arguments (line 386)
        unicode_137972 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 27), 'unicode', u'latex')
        # Processing the call keyword arguments (line 386)
        kwargs_137973 = {}
        # Getting the type of 'str' (line 386)
        str_137971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 23), 'str', False)
        # Calling str(args, kwargs) (line 386)
        str_call_result_137974 = invoke(stypy.reporting.localization.Localization(__file__, 386, 23), str_137971, *[unicode_137972], **kwargs_137973)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 386, 22), list_137970, str_call_result_137974)
        # Adding element type (line 386)
        unicode_137975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 37), 'unicode', u'-interaction=nonstopmode')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 386, 22), list_137970, unicode_137975)
        # Adding element type (line 386)
        
        # Call to basename(...): (line 387)
        # Processing the call arguments (line 387)
        # Getting the type of 'texfile' (line 387)
        texfile_137979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 40), 'texfile', False)
        # Processing the call keyword arguments (line 387)
        kwargs_137980 = {}
        # Getting the type of 'os' (line 387)
        os_137976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 23), 'os', False)
        # Obtaining the member 'path' of a type (line 387)
        path_137977 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 387, 23), os_137976, 'path')
        # Obtaining the member 'basename' of a type (line 387)
        basename_137978 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 387, 23), path_137977, 'basename')
        # Calling basename(args, kwargs) (line 387)
        basename_call_result_137981 = invoke(stypy.reporting.localization.Localization(__file__, 387, 23), basename_137978, *[texfile_137979], **kwargs_137980)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 386, 22), list_137970, basename_call_result_137981)
        
        # Assigning a type to the variable 'command' (line 386)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 386, 12), 'command', list_137970)
        
        # Call to report(...): (line 388)
        # Processing the call arguments (line 388)
        # Getting the type of 'command' (line 388)
        command_137985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 31), 'command', False)
        unicode_137986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 388, 40), 'unicode', u'debug')
        # Processing the call keyword arguments (line 388)
        kwargs_137987 = {}
        # Getting the type of 'mpl' (line 388)
        mpl_137982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 12), 'mpl', False)
        # Obtaining the member 'verbose' of a type (line 388)
        verbose_137983 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 12), mpl_137982, 'verbose')
        # Obtaining the member 'report' of a type (line 388)
        report_137984 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 12), verbose_137983, 'report')
        # Calling report(args, kwargs) (line 388)
        report_call_result_137988 = invoke(stypy.reporting.localization.Localization(__file__, 388, 12), report_137984, *[command_137985, unicode_137986], **kwargs_137987)
        
        
        # Call to Locked(...): (line 389)
        # Processing the call arguments (line 389)
        # Getting the type of 'self' (line 389)
        self_137990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 24), 'self', False)
        # Obtaining the member 'texcache' of a type (line 389)
        texcache_137991 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 24), self_137990, 'texcache')
        # Processing the call keyword arguments (line 389)
        kwargs_137992 = {}
        # Getting the type of 'Locked' (line 389)
        Locked_137989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 17), 'Locked', False)
        # Calling Locked(args, kwargs) (line 389)
        Locked_call_result_137993 = invoke(stypy.reporting.localization.Localization(__file__, 389, 17), Locked_137989, *[texcache_137991], **kwargs_137992)
        
        with_137994 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 389, 17), Locked_call_result_137993, 'with parameter', '__enter__', '__exit__')

        if with_137994:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 389)
            enter___137995 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 17), Locked_call_result_137993, '__enter__')
            with_enter_137996 = invoke(stypy.reporting.localization.Localization(__file__, 389, 17), enter___137995)
            
            
            # SSA begins for try-except statement (line 390)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
            
            # Assigning a Call to a Name (line 391):
            
            # Assigning a Call to a Name (line 391):
            
            # Call to check_output(...): (line 391)
            # Processing the call arguments (line 391)
            # Getting the type of 'command' (line 391)
            command_137999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 53), 'command', False)
            # Processing the call keyword arguments (line 391)
            # Getting the type of 'self' (line 392)
            self_138000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 57), 'self', False)
            # Obtaining the member 'texcache' of a type (line 392)
            texcache_138001 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 392, 57), self_138000, 'texcache')
            keyword_138002 = texcache_138001
            # Getting the type of 'subprocess' (line 393)
            subprocess_138003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 60), 'subprocess', False)
            # Obtaining the member 'STDOUT' of a type (line 393)
            STDOUT_138004 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 393, 60), subprocess_138003, 'STDOUT')
            keyword_138005 = STDOUT_138004
            kwargs_138006 = {'cwd': keyword_138002, 'stderr': keyword_138005}
            # Getting the type of 'subprocess' (line 391)
            subprocess_137997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 29), 'subprocess', False)
            # Obtaining the member 'check_output' of a type (line 391)
            check_output_137998 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 391, 29), subprocess_137997, 'check_output')
            # Calling check_output(args, kwargs) (line 391)
            check_output_call_result_138007 = invoke(stypy.reporting.localization.Localization(__file__, 391, 29), check_output_137998, *[command_137999], **kwargs_138006)
            
            # Assigning a type to the variable 'report' (line 391)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 391, 20), 'report', check_output_call_result_138007)
            # SSA branch for the except part of a try statement (line 390)
            # SSA branch for the except 'Attribute' branch of a try statement (line 390)
            # Storing handler type
            module_type_store.open_ssa_branch('except')
            # Getting the type of 'subprocess' (line 394)
            subprocess_138008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 23), 'subprocess')
            # Obtaining the member 'CalledProcessError' of a type (line 394)
            CalledProcessError_138009 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 23), subprocess_138008, 'CalledProcessError')
            # Assigning a type to the variable 'exc' (line 394)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 394, 16), 'exc', CalledProcessError_138009)
            
            # Call to RuntimeError(...): (line 395)
            # Processing the call arguments (line 395)
            unicode_138011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 396, 25), 'unicode', u'LaTeX was not able to process the following string:\n%s\n\nHere is the full report generated by LaTeX:\n%s \n\n')
            
            # Obtaining an instance of the builtin type 'tuple' (line 399)
            tuple_138012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 35), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 399)
            # Adding element type (line 399)
            
            # Call to repr(...): (line 399)
            # Processing the call arguments (line 399)
            
            # Call to encode(...): (line 399)
            # Processing the call arguments (line 399)
            unicode_138016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 51), 'unicode', u'unicode_escape')
            # Processing the call keyword arguments (line 399)
            kwargs_138017 = {}
            # Getting the type of 'tex' (line 399)
            tex_138014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 40), 'tex', False)
            # Obtaining the member 'encode' of a type (line 399)
            encode_138015 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 399, 40), tex_138014, 'encode')
            # Calling encode(args, kwargs) (line 399)
            encode_call_result_138018 = invoke(stypy.reporting.localization.Localization(__file__, 399, 40), encode_138015, *[unicode_138016], **kwargs_138017)
            
            # Processing the call keyword arguments (line 399)
            kwargs_138019 = {}
            # Getting the type of 'repr' (line 399)
            repr_138013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 35), 'repr', False)
            # Calling repr(args, kwargs) (line 399)
            repr_call_result_138020 = invoke(stypy.reporting.localization.Localization(__file__, 399, 35), repr_138013, *[encode_call_result_138018], **kwargs_138019)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 399, 35), tuple_138012, repr_call_result_138020)
            # Adding element type (line 399)
            
            # Call to decode(...): (line 400)
            # Processing the call arguments (line 400)
            unicode_138024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 53), 'unicode', u'utf-8')
            # Processing the call keyword arguments (line 400)
            kwargs_138025 = {}
            # Getting the type of 'exc' (line 400)
            exc_138021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 35), 'exc', False)
            # Obtaining the member 'output' of a type (line 400)
            output_138022 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 400, 35), exc_138021, 'output')
            # Obtaining the member 'decode' of a type (line 400)
            decode_138023 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 400, 35), output_138022, 'decode')
            # Calling decode(args, kwargs) (line 400)
            decode_call_result_138026 = invoke(stypy.reporting.localization.Localization(__file__, 400, 35), decode_138023, *[unicode_138024], **kwargs_138025)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 399, 35), tuple_138012, decode_call_result_138026)
            
            # Applying the binary operator '%' (line 396)
            result_mod_138027 = python_operator(stypy.reporting.localization.Localization(__file__, 396, 25), '%', unicode_138011, tuple_138012)
            
            # Processing the call keyword arguments (line 395)
            kwargs_138028 = {}
            # Getting the type of 'RuntimeError' (line 395)
            RuntimeError_138010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 26), 'RuntimeError', False)
            # Calling RuntimeError(args, kwargs) (line 395)
            RuntimeError_call_result_138029 = invoke(stypy.reporting.localization.Localization(__file__, 395, 26), RuntimeError_138010, *[result_mod_138027], **kwargs_138028)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 395, 20), RuntimeError_call_result_138029, 'raise parameter', BaseException)
            # SSA join for try-except statement (line 390)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Call to report(...): (line 401)
            # Processing the call arguments (line 401)
            # Getting the type of 'report' (line 401)
            report_138033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 35), 'report', False)
            unicode_138034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 401, 43), 'unicode', u'debug')
            # Processing the call keyword arguments (line 401)
            kwargs_138035 = {}
            # Getting the type of 'mpl' (line 401)
            mpl_138030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 16), 'mpl', False)
            # Obtaining the member 'verbose' of a type (line 401)
            verbose_138031 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 401, 16), mpl_138030, 'verbose')
            # Obtaining the member 'report' of a type (line 401)
            report_138032 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 401, 16), verbose_138031, 'report')
            # Calling report(args, kwargs) (line 401)
            report_call_result_138036 = invoke(stypy.reporting.localization.Localization(__file__, 401, 16), report_138032, *[report_138033, unicode_138034], **kwargs_138035)
            
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 389)
            exit___138037 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 17), Locked_call_result_137993, '__exit__')
            with_exit_138038 = invoke(stypy.reporting.localization.Localization(__file__, 389, 17), exit___138037, None, None, None)

        
        
        # Call to glob(...): (line 402)
        # Processing the call arguments (line 402)
        # Getting the type of 'basefile' (line 402)
        basefile_138041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 35), 'basefile', False)
        unicode_138042 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 402, 46), 'unicode', u'*')
        # Applying the binary operator '+' (line 402)
        result_add_138043 = python_operator(stypy.reporting.localization.Localization(__file__, 402, 35), '+', basefile_138041, unicode_138042)
        
        # Processing the call keyword arguments (line 402)
        kwargs_138044 = {}
        # Getting the type of 'glob' (line 402)
        glob_138039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 25), 'glob', False)
        # Obtaining the member 'glob' of a type (line 402)
        glob_138040 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 402, 25), glob_138039, 'glob')
        # Calling glob(args, kwargs) (line 402)
        glob_call_result_138045 = invoke(stypy.reporting.localization.Localization(__file__, 402, 25), glob_138040, *[result_add_138043], **kwargs_138044)
        
        # Testing the type of a for loop iterable (line 402)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 402, 12), glob_call_result_138045)
        # Getting the type of the for loop variable (line 402)
        for_loop_var_138046 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 402, 12), glob_call_result_138045)
        # Assigning a type to the variable 'fname' (line 402)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 402, 12), 'fname', for_loop_var_138046)
        # SSA begins for a for statement (line 402)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Call to endswith(...): (line 403)
        # Processing the call arguments (line 403)
        unicode_138049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 403, 34), 'unicode', u'dvi')
        # Processing the call keyword arguments (line 403)
        kwargs_138050 = {}
        # Getting the type of 'fname' (line 403)
        fname_138047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 19), 'fname', False)
        # Obtaining the member 'endswith' of a type (line 403)
        endswith_138048 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 403, 19), fname_138047, 'endswith')
        # Calling endswith(args, kwargs) (line 403)
        endswith_call_result_138051 = invoke(stypy.reporting.localization.Localization(__file__, 403, 19), endswith_138048, *[unicode_138049], **kwargs_138050)
        
        # Testing the type of an if condition (line 403)
        if_condition_138052 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 403, 16), endswith_call_result_138051)
        # Assigning a type to the variable 'if_condition_138052' (line 403)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 403, 16), 'if_condition_138052', if_condition_138052)
        # SSA begins for if statement (line 403)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        pass
        # SSA branch for the else part of an if statement (line 403)
        module_type_store.open_ssa_branch('else')
        
        
        # Call to endswith(...): (line 405)
        # Processing the call arguments (line 405)
        unicode_138055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 36), 'unicode', u'tex')
        # Processing the call keyword arguments (line 405)
        kwargs_138056 = {}
        # Getting the type of 'fname' (line 405)
        fname_138053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 21), 'fname', False)
        # Obtaining the member 'endswith' of a type (line 405)
        endswith_138054 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 405, 21), fname_138053, 'endswith')
        # Calling endswith(args, kwargs) (line 405)
        endswith_call_result_138057 = invoke(stypy.reporting.localization.Localization(__file__, 405, 21), endswith_138054, *[unicode_138055], **kwargs_138056)
        
        # Testing the type of an if condition (line 405)
        if_condition_138058 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 405, 21), endswith_call_result_138057)
        # Assigning a type to the variable 'if_condition_138058' (line 405)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 405, 21), 'if_condition_138058', if_condition_138058)
        # SSA begins for if statement (line 405)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        pass
        # SSA branch for the else part of an if statement (line 405)
        module_type_store.open_ssa_branch('else')
        
        
        # SSA begins for try-except statement (line 408)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to remove(...): (line 409)
        # Processing the call arguments (line 409)
        # Getting the type of 'fname' (line 409)
        fname_138061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 34), 'fname', False)
        # Processing the call keyword arguments (line 409)
        kwargs_138062 = {}
        # Getting the type of 'os' (line 409)
        os_138059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 24), 'os', False)
        # Obtaining the member 'remove' of a type (line 409)
        remove_138060 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 409, 24), os_138059, 'remove')
        # Calling remove(args, kwargs) (line 409)
        remove_call_result_138063 = invoke(stypy.reporting.localization.Localization(__file__, 409, 24), remove_138060, *[fname_138061], **kwargs_138062)
        
        # SSA branch for the except part of a try statement (line 408)
        # SSA branch for the except 'OSError' branch of a try statement (line 408)
        module_type_store.open_ssa_branch('except')
        pass
        # SSA join for try-except statement (line 408)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 405)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 403)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 384)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'dvifile' (line 413)
        dvifile_138064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 15), 'dvifile')
        # Assigning a type to the variable 'stypy_return_type' (line 413)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 413, 8), 'stypy_return_type', dvifile_138064)
        
        # ################# End of 'make_dvi(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'make_dvi' in the type store
        # Getting the type of 'stypy_return_type' (line 371)
        stypy_return_type_138065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_138065)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'make_dvi'
        return stypy_return_type_138065


    @norecursion
    def make_dvi_preview(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'make_dvi_preview'
        module_type_store = module_type_store.open_function_context('make_dvi_preview', 415, 4, False)
        # Assigning a type to the variable 'self' (line 416)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 416, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TexManager.make_dvi_preview.__dict__.__setitem__('stypy_localization', localization)
        TexManager.make_dvi_preview.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TexManager.make_dvi_preview.__dict__.__setitem__('stypy_type_store', module_type_store)
        TexManager.make_dvi_preview.__dict__.__setitem__('stypy_function_name', 'TexManager.make_dvi_preview')
        TexManager.make_dvi_preview.__dict__.__setitem__('stypy_param_names_list', ['tex', 'fontsize'])
        TexManager.make_dvi_preview.__dict__.__setitem__('stypy_varargs_param_name', None)
        TexManager.make_dvi_preview.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TexManager.make_dvi_preview.__dict__.__setitem__('stypy_call_defaults', defaults)
        TexManager.make_dvi_preview.__dict__.__setitem__('stypy_call_varargs', varargs)
        TexManager.make_dvi_preview.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TexManager.make_dvi_preview.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TexManager.make_dvi_preview', ['tex', 'fontsize'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'make_dvi_preview', localization, ['tex', 'fontsize'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'make_dvi_preview(...)' code ##################

        unicode_138066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 422, (-1)), 'unicode', u"\n        generates a dvi file containing latex's layout of tex\n        string. It calls make_tex_preview() method and store the size\n        information (width, height, descent) in a separte file.\n\n        returns the file name\n        ")
        
        # Assigning a Call to a Name (line 423):
        
        # Assigning a Call to a Name (line 423):
        
        # Call to get_basefile(...): (line 423)
        # Processing the call arguments (line 423)
        # Getting the type of 'tex' (line 423)
        tex_138069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 37), 'tex', False)
        # Getting the type of 'fontsize' (line 423)
        fontsize_138070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 42), 'fontsize', False)
        # Processing the call keyword arguments (line 423)
        kwargs_138071 = {}
        # Getting the type of 'self' (line 423)
        self_138067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 19), 'self', False)
        # Obtaining the member 'get_basefile' of a type (line 423)
        get_basefile_138068 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 423, 19), self_138067, 'get_basefile')
        # Calling get_basefile(args, kwargs) (line 423)
        get_basefile_call_result_138072 = invoke(stypy.reporting.localization.Localization(__file__, 423, 19), get_basefile_138068, *[tex_138069, fontsize_138070], **kwargs_138071)
        
        # Assigning a type to the variable 'basefile' (line 423)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 423, 8), 'basefile', get_basefile_call_result_138072)
        
        # Assigning a BinOp to a Name (line 424):
        
        # Assigning a BinOp to a Name (line 424):
        unicode_138073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 424, 18), 'unicode', u'%s.dvi')
        # Getting the type of 'basefile' (line 424)
        basefile_138074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 29), 'basefile')
        # Applying the binary operator '%' (line 424)
        result_mod_138075 = python_operator(stypy.reporting.localization.Localization(__file__, 424, 18), '%', unicode_138073, basefile_138074)
        
        # Assigning a type to the variable 'dvifile' (line 424)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 424, 8), 'dvifile', result_mod_138075)
        
        # Assigning a BinOp to a Name (line 425):
        
        # Assigning a BinOp to a Name (line 425):
        unicode_138076 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 425, 23), 'unicode', u'%s.baseline')
        # Getting the type of 'basefile' (line 425)
        basefile_138077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 39), 'basefile')
        # Applying the binary operator '%' (line 425)
        result_mod_138078 = python_operator(stypy.reporting.localization.Localization(__file__, 425, 23), '%', unicode_138076, basefile_138077)
        
        # Assigning a type to the variable 'baselinefile' (line 425)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 425, 8), 'baselinefile', result_mod_138078)
        
        
        # Evaluating a boolean operation
        # Getting the type of 'DEBUG' (line 427)
        DEBUG_138079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 12), 'DEBUG')
        
        
        # Call to exists(...): (line 427)
        # Processing the call arguments (line 427)
        # Getting the type of 'dvifile' (line 427)
        dvifile_138083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 40), 'dvifile', False)
        # Processing the call keyword arguments (line 427)
        kwargs_138084 = {}
        # Getting the type of 'os' (line 427)
        os_138080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 25), 'os', False)
        # Obtaining the member 'path' of a type (line 427)
        path_138081 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 427, 25), os_138080, 'path')
        # Obtaining the member 'exists' of a type (line 427)
        exists_138082 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 427, 25), path_138081, 'exists')
        # Calling exists(args, kwargs) (line 427)
        exists_call_result_138085 = invoke(stypy.reporting.localization.Localization(__file__, 427, 25), exists_138082, *[dvifile_138083], **kwargs_138084)
        
        # Applying the 'not' unary operator (line 427)
        result_not__138086 = python_operator(stypy.reporting.localization.Localization(__file__, 427, 21), 'not', exists_call_result_138085)
        
        # Applying the binary operator 'or' (line 427)
        result_or_keyword_138087 = python_operator(stypy.reporting.localization.Localization(__file__, 427, 12), 'or', DEBUG_138079, result_not__138086)
        
        
        # Call to exists(...): (line 428)
        # Processing the call arguments (line 428)
        # Getting the type of 'baselinefile' (line 428)
        baselinefile_138091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 35), 'baselinefile', False)
        # Processing the call keyword arguments (line 428)
        kwargs_138092 = {}
        # Getting the type of 'os' (line 428)
        os_138088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 20), 'os', False)
        # Obtaining the member 'path' of a type (line 428)
        path_138089 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 428, 20), os_138088, 'path')
        # Obtaining the member 'exists' of a type (line 428)
        exists_138090 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 428, 20), path_138089, 'exists')
        # Calling exists(args, kwargs) (line 428)
        exists_call_result_138093 = invoke(stypy.reporting.localization.Localization(__file__, 428, 20), exists_138090, *[baselinefile_138091], **kwargs_138092)
        
        # Applying the 'not' unary operator (line 428)
        result_not__138094 = python_operator(stypy.reporting.localization.Localization(__file__, 428, 16), 'not', exists_call_result_138093)
        
        # Applying the binary operator 'or' (line 427)
        result_or_keyword_138095 = python_operator(stypy.reporting.localization.Localization(__file__, 427, 12), 'or', result_or_keyword_138087, result_not__138094)
        
        # Testing the type of an if condition (line 427)
        if_condition_138096 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 427, 8), result_or_keyword_138095)
        # Assigning a type to the variable 'if_condition_138096' (line 427)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 427, 8), 'if_condition_138096', if_condition_138096)
        # SSA begins for if statement (line 427)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 429):
        
        # Assigning a Call to a Name (line 429):
        
        # Call to make_tex_preview(...): (line 429)
        # Processing the call arguments (line 429)
        # Getting the type of 'tex' (line 429)
        tex_138099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 44), 'tex', False)
        # Getting the type of 'fontsize' (line 429)
        fontsize_138100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 49), 'fontsize', False)
        # Processing the call keyword arguments (line 429)
        kwargs_138101 = {}
        # Getting the type of 'self' (line 429)
        self_138097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 22), 'self', False)
        # Obtaining the member 'make_tex_preview' of a type (line 429)
        make_tex_preview_138098 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 429, 22), self_138097, 'make_tex_preview')
        # Calling make_tex_preview(args, kwargs) (line 429)
        make_tex_preview_call_result_138102 = invoke(stypy.reporting.localization.Localization(__file__, 429, 22), make_tex_preview_138098, *[tex_138099, fontsize_138100], **kwargs_138101)
        
        # Assigning a type to the variable 'texfile' (line 429)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 429, 12), 'texfile', make_tex_preview_call_result_138102)
        
        # Assigning a List to a Name (line 430):
        
        # Assigning a List to a Name (line 430):
        
        # Obtaining an instance of the builtin type 'list' (line 430)
        list_138103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 430, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 430)
        # Adding element type (line 430)
        
        # Call to str(...): (line 430)
        # Processing the call arguments (line 430)
        unicode_138105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 430, 27), 'unicode', u'latex')
        # Processing the call keyword arguments (line 430)
        kwargs_138106 = {}
        # Getting the type of 'str' (line 430)
        str_138104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 23), 'str', False)
        # Calling str(args, kwargs) (line 430)
        str_call_result_138107 = invoke(stypy.reporting.localization.Localization(__file__, 430, 23), str_138104, *[unicode_138105], **kwargs_138106)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 430, 22), list_138103, str_call_result_138107)
        # Adding element type (line 430)
        unicode_138108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 430, 37), 'unicode', u'-interaction=nonstopmode')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 430, 22), list_138103, unicode_138108)
        # Adding element type (line 430)
        
        # Call to basename(...): (line 431)
        # Processing the call arguments (line 431)
        # Getting the type of 'texfile' (line 431)
        texfile_138112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 40), 'texfile', False)
        # Processing the call keyword arguments (line 431)
        kwargs_138113 = {}
        # Getting the type of 'os' (line 431)
        os_138109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 23), 'os', False)
        # Obtaining the member 'path' of a type (line 431)
        path_138110 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 431, 23), os_138109, 'path')
        # Obtaining the member 'basename' of a type (line 431)
        basename_138111 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 431, 23), path_138110, 'basename')
        # Calling basename(args, kwargs) (line 431)
        basename_call_result_138114 = invoke(stypy.reporting.localization.Localization(__file__, 431, 23), basename_138111, *[texfile_138112], **kwargs_138113)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 430, 22), list_138103, basename_call_result_138114)
        
        # Assigning a type to the variable 'command' (line 430)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 430, 12), 'command', list_138103)
        
        # Call to report(...): (line 432)
        # Processing the call arguments (line 432)
        # Getting the type of 'command' (line 432)
        command_138118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 31), 'command', False)
        unicode_138119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 432, 40), 'unicode', u'debug')
        # Processing the call keyword arguments (line 432)
        kwargs_138120 = {}
        # Getting the type of 'mpl' (line 432)
        mpl_138115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 12), 'mpl', False)
        # Obtaining the member 'verbose' of a type (line 432)
        verbose_138116 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 432, 12), mpl_138115, 'verbose')
        # Obtaining the member 'report' of a type (line 432)
        report_138117 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 432, 12), verbose_138116, 'report')
        # Calling report(args, kwargs) (line 432)
        report_call_result_138121 = invoke(stypy.reporting.localization.Localization(__file__, 432, 12), report_138117, *[command_138118, unicode_138119], **kwargs_138120)
        
        
        
        # SSA begins for try-except statement (line 433)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Name (line 434):
        
        # Assigning a Call to a Name (line 434):
        
        # Call to check_output(...): (line 434)
        # Processing the call arguments (line 434)
        # Getting the type of 'command' (line 434)
        command_138124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 49), 'command', False)
        # Processing the call keyword arguments (line 434)
        # Getting the type of 'self' (line 435)
        self_138125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 53), 'self', False)
        # Obtaining the member 'texcache' of a type (line 435)
        texcache_138126 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 435, 53), self_138125, 'texcache')
        keyword_138127 = texcache_138126
        # Getting the type of 'subprocess' (line 436)
        subprocess_138128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 56), 'subprocess', False)
        # Obtaining the member 'STDOUT' of a type (line 436)
        STDOUT_138129 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 436, 56), subprocess_138128, 'STDOUT')
        keyword_138130 = STDOUT_138129
        kwargs_138131 = {'cwd': keyword_138127, 'stderr': keyword_138130}
        # Getting the type of 'subprocess' (line 434)
        subprocess_138122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 25), 'subprocess', False)
        # Obtaining the member 'check_output' of a type (line 434)
        check_output_138123 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 434, 25), subprocess_138122, 'check_output')
        # Calling check_output(args, kwargs) (line 434)
        check_output_call_result_138132 = invoke(stypy.reporting.localization.Localization(__file__, 434, 25), check_output_138123, *[command_138124], **kwargs_138131)
        
        # Assigning a type to the variable 'report' (line 434)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 434, 16), 'report', check_output_call_result_138132)
        # SSA branch for the except part of a try statement (line 433)
        # SSA branch for the except 'Attribute' branch of a try statement (line 433)
        # Storing handler type
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'subprocess' (line 437)
        subprocess_138133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 19), 'subprocess')
        # Obtaining the member 'CalledProcessError' of a type (line 437)
        CalledProcessError_138134 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 437, 19), subprocess_138133, 'CalledProcessError')
        # Assigning a type to the variable 'exc' (line 437)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 437, 12), 'exc', CalledProcessError_138134)
        
        # Call to RuntimeError(...): (line 438)
        # Processing the call arguments (line 438)
        unicode_138136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 439, 21), 'unicode', u'LaTeX was not able to process the following string:\n%s\n\nHere is the full report generated by LaTeX:\n%s \n\n')
        
        # Obtaining an instance of the builtin type 'tuple' (line 442)
        tuple_138137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 442, 31), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 442)
        # Adding element type (line 442)
        
        # Call to repr(...): (line 442)
        # Processing the call arguments (line 442)
        
        # Call to encode(...): (line 442)
        # Processing the call arguments (line 442)
        unicode_138141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 442, 47), 'unicode', u'unicode_escape')
        # Processing the call keyword arguments (line 442)
        kwargs_138142 = {}
        # Getting the type of 'tex' (line 442)
        tex_138139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 36), 'tex', False)
        # Obtaining the member 'encode' of a type (line 442)
        encode_138140 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 442, 36), tex_138139, 'encode')
        # Calling encode(args, kwargs) (line 442)
        encode_call_result_138143 = invoke(stypy.reporting.localization.Localization(__file__, 442, 36), encode_138140, *[unicode_138141], **kwargs_138142)
        
        # Processing the call keyword arguments (line 442)
        kwargs_138144 = {}
        # Getting the type of 'repr' (line 442)
        repr_138138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 31), 'repr', False)
        # Calling repr(args, kwargs) (line 442)
        repr_call_result_138145 = invoke(stypy.reporting.localization.Localization(__file__, 442, 31), repr_138138, *[encode_call_result_138143], **kwargs_138144)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 442, 31), tuple_138137, repr_call_result_138145)
        # Adding element type (line 442)
        
        # Call to decode(...): (line 443)
        # Processing the call arguments (line 443)
        unicode_138149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 443, 49), 'unicode', u'utf-8')
        # Processing the call keyword arguments (line 443)
        kwargs_138150 = {}
        # Getting the type of 'exc' (line 443)
        exc_138146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 31), 'exc', False)
        # Obtaining the member 'output' of a type (line 443)
        output_138147 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 443, 31), exc_138146, 'output')
        # Obtaining the member 'decode' of a type (line 443)
        decode_138148 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 443, 31), output_138147, 'decode')
        # Calling decode(args, kwargs) (line 443)
        decode_call_result_138151 = invoke(stypy.reporting.localization.Localization(__file__, 443, 31), decode_138148, *[unicode_138149], **kwargs_138150)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 442, 31), tuple_138137, decode_call_result_138151)
        
        # Applying the binary operator '%' (line 439)
        result_mod_138152 = python_operator(stypy.reporting.localization.Localization(__file__, 439, 21), '%', unicode_138136, tuple_138137)
        
        # Processing the call keyword arguments (line 438)
        kwargs_138153 = {}
        # Getting the type of 'RuntimeError' (line 438)
        RuntimeError_138135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 22), 'RuntimeError', False)
        # Calling RuntimeError(args, kwargs) (line 438)
        RuntimeError_call_result_138154 = invoke(stypy.reporting.localization.Localization(__file__, 438, 22), RuntimeError_138135, *[result_mod_138152], **kwargs_138153)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 438, 16), RuntimeError_call_result_138154, 'raise parameter', BaseException)
        # SSA join for try-except statement (line 433)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to report(...): (line 444)
        # Processing the call arguments (line 444)
        # Getting the type of 'report' (line 444)
        report_138158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 31), 'report', False)
        unicode_138159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 444, 39), 'unicode', u'debug')
        # Processing the call keyword arguments (line 444)
        kwargs_138160 = {}
        # Getting the type of 'mpl' (line 444)
        mpl_138155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 12), 'mpl', False)
        # Obtaining the member 'verbose' of a type (line 444)
        verbose_138156 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 444, 12), mpl_138155, 'verbose')
        # Obtaining the member 'report' of a type (line 444)
        report_138157 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 444, 12), verbose_138156, 'report')
        # Calling report(args, kwargs) (line 444)
        report_call_result_138161 = invoke(stypy.reporting.localization.Localization(__file__, 444, 12), report_138157, *[report_138158, unicode_138159], **kwargs_138160)
        
        
        # Assigning a Call to a Name (line 448):
        
        # Assigning a Call to a Name (line 448):
        
        # Call to search(...): (line 448)
        # Processing the call arguments (line 448)
        
        # Call to decode(...): (line 448)
        # Processing the call arguments (line 448)
        unicode_138167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 448, 57), 'unicode', u'utf-8')
        # Processing the call keyword arguments (line 448)
        kwargs_138168 = {}
        # Getting the type of 'report' (line 448)
        report_138165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 43), 'report', False)
        # Obtaining the member 'decode' of a type (line 448)
        decode_138166 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 448, 43), report_138165, 'decode')
        # Calling decode(args, kwargs) (line 448)
        decode_call_result_138169 = invoke(stypy.reporting.localization.Localization(__file__, 448, 43), decode_138166, *[unicode_138167], **kwargs_138168)
        
        # Processing the call keyword arguments (line 448)
        kwargs_138170 = {}
        # Getting the type of 'TexManager' (line 448)
        TexManager_138162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 16), 'TexManager', False)
        # Obtaining the member '_re_vbox' of a type (line 448)
        _re_vbox_138163 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 448, 16), TexManager_138162, '_re_vbox')
        # Obtaining the member 'search' of a type (line 448)
        search_138164 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 448, 16), _re_vbox_138163, 'search')
        # Calling search(args, kwargs) (line 448)
        search_call_result_138171 = invoke(stypy.reporting.localization.Localization(__file__, 448, 16), search_138164, *[decode_call_result_138169], **kwargs_138170)
        
        # Assigning a type to the variable 'm' (line 448)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 448, 12), 'm', search_call_result_138171)
        
        # Call to open(...): (line 449)
        # Processing the call arguments (line 449)
        # Getting the type of 'basefile' (line 449)
        basefile_138173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 22), 'basefile', False)
        unicode_138174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 449, 33), 'unicode', u'.baseline')
        # Applying the binary operator '+' (line 449)
        result_add_138175 = python_operator(stypy.reporting.localization.Localization(__file__, 449, 22), '+', basefile_138173, unicode_138174)
        
        unicode_138176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 449, 46), 'unicode', u'w')
        # Processing the call keyword arguments (line 449)
        kwargs_138177 = {}
        # Getting the type of 'open' (line 449)
        open_138172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 17), 'open', False)
        # Calling open(args, kwargs) (line 449)
        open_call_result_138178 = invoke(stypy.reporting.localization.Localization(__file__, 449, 17), open_138172, *[result_add_138175, unicode_138176], **kwargs_138177)
        
        with_138179 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 449, 17), open_call_result_138178, 'with parameter', '__enter__', '__exit__')

        if with_138179:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 449)
            enter___138180 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 449, 17), open_call_result_138178, '__enter__')
            with_enter_138181 = invoke(stypy.reporting.localization.Localization(__file__, 449, 17), enter___138180)
            # Assigning a type to the variable 'fh' (line 449)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 449, 17), 'fh', with_enter_138181)
            
            # Call to write(...): (line 450)
            # Processing the call arguments (line 450)
            
            # Call to join(...): (line 450)
            # Processing the call arguments (line 450)
            
            # Call to groups(...): (line 450)
            # Processing the call keyword arguments (line 450)
            kwargs_138188 = {}
            # Getting the type of 'm' (line 450)
            m_138186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 34), 'm', False)
            # Obtaining the member 'groups' of a type (line 450)
            groups_138187 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 450, 34), m_138186, 'groups')
            # Calling groups(args, kwargs) (line 450)
            groups_call_result_138189 = invoke(stypy.reporting.localization.Localization(__file__, 450, 34), groups_138187, *[], **kwargs_138188)
            
            # Processing the call keyword arguments (line 450)
            kwargs_138190 = {}
            unicode_138184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 450, 25), 'unicode', u' ')
            # Obtaining the member 'join' of a type (line 450)
            join_138185 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 450, 25), unicode_138184, 'join')
            # Calling join(args, kwargs) (line 450)
            join_call_result_138191 = invoke(stypy.reporting.localization.Localization(__file__, 450, 25), join_138185, *[groups_call_result_138189], **kwargs_138190)
            
            # Processing the call keyword arguments (line 450)
            kwargs_138192 = {}
            # Getting the type of 'fh' (line 450)
            fh_138182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 16), 'fh', False)
            # Obtaining the member 'write' of a type (line 450)
            write_138183 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 450, 16), fh_138182, 'write')
            # Calling write(args, kwargs) (line 450)
            write_call_result_138193 = invoke(stypy.reporting.localization.Localization(__file__, 450, 16), write_138183, *[join_call_result_138191], **kwargs_138192)
            
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 449)
            exit___138194 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 449, 17), open_call_result_138178, '__exit__')
            with_exit_138195 = invoke(stypy.reporting.localization.Localization(__file__, 449, 17), exit___138194, None, None, None)

        
        
        # Call to glob(...): (line 452)
        # Processing the call arguments (line 452)
        # Getting the type of 'basefile' (line 452)
        basefile_138198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 35), 'basefile', False)
        unicode_138199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 452, 46), 'unicode', u'*')
        # Applying the binary operator '+' (line 452)
        result_add_138200 = python_operator(stypy.reporting.localization.Localization(__file__, 452, 35), '+', basefile_138198, unicode_138199)
        
        # Processing the call keyword arguments (line 452)
        kwargs_138201 = {}
        # Getting the type of 'glob' (line 452)
        glob_138196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 25), 'glob', False)
        # Obtaining the member 'glob' of a type (line 452)
        glob_138197 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 452, 25), glob_138196, 'glob')
        # Calling glob(args, kwargs) (line 452)
        glob_call_result_138202 = invoke(stypy.reporting.localization.Localization(__file__, 452, 25), glob_138197, *[result_add_138200], **kwargs_138201)
        
        # Testing the type of a for loop iterable (line 452)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 452, 12), glob_call_result_138202)
        # Getting the type of the for loop variable (line 452)
        for_loop_var_138203 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 452, 12), glob_call_result_138202)
        # Assigning a type to the variable 'fname' (line 452)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 452, 12), 'fname', for_loop_var_138203)
        # SSA begins for a for statement (line 452)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Call to endswith(...): (line 453)
        # Processing the call arguments (line 453)
        unicode_138206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 453, 34), 'unicode', u'dvi')
        # Processing the call keyword arguments (line 453)
        kwargs_138207 = {}
        # Getting the type of 'fname' (line 453)
        fname_138204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 19), 'fname', False)
        # Obtaining the member 'endswith' of a type (line 453)
        endswith_138205 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 453, 19), fname_138204, 'endswith')
        # Calling endswith(args, kwargs) (line 453)
        endswith_call_result_138208 = invoke(stypy.reporting.localization.Localization(__file__, 453, 19), endswith_138205, *[unicode_138206], **kwargs_138207)
        
        # Testing the type of an if condition (line 453)
        if_condition_138209 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 453, 16), endswith_call_result_138208)
        # Assigning a type to the variable 'if_condition_138209' (line 453)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 453, 16), 'if_condition_138209', if_condition_138209)
        # SSA begins for if statement (line 453)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        pass
        # SSA branch for the else part of an if statement (line 453)
        module_type_store.open_ssa_branch('else')
        
        
        # Call to endswith(...): (line 455)
        # Processing the call arguments (line 455)
        unicode_138212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 455, 36), 'unicode', u'tex')
        # Processing the call keyword arguments (line 455)
        kwargs_138213 = {}
        # Getting the type of 'fname' (line 455)
        fname_138210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 21), 'fname', False)
        # Obtaining the member 'endswith' of a type (line 455)
        endswith_138211 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 455, 21), fname_138210, 'endswith')
        # Calling endswith(args, kwargs) (line 455)
        endswith_call_result_138214 = invoke(stypy.reporting.localization.Localization(__file__, 455, 21), endswith_138211, *[unicode_138212], **kwargs_138213)
        
        # Testing the type of an if condition (line 455)
        if_condition_138215 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 455, 21), endswith_call_result_138214)
        # Assigning a type to the variable 'if_condition_138215' (line 455)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 455, 21), 'if_condition_138215', if_condition_138215)
        # SSA begins for if statement (line 455)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        pass
        # SSA branch for the else part of an if statement (line 455)
        module_type_store.open_ssa_branch('else')
        
        
        # Call to endswith(...): (line 457)
        # Processing the call arguments (line 457)
        unicode_138218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 457, 36), 'unicode', u'baseline')
        # Processing the call keyword arguments (line 457)
        kwargs_138219 = {}
        # Getting the type of 'fname' (line 457)
        fname_138216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 21), 'fname', False)
        # Obtaining the member 'endswith' of a type (line 457)
        endswith_138217 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 457, 21), fname_138216, 'endswith')
        # Calling endswith(args, kwargs) (line 457)
        endswith_call_result_138220 = invoke(stypy.reporting.localization.Localization(__file__, 457, 21), endswith_138217, *[unicode_138218], **kwargs_138219)
        
        # Testing the type of an if condition (line 457)
        if_condition_138221 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 457, 21), endswith_call_result_138220)
        # Assigning a type to the variable 'if_condition_138221' (line 457)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 457, 21), 'if_condition_138221', if_condition_138221)
        # SSA begins for if statement (line 457)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        pass
        # SSA branch for the else part of an if statement (line 457)
        module_type_store.open_ssa_branch('else')
        
        
        # SSA begins for try-except statement (line 460)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to remove(...): (line 461)
        # Processing the call arguments (line 461)
        # Getting the type of 'fname' (line 461)
        fname_138224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 34), 'fname', False)
        # Processing the call keyword arguments (line 461)
        kwargs_138225 = {}
        # Getting the type of 'os' (line 461)
        os_138222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 24), 'os', False)
        # Obtaining the member 'remove' of a type (line 461)
        remove_138223 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 461, 24), os_138222, 'remove')
        # Calling remove(args, kwargs) (line 461)
        remove_call_result_138226 = invoke(stypy.reporting.localization.Localization(__file__, 461, 24), remove_138223, *[fname_138224], **kwargs_138225)
        
        # SSA branch for the except part of a try statement (line 460)
        # SSA branch for the except 'OSError' branch of a try statement (line 460)
        module_type_store.open_ssa_branch('except')
        pass
        # SSA join for try-except statement (line 460)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 457)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 455)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 453)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 427)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'dvifile' (line 465)
        dvifile_138227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 15), 'dvifile')
        # Assigning a type to the variable 'stypy_return_type' (line 465)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 465, 8), 'stypy_return_type', dvifile_138227)
        
        # ################# End of 'make_dvi_preview(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'make_dvi_preview' in the type store
        # Getting the type of 'stypy_return_type' (line 415)
        stypy_return_type_138228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_138228)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'make_dvi_preview'
        return stypy_return_type_138228


    @norecursion
    def make_png(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'make_png'
        module_type_store = module_type_store.open_function_context('make_png', 467, 4, False)
        # Assigning a type to the variable 'self' (line 468)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 468, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TexManager.make_png.__dict__.__setitem__('stypy_localization', localization)
        TexManager.make_png.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TexManager.make_png.__dict__.__setitem__('stypy_type_store', module_type_store)
        TexManager.make_png.__dict__.__setitem__('stypy_function_name', 'TexManager.make_png')
        TexManager.make_png.__dict__.__setitem__('stypy_param_names_list', ['tex', 'fontsize', 'dpi'])
        TexManager.make_png.__dict__.__setitem__('stypy_varargs_param_name', None)
        TexManager.make_png.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TexManager.make_png.__dict__.__setitem__('stypy_call_defaults', defaults)
        TexManager.make_png.__dict__.__setitem__('stypy_call_varargs', varargs)
        TexManager.make_png.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TexManager.make_png.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TexManager.make_png', ['tex', 'fontsize', 'dpi'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'make_png', localization, ['tex', 'fontsize', 'dpi'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'make_png(...)' code ##################

        unicode_138229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 472, (-1)), 'unicode', u"\n        generates a png file containing latex's rendering of tex string\n\n        returns the filename\n        ")
        
        # Assigning a Call to a Name (line 473):
        
        # Assigning a Call to a Name (line 473):
        
        # Call to get_basefile(...): (line 473)
        # Processing the call arguments (line 473)
        # Getting the type of 'tex' (line 473)
        tex_138232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 37), 'tex', False)
        # Getting the type of 'fontsize' (line 473)
        fontsize_138233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 42), 'fontsize', False)
        # Getting the type of 'dpi' (line 473)
        dpi_138234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 52), 'dpi', False)
        # Processing the call keyword arguments (line 473)
        kwargs_138235 = {}
        # Getting the type of 'self' (line 473)
        self_138230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 19), 'self', False)
        # Obtaining the member 'get_basefile' of a type (line 473)
        get_basefile_138231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 473, 19), self_138230, 'get_basefile')
        # Calling get_basefile(args, kwargs) (line 473)
        get_basefile_call_result_138236 = invoke(stypy.reporting.localization.Localization(__file__, 473, 19), get_basefile_138231, *[tex_138232, fontsize_138233, dpi_138234], **kwargs_138235)
        
        # Assigning a type to the variable 'basefile' (line 473)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 473, 8), 'basefile', get_basefile_call_result_138236)
        
        # Assigning a BinOp to a Name (line 474):
        
        # Assigning a BinOp to a Name (line 474):
        unicode_138237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 474, 18), 'unicode', u'%s.png')
        # Getting the type of 'basefile' (line 474)
        basefile_138238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 29), 'basefile')
        # Applying the binary operator '%' (line 474)
        result_mod_138239 = python_operator(stypy.reporting.localization.Localization(__file__, 474, 18), '%', unicode_138237, basefile_138238)
        
        # Assigning a type to the variable 'pngfile' (line 474)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 474, 8), 'pngfile', result_mod_138239)
        
        
        # Evaluating a boolean operation
        # Getting the type of 'DEBUG' (line 477)
        DEBUG_138240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 11), 'DEBUG')
        
        
        # Call to exists(...): (line 477)
        # Processing the call arguments (line 477)
        # Getting the type of 'pngfile' (line 477)
        pngfile_138244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 39), 'pngfile', False)
        # Processing the call keyword arguments (line 477)
        kwargs_138245 = {}
        # Getting the type of 'os' (line 477)
        os_138241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 24), 'os', False)
        # Obtaining the member 'path' of a type (line 477)
        path_138242 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 477, 24), os_138241, 'path')
        # Obtaining the member 'exists' of a type (line 477)
        exists_138243 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 477, 24), path_138242, 'exists')
        # Calling exists(args, kwargs) (line 477)
        exists_call_result_138246 = invoke(stypy.reporting.localization.Localization(__file__, 477, 24), exists_138243, *[pngfile_138244], **kwargs_138245)
        
        # Applying the 'not' unary operator (line 477)
        result_not__138247 = python_operator(stypy.reporting.localization.Localization(__file__, 477, 20), 'not', exists_call_result_138246)
        
        # Applying the binary operator 'or' (line 477)
        result_or_keyword_138248 = python_operator(stypy.reporting.localization.Localization(__file__, 477, 11), 'or', DEBUG_138240, result_not__138247)
        
        # Testing the type of an if condition (line 477)
        if_condition_138249 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 477, 8), result_or_keyword_138248)
        # Assigning a type to the variable 'if_condition_138249' (line 477)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 477, 8), 'if_condition_138249', if_condition_138249)
        # SSA begins for if statement (line 477)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 478):
        
        # Assigning a Call to a Name (line 478):
        
        # Call to make_dvi(...): (line 478)
        # Processing the call arguments (line 478)
        # Getting the type of 'tex' (line 478)
        tex_138252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 36), 'tex', False)
        # Getting the type of 'fontsize' (line 478)
        fontsize_138253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 41), 'fontsize', False)
        # Processing the call keyword arguments (line 478)
        kwargs_138254 = {}
        # Getting the type of 'self' (line 478)
        self_138250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 22), 'self', False)
        # Obtaining the member 'make_dvi' of a type (line 478)
        make_dvi_138251 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 478, 22), self_138250, 'make_dvi')
        # Calling make_dvi(args, kwargs) (line 478)
        make_dvi_call_result_138255 = invoke(stypy.reporting.localization.Localization(__file__, 478, 22), make_dvi_138251, *[tex_138252, fontsize_138253], **kwargs_138254)
        
        # Assigning a type to the variable 'dvifile' (line 478)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 478, 12), 'dvifile', make_dvi_call_result_138255)
        
        # Assigning a List to a Name (line 479):
        
        # Assigning a List to a Name (line 479):
        
        # Obtaining an instance of the builtin type 'list' (line 479)
        list_138256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 479, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 479)
        # Adding element type (line 479)
        
        # Call to str(...): (line 479)
        # Processing the call arguments (line 479)
        unicode_138258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 479, 27), 'unicode', u'dvipng')
        # Processing the call keyword arguments (line 479)
        kwargs_138259 = {}
        # Getting the type of 'str' (line 479)
        str_138257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 23), 'str', False)
        # Calling str(args, kwargs) (line 479)
        str_call_result_138260 = invoke(stypy.reporting.localization.Localization(__file__, 479, 23), str_138257, *[unicode_138258], **kwargs_138259)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 479, 22), list_138256, str_call_result_138260)
        # Adding element type (line 479)
        unicode_138261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 479, 38), 'unicode', u'-bg')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 479, 22), list_138256, unicode_138261)
        # Adding element type (line 479)
        unicode_138262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 479, 45), 'unicode', u'Transparent')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 479, 22), list_138256, unicode_138262)
        # Adding element type (line 479)
        unicode_138263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 479, 60), 'unicode', u'-D')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 479, 22), list_138256, unicode_138263)
        # Adding element type (line 479)
        
        # Call to str(...): (line 479)
        # Processing the call arguments (line 479)
        # Getting the type of 'dpi' (line 479)
        dpi_138265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 70), 'dpi', False)
        # Processing the call keyword arguments (line 479)
        kwargs_138266 = {}
        # Getting the type of 'str' (line 479)
        str_138264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 66), 'str', False)
        # Calling str(args, kwargs) (line 479)
        str_call_result_138267 = invoke(stypy.reporting.localization.Localization(__file__, 479, 66), str_138264, *[dpi_138265], **kwargs_138266)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 479, 22), list_138256, str_call_result_138267)
        # Adding element type (line 479)
        unicode_138268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 480, 23), 'unicode', u'-T')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 479, 22), list_138256, unicode_138268)
        # Adding element type (line 479)
        unicode_138269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 480, 29), 'unicode', u'tight')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 479, 22), list_138256, unicode_138269)
        # Adding element type (line 479)
        unicode_138270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 480, 38), 'unicode', u'-o')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 479, 22), list_138256, unicode_138270)
        # Adding element type (line 479)
        
        # Call to basename(...): (line 480)
        # Processing the call arguments (line 480)
        # Getting the type of 'pngfile' (line 480)
        pngfile_138274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 61), 'pngfile', False)
        # Processing the call keyword arguments (line 480)
        kwargs_138275 = {}
        # Getting the type of 'os' (line 480)
        os_138271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 44), 'os', False)
        # Obtaining the member 'path' of a type (line 480)
        path_138272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 480, 44), os_138271, 'path')
        # Obtaining the member 'basename' of a type (line 480)
        basename_138273 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 480, 44), path_138272, 'basename')
        # Calling basename(args, kwargs) (line 480)
        basename_call_result_138276 = invoke(stypy.reporting.localization.Localization(__file__, 480, 44), basename_138273, *[pngfile_138274], **kwargs_138275)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 479, 22), list_138256, basename_call_result_138276)
        # Adding element type (line 479)
        
        # Call to basename(...): (line 481)
        # Processing the call arguments (line 481)
        # Getting the type of 'dvifile' (line 481)
        dvifile_138280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 40), 'dvifile', False)
        # Processing the call keyword arguments (line 481)
        kwargs_138281 = {}
        # Getting the type of 'os' (line 481)
        os_138277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 23), 'os', False)
        # Obtaining the member 'path' of a type (line 481)
        path_138278 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 481, 23), os_138277, 'path')
        # Obtaining the member 'basename' of a type (line 481)
        basename_138279 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 481, 23), path_138278, 'basename')
        # Calling basename(args, kwargs) (line 481)
        basename_call_result_138282 = invoke(stypy.reporting.localization.Localization(__file__, 481, 23), basename_138279, *[dvifile_138280], **kwargs_138281)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 479, 22), list_138256, basename_call_result_138282)
        
        # Assigning a type to the variable 'command' (line 479)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 479, 12), 'command', list_138256)
        
        # Call to report(...): (line 482)
        # Processing the call arguments (line 482)
        # Getting the type of 'command' (line 482)
        command_138286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 31), 'command', False)
        unicode_138287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 482, 40), 'unicode', u'debug')
        # Processing the call keyword arguments (line 482)
        kwargs_138288 = {}
        # Getting the type of 'mpl' (line 482)
        mpl_138283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 12), 'mpl', False)
        # Obtaining the member 'verbose' of a type (line 482)
        verbose_138284 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 482, 12), mpl_138283, 'verbose')
        # Obtaining the member 'report' of a type (line 482)
        report_138285 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 482, 12), verbose_138284, 'report')
        # Calling report(args, kwargs) (line 482)
        report_call_result_138289 = invoke(stypy.reporting.localization.Localization(__file__, 482, 12), report_138285, *[command_138286, unicode_138287], **kwargs_138288)
        
        
        
        # SSA begins for try-except statement (line 483)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Name (line 484):
        
        # Assigning a Call to a Name (line 484):
        
        # Call to check_output(...): (line 484)
        # Processing the call arguments (line 484)
        # Getting the type of 'command' (line 484)
        command_138292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 49), 'command', False)
        # Processing the call keyword arguments (line 484)
        # Getting the type of 'self' (line 485)
        self_138293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 53), 'self', False)
        # Obtaining the member 'texcache' of a type (line 485)
        texcache_138294 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 485, 53), self_138293, 'texcache')
        keyword_138295 = texcache_138294
        # Getting the type of 'subprocess' (line 486)
        subprocess_138296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 56), 'subprocess', False)
        # Obtaining the member 'STDOUT' of a type (line 486)
        STDOUT_138297 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 486, 56), subprocess_138296, 'STDOUT')
        keyword_138298 = STDOUT_138297
        kwargs_138299 = {'cwd': keyword_138295, 'stderr': keyword_138298}
        # Getting the type of 'subprocess' (line 484)
        subprocess_138290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 25), 'subprocess', False)
        # Obtaining the member 'check_output' of a type (line 484)
        check_output_138291 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 484, 25), subprocess_138290, 'check_output')
        # Calling check_output(args, kwargs) (line 484)
        check_output_call_result_138300 = invoke(stypy.reporting.localization.Localization(__file__, 484, 25), check_output_138291, *[command_138292], **kwargs_138299)
        
        # Assigning a type to the variable 'report' (line 484)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 484, 16), 'report', check_output_call_result_138300)
        # SSA branch for the except part of a try statement (line 483)
        # SSA branch for the except 'Attribute' branch of a try statement (line 483)
        # Storing handler type
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'subprocess' (line 487)
        subprocess_138301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 19), 'subprocess')
        # Obtaining the member 'CalledProcessError' of a type (line 487)
        CalledProcessError_138302 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 487, 19), subprocess_138301, 'CalledProcessError')
        # Assigning a type to the variable 'exc' (line 487)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 487, 12), 'exc', CalledProcessError_138302)
        
        # Call to RuntimeError(...): (line 488)
        # Processing the call arguments (line 488)
        unicode_138304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 489, 21), 'unicode', u'dvipng was not able to process the following string:\n%s\n\nHere is the full report generated by dvipng:\n%s \n\n')
        
        # Obtaining an instance of the builtin type 'tuple' (line 492)
        tuple_138305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 492, 31), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 492)
        # Adding element type (line 492)
        
        # Call to repr(...): (line 492)
        # Processing the call arguments (line 492)
        
        # Call to encode(...): (line 492)
        # Processing the call arguments (line 492)
        unicode_138309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 492, 47), 'unicode', u'unicode_escape')
        # Processing the call keyword arguments (line 492)
        kwargs_138310 = {}
        # Getting the type of 'tex' (line 492)
        tex_138307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 36), 'tex', False)
        # Obtaining the member 'encode' of a type (line 492)
        encode_138308 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 492, 36), tex_138307, 'encode')
        # Calling encode(args, kwargs) (line 492)
        encode_call_result_138311 = invoke(stypy.reporting.localization.Localization(__file__, 492, 36), encode_138308, *[unicode_138309], **kwargs_138310)
        
        # Processing the call keyword arguments (line 492)
        kwargs_138312 = {}
        # Getting the type of 'repr' (line 492)
        repr_138306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 31), 'repr', False)
        # Calling repr(args, kwargs) (line 492)
        repr_call_result_138313 = invoke(stypy.reporting.localization.Localization(__file__, 492, 31), repr_138306, *[encode_call_result_138311], **kwargs_138312)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 492, 31), tuple_138305, repr_call_result_138313)
        # Adding element type (line 492)
        
        # Call to decode(...): (line 493)
        # Processing the call arguments (line 493)
        unicode_138317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 493, 49), 'unicode', u'utf-8')
        # Processing the call keyword arguments (line 493)
        kwargs_138318 = {}
        # Getting the type of 'exc' (line 493)
        exc_138314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 31), 'exc', False)
        # Obtaining the member 'output' of a type (line 493)
        output_138315 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 493, 31), exc_138314, 'output')
        # Obtaining the member 'decode' of a type (line 493)
        decode_138316 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 493, 31), output_138315, 'decode')
        # Calling decode(args, kwargs) (line 493)
        decode_call_result_138319 = invoke(stypy.reporting.localization.Localization(__file__, 493, 31), decode_138316, *[unicode_138317], **kwargs_138318)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 492, 31), tuple_138305, decode_call_result_138319)
        
        # Applying the binary operator '%' (line 489)
        result_mod_138320 = python_operator(stypy.reporting.localization.Localization(__file__, 489, 21), '%', unicode_138304, tuple_138305)
        
        # Processing the call keyword arguments (line 488)
        kwargs_138321 = {}
        # Getting the type of 'RuntimeError' (line 488)
        RuntimeError_138303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 22), 'RuntimeError', False)
        # Calling RuntimeError(args, kwargs) (line 488)
        RuntimeError_call_result_138322 = invoke(stypy.reporting.localization.Localization(__file__, 488, 22), RuntimeError_138303, *[result_mod_138320], **kwargs_138321)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 488, 16), RuntimeError_call_result_138322, 'raise parameter', BaseException)
        # SSA join for try-except statement (line 483)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to report(...): (line 494)
        # Processing the call arguments (line 494)
        # Getting the type of 'report' (line 494)
        report_138326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 31), 'report', False)
        unicode_138327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 494, 39), 'unicode', u'debug')
        # Processing the call keyword arguments (line 494)
        kwargs_138328 = {}
        # Getting the type of 'mpl' (line 494)
        mpl_138323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 12), 'mpl', False)
        # Obtaining the member 'verbose' of a type (line 494)
        verbose_138324 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 494, 12), mpl_138323, 'verbose')
        # Obtaining the member 'report' of a type (line 494)
        report_138325 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 494, 12), verbose_138324, 'report')
        # Calling report(args, kwargs) (line 494)
        report_call_result_138329 = invoke(stypy.reporting.localization.Localization(__file__, 494, 12), report_138325, *[report_138326, unicode_138327], **kwargs_138328)
        
        # SSA join for if statement (line 477)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'pngfile' (line 496)
        pngfile_138330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 15), 'pngfile')
        # Assigning a type to the variable 'stypy_return_type' (line 496)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 496, 8), 'stypy_return_type', pngfile_138330)
        
        # ################# End of 'make_png(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'make_png' in the type store
        # Getting the type of 'stypy_return_type' (line 467)
        stypy_return_type_138331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_138331)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'make_png'
        return stypy_return_type_138331


    @norecursion
    def make_ps(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'make_ps'
        module_type_store = module_type_store.open_function_context('make_ps', 498, 4, False)
        # Assigning a type to the variable 'self' (line 499)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 499, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TexManager.make_ps.__dict__.__setitem__('stypy_localization', localization)
        TexManager.make_ps.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TexManager.make_ps.__dict__.__setitem__('stypy_type_store', module_type_store)
        TexManager.make_ps.__dict__.__setitem__('stypy_function_name', 'TexManager.make_ps')
        TexManager.make_ps.__dict__.__setitem__('stypy_param_names_list', ['tex', 'fontsize'])
        TexManager.make_ps.__dict__.__setitem__('stypy_varargs_param_name', None)
        TexManager.make_ps.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TexManager.make_ps.__dict__.__setitem__('stypy_call_defaults', defaults)
        TexManager.make_ps.__dict__.__setitem__('stypy_call_varargs', varargs)
        TexManager.make_ps.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TexManager.make_ps.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TexManager.make_ps', ['tex', 'fontsize'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'make_ps', localization, ['tex', 'fontsize'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'make_ps(...)' code ##################

        unicode_138332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 503, (-1)), 'unicode', u"\n        generates a postscript file containing latex's rendering of tex string\n\n        returns the file name\n        ")
        
        # Assigning a Call to a Name (line 504):
        
        # Assigning a Call to a Name (line 504):
        
        # Call to get_basefile(...): (line 504)
        # Processing the call arguments (line 504)
        # Getting the type of 'tex' (line 504)
        tex_138335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 37), 'tex', False)
        # Getting the type of 'fontsize' (line 504)
        fontsize_138336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 42), 'fontsize', False)
        # Processing the call keyword arguments (line 504)
        kwargs_138337 = {}
        # Getting the type of 'self' (line 504)
        self_138333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 19), 'self', False)
        # Obtaining the member 'get_basefile' of a type (line 504)
        get_basefile_138334 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 504, 19), self_138333, 'get_basefile')
        # Calling get_basefile(args, kwargs) (line 504)
        get_basefile_call_result_138338 = invoke(stypy.reporting.localization.Localization(__file__, 504, 19), get_basefile_138334, *[tex_138335, fontsize_138336], **kwargs_138337)
        
        # Assigning a type to the variable 'basefile' (line 504)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 504, 8), 'basefile', get_basefile_call_result_138338)
        
        # Assigning a BinOp to a Name (line 505):
        
        # Assigning a BinOp to a Name (line 505):
        unicode_138339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 505, 17), 'unicode', u'%s.epsf')
        # Getting the type of 'basefile' (line 505)
        basefile_138340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 29), 'basefile')
        # Applying the binary operator '%' (line 505)
        result_mod_138341 = python_operator(stypy.reporting.localization.Localization(__file__, 505, 17), '%', unicode_138339, basefile_138340)
        
        # Assigning a type to the variable 'psfile' (line 505)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 505, 8), 'psfile', result_mod_138341)
        
        
        # Evaluating a boolean operation
        # Getting the type of 'DEBUG' (line 507)
        DEBUG_138342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 11), 'DEBUG')
        
        
        # Call to exists(...): (line 507)
        # Processing the call arguments (line 507)
        # Getting the type of 'psfile' (line 507)
        psfile_138346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 39), 'psfile', False)
        # Processing the call keyword arguments (line 507)
        kwargs_138347 = {}
        # Getting the type of 'os' (line 507)
        os_138343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 24), 'os', False)
        # Obtaining the member 'path' of a type (line 507)
        path_138344 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 507, 24), os_138343, 'path')
        # Obtaining the member 'exists' of a type (line 507)
        exists_138345 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 507, 24), path_138344, 'exists')
        # Calling exists(args, kwargs) (line 507)
        exists_call_result_138348 = invoke(stypy.reporting.localization.Localization(__file__, 507, 24), exists_138345, *[psfile_138346], **kwargs_138347)
        
        # Applying the 'not' unary operator (line 507)
        result_not__138349 = python_operator(stypy.reporting.localization.Localization(__file__, 507, 20), 'not', exists_call_result_138348)
        
        # Applying the binary operator 'or' (line 507)
        result_or_keyword_138350 = python_operator(stypy.reporting.localization.Localization(__file__, 507, 11), 'or', DEBUG_138342, result_not__138349)
        
        # Testing the type of an if condition (line 507)
        if_condition_138351 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 507, 8), result_or_keyword_138350)
        # Assigning a type to the variable 'if_condition_138351' (line 507)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 507, 8), 'if_condition_138351', if_condition_138351)
        # SSA begins for if statement (line 507)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 508):
        
        # Assigning a Call to a Name (line 508):
        
        # Call to make_dvi(...): (line 508)
        # Processing the call arguments (line 508)
        # Getting the type of 'tex' (line 508)
        tex_138354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 36), 'tex', False)
        # Getting the type of 'fontsize' (line 508)
        fontsize_138355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 41), 'fontsize', False)
        # Processing the call keyword arguments (line 508)
        kwargs_138356 = {}
        # Getting the type of 'self' (line 508)
        self_138352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 22), 'self', False)
        # Obtaining the member 'make_dvi' of a type (line 508)
        make_dvi_138353 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 508, 22), self_138352, 'make_dvi')
        # Calling make_dvi(args, kwargs) (line 508)
        make_dvi_call_result_138357 = invoke(stypy.reporting.localization.Localization(__file__, 508, 22), make_dvi_138353, *[tex_138354, fontsize_138355], **kwargs_138356)
        
        # Assigning a type to the variable 'dvifile' (line 508)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 508, 12), 'dvifile', make_dvi_call_result_138357)
        
        # Assigning a List to a Name (line 509):
        
        # Assigning a List to a Name (line 509):
        
        # Obtaining an instance of the builtin type 'list' (line 509)
        list_138358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 509, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 509)
        # Adding element type (line 509)
        
        # Call to str(...): (line 509)
        # Processing the call arguments (line 509)
        unicode_138360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 509, 27), 'unicode', u'dvips')
        # Processing the call keyword arguments (line 509)
        kwargs_138361 = {}
        # Getting the type of 'str' (line 509)
        str_138359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 23), 'str', False)
        # Calling str(args, kwargs) (line 509)
        str_call_result_138362 = invoke(stypy.reporting.localization.Localization(__file__, 509, 23), str_138359, *[unicode_138360], **kwargs_138361)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 509, 22), list_138358, str_call_result_138362)
        # Adding element type (line 509)
        unicode_138363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 509, 37), 'unicode', u'-q')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 509, 22), list_138358, unicode_138363)
        # Adding element type (line 509)
        unicode_138364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 509, 43), 'unicode', u'-E')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 509, 22), list_138358, unicode_138364)
        # Adding element type (line 509)
        unicode_138365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 509, 49), 'unicode', u'-o')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 509, 22), list_138358, unicode_138365)
        # Adding element type (line 509)
        
        # Call to basename(...): (line 510)
        # Processing the call arguments (line 510)
        # Getting the type of 'psfile' (line 510)
        psfile_138369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 40), 'psfile', False)
        # Processing the call keyword arguments (line 510)
        kwargs_138370 = {}
        # Getting the type of 'os' (line 510)
        os_138366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 23), 'os', False)
        # Obtaining the member 'path' of a type (line 510)
        path_138367 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 510, 23), os_138366, 'path')
        # Obtaining the member 'basename' of a type (line 510)
        basename_138368 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 510, 23), path_138367, 'basename')
        # Calling basename(args, kwargs) (line 510)
        basename_call_result_138371 = invoke(stypy.reporting.localization.Localization(__file__, 510, 23), basename_138368, *[psfile_138369], **kwargs_138370)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 509, 22), list_138358, basename_call_result_138371)
        # Adding element type (line 509)
        
        # Call to basename(...): (line 511)
        # Processing the call arguments (line 511)
        # Getting the type of 'dvifile' (line 511)
        dvifile_138375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 40), 'dvifile', False)
        # Processing the call keyword arguments (line 511)
        kwargs_138376 = {}
        # Getting the type of 'os' (line 511)
        os_138372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 23), 'os', False)
        # Obtaining the member 'path' of a type (line 511)
        path_138373 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 511, 23), os_138372, 'path')
        # Obtaining the member 'basename' of a type (line 511)
        basename_138374 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 511, 23), path_138373, 'basename')
        # Calling basename(args, kwargs) (line 511)
        basename_call_result_138377 = invoke(stypy.reporting.localization.Localization(__file__, 511, 23), basename_138374, *[dvifile_138375], **kwargs_138376)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 509, 22), list_138358, basename_call_result_138377)
        
        # Assigning a type to the variable 'command' (line 509)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 509, 12), 'command', list_138358)
        
        # Call to report(...): (line 512)
        # Processing the call arguments (line 512)
        # Getting the type of 'command' (line 512)
        command_138381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 31), 'command', False)
        unicode_138382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 512, 40), 'unicode', u'debug')
        # Processing the call keyword arguments (line 512)
        kwargs_138383 = {}
        # Getting the type of 'mpl' (line 512)
        mpl_138378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 12), 'mpl', False)
        # Obtaining the member 'verbose' of a type (line 512)
        verbose_138379 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 512, 12), mpl_138378, 'verbose')
        # Obtaining the member 'report' of a type (line 512)
        report_138380 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 512, 12), verbose_138379, 'report')
        # Calling report(args, kwargs) (line 512)
        report_call_result_138384 = invoke(stypy.reporting.localization.Localization(__file__, 512, 12), report_138380, *[command_138381, unicode_138382], **kwargs_138383)
        
        
        
        # SSA begins for try-except statement (line 513)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Name (line 514):
        
        # Assigning a Call to a Name (line 514):
        
        # Call to check_output(...): (line 514)
        # Processing the call arguments (line 514)
        # Getting the type of 'command' (line 514)
        command_138387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 49), 'command', False)
        # Processing the call keyword arguments (line 514)
        # Getting the type of 'self' (line 515)
        self_138388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 53), 'self', False)
        # Obtaining the member 'texcache' of a type (line 515)
        texcache_138389 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 515, 53), self_138388, 'texcache')
        keyword_138390 = texcache_138389
        # Getting the type of 'subprocess' (line 516)
        subprocess_138391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 56), 'subprocess', False)
        # Obtaining the member 'STDOUT' of a type (line 516)
        STDOUT_138392 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 516, 56), subprocess_138391, 'STDOUT')
        keyword_138393 = STDOUT_138392
        kwargs_138394 = {'cwd': keyword_138390, 'stderr': keyword_138393}
        # Getting the type of 'subprocess' (line 514)
        subprocess_138385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 25), 'subprocess', False)
        # Obtaining the member 'check_output' of a type (line 514)
        check_output_138386 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 514, 25), subprocess_138385, 'check_output')
        # Calling check_output(args, kwargs) (line 514)
        check_output_call_result_138395 = invoke(stypy.reporting.localization.Localization(__file__, 514, 25), check_output_138386, *[command_138387], **kwargs_138394)
        
        # Assigning a type to the variable 'report' (line 514)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 514, 16), 'report', check_output_call_result_138395)
        # SSA branch for the except part of a try statement (line 513)
        # SSA branch for the except 'Attribute' branch of a try statement (line 513)
        # Storing handler type
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'subprocess' (line 517)
        subprocess_138396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 19), 'subprocess')
        # Obtaining the member 'CalledProcessError' of a type (line 517)
        CalledProcessError_138397 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 517, 19), subprocess_138396, 'CalledProcessError')
        # Assigning a type to the variable 'exc' (line 517)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 517, 12), 'exc', CalledProcessError_138397)
        
        # Call to RuntimeError(...): (line 518)
        # Processing the call arguments (line 518)
        unicode_138399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 519, 21), 'unicode', u'dvips was not able to process the following string:\n%s\n\nHere is the full report generated by dvips:\n%s \n\n')
        
        # Obtaining an instance of the builtin type 'tuple' (line 522)
        tuple_138400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 522, 31), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 522)
        # Adding element type (line 522)
        
        # Call to repr(...): (line 522)
        # Processing the call arguments (line 522)
        
        # Call to encode(...): (line 522)
        # Processing the call arguments (line 522)
        unicode_138404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 522, 47), 'unicode', u'unicode_escape')
        # Processing the call keyword arguments (line 522)
        kwargs_138405 = {}
        # Getting the type of 'tex' (line 522)
        tex_138402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 36), 'tex', False)
        # Obtaining the member 'encode' of a type (line 522)
        encode_138403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 522, 36), tex_138402, 'encode')
        # Calling encode(args, kwargs) (line 522)
        encode_call_result_138406 = invoke(stypy.reporting.localization.Localization(__file__, 522, 36), encode_138403, *[unicode_138404], **kwargs_138405)
        
        # Processing the call keyword arguments (line 522)
        kwargs_138407 = {}
        # Getting the type of 'repr' (line 522)
        repr_138401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 31), 'repr', False)
        # Calling repr(args, kwargs) (line 522)
        repr_call_result_138408 = invoke(stypy.reporting.localization.Localization(__file__, 522, 31), repr_138401, *[encode_call_result_138406], **kwargs_138407)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 522, 31), tuple_138400, repr_call_result_138408)
        # Adding element type (line 522)
        
        # Call to decode(...): (line 523)
        # Processing the call arguments (line 523)
        unicode_138412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 523, 49), 'unicode', u'utf-8')
        # Processing the call keyword arguments (line 523)
        kwargs_138413 = {}
        # Getting the type of 'exc' (line 523)
        exc_138409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 31), 'exc', False)
        # Obtaining the member 'output' of a type (line 523)
        output_138410 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 523, 31), exc_138409, 'output')
        # Obtaining the member 'decode' of a type (line 523)
        decode_138411 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 523, 31), output_138410, 'decode')
        # Calling decode(args, kwargs) (line 523)
        decode_call_result_138414 = invoke(stypy.reporting.localization.Localization(__file__, 523, 31), decode_138411, *[unicode_138412], **kwargs_138413)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 522, 31), tuple_138400, decode_call_result_138414)
        
        # Applying the binary operator '%' (line 519)
        result_mod_138415 = python_operator(stypy.reporting.localization.Localization(__file__, 519, 21), '%', unicode_138399, tuple_138400)
        
        # Processing the call keyword arguments (line 518)
        kwargs_138416 = {}
        # Getting the type of 'RuntimeError' (line 518)
        RuntimeError_138398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 22), 'RuntimeError', False)
        # Calling RuntimeError(args, kwargs) (line 518)
        RuntimeError_call_result_138417 = invoke(stypy.reporting.localization.Localization(__file__, 518, 22), RuntimeError_138398, *[result_mod_138415], **kwargs_138416)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 518, 16), RuntimeError_call_result_138417, 'raise parameter', BaseException)
        # SSA join for try-except statement (line 513)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to report(...): (line 524)
        # Processing the call arguments (line 524)
        # Getting the type of 'report' (line 524)
        report_138421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 31), 'report', False)
        unicode_138422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 524, 39), 'unicode', u'debug')
        # Processing the call keyword arguments (line 524)
        kwargs_138423 = {}
        # Getting the type of 'mpl' (line 524)
        mpl_138418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 12), 'mpl', False)
        # Obtaining the member 'verbose' of a type (line 524)
        verbose_138419 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 524, 12), mpl_138418, 'verbose')
        # Obtaining the member 'report' of a type (line 524)
        report_138420 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 524, 12), verbose_138419, 'report')
        # Calling report(args, kwargs) (line 524)
        report_call_result_138424 = invoke(stypy.reporting.localization.Localization(__file__, 524, 12), report_138420, *[report_138421, unicode_138422], **kwargs_138423)
        
        # SSA join for if statement (line 507)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'psfile' (line 526)
        psfile_138425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 15), 'psfile')
        # Assigning a type to the variable 'stypy_return_type' (line 526)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 526, 8), 'stypy_return_type', psfile_138425)
        
        # ################# End of 'make_ps(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'make_ps' in the type store
        # Getting the type of 'stypy_return_type' (line 498)
        stypy_return_type_138426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_138426)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'make_ps'
        return stypy_return_type_138426


    @norecursion
    def get_ps_bbox(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_ps_bbox'
        module_type_store = module_type_store.open_function_context('get_ps_bbox', 528, 4, False)
        # Assigning a type to the variable 'self' (line 529)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 529, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TexManager.get_ps_bbox.__dict__.__setitem__('stypy_localization', localization)
        TexManager.get_ps_bbox.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TexManager.get_ps_bbox.__dict__.__setitem__('stypy_type_store', module_type_store)
        TexManager.get_ps_bbox.__dict__.__setitem__('stypy_function_name', 'TexManager.get_ps_bbox')
        TexManager.get_ps_bbox.__dict__.__setitem__('stypy_param_names_list', ['tex', 'fontsize'])
        TexManager.get_ps_bbox.__dict__.__setitem__('stypy_varargs_param_name', None)
        TexManager.get_ps_bbox.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TexManager.get_ps_bbox.__dict__.__setitem__('stypy_call_defaults', defaults)
        TexManager.get_ps_bbox.__dict__.__setitem__('stypy_call_varargs', varargs)
        TexManager.get_ps_bbox.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TexManager.get_ps_bbox.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TexManager.get_ps_bbox', ['tex', 'fontsize'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_ps_bbox', localization, ['tex', 'fontsize'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_ps_bbox(...)' code ##################

        unicode_138427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 532, (-1)), 'unicode', u"\n        returns a list containing the postscript bounding box for latex's\n        rendering of the tex string\n        ")
        
        # Assigning a Call to a Name (line 533):
        
        # Assigning a Call to a Name (line 533):
        
        # Call to make_ps(...): (line 533)
        # Processing the call arguments (line 533)
        # Getting the type of 'tex' (line 533)
        tex_138430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 30), 'tex', False)
        # Getting the type of 'fontsize' (line 533)
        fontsize_138431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 35), 'fontsize', False)
        # Processing the call keyword arguments (line 533)
        kwargs_138432 = {}
        # Getting the type of 'self' (line 533)
        self_138428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 17), 'self', False)
        # Obtaining the member 'make_ps' of a type (line 533)
        make_ps_138429 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 533, 17), self_138428, 'make_ps')
        # Calling make_ps(args, kwargs) (line 533)
        make_ps_call_result_138433 = invoke(stypy.reporting.localization.Localization(__file__, 533, 17), make_ps_138429, *[tex_138430, fontsize_138431], **kwargs_138432)
        
        # Assigning a type to the variable 'psfile' (line 533)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 533, 8), 'psfile', make_ps_call_result_138433)
        
        # Call to open(...): (line 534)
        # Processing the call arguments (line 534)
        # Getting the type of 'psfile' (line 534)
        psfile_138435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 18), 'psfile', False)
        # Processing the call keyword arguments (line 534)
        kwargs_138436 = {}
        # Getting the type of 'open' (line 534)
        open_138434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 13), 'open', False)
        # Calling open(args, kwargs) (line 534)
        open_call_result_138437 = invoke(stypy.reporting.localization.Localization(__file__, 534, 13), open_138434, *[psfile_138435], **kwargs_138436)
        
        with_138438 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 534, 13), open_call_result_138437, 'with parameter', '__enter__', '__exit__')

        if with_138438:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 534)
            enter___138439 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 534, 13), open_call_result_138437, '__enter__')
            with_enter_138440 = invoke(stypy.reporting.localization.Localization(__file__, 534, 13), enter___138439)
            # Assigning a type to the variable 'ps' (line 534)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 534, 13), 'ps', with_enter_138440)
            
            # Getting the type of 'ps' (line 535)
            ps_138441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 24), 'ps')
            # Testing the type of a for loop iterable (line 535)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 535, 12), ps_138441)
            # Getting the type of the for loop variable (line 535)
            for_loop_var_138442 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 535, 12), ps_138441)
            # Assigning a type to the variable 'line' (line 535)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 535, 12), 'line', for_loop_var_138442)
            # SSA begins for a for statement (line 535)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            
            # Call to startswith(...): (line 536)
            # Processing the call arguments (line 536)
            unicode_138445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 536, 35), 'unicode', u'%%BoundingBox:')
            # Processing the call keyword arguments (line 536)
            kwargs_138446 = {}
            # Getting the type of 'line' (line 536)
            line_138443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 19), 'line', False)
            # Obtaining the member 'startswith' of a type (line 536)
            startswith_138444 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 536, 19), line_138443, 'startswith')
            # Calling startswith(args, kwargs) (line 536)
            startswith_call_result_138447 = invoke(stypy.reporting.localization.Localization(__file__, 536, 19), startswith_138444, *[unicode_138445], **kwargs_138446)
            
            # Testing the type of an if condition (line 536)
            if_condition_138448 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 536, 16), startswith_call_result_138447)
            # Assigning a type to the variable 'if_condition_138448' (line 536)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 536, 16), 'if_condition_138448', if_condition_138448)
            # SSA begins for if statement (line 536)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Calculating list comprehension
            # Calculating comprehension expression
            
            # Obtaining the type of the subscript
            int_138453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 537, 61), 'int')
            slice_138454 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 537, 48), int_138453, None, None)
            
            # Call to split(...): (line 537)
            # Processing the call keyword arguments (line 537)
            kwargs_138457 = {}
            # Getting the type of 'line' (line 537)
            line_138455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 48), 'line', False)
            # Obtaining the member 'split' of a type (line 537)
            split_138456 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 537, 48), line_138455, 'split')
            # Calling split(args, kwargs) (line 537)
            split_call_result_138458 = invoke(stypy.reporting.localization.Localization(__file__, 537, 48), split_138456, *[], **kwargs_138457)
            
            # Obtaining the member '__getitem__' of a type (line 537)
            getitem___138459 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 537, 48), split_call_result_138458, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 537)
            subscript_call_result_138460 = invoke(stypy.reporting.localization.Localization(__file__, 537, 48), getitem___138459, slice_138454)
            
            comprehension_138461 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 537, 28), subscript_call_result_138460)
            # Assigning a type to the variable 'val' (line 537)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 537, 28), 'val', comprehension_138461)
            
            # Call to int(...): (line 537)
            # Processing the call arguments (line 537)
            # Getting the type of 'val' (line 537)
            val_138450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 32), 'val', False)
            # Processing the call keyword arguments (line 537)
            kwargs_138451 = {}
            # Getting the type of 'int' (line 537)
            int_138449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 28), 'int', False)
            # Calling int(args, kwargs) (line 537)
            int_call_result_138452 = invoke(stypy.reporting.localization.Localization(__file__, 537, 28), int_138449, *[val_138450], **kwargs_138451)
            
            list_138462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 537, 28), 'list')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 537, 28), list_138462, int_call_result_138452)
            # Assigning a type to the variable 'stypy_return_type' (line 537)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 537, 20), 'stypy_return_type', list_138462)
            # SSA join for if statement (line 536)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()
            
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 534)
            exit___138463 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 534, 13), open_call_result_138437, '__exit__')
            with_exit_138464 = invoke(stypy.reporting.localization.Localization(__file__, 534, 13), exit___138463, None, None, None)

        
        # Call to RuntimeError(...): (line 538)
        # Processing the call arguments (line 538)
        unicode_138466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 538, 27), 'unicode', u'Could not parse %s')
        # Getting the type of 'psfile' (line 538)
        psfile_138467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 50), 'psfile', False)
        # Applying the binary operator '%' (line 538)
        result_mod_138468 = python_operator(stypy.reporting.localization.Localization(__file__, 538, 27), '%', unicode_138466, psfile_138467)
        
        # Processing the call keyword arguments (line 538)
        kwargs_138469 = {}
        # Getting the type of 'RuntimeError' (line 538)
        RuntimeError_138465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 14), 'RuntimeError', False)
        # Calling RuntimeError(args, kwargs) (line 538)
        RuntimeError_call_result_138470 = invoke(stypy.reporting.localization.Localization(__file__, 538, 14), RuntimeError_138465, *[result_mod_138468], **kwargs_138469)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 538, 8), RuntimeError_call_result_138470, 'raise parameter', BaseException)
        
        # ################# End of 'get_ps_bbox(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_ps_bbox' in the type store
        # Getting the type of 'stypy_return_type' (line 528)
        stypy_return_type_138471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_138471)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_ps_bbox'
        return stypy_return_type_138471


    @norecursion
    def get_grey(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 540)
        None_138472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 37), 'None')
        # Getting the type of 'None' (line 540)
        None_138473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 47), 'None')
        defaults = [None_138472, None_138473]
        # Create a new context for function 'get_grey'
        module_type_store = module_type_store.open_function_context('get_grey', 540, 4, False)
        # Assigning a type to the variable 'self' (line 541)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 541, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TexManager.get_grey.__dict__.__setitem__('stypy_localization', localization)
        TexManager.get_grey.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TexManager.get_grey.__dict__.__setitem__('stypy_type_store', module_type_store)
        TexManager.get_grey.__dict__.__setitem__('stypy_function_name', 'TexManager.get_grey')
        TexManager.get_grey.__dict__.__setitem__('stypy_param_names_list', ['tex', 'fontsize', 'dpi'])
        TexManager.get_grey.__dict__.__setitem__('stypy_varargs_param_name', None)
        TexManager.get_grey.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TexManager.get_grey.__dict__.__setitem__('stypy_call_defaults', defaults)
        TexManager.get_grey.__dict__.__setitem__('stypy_call_varargs', varargs)
        TexManager.get_grey.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TexManager.get_grey.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TexManager.get_grey', ['tex', 'fontsize', 'dpi'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_grey', localization, ['tex', 'fontsize', 'dpi'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_grey(...)' code ##################

        unicode_138474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 541, 8), 'unicode', u'returns the alpha channel')
        
        # Assigning a Tuple to a Name (line 542):
        
        # Assigning a Tuple to a Name (line 542):
        
        # Obtaining an instance of the builtin type 'tuple' (line 542)
        tuple_138475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 542, 14), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 542)
        # Adding element type (line 542)
        # Getting the type of 'tex' (line 542)
        tex_138476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 14), 'tex')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 542, 14), tuple_138475, tex_138476)
        # Adding element type (line 542)
        
        # Call to get_font_config(...): (line 542)
        # Processing the call keyword arguments (line 542)
        kwargs_138479 = {}
        # Getting the type of 'self' (line 542)
        self_138477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 19), 'self', False)
        # Obtaining the member 'get_font_config' of a type (line 542)
        get_font_config_138478 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 542, 19), self_138477, 'get_font_config')
        # Calling get_font_config(args, kwargs) (line 542)
        get_font_config_call_result_138480 = invoke(stypy.reporting.localization.Localization(__file__, 542, 19), get_font_config_138478, *[], **kwargs_138479)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 542, 14), tuple_138475, get_font_config_call_result_138480)
        # Adding element type (line 542)
        # Getting the type of 'fontsize' (line 542)
        fontsize_138481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 43), 'fontsize')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 542, 14), tuple_138475, fontsize_138481)
        # Adding element type (line 542)
        # Getting the type of 'dpi' (line 542)
        dpi_138482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 53), 'dpi')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 542, 14), tuple_138475, dpi_138482)
        
        # Assigning a type to the variable 'key' (line 542)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 542, 8), 'key', tuple_138475)
        
        # Assigning a Call to a Name (line 543):
        
        # Assigning a Call to a Name (line 543):
        
        # Call to get(...): (line 543)
        # Processing the call arguments (line 543)
        # Getting the type of 'key' (line 543)
        key_138486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 37), 'key', False)
        # Processing the call keyword arguments (line 543)
        kwargs_138487 = {}
        # Getting the type of 'self' (line 543)
        self_138483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 16), 'self', False)
        # Obtaining the member 'grey_arrayd' of a type (line 543)
        grey_arrayd_138484 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 543, 16), self_138483, 'grey_arrayd')
        # Obtaining the member 'get' of a type (line 543)
        get_138485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 543, 16), grey_arrayd_138484, 'get')
        # Calling get(args, kwargs) (line 543)
        get_call_result_138488 = invoke(stypy.reporting.localization.Localization(__file__, 543, 16), get_138485, *[key_138486], **kwargs_138487)
        
        # Assigning a type to the variable 'alpha' (line 543)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 543, 8), 'alpha', get_call_result_138488)
        
        # Type idiom detected: calculating its left and rigth part (line 544)
        # Getting the type of 'alpha' (line 544)
        alpha_138489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 11), 'alpha')
        # Getting the type of 'None' (line 544)
        None_138490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 20), 'None')
        
        (may_be_138491, more_types_in_union_138492) = may_be_none(alpha_138489, None_138490)

        if may_be_138491:

            if more_types_in_union_138492:
                # Runtime conditional SSA (line 544)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 545):
            
            # Assigning a Call to a Name (line 545):
            
            # Call to make_png(...): (line 545)
            # Processing the call arguments (line 545)
            # Getting the type of 'tex' (line 545)
            tex_138495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 36), 'tex', False)
            # Getting the type of 'fontsize' (line 545)
            fontsize_138496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 41), 'fontsize', False)
            # Getting the type of 'dpi' (line 545)
            dpi_138497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 51), 'dpi', False)
            # Processing the call keyword arguments (line 545)
            kwargs_138498 = {}
            # Getting the type of 'self' (line 545)
            self_138493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 22), 'self', False)
            # Obtaining the member 'make_png' of a type (line 545)
            make_png_138494 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 545, 22), self_138493, 'make_png')
            # Calling make_png(args, kwargs) (line 545)
            make_png_call_result_138499 = invoke(stypy.reporting.localization.Localization(__file__, 545, 22), make_png_138494, *[tex_138495, fontsize_138496, dpi_138497], **kwargs_138498)
            
            # Assigning a type to the variable 'pngfile' (line 545)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 545, 12), 'pngfile', make_png_call_result_138499)
            
            # Assigning a Call to a Name (line 546):
            
            # Assigning a Call to a Name (line 546):
            
            # Call to read_png(...): (line 546)
            # Processing the call arguments (line 546)
            
            # Call to join(...): (line 546)
            # Processing the call arguments (line 546)
            # Getting the type of 'self' (line 546)
            self_138504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 38), 'self', False)
            # Obtaining the member 'texcache' of a type (line 546)
            texcache_138505 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 546, 38), self_138504, 'texcache')
            # Getting the type of 'pngfile' (line 546)
            pngfile_138506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 53), 'pngfile', False)
            # Processing the call keyword arguments (line 546)
            kwargs_138507 = {}
            # Getting the type of 'os' (line 546)
            os_138501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 25), 'os', False)
            # Obtaining the member 'path' of a type (line 546)
            path_138502 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 546, 25), os_138501, 'path')
            # Obtaining the member 'join' of a type (line 546)
            join_138503 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 546, 25), path_138502, 'join')
            # Calling join(args, kwargs) (line 546)
            join_call_result_138508 = invoke(stypy.reporting.localization.Localization(__file__, 546, 25), join_138503, *[texcache_138505, pngfile_138506], **kwargs_138507)
            
            # Processing the call keyword arguments (line 546)
            kwargs_138509 = {}
            # Getting the type of 'read_png' (line 546)
            read_png_138500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 16), 'read_png', False)
            # Calling read_png(args, kwargs) (line 546)
            read_png_call_result_138510 = invoke(stypy.reporting.localization.Localization(__file__, 546, 16), read_png_138500, *[join_call_result_138508], **kwargs_138509)
            
            # Assigning a type to the variable 'X' (line 546)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 546, 12), 'X', read_png_call_result_138510)
            
            # Multiple assignment of 2 elements.
            
            # Assigning a Subscript to a Name (line 547):
            
            # Obtaining the type of the subscript
            slice_138511 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 547, 44), None, None, None)
            slice_138512 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 547, 44), None, None, None)
            int_138513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 547, 52), 'int')
            # Getting the type of 'X' (line 547)
            X_138514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 44), 'X')
            # Obtaining the member '__getitem__' of a type (line 547)
            getitem___138515 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 547, 44), X_138514, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 547)
            subscript_call_result_138516 = invoke(stypy.reporting.localization.Localization(__file__, 547, 44), getitem___138515, (slice_138511, slice_138512, int_138513))
            
            # Assigning a type to the variable 'alpha' (line 547)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 547, 36), 'alpha', subscript_call_result_138516)
            
            # Assigning a Name to a Subscript (line 547):
            # Getting the type of 'alpha' (line 547)
            alpha_138517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 36), 'alpha')
            # Getting the type of 'self' (line 547)
            self_138518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 12), 'self')
            # Obtaining the member 'grey_arrayd' of a type (line 547)
            grey_arrayd_138519 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 547, 12), self_138518, 'grey_arrayd')
            # Getting the type of 'key' (line 547)
            key_138520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 29), 'key')
            # Storing an element on a container (line 547)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 547, 12), grey_arrayd_138519, (key_138520, alpha_138517))

            if more_types_in_union_138492:
                # SSA join for if statement (line 544)
                module_type_store = module_type_store.join_ssa_context()


        
        # Getting the type of 'alpha' (line 548)
        alpha_138521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 15), 'alpha')
        # Assigning a type to the variable 'stypy_return_type' (line 548)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 548, 8), 'stypy_return_type', alpha_138521)
        
        # ################# End of 'get_grey(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_grey' in the type store
        # Getting the type of 'stypy_return_type' (line 540)
        stypy_return_type_138522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_138522)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_grey'
        return stypy_return_type_138522


    @norecursion
    def get_rgba(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 550)
        None_138523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 37), 'None')
        # Getting the type of 'None' (line 550)
        None_138524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 47), 'None')
        
        # Obtaining an instance of the builtin type 'tuple' (line 550)
        tuple_138525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 550, 58), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 550)
        # Adding element type (line 550)
        int_138526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 550, 58), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 550, 58), tuple_138525, int_138526)
        # Adding element type (line 550)
        int_138527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 550, 61), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 550, 58), tuple_138525, int_138527)
        # Adding element type (line 550)
        int_138528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 550, 64), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 550, 58), tuple_138525, int_138528)
        
        defaults = [None_138523, None_138524, tuple_138525]
        # Create a new context for function 'get_rgba'
        module_type_store = module_type_store.open_function_context('get_rgba', 550, 4, False)
        # Assigning a type to the variable 'self' (line 551)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 551, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TexManager.get_rgba.__dict__.__setitem__('stypy_localization', localization)
        TexManager.get_rgba.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TexManager.get_rgba.__dict__.__setitem__('stypy_type_store', module_type_store)
        TexManager.get_rgba.__dict__.__setitem__('stypy_function_name', 'TexManager.get_rgba')
        TexManager.get_rgba.__dict__.__setitem__('stypy_param_names_list', ['tex', 'fontsize', 'dpi', 'rgb'])
        TexManager.get_rgba.__dict__.__setitem__('stypy_varargs_param_name', None)
        TexManager.get_rgba.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TexManager.get_rgba.__dict__.__setitem__('stypy_call_defaults', defaults)
        TexManager.get_rgba.__dict__.__setitem__('stypy_call_varargs', varargs)
        TexManager.get_rgba.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TexManager.get_rgba.__dict__.__setitem__('stypy_declared_arg_number', 5)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TexManager.get_rgba', ['tex', 'fontsize', 'dpi', 'rgb'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_rgba', localization, ['tex', 'fontsize', 'dpi', 'rgb'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_rgba(...)' code ##################

        unicode_138529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 553, (-1)), 'unicode', u"\n        Returns latex's rendering of the tex string as an rgba array\n        ")
        
        
        # Getting the type of 'fontsize' (line 554)
        fontsize_138530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 15), 'fontsize')
        # Applying the 'not' unary operator (line 554)
        result_not__138531 = python_operator(stypy.reporting.localization.Localization(__file__, 554, 11), 'not', fontsize_138530)
        
        # Testing the type of an if condition (line 554)
        if_condition_138532 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 554, 8), result_not__138531)
        # Assigning a type to the variable 'if_condition_138532' (line 554)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 554, 8), 'if_condition_138532', if_condition_138532)
        # SSA begins for if statement (line 554)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Name (line 555):
        
        # Assigning a Subscript to a Name (line 555):
        
        # Obtaining the type of the subscript
        unicode_138533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 555, 32), 'unicode', u'font.size')
        # Getting the type of 'rcParams' (line 555)
        rcParams_138534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 23), 'rcParams')
        # Obtaining the member '__getitem__' of a type (line 555)
        getitem___138535 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 555, 23), rcParams_138534, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 555)
        subscript_call_result_138536 = invoke(stypy.reporting.localization.Localization(__file__, 555, 23), getitem___138535, unicode_138533)
        
        # Assigning a type to the variable 'fontsize' (line 555)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 555, 12), 'fontsize', subscript_call_result_138536)
        # SSA join for if statement (line 554)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'dpi' (line 556)
        dpi_138537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 15), 'dpi')
        # Applying the 'not' unary operator (line 556)
        result_not__138538 = python_operator(stypy.reporting.localization.Localization(__file__, 556, 11), 'not', dpi_138537)
        
        # Testing the type of an if condition (line 556)
        if_condition_138539 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 556, 8), result_not__138538)
        # Assigning a type to the variable 'if_condition_138539' (line 556)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 556, 8), 'if_condition_138539', if_condition_138539)
        # SSA begins for if statement (line 556)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Name (line 557):
        
        # Assigning a Subscript to a Name (line 557):
        
        # Obtaining the type of the subscript
        unicode_138540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 557, 27), 'unicode', u'savefig.dpi')
        # Getting the type of 'rcParams' (line 557)
        rcParams_138541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 18), 'rcParams')
        # Obtaining the member '__getitem__' of a type (line 557)
        getitem___138542 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 557, 18), rcParams_138541, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 557)
        subscript_call_result_138543 = invoke(stypy.reporting.localization.Localization(__file__, 557, 18), getitem___138542, unicode_138540)
        
        # Assigning a type to the variable 'dpi' (line 557)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 557, 12), 'dpi', subscript_call_result_138543)
        # SSA join for if statement (line 556)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Tuple (line 558):
        
        # Assigning a Subscript to a Name (line 558):
        
        # Obtaining the type of the subscript
        int_138544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 558, 8), 'int')
        # Getting the type of 'rgb' (line 558)
        rgb_138545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 18), 'rgb')
        # Obtaining the member '__getitem__' of a type (line 558)
        getitem___138546 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 558, 8), rgb_138545, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 558)
        subscript_call_result_138547 = invoke(stypy.reporting.localization.Localization(__file__, 558, 8), getitem___138546, int_138544)
        
        # Assigning a type to the variable 'tuple_var_assignment_137200' (line 558)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 558, 8), 'tuple_var_assignment_137200', subscript_call_result_138547)
        
        # Assigning a Subscript to a Name (line 558):
        
        # Obtaining the type of the subscript
        int_138548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 558, 8), 'int')
        # Getting the type of 'rgb' (line 558)
        rgb_138549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 18), 'rgb')
        # Obtaining the member '__getitem__' of a type (line 558)
        getitem___138550 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 558, 8), rgb_138549, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 558)
        subscript_call_result_138551 = invoke(stypy.reporting.localization.Localization(__file__, 558, 8), getitem___138550, int_138548)
        
        # Assigning a type to the variable 'tuple_var_assignment_137201' (line 558)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 558, 8), 'tuple_var_assignment_137201', subscript_call_result_138551)
        
        # Assigning a Subscript to a Name (line 558):
        
        # Obtaining the type of the subscript
        int_138552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 558, 8), 'int')
        # Getting the type of 'rgb' (line 558)
        rgb_138553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 18), 'rgb')
        # Obtaining the member '__getitem__' of a type (line 558)
        getitem___138554 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 558, 8), rgb_138553, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 558)
        subscript_call_result_138555 = invoke(stypy.reporting.localization.Localization(__file__, 558, 8), getitem___138554, int_138552)
        
        # Assigning a type to the variable 'tuple_var_assignment_137202' (line 558)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 558, 8), 'tuple_var_assignment_137202', subscript_call_result_138555)
        
        # Assigning a Name to a Name (line 558):
        # Getting the type of 'tuple_var_assignment_137200' (line 558)
        tuple_var_assignment_137200_138556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 8), 'tuple_var_assignment_137200')
        # Assigning a type to the variable 'r' (line 558)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 558, 8), 'r', tuple_var_assignment_137200_138556)
        
        # Assigning a Name to a Name (line 558):
        # Getting the type of 'tuple_var_assignment_137201' (line 558)
        tuple_var_assignment_137201_138557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 8), 'tuple_var_assignment_137201')
        # Assigning a type to the variable 'g' (line 558)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 558, 11), 'g', tuple_var_assignment_137201_138557)
        
        # Assigning a Name to a Name (line 558):
        # Getting the type of 'tuple_var_assignment_137202' (line 558)
        tuple_var_assignment_137202_138558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 8), 'tuple_var_assignment_137202')
        # Assigning a type to the variable 'b' (line 558)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 558, 14), 'b', tuple_var_assignment_137202_138558)
        
        # Assigning a Tuple to a Name (line 559):
        
        # Assigning a Tuple to a Name (line 559):
        
        # Obtaining an instance of the builtin type 'tuple' (line 559)
        tuple_138559 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 559, 14), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 559)
        # Adding element type (line 559)
        # Getting the type of 'tex' (line 559)
        tex_138560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 14), 'tex')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 559, 14), tuple_138559, tex_138560)
        # Adding element type (line 559)
        
        # Call to get_font_config(...): (line 559)
        # Processing the call keyword arguments (line 559)
        kwargs_138563 = {}
        # Getting the type of 'self' (line 559)
        self_138561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 19), 'self', False)
        # Obtaining the member 'get_font_config' of a type (line 559)
        get_font_config_138562 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 559, 19), self_138561, 'get_font_config')
        # Calling get_font_config(args, kwargs) (line 559)
        get_font_config_call_result_138564 = invoke(stypy.reporting.localization.Localization(__file__, 559, 19), get_font_config_138562, *[], **kwargs_138563)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 559, 14), tuple_138559, get_font_config_call_result_138564)
        # Adding element type (line 559)
        # Getting the type of 'fontsize' (line 559)
        fontsize_138565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 43), 'fontsize')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 559, 14), tuple_138559, fontsize_138565)
        # Adding element type (line 559)
        # Getting the type of 'dpi' (line 559)
        dpi_138566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 53), 'dpi')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 559, 14), tuple_138559, dpi_138566)
        # Adding element type (line 559)
        
        # Call to tuple(...): (line 559)
        # Processing the call arguments (line 559)
        # Getting the type of 'rgb' (line 559)
        rgb_138568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 64), 'rgb', False)
        # Processing the call keyword arguments (line 559)
        kwargs_138569 = {}
        # Getting the type of 'tuple' (line 559)
        tuple_138567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 58), 'tuple', False)
        # Calling tuple(args, kwargs) (line 559)
        tuple_call_result_138570 = invoke(stypy.reporting.localization.Localization(__file__, 559, 58), tuple_138567, *[rgb_138568], **kwargs_138569)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 559, 14), tuple_138559, tuple_call_result_138570)
        
        # Assigning a type to the variable 'key' (line 559)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 559, 8), 'key', tuple_138559)
        
        # Assigning a Call to a Name (line 560):
        
        # Assigning a Call to a Name (line 560):
        
        # Call to get(...): (line 560)
        # Processing the call arguments (line 560)
        # Getting the type of 'key' (line 560)
        key_138574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 33), 'key', False)
        # Processing the call keyword arguments (line 560)
        kwargs_138575 = {}
        # Getting the type of 'self' (line 560)
        self_138571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 12), 'self', False)
        # Obtaining the member 'rgba_arrayd' of a type (line 560)
        rgba_arrayd_138572 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 560, 12), self_138571, 'rgba_arrayd')
        # Obtaining the member 'get' of a type (line 560)
        get_138573 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 560, 12), rgba_arrayd_138572, 'get')
        # Calling get(args, kwargs) (line 560)
        get_call_result_138576 = invoke(stypy.reporting.localization.Localization(__file__, 560, 12), get_138573, *[key_138574], **kwargs_138575)
        
        # Assigning a type to the variable 'Z' (line 560)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 560, 8), 'Z', get_call_result_138576)
        
        # Type idiom detected: calculating its left and rigth part (line 562)
        # Getting the type of 'Z' (line 562)
        Z_138577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 11), 'Z')
        # Getting the type of 'None' (line 562)
        None_138578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 16), 'None')
        
        (may_be_138579, more_types_in_union_138580) = may_be_none(Z_138577, None_138578)

        if may_be_138579:

            if more_types_in_union_138580:
                # Runtime conditional SSA (line 562)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 563):
            
            # Assigning a Call to a Name (line 563):
            
            # Call to get_grey(...): (line 563)
            # Processing the call arguments (line 563)
            # Getting the type of 'tex' (line 563)
            tex_138583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 34), 'tex', False)
            # Getting the type of 'fontsize' (line 563)
            fontsize_138584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 39), 'fontsize', False)
            # Getting the type of 'dpi' (line 563)
            dpi_138585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 49), 'dpi', False)
            # Processing the call keyword arguments (line 563)
            kwargs_138586 = {}
            # Getting the type of 'self' (line 563)
            self_138581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 20), 'self', False)
            # Obtaining the member 'get_grey' of a type (line 563)
            get_grey_138582 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 563, 20), self_138581, 'get_grey')
            # Calling get_grey(args, kwargs) (line 563)
            get_grey_call_result_138587 = invoke(stypy.reporting.localization.Localization(__file__, 563, 20), get_grey_138582, *[tex_138583, fontsize_138584, dpi_138585], **kwargs_138586)
            
            # Assigning a type to the variable 'alpha' (line 563)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 563, 12), 'alpha', get_grey_call_result_138587)
            
            # Assigning a Call to a Name (line 565):
            
            # Assigning a Call to a Name (line 565):
            
            # Call to zeros(...): (line 565)
            # Processing the call arguments (line 565)
            
            # Obtaining an instance of the builtin type 'tuple' (line 565)
            tuple_138590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 565, 26), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 565)
            # Adding element type (line 565)
            
            # Obtaining the type of the subscript
            int_138591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 565, 38), 'int')
            # Getting the type of 'alpha' (line 565)
            alpha_138592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 26), 'alpha', False)
            # Obtaining the member 'shape' of a type (line 565)
            shape_138593 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 565, 26), alpha_138592, 'shape')
            # Obtaining the member '__getitem__' of a type (line 565)
            getitem___138594 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 565, 26), shape_138593, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 565)
            subscript_call_result_138595 = invoke(stypy.reporting.localization.Localization(__file__, 565, 26), getitem___138594, int_138591)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 565, 26), tuple_138590, subscript_call_result_138595)
            # Adding element type (line 565)
            
            # Obtaining the type of the subscript
            int_138596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 565, 54), 'int')
            # Getting the type of 'alpha' (line 565)
            alpha_138597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 42), 'alpha', False)
            # Obtaining the member 'shape' of a type (line 565)
            shape_138598 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 565, 42), alpha_138597, 'shape')
            # Obtaining the member '__getitem__' of a type (line 565)
            getitem___138599 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 565, 42), shape_138598, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 565)
            subscript_call_result_138600 = invoke(stypy.reporting.localization.Localization(__file__, 565, 42), getitem___138599, int_138596)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 565, 26), tuple_138590, subscript_call_result_138600)
            # Adding element type (line 565)
            int_138601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 565, 58), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 565, 26), tuple_138590, int_138601)
            
            # Getting the type of 'float' (line 565)
            float_138602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 62), 'float', False)
            # Processing the call keyword arguments (line 565)
            kwargs_138603 = {}
            # Getting the type of 'np' (line 565)
            np_138588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 16), 'np', False)
            # Obtaining the member 'zeros' of a type (line 565)
            zeros_138589 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 565, 16), np_138588, 'zeros')
            # Calling zeros(args, kwargs) (line 565)
            zeros_call_result_138604 = invoke(stypy.reporting.localization.Localization(__file__, 565, 16), zeros_138589, *[tuple_138590, float_138602], **kwargs_138603)
            
            # Assigning a type to the variable 'Z' (line 565)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 565, 12), 'Z', zeros_call_result_138604)
            
            # Assigning a Name to a Subscript (line 567):
            
            # Assigning a Name to a Subscript (line 567):
            # Getting the type of 'r' (line 567)
            r_138605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 25), 'r')
            # Getting the type of 'Z' (line 567)
            Z_138606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 12), 'Z')
            slice_138607 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 567, 12), None, None, None)
            slice_138608 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 567, 12), None, None, None)
            int_138609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 567, 20), 'int')
            # Storing an element on a container (line 567)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 567, 12), Z_138606, ((slice_138607, slice_138608, int_138609), r_138605))
            
            # Assigning a Name to a Subscript (line 568):
            
            # Assigning a Name to a Subscript (line 568):
            # Getting the type of 'g' (line 568)
            g_138610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 25), 'g')
            # Getting the type of 'Z' (line 568)
            Z_138611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 12), 'Z')
            slice_138612 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 568, 12), None, None, None)
            slice_138613 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 568, 12), None, None, None)
            int_138614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 568, 20), 'int')
            # Storing an element on a container (line 568)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 568, 12), Z_138611, ((slice_138612, slice_138613, int_138614), g_138610))
            
            # Assigning a Name to a Subscript (line 569):
            
            # Assigning a Name to a Subscript (line 569):
            # Getting the type of 'b' (line 569)
            b_138615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 25), 'b')
            # Getting the type of 'Z' (line 569)
            Z_138616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 12), 'Z')
            slice_138617 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 569, 12), None, None, None)
            slice_138618 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 569, 12), None, None, None)
            int_138619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 569, 20), 'int')
            # Storing an element on a container (line 569)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 569, 12), Z_138616, ((slice_138617, slice_138618, int_138619), b_138615))
            
            # Assigning a Name to a Subscript (line 570):
            
            # Assigning a Name to a Subscript (line 570):
            # Getting the type of 'alpha' (line 570)
            alpha_138620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 25), 'alpha')
            # Getting the type of 'Z' (line 570)
            Z_138621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 12), 'Z')
            slice_138622 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 570, 12), None, None, None)
            slice_138623 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 570, 12), None, None, None)
            int_138624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 570, 20), 'int')
            # Storing an element on a container (line 570)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 570, 12), Z_138621, ((slice_138622, slice_138623, int_138624), alpha_138620))
            
            # Assigning a Name to a Subscript (line 571):
            
            # Assigning a Name to a Subscript (line 571):
            # Getting the type of 'Z' (line 571)
            Z_138625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 36), 'Z')
            # Getting the type of 'self' (line 571)
            self_138626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 12), 'self')
            # Obtaining the member 'rgba_arrayd' of a type (line 571)
            rgba_arrayd_138627 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 571, 12), self_138626, 'rgba_arrayd')
            # Getting the type of 'key' (line 571)
            key_138628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 29), 'key')
            # Storing an element on a container (line 571)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 571, 12), rgba_arrayd_138627, (key_138628, Z_138625))

            if more_types_in_union_138580:
                # SSA join for if statement (line 562)
                module_type_store = module_type_store.join_ssa_context()


        
        # Getting the type of 'Z' (line 573)
        Z_138629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 15), 'Z')
        # Assigning a type to the variable 'stypy_return_type' (line 573)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 573, 8), 'stypy_return_type', Z_138629)
        
        # ################# End of 'get_rgba(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_rgba' in the type store
        # Getting the type of 'stypy_return_type' (line 550)
        stypy_return_type_138630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_138630)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_rgba'
        return stypy_return_type_138630


    @norecursion
    def get_text_width_height_descent(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 575)
        None_138631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 68), 'None')
        defaults = [None_138631]
        # Create a new context for function 'get_text_width_height_descent'
        module_type_store = module_type_store.open_function_context('get_text_width_height_descent', 575, 4, False)
        # Assigning a type to the variable 'self' (line 576)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 576, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TexManager.get_text_width_height_descent.__dict__.__setitem__('stypy_localization', localization)
        TexManager.get_text_width_height_descent.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TexManager.get_text_width_height_descent.__dict__.__setitem__('stypy_type_store', module_type_store)
        TexManager.get_text_width_height_descent.__dict__.__setitem__('stypy_function_name', 'TexManager.get_text_width_height_descent')
        TexManager.get_text_width_height_descent.__dict__.__setitem__('stypy_param_names_list', ['tex', 'fontsize', 'renderer'])
        TexManager.get_text_width_height_descent.__dict__.__setitem__('stypy_varargs_param_name', None)
        TexManager.get_text_width_height_descent.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TexManager.get_text_width_height_descent.__dict__.__setitem__('stypy_call_defaults', defaults)
        TexManager.get_text_width_height_descent.__dict__.__setitem__('stypy_call_varargs', varargs)
        TexManager.get_text_width_height_descent.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TexManager.get_text_width_height_descent.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TexManager.get_text_width_height_descent', ['tex', 'fontsize', 'renderer'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_text_width_height_descent', localization, ['tex', 'fontsize', 'renderer'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_text_width_height_descent(...)' code ##################

        unicode_138632 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 578, (-1)), 'unicode', u'\n        return width, heigth and descent of the text.\n        ')
        
        
        
        # Call to strip(...): (line 579)
        # Processing the call keyword arguments (line 579)
        kwargs_138635 = {}
        # Getting the type of 'tex' (line 579)
        tex_138633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 11), 'tex', False)
        # Obtaining the member 'strip' of a type (line 579)
        strip_138634 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 579, 11), tex_138633, 'strip')
        # Calling strip(args, kwargs) (line 579)
        strip_call_result_138636 = invoke(stypy.reporting.localization.Localization(__file__, 579, 11), strip_138634, *[], **kwargs_138635)
        
        unicode_138637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 579, 26), 'unicode', u'')
        # Applying the binary operator '==' (line 579)
        result_eq_138638 = python_operator(stypy.reporting.localization.Localization(__file__, 579, 11), '==', strip_call_result_138636, unicode_138637)
        
        # Testing the type of an if condition (line 579)
        if_condition_138639 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 579, 8), result_eq_138638)
        # Assigning a type to the variable 'if_condition_138639' (line 579)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 579, 8), 'if_condition_138639', if_condition_138639)
        # SSA begins for if statement (line 579)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Obtaining an instance of the builtin type 'tuple' (line 580)
        tuple_138640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 580, 19), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 580)
        # Adding element type (line 580)
        int_138641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 580, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 580, 19), tuple_138640, int_138641)
        # Adding element type (line 580)
        int_138642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 580, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 580, 19), tuple_138640, int_138642)
        # Adding element type (line 580)
        int_138643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 580, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 580, 19), tuple_138640, int_138643)
        
        # Assigning a type to the variable 'stypy_return_type' (line 580)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 580, 12), 'stypy_return_type', tuple_138640)
        # SSA join for if statement (line 579)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'renderer' (line 582)
        renderer_138644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 11), 'renderer')
        # Testing the type of an if condition (line 582)
        if_condition_138645 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 582, 8), renderer_138644)
        # Assigning a type to the variable 'if_condition_138645' (line 582)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 582, 8), 'if_condition_138645', if_condition_138645)
        # SSA begins for if statement (line 582)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 583):
        
        # Assigning a Call to a Name (line 583):
        
        # Call to points_to_pixels(...): (line 583)
        # Processing the call arguments (line 583)
        float_138648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 583, 53), 'float')
        # Processing the call keyword arguments (line 583)
        kwargs_138649 = {}
        # Getting the type of 'renderer' (line 583)
        renderer_138646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 27), 'renderer', False)
        # Obtaining the member 'points_to_pixels' of a type (line 583)
        points_to_pixels_138647 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 583, 27), renderer_138646, 'points_to_pixels')
        # Calling points_to_pixels(args, kwargs) (line 583)
        points_to_pixels_call_result_138650 = invoke(stypy.reporting.localization.Localization(__file__, 583, 27), points_to_pixels_138647, *[float_138648], **kwargs_138649)
        
        # Assigning a type to the variable 'dpi_fraction' (line 583)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 583, 12), 'dpi_fraction', points_to_pixels_call_result_138650)
        # SSA branch for the else part of an if statement (line 582)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Num to a Name (line 585):
        
        # Assigning a Num to a Name (line 585):
        float_138651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 585, 27), 'float')
        # Assigning a type to the variable 'dpi_fraction' (line 585)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 585, 12), 'dpi_fraction', float_138651)
        # SSA join for if statement (line 582)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Obtaining the type of the subscript
        unicode_138652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 587, 20), 'unicode', u'text.latex.preview')
        # Getting the type of 'rcParams' (line 587)
        rcParams_138653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 11), 'rcParams')
        # Obtaining the member '__getitem__' of a type (line 587)
        getitem___138654 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 587, 11), rcParams_138653, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 587)
        subscript_call_result_138655 = invoke(stypy.reporting.localization.Localization(__file__, 587, 11), getitem___138654, unicode_138652)
        
        # Testing the type of an if condition (line 587)
        if_condition_138656 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 587, 8), subscript_call_result_138655)
        # Assigning a type to the variable 'if_condition_138656' (line 587)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 587, 8), 'if_condition_138656', if_condition_138656)
        # SSA begins for if statement (line 587)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 589):
        
        # Assigning a Call to a Name (line 589):
        
        # Call to get_basefile(...): (line 589)
        # Processing the call arguments (line 589)
        # Getting the type of 'tex' (line 589)
        tex_138659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 41), 'tex', False)
        # Getting the type of 'fontsize' (line 589)
        fontsize_138660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 46), 'fontsize', False)
        # Processing the call keyword arguments (line 589)
        kwargs_138661 = {}
        # Getting the type of 'self' (line 589)
        self_138657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 23), 'self', False)
        # Obtaining the member 'get_basefile' of a type (line 589)
        get_basefile_138658 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 589, 23), self_138657, 'get_basefile')
        # Calling get_basefile(args, kwargs) (line 589)
        get_basefile_call_result_138662 = invoke(stypy.reporting.localization.Localization(__file__, 589, 23), get_basefile_138658, *[tex_138659, fontsize_138660], **kwargs_138661)
        
        # Assigning a type to the variable 'basefile' (line 589)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 589, 12), 'basefile', get_basefile_call_result_138662)
        
        # Assigning a BinOp to a Name (line 590):
        
        # Assigning a BinOp to a Name (line 590):
        unicode_138663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 590, 27), 'unicode', u'%s.baseline')
        # Getting the type of 'basefile' (line 590)
        basefile_138664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 43), 'basefile')
        # Applying the binary operator '%' (line 590)
        result_mod_138665 = python_operator(stypy.reporting.localization.Localization(__file__, 590, 27), '%', unicode_138663, basefile_138664)
        
        # Assigning a type to the variable 'baselinefile' (line 590)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 590, 12), 'baselinefile', result_mod_138665)
        
        
        # Evaluating a boolean operation
        # Getting the type of 'DEBUG' (line 592)
        DEBUG_138666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 15), 'DEBUG')
        
        
        # Call to exists(...): (line 592)
        # Processing the call arguments (line 592)
        # Getting the type of 'baselinefile' (line 592)
        baselinefile_138670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 43), 'baselinefile', False)
        # Processing the call keyword arguments (line 592)
        kwargs_138671 = {}
        # Getting the type of 'os' (line 592)
        os_138667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 28), 'os', False)
        # Obtaining the member 'path' of a type (line 592)
        path_138668 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 592, 28), os_138667, 'path')
        # Obtaining the member 'exists' of a type (line 592)
        exists_138669 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 592, 28), path_138668, 'exists')
        # Calling exists(args, kwargs) (line 592)
        exists_call_result_138672 = invoke(stypy.reporting.localization.Localization(__file__, 592, 28), exists_138669, *[baselinefile_138670], **kwargs_138671)
        
        # Applying the 'not' unary operator (line 592)
        result_not__138673 = python_operator(stypy.reporting.localization.Localization(__file__, 592, 24), 'not', exists_call_result_138672)
        
        # Applying the binary operator 'or' (line 592)
        result_or_keyword_138674 = python_operator(stypy.reporting.localization.Localization(__file__, 592, 15), 'or', DEBUG_138666, result_not__138673)
        
        # Testing the type of an if condition (line 592)
        if_condition_138675 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 592, 12), result_or_keyword_138674)
        # Assigning a type to the variable 'if_condition_138675' (line 592)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 592, 12), 'if_condition_138675', if_condition_138675)
        # SSA begins for if statement (line 592)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 593):
        
        # Assigning a Call to a Name (line 593):
        
        # Call to make_dvi_preview(...): (line 593)
        # Processing the call arguments (line 593)
        # Getting the type of 'tex' (line 593)
        tex_138678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 48), 'tex', False)
        # Getting the type of 'fontsize' (line 593)
        fontsize_138679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 53), 'fontsize', False)
        # Processing the call keyword arguments (line 593)
        kwargs_138680 = {}
        # Getting the type of 'self' (line 593)
        self_138676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 26), 'self', False)
        # Obtaining the member 'make_dvi_preview' of a type (line 593)
        make_dvi_preview_138677 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 593, 26), self_138676, 'make_dvi_preview')
        # Calling make_dvi_preview(args, kwargs) (line 593)
        make_dvi_preview_call_result_138681 = invoke(stypy.reporting.localization.Localization(__file__, 593, 26), make_dvi_preview_138677, *[tex_138678, fontsize_138679], **kwargs_138680)
        
        # Assigning a type to the variable 'dvifile' (line 593)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 593, 16), 'dvifile', make_dvi_preview_call_result_138681)
        # SSA join for if statement (line 592)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to open(...): (line 595)
        # Processing the call arguments (line 595)
        # Getting the type of 'baselinefile' (line 595)
        baselinefile_138683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 22), 'baselinefile', False)
        # Processing the call keyword arguments (line 595)
        kwargs_138684 = {}
        # Getting the type of 'open' (line 595)
        open_138682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 17), 'open', False)
        # Calling open(args, kwargs) (line 595)
        open_call_result_138685 = invoke(stypy.reporting.localization.Localization(__file__, 595, 17), open_138682, *[baselinefile_138683], **kwargs_138684)
        
        with_138686 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 595, 17), open_call_result_138685, 'with parameter', '__enter__', '__exit__')

        if with_138686:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 595)
            enter___138687 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 595, 17), open_call_result_138685, '__enter__')
            with_enter_138688 = invoke(stypy.reporting.localization.Localization(__file__, 595, 17), enter___138687)
            # Assigning a type to the variable 'fh' (line 595)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 595, 17), 'fh', with_enter_138688)
            
            # Assigning a Call to a Name (line 596):
            
            # Assigning a Call to a Name (line 596):
            
            # Call to split(...): (line 596)
            # Processing the call keyword arguments (line 596)
            kwargs_138694 = {}
            
            # Call to read(...): (line 596)
            # Processing the call keyword arguments (line 596)
            kwargs_138691 = {}
            # Getting the type of 'fh' (line 596)
            fh_138689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 20), 'fh', False)
            # Obtaining the member 'read' of a type (line 596)
            read_138690 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 596, 20), fh_138689, 'read')
            # Calling read(args, kwargs) (line 596)
            read_call_result_138692 = invoke(stypy.reporting.localization.Localization(__file__, 596, 20), read_138690, *[], **kwargs_138691)
            
            # Obtaining the member 'split' of a type (line 596)
            split_138693 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 596, 20), read_call_result_138692, 'split')
            # Calling split(args, kwargs) (line 596)
            split_call_result_138695 = invoke(stypy.reporting.localization.Localization(__file__, 596, 20), split_138693, *[], **kwargs_138694)
            
            # Assigning a type to the variable 'l' (line 596)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 596, 16), 'l', split_call_result_138695)
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 595)
            exit___138696 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 595, 17), open_call_result_138685, '__exit__')
            with_exit_138697 = invoke(stypy.reporting.localization.Localization(__file__, 595, 17), exit___138696, None, None, None)

        
        # Assigning a ListComp to a Tuple (line 597):
        
        # Assigning a Subscript to a Name (line 597):
        
        # Obtaining the type of the subscript
        int_138698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 597, 12), 'int')
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'l' (line 597)
        l_138705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 71), 'l')
        comprehension_138706 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 597, 36), l_138705)
        # Assigning a type to the variable 'l1' (line 597)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 597, 36), 'l1', comprehension_138706)
        
        # Call to float(...): (line 597)
        # Processing the call arguments (line 597)
        # Getting the type of 'l1' (line 597)
        l1_138700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 42), 'l1', False)
        # Processing the call keyword arguments (line 597)
        kwargs_138701 = {}
        # Getting the type of 'float' (line 597)
        float_138699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 36), 'float', False)
        # Calling float(args, kwargs) (line 597)
        float_call_result_138702 = invoke(stypy.reporting.localization.Localization(__file__, 597, 36), float_138699, *[l1_138700], **kwargs_138701)
        
        # Getting the type of 'dpi_fraction' (line 597)
        dpi_fraction_138703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 48), 'dpi_fraction')
        # Applying the binary operator '*' (line 597)
        result_mul_138704 = python_operator(stypy.reporting.localization.Localization(__file__, 597, 36), '*', float_call_result_138702, dpi_fraction_138703)
        
        list_138707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 597, 36), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 597, 36), list_138707, result_mul_138704)
        # Obtaining the member '__getitem__' of a type (line 597)
        getitem___138708 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 597, 12), list_138707, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 597)
        subscript_call_result_138709 = invoke(stypy.reporting.localization.Localization(__file__, 597, 12), getitem___138708, int_138698)
        
        # Assigning a type to the variable 'tuple_var_assignment_137203' (line 597)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 597, 12), 'tuple_var_assignment_137203', subscript_call_result_138709)
        
        # Assigning a Subscript to a Name (line 597):
        
        # Obtaining the type of the subscript
        int_138710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 597, 12), 'int')
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'l' (line 597)
        l_138717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 71), 'l')
        comprehension_138718 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 597, 36), l_138717)
        # Assigning a type to the variable 'l1' (line 597)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 597, 36), 'l1', comprehension_138718)
        
        # Call to float(...): (line 597)
        # Processing the call arguments (line 597)
        # Getting the type of 'l1' (line 597)
        l1_138712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 42), 'l1', False)
        # Processing the call keyword arguments (line 597)
        kwargs_138713 = {}
        # Getting the type of 'float' (line 597)
        float_138711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 36), 'float', False)
        # Calling float(args, kwargs) (line 597)
        float_call_result_138714 = invoke(stypy.reporting.localization.Localization(__file__, 597, 36), float_138711, *[l1_138712], **kwargs_138713)
        
        # Getting the type of 'dpi_fraction' (line 597)
        dpi_fraction_138715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 48), 'dpi_fraction')
        # Applying the binary operator '*' (line 597)
        result_mul_138716 = python_operator(stypy.reporting.localization.Localization(__file__, 597, 36), '*', float_call_result_138714, dpi_fraction_138715)
        
        list_138719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 597, 36), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 597, 36), list_138719, result_mul_138716)
        # Obtaining the member '__getitem__' of a type (line 597)
        getitem___138720 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 597, 12), list_138719, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 597)
        subscript_call_result_138721 = invoke(stypy.reporting.localization.Localization(__file__, 597, 12), getitem___138720, int_138710)
        
        # Assigning a type to the variable 'tuple_var_assignment_137204' (line 597)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 597, 12), 'tuple_var_assignment_137204', subscript_call_result_138721)
        
        # Assigning a Subscript to a Name (line 597):
        
        # Obtaining the type of the subscript
        int_138722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 597, 12), 'int')
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'l' (line 597)
        l_138729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 71), 'l')
        comprehension_138730 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 597, 36), l_138729)
        # Assigning a type to the variable 'l1' (line 597)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 597, 36), 'l1', comprehension_138730)
        
        # Call to float(...): (line 597)
        # Processing the call arguments (line 597)
        # Getting the type of 'l1' (line 597)
        l1_138724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 42), 'l1', False)
        # Processing the call keyword arguments (line 597)
        kwargs_138725 = {}
        # Getting the type of 'float' (line 597)
        float_138723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 36), 'float', False)
        # Calling float(args, kwargs) (line 597)
        float_call_result_138726 = invoke(stypy.reporting.localization.Localization(__file__, 597, 36), float_138723, *[l1_138724], **kwargs_138725)
        
        # Getting the type of 'dpi_fraction' (line 597)
        dpi_fraction_138727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 48), 'dpi_fraction')
        # Applying the binary operator '*' (line 597)
        result_mul_138728 = python_operator(stypy.reporting.localization.Localization(__file__, 597, 36), '*', float_call_result_138726, dpi_fraction_138727)
        
        list_138731 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 597, 36), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 597, 36), list_138731, result_mul_138728)
        # Obtaining the member '__getitem__' of a type (line 597)
        getitem___138732 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 597, 12), list_138731, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 597)
        subscript_call_result_138733 = invoke(stypy.reporting.localization.Localization(__file__, 597, 12), getitem___138732, int_138722)
        
        # Assigning a type to the variable 'tuple_var_assignment_137205' (line 597)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 597, 12), 'tuple_var_assignment_137205', subscript_call_result_138733)
        
        # Assigning a Name to a Name (line 597):
        # Getting the type of 'tuple_var_assignment_137203' (line 597)
        tuple_var_assignment_137203_138734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 12), 'tuple_var_assignment_137203')
        # Assigning a type to the variable 'height' (line 597)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 597, 12), 'height', tuple_var_assignment_137203_138734)
        
        # Assigning a Name to a Name (line 597):
        # Getting the type of 'tuple_var_assignment_137204' (line 597)
        tuple_var_assignment_137204_138735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 12), 'tuple_var_assignment_137204')
        # Assigning a type to the variable 'depth' (line 597)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 597, 20), 'depth', tuple_var_assignment_137204_138735)
        
        # Assigning a Name to a Name (line 597):
        # Getting the type of 'tuple_var_assignment_137205' (line 597)
        tuple_var_assignment_137205_138736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 12), 'tuple_var_assignment_137205')
        # Assigning a type to the variable 'width' (line 597)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 597, 27), 'width', tuple_var_assignment_137205_138736)
        
        # Obtaining an instance of the builtin type 'tuple' (line 598)
        tuple_138737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 598, 19), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 598)
        # Adding element type (line 598)
        # Getting the type of 'width' (line 598)
        width_138738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 19), 'width')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 598, 19), tuple_138737, width_138738)
        # Adding element type (line 598)
        # Getting the type of 'height' (line 598)
        height_138739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 26), 'height')
        # Getting the type of 'depth' (line 598)
        depth_138740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 35), 'depth')
        # Applying the binary operator '+' (line 598)
        result_add_138741 = python_operator(stypy.reporting.localization.Localization(__file__, 598, 26), '+', height_138739, depth_138740)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 598, 19), tuple_138737, result_add_138741)
        # Adding element type (line 598)
        # Getting the type of 'depth' (line 598)
        depth_138742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 42), 'depth')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 598, 19), tuple_138737, depth_138742)
        
        # Assigning a type to the variable 'stypy_return_type' (line 598)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 598, 12), 'stypy_return_type', tuple_138737)
        # SSA branch for the else part of an if statement (line 587)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 602):
        
        # Assigning a Call to a Name (line 602):
        
        # Call to make_dvi(...): (line 602)
        # Processing the call arguments (line 602)
        # Getting the type of 'tex' (line 602)
        tex_138745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 36), 'tex', False)
        # Getting the type of 'fontsize' (line 602)
        fontsize_138746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 41), 'fontsize', False)
        # Processing the call keyword arguments (line 602)
        kwargs_138747 = {}
        # Getting the type of 'self' (line 602)
        self_138743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 22), 'self', False)
        # Obtaining the member 'make_dvi' of a type (line 602)
        make_dvi_138744 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 602, 22), self_138743, 'make_dvi')
        # Calling make_dvi(args, kwargs) (line 602)
        make_dvi_call_result_138748 = invoke(stypy.reporting.localization.Localization(__file__, 602, 22), make_dvi_138744, *[tex_138745, fontsize_138746], **kwargs_138747)
        
        # Assigning a type to the variable 'dvifile' (line 602)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 602, 12), 'dvifile', make_dvi_call_result_138748)
        
        # Call to Dvi(...): (line 603)
        # Processing the call arguments (line 603)
        # Getting the type of 'dvifile' (line 603)
        dvifile_138751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 29), 'dvifile', False)
        int_138752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 603, 38), 'int')
        # Getting the type of 'dpi_fraction' (line 603)
        dpi_fraction_138753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 43), 'dpi_fraction', False)
        # Applying the binary operator '*' (line 603)
        result_mul_138754 = python_operator(stypy.reporting.localization.Localization(__file__, 603, 38), '*', int_138752, dpi_fraction_138753)
        
        # Processing the call keyword arguments (line 603)
        kwargs_138755 = {}
        # Getting the type of 'dviread' (line 603)
        dviread_138749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 17), 'dviread', False)
        # Obtaining the member 'Dvi' of a type (line 603)
        Dvi_138750 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 603, 17), dviread_138749, 'Dvi')
        # Calling Dvi(args, kwargs) (line 603)
        Dvi_call_result_138756 = invoke(stypy.reporting.localization.Localization(__file__, 603, 17), Dvi_138750, *[dvifile_138751, result_mul_138754], **kwargs_138755)
        
        with_138757 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 603, 17), Dvi_call_result_138756, 'with parameter', '__enter__', '__exit__')

        if with_138757:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 603)
            enter___138758 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 603, 17), Dvi_call_result_138756, '__enter__')
            with_enter_138759 = invoke(stypy.reporting.localization.Localization(__file__, 603, 17), enter___138758)
            # Assigning a type to the variable 'dvi' (line 603)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 603, 17), 'dvi', with_enter_138759)
            
            # Assigning a Call to a Name (line 604):
            
            # Assigning a Call to a Name (line 604):
            
            # Call to next(...): (line 604)
            # Processing the call arguments (line 604)
            
            # Call to iter(...): (line 604)
            # Processing the call arguments (line 604)
            # Getting the type of 'dvi' (line 604)
            dvi_138762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 33), 'dvi', False)
            # Processing the call keyword arguments (line 604)
            kwargs_138763 = {}
            # Getting the type of 'iter' (line 604)
            iter_138761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 28), 'iter', False)
            # Calling iter(args, kwargs) (line 604)
            iter_call_result_138764 = invoke(stypy.reporting.localization.Localization(__file__, 604, 28), iter_138761, *[dvi_138762], **kwargs_138763)
            
            # Processing the call keyword arguments (line 604)
            kwargs_138765 = {}
            # Getting the type of 'next' (line 604)
            next_138760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 23), 'next', False)
            # Calling next(args, kwargs) (line 604)
            next_call_result_138766 = invoke(stypy.reporting.localization.Localization(__file__, 604, 23), next_138760, *[iter_call_result_138764], **kwargs_138765)
            
            # Assigning a type to the variable 'page' (line 604)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 604, 16), 'page', next_call_result_138766)
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 603)
            exit___138767 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 603, 17), Dvi_call_result_138756, '__exit__')
            with_exit_138768 = invoke(stypy.reporting.localization.Localization(__file__, 603, 17), exit___138767, None, None, None)

        
        # Obtaining an instance of the builtin type 'tuple' (line 606)
        tuple_138769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 606, 19), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 606)
        # Adding element type (line 606)
        # Getting the type of 'page' (line 606)
        page_138770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 606, 19), 'page')
        # Obtaining the member 'width' of a type (line 606)
        width_138771 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 606, 19), page_138770, 'width')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 606, 19), tuple_138769, width_138771)
        # Adding element type (line 606)
        # Getting the type of 'page' (line 606)
        page_138772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 606, 31), 'page')
        # Obtaining the member 'height' of a type (line 606)
        height_138773 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 606, 31), page_138772, 'height')
        # Getting the type of 'page' (line 606)
        page_138774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 606, 45), 'page')
        # Obtaining the member 'descent' of a type (line 606)
        descent_138775 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 606, 45), page_138774, 'descent')
        # Applying the binary operator '+' (line 606)
        result_add_138776 = python_operator(stypy.reporting.localization.Localization(__file__, 606, 31), '+', height_138773, descent_138775)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 606, 19), tuple_138769, result_add_138776)
        # Adding element type (line 606)
        # Getting the type of 'page' (line 606)
        page_138777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 606, 59), 'page')
        # Obtaining the member 'descent' of a type (line 606)
        descent_138778 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 606, 59), page_138777, 'descent')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 606, 19), tuple_138769, descent_138778)
        
        # Assigning a type to the variable 'stypy_return_type' (line 606)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 606, 12), 'stypy_return_type', tuple_138769)
        # SSA join for if statement (line 587)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'get_text_width_height_descent(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_text_width_height_descent' in the type store
        # Getting the type of 'stypy_return_type' (line 575)
        stypy_return_type_138779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_138779)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_text_width_height_descent'
        return stypy_return_type_138779


# Assigning a type to the variable 'TexManager' (line 89)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 0), 'TexManager', TexManager)

# Assigning a Call to a Name (line 95):

# Call to get_home(...): (line 95)
# Processing the call keyword arguments (line 95)
kwargs_138782 = {}
# Getting the type of 'mpl' (line 95)
mpl_138780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 14), 'mpl', False)
# Obtaining the member 'get_home' of a type (line 95)
get_home_138781 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 14), mpl_138780, 'get_home')
# Calling get_home(args, kwargs) (line 95)
get_home_call_result_138783 = invoke(stypy.reporting.localization.Localization(__file__, 95, 14), get_home_138781, *[], **kwargs_138782)

# Getting the type of 'TexManager'
TexManager_138784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TexManager')
# Setting the type of the member 'oldpath' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TexManager_138784, 'oldpath', get_home_call_result_138783)

# Assigning a Call to a Name (line 95):

# Type idiom detected: calculating its left and rigth part (line 96)
# Getting the type of 'TexManager'
TexManager_138785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TexManager')
# Obtaining the member 'oldpath' of a type
oldpath_138786 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TexManager_138785, 'oldpath')
# Getting the type of 'None' (line 96)
None_138787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 18), 'None')

(may_be_138788, more_types_in_union_138789) = may_be_none(oldpath_138786, None_138787)

if may_be_138788:

    if more_types_in_union_138789:
        # Runtime conditional SSA (line 96)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
    else:
        module_type_store = module_type_store

    
    # Assigning a Call to a Name (line 97):
    
    # Assigning a Call to a Name (line 97):
    
    # Call to get_data_path(...): (line 97)
    # Processing the call keyword arguments (line 97)
    kwargs_138792 = {}
    # Getting the type of 'mpl' (line 97)
    mpl_138790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 18), 'mpl', False)
    # Obtaining the member 'get_data_path' of a type (line 97)
    get_data_path_138791 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 18), mpl_138790, 'get_data_path')
    # Calling get_data_path(args, kwargs) (line 97)
    get_data_path_call_result_138793 = invoke(stypy.reporting.localization.Localization(__file__, 97, 18), get_data_path_138791, *[], **kwargs_138792)
    
    # Getting the type of 'TexManager'
    TexManager_138794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TexManager')
    # Setting the type of the member 'oldpath' of a type
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TexManager_138794, 'oldpath', get_data_path_call_result_138793)

    if more_types_in_union_138789:
        # SSA join for if statement (line 96)
        module_type_store = module_type_store.join_ssa_context()




# Assigning a Call to a Name (line 98):

# Call to join(...): (line 98)
# Processing the call arguments (line 98)
# Getting the type of 'TexManager'
TexManager_138798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TexManager', False)
# Obtaining the member 'oldpath' of a type
oldpath_138799 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TexManager_138798, 'oldpath')
unicode_138800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 37), 'unicode', u'.tex.cache')
# Processing the call keyword arguments (line 98)
kwargs_138801 = {}
# Getting the type of 'os' (line 98)
os_138795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 15), 'os', False)
# Obtaining the member 'path' of a type (line 98)
path_138796 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 15), os_138795, 'path')
# Obtaining the member 'join' of a type (line 98)
join_138797 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 15), path_138796, 'join')
# Calling join(args, kwargs) (line 98)
join_call_result_138802 = invoke(stypy.reporting.localization.Localization(__file__, 98, 15), join_138797, *[oldpath_138799, unicode_138800], **kwargs_138801)

# Getting the type of 'TexManager'
TexManager_138803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TexManager')
# Setting the type of the member 'oldcache' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TexManager_138803, 'oldcache', join_call_result_138802)

# Assigning a Call to a Name (line 100):

# Call to get_cachedir(...): (line 100)
# Processing the call keyword arguments (line 100)
kwargs_138806 = {}
# Getting the type of 'mpl' (line 100)
mpl_138804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 15), 'mpl', False)
# Obtaining the member 'get_cachedir' of a type (line 100)
get_cachedir_138805 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 15), mpl_138804, 'get_cachedir')
# Calling get_cachedir(args, kwargs) (line 100)
get_cachedir_call_result_138807 = invoke(stypy.reporting.localization.Localization(__file__, 100, 15), get_cachedir_138805, *[], **kwargs_138806)

# Getting the type of 'TexManager'
TexManager_138808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TexManager')
# Setting the type of the member 'cachedir' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TexManager_138808, 'cachedir', get_cachedir_call_result_138807)

# Assigning a Call to a Name (line 100):

# Type idiom detected: calculating its left and rigth part (line 101)
# Getting the type of 'TexManager'
TexManager_138809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TexManager')
# Obtaining the member 'cachedir' of a type
cachedir_138810 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TexManager_138809, 'cachedir')
# Getting the type of 'None' (line 101)
None_138811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 23), 'None')

(may_be_138812, more_types_in_union_138813) = may_not_be_none(cachedir_138810, None_138811)

if may_be_138812:

    if more_types_in_union_138813:
        # Runtime conditional SSA (line 101)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
    else:
        module_type_store = module_type_store

    
    # Assigning a Call to a Name (line 102):
    
    # Assigning a Call to a Name (line 102):
    
    # Call to join(...): (line 102)
    # Processing the call arguments (line 102)
    # Getting the type of 'TexManager'
    TexManager_138817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TexManager', False)
    # Obtaining the member 'cachedir' of a type
    cachedir_138818 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TexManager_138817, 'cachedir')
    unicode_138819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 42), 'unicode', u'tex.cache')
    # Processing the call keyword arguments (line 102)
    kwargs_138820 = {}
    # Getting the type of 'os' (line 102)
    os_138814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 19), 'os', False)
    # Obtaining the member 'path' of a type (line 102)
    path_138815 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 19), os_138814, 'path')
    # Obtaining the member 'join' of a type (line 102)
    join_138816 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 19), path_138815, 'join')
    # Calling join(args, kwargs) (line 102)
    join_call_result_138821 = invoke(stypy.reporting.localization.Localization(__file__, 102, 19), join_138816, *[cachedir_138818, unicode_138819], **kwargs_138820)
    
    # Assigning a type to the variable 'texcache' (line 102)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 8), 'texcache', join_call_result_138821)

    if more_types_in_union_138813:
        # Runtime conditional SSA for else branch (line 101)
        module_type_store.open_ssa_branch('idiom else')



if ((not may_be_138812) or more_types_in_union_138813):
    
    # Assigning a Name to a Name (line 106):
    
    # Assigning a Name to a Name (line 106):
    # Getting the type of 'None' (line 106)
    None_138822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 19), 'None')
    # Assigning a type to the variable 'texcache' (line 106)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'texcache', None_138822)

    if (may_be_138812 and more_types_in_union_138813):
        # SSA join for if statement (line 101)
        module_type_store = module_type_store.join_ssa_context()




# Assigning a Call to a Name (line 98):


# Call to exists(...): (line 108)
# Processing the call arguments (line 108)
# Getting the type of 'TexManager'
TexManager_138826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TexManager', False)
# Obtaining the member 'oldcache' of a type
oldcache_138827 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TexManager_138826, 'oldcache')
# Processing the call keyword arguments (line 108)
kwargs_138828 = {}
# Getting the type of 'os' (line 108)
os_138823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 7), 'os', False)
# Obtaining the member 'path' of a type (line 108)
path_138824 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 7), os_138823, 'path')
# Obtaining the member 'exists' of a type (line 108)
exists_138825 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 7), path_138824, 'exists')
# Calling exists(args, kwargs) (line 108)
exists_call_result_138829 = invoke(stypy.reporting.localization.Localization(__file__, 108, 7), exists_138825, *[oldcache_138827], **kwargs_138828)

# Testing the type of an if condition (line 108)
if_condition_138830 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 108, 4), exists_call_result_138829)
# Assigning a type to the variable 'if_condition_138830' (line 108)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), 'if_condition_138830', if_condition_138830)
# SSA begins for if statement (line 108)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Type idiom detected: calculating its left and rigth part (line 109)
# Getting the type of 'texcache' (line 109)
texcache_138831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'texcache')
# Getting the type of 'None' (line 109)
None_138832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 27), 'None')

(may_be_138833, more_types_in_union_138834) = may_not_be_none(texcache_138831, None_138832)

if may_be_138833:

    if more_types_in_union_138834:
        # Runtime conditional SSA (line 109)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
    else:
        module_type_store = module_type_store

    
    
    # SSA begins for try-except statement (line 110)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Call to move(...): (line 111)
    # Processing the call arguments (line 111)
    # Getting the type of 'TexManager'
    TexManager_138837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TexManager', False)
    # Obtaining the member 'oldcache' of a type
    oldcache_138838 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TexManager_138837, 'oldcache')
    # Getting the type of 'texcache' (line 111)
    texcache_138839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 38), 'texcache', False)
    # Processing the call keyword arguments (line 111)
    kwargs_138840 = {}
    # Getting the type of 'shutil' (line 111)
    shutil_138835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 16), 'shutil', False)
    # Obtaining the member 'move' of a type (line 111)
    move_138836 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 16), shutil_138835, 'move')
    # Calling move(args, kwargs) (line 111)
    move_call_result_138841 = invoke(stypy.reporting.localization.Localization(__file__, 111, 16), move_138836, *[oldcache_138838, texcache_138839], **kwargs_138840)
    
    # SSA branch for the except part of a try statement (line 110)
    # SSA branch for the except 'IOError' branch of a try statement (line 110)
    # Storing handler type
    module_type_store.open_ssa_branch('except')
    # Getting the type of 'IOError' (line 112)
    IOError_138842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 19), 'IOError')
    # Assigning a type to the variable 'e' (line 112)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 12), 'e', IOError_138842)
    
    # Call to warn(...): (line 113)
    # Processing the call arguments (line 113)
    unicode_138845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 30), 'unicode', u'File could not be renamed: %s')
    # Getting the type of 'e' (line 113)
    e_138846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 64), 'e', False)
    # Applying the binary operator '%' (line 113)
    result_mod_138847 = python_operator(stypy.reporting.localization.Localization(__file__, 113, 30), '%', unicode_138845, e_138846)
    
    # Processing the call keyword arguments (line 113)
    kwargs_138848 = {}
    # Getting the type of 'warnings' (line 113)
    warnings_138843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 16), 'warnings', False)
    # Obtaining the member 'warn' of a type (line 113)
    warn_138844 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 16), warnings_138843, 'warn')
    # Calling warn(args, kwargs) (line 113)
    warn_call_result_138849 = invoke(stypy.reporting.localization.Localization(__file__, 113, 16), warn_138844, *[result_mod_138847], **kwargs_138848)
    
    # SSA branch for the else branch of a try statement (line 110)
    module_type_store.open_ssa_branch('except else')
    
    # Call to warn(...): (line 115)
    # Processing the call arguments (line 115)
    unicode_138852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, (-1)), 'unicode', u'Found a TeX cache dir in the deprecated location "%s".\n    Moving it to the new default location "%s".')
    
    # Obtaining an instance of the builtin type 'tuple' (line 117)
    tuple_138853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 54), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 117)
    # Adding element type (line 117)
    # Getting the type of 'TexManager'
    TexManager_138854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TexManager', False)
    # Obtaining the member 'oldcache' of a type
    oldcache_138855 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TexManager_138854, 'oldcache')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 54), tuple_138853, oldcache_138855)
    # Adding element type (line 117)
    # Getting the type of 'texcache' (line 117)
    texcache_138856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 64), 'texcache', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 54), tuple_138853, texcache_138856)
    
    # Applying the binary operator '%' (line 117)
    result_mod_138857 = python_operator(stypy.reporting.localization.Localization(__file__, 117, (-1)), '%', unicode_138852, tuple_138853)
    
    # Processing the call keyword arguments (line 115)
    kwargs_138858 = {}
    # Getting the type of 'warnings' (line 115)
    warnings_138850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 16), 'warnings', False)
    # Obtaining the member 'warn' of a type (line 115)
    warn_138851 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 16), warnings_138850, 'warn')
    # Calling warn(args, kwargs) (line 115)
    warn_call_result_138859 = invoke(stypy.reporting.localization.Localization(__file__, 115, 16), warn_138851, *[result_mod_138857], **kwargs_138858)
    
    # SSA join for try-except statement (line 110)
    module_type_store = module_type_store.join_ssa_context()
    

    if more_types_in_union_138834:
        # Runtime conditional SSA for else branch (line 109)
        module_type_store.open_ssa_branch('idiom else')



if ((not may_be_138833) or more_types_in_union_138834):
    
    # Call to warn(...): (line 119)
    # Processing the call arguments (line 119)
    unicode_138862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, (-1)), 'unicode', u'Could not rename old TeX cache dir "%s": a suitable configuration\n    directory could not be found.')
    # Getting the type of 'TexManager'
    TexManager_138863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TexManager', False)
    # Obtaining the member 'oldcache' of a type
    oldcache_138864 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TexManager_138863, 'oldcache')
    # Applying the binary operator '%' (line 121)
    result_mod_138865 = python_operator(stypy.reporting.localization.Localization(__file__, 121, (-1)), '%', unicode_138862, oldcache_138864)
    
    # Processing the call keyword arguments (line 119)
    kwargs_138866 = {}
    # Getting the type of 'warnings' (line 119)
    warnings_138860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 12), 'warnings', False)
    # Obtaining the member 'warn' of a type (line 119)
    warn_138861 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 12), warnings_138860, 'warn')
    # Calling warn(args, kwargs) (line 119)
    warn_call_result_138867 = invoke(stypy.reporting.localization.Localization(__file__, 119, 12), warn_138861, *[result_mod_138865], **kwargs_138866)
    

    if (may_be_138833 and more_types_in_union_138834):
        # SSA join for if statement (line 109)
        module_type_store = module_type_store.join_ssa_context()



# SSA join for if statement (line 108)
module_type_store = module_type_store.join_ssa_context()


# Type idiom detected: calculating its left and rigth part (line 123)
# Getting the type of 'texcache' (line 123)
texcache_138868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 4), 'texcache')
# Getting the type of 'None' (line 123)
None_138869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 23), 'None')

(may_be_138870, more_types_in_union_138871) = may_not_be_none(texcache_138868, None_138869)

if may_be_138870:

    if more_types_in_union_138871:
        # Runtime conditional SSA (line 123)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
    else:
        module_type_store = module_type_store

    
    # Call to mkdirs(...): (line 124)
    # Processing the call arguments (line 124)
    # Getting the type of 'texcache' (line 124)
    texcache_138873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 15), 'texcache', False)
    # Processing the call keyword arguments (line 124)
    kwargs_138874 = {}
    # Getting the type of 'mkdirs' (line 124)
    mkdirs_138872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'mkdirs', False)
    # Calling mkdirs(args, kwargs) (line 124)
    mkdirs_call_result_138875 = invoke(stypy.reporting.localization.Localization(__file__, 124, 8), mkdirs_138872, *[texcache_138873], **kwargs_138874)
    

    if more_types_in_union_138871:
        # SSA join for if statement (line 123)
        module_type_store = module_type_store.join_ssa_context()




# Assigning a Dict to a Name (line 127):

# Obtaining an instance of the builtin type 'dict' (line 127)
dict_138876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 18), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 127)

# Getting the type of 'TexManager'
TexManager_138877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TexManager')
# Setting the type of the member 'rgba_arrayd' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TexManager_138877, 'rgba_arrayd', dict_138876)

# Assigning a Dict to a Name (line 128):

# Obtaining an instance of the builtin type 'dict' (line 128)
dict_138878 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 18), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 128)

# Getting the type of 'TexManager'
TexManager_138879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TexManager')
# Setting the type of the member 'grey_arrayd' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TexManager_138879, 'grey_arrayd', dict_138878)

# Assigning a Dict to a Name (line 129):

# Obtaining an instance of the builtin type 'dict' (line 129)
dict_138880 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 18), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 129)

# Getting the type of 'TexManager'
TexManager_138881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TexManager')
# Setting the type of the member 'postscriptd' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TexManager_138881, 'postscriptd', dict_138880)

# Assigning a Num to a Name (line 130):
int_138882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 12), 'int')
# Getting the type of 'TexManager'
TexManager_138883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TexManager')
# Setting the type of the member 'pscnt' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TexManager_138883, 'pscnt', int_138882)

# Assigning a Tuple to a Name (line 132):

# Obtaining an instance of the builtin type 'tuple' (line 132)
tuple_138884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 13), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 132)
# Adding element type (line 132)
unicode_138885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 13), 'unicode', u'cmr')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 13), tuple_138884, unicode_138885)
# Adding element type (line 132)
unicode_138886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 20), 'unicode', u'')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 13), tuple_138884, unicode_138886)

# Getting the type of 'TexManager'
TexManager_138887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TexManager')
# Setting the type of the member 'serif' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TexManager_138887, 'serif', tuple_138884)

# Assigning a Tuple to a Name (line 133):

# Obtaining an instance of the builtin type 'tuple' (line 133)
tuple_138888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 18), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 133)
# Adding element type (line 133)
unicode_138889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 18), 'unicode', u'cmss')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 18), tuple_138888, unicode_138889)
# Adding element type (line 133)
unicode_138890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 26), 'unicode', u'')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 18), tuple_138888, unicode_138890)

# Getting the type of 'TexManager'
TexManager_138891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TexManager')
# Setting the type of the member 'sans_serif' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TexManager_138891, 'sans_serif', tuple_138888)

# Assigning a Tuple to a Name (line 134):

# Obtaining an instance of the builtin type 'tuple' (line 134)
tuple_138892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 17), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 134)
# Adding element type (line 134)
unicode_138893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 17), 'unicode', u'cmtt')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 134, 17), tuple_138892, unicode_138893)
# Adding element type (line 134)
unicode_138894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 25), 'unicode', u'')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 134, 17), tuple_138892, unicode_138894)

# Getting the type of 'TexManager'
TexManager_138895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TexManager')
# Setting the type of the member 'monospace' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TexManager_138895, 'monospace', tuple_138892)

# Assigning a Tuple to a Name (line 135):

# Obtaining an instance of the builtin type 'tuple' (line 135)
tuple_138896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 15), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 135)
# Adding element type (line 135)
unicode_138897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 15), 'unicode', u'pzc')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 15), tuple_138896, unicode_138897)
# Adding element type (line 135)
unicode_138898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 22), 'unicode', u'\\usepackage{chancery}')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 15), tuple_138896, unicode_138898)

# Getting the type of 'TexManager'
TexManager_138899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TexManager')
# Setting the type of the member 'cursive' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TexManager_138899, 'cursive', tuple_138896)

# Assigning a Str to a Name (line 136):
unicode_138900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 18), 'unicode', u'serif')
# Getting the type of 'TexManager'
TexManager_138901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TexManager')
# Setting the type of the member 'font_family' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TexManager_138901, 'font_family', unicode_138900)

# Assigning a Tuple to a Name (line 137):

# Obtaining an instance of the builtin type 'tuple' (line 137)
tuple_138902 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 21), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 137)
# Adding element type (line 137)
unicode_138903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 21), 'unicode', u'serif')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 137, 21), tuple_138902, unicode_138903)
# Adding element type (line 137)
unicode_138904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 30), 'unicode', u'sans-serif')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 137, 21), tuple_138902, unicode_138904)
# Adding element type (line 137)
unicode_138905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 44), 'unicode', u'cursive')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 137, 21), tuple_138902, unicode_138905)
# Adding element type (line 137)
unicode_138906 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 55), 'unicode', u'monospace')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 137, 21), tuple_138902, unicode_138906)

# Getting the type of 'TexManager'
TexManager_138907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TexManager')
# Setting the type of the member 'font_families' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TexManager_138907, 'font_families', tuple_138902)

# Assigning a Dict to a Name (line 139):

# Obtaining an instance of the builtin type 'dict' (line 139)
dict_138908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 16), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 139)
# Adding element type (key, value) (line 139)
unicode_138909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 17), 'unicode', u'new century schoolbook')

# Obtaining an instance of the builtin type 'tuple' (line 139)
tuple_138910 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 44), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 139)
# Adding element type (line 139)
unicode_138911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 44), 'unicode', u'pnc')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 139, 44), tuple_138910, unicode_138911)
# Adding element type (line 139)
unicode_138912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 44), 'unicode', u'\\renewcommand{\\rmdefault}{pnc}')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 139, 44), tuple_138910, unicode_138912)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 139, 16), dict_138908, (unicode_138909, tuple_138910))
# Adding element type (key, value) (line 139)
unicode_138913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 17), 'unicode', u'bookman')

# Obtaining an instance of the builtin type 'tuple' (line 141)
tuple_138914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 29), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 141)
# Adding element type (line 141)
unicode_138915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 29), 'unicode', u'pbk')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 141, 29), tuple_138914, unicode_138915)
# Adding element type (line 141)
unicode_138916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 36), 'unicode', u'\\renewcommand{\\rmdefault}{pbk}')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 141, 29), tuple_138914, unicode_138916)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 139, 16), dict_138908, (unicode_138913, tuple_138914))
# Adding element type (key, value) (line 139)
unicode_138917 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 17), 'unicode', u'times')

# Obtaining an instance of the builtin type 'tuple' (line 142)
tuple_138918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 27), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 142)
# Adding element type (line 142)
unicode_138919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 27), 'unicode', u'ptm')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 142, 27), tuple_138918, unicode_138919)
# Adding element type (line 142)
unicode_138920 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 34), 'unicode', u'\\usepackage{mathptmx}')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 142, 27), tuple_138918, unicode_138920)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 139, 16), dict_138908, (unicode_138917, tuple_138918))
# Adding element type (key, value) (line 139)
unicode_138921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 17), 'unicode', u'palatino')

# Obtaining an instance of the builtin type 'tuple' (line 143)
tuple_138922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 30), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 143)
# Adding element type (line 143)
unicode_138923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 30), 'unicode', u'ppl')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 143, 30), tuple_138922, unicode_138923)
# Adding element type (line 143)
unicode_138924 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 37), 'unicode', u'\\usepackage{mathpazo}')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 143, 30), tuple_138922, unicode_138924)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 139, 16), dict_138908, (unicode_138921, tuple_138922))
# Adding element type (key, value) (line 139)
unicode_138925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 17), 'unicode', u'zapf chancery')

# Obtaining an instance of the builtin type 'tuple' (line 144)
tuple_138926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 35), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 144)
# Adding element type (line 144)
unicode_138927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 35), 'unicode', u'pzc')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 35), tuple_138926, unicode_138927)
# Adding element type (line 144)
unicode_138928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 42), 'unicode', u'\\usepackage{chancery}')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 35), tuple_138926, unicode_138928)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 139, 16), dict_138908, (unicode_138925, tuple_138926))
# Adding element type (key, value) (line 139)
unicode_138929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 17), 'unicode', u'cursive')

# Obtaining an instance of the builtin type 'tuple' (line 145)
tuple_138930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 29), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 145)
# Adding element type (line 145)
unicode_138931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 29), 'unicode', u'pzc')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 29), tuple_138930, unicode_138931)
# Adding element type (line 145)
unicode_138932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 36), 'unicode', u'\\usepackage{chancery}')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 29), tuple_138930, unicode_138932)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 139, 16), dict_138908, (unicode_138929, tuple_138930))
# Adding element type (key, value) (line 139)
unicode_138933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 17), 'unicode', u'charter')

# Obtaining an instance of the builtin type 'tuple' (line 146)
tuple_138934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 29), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 146)
# Adding element type (line 146)
unicode_138935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 29), 'unicode', u'pch')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 146, 29), tuple_138934, unicode_138935)
# Adding element type (line 146)
unicode_138936 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 36), 'unicode', u'\\usepackage{charter}')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 146, 29), tuple_138934, unicode_138936)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 139, 16), dict_138908, (unicode_138933, tuple_138934))
# Adding element type (key, value) (line 139)
unicode_138937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 17), 'unicode', u'serif')

# Obtaining an instance of the builtin type 'tuple' (line 147)
tuple_138938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 27), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 147)
# Adding element type (line 147)
unicode_138939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 27), 'unicode', u'cmr')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 27), tuple_138938, unicode_138939)
# Adding element type (line 147)
unicode_138940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 34), 'unicode', u'')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 27), tuple_138938, unicode_138940)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 139, 16), dict_138908, (unicode_138937, tuple_138938))
# Adding element type (key, value) (line 139)
unicode_138941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 17), 'unicode', u'sans-serif')

# Obtaining an instance of the builtin type 'tuple' (line 148)
tuple_138942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 32), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 148)
# Adding element type (line 148)
unicode_138943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 32), 'unicode', u'cmss')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 148, 32), tuple_138942, unicode_138943)
# Adding element type (line 148)
unicode_138944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 40), 'unicode', u'')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 148, 32), tuple_138942, unicode_138944)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 139, 16), dict_138908, (unicode_138941, tuple_138942))
# Adding element type (key, value) (line 139)
unicode_138945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 17), 'unicode', u'helvetica')

# Obtaining an instance of the builtin type 'tuple' (line 149)
tuple_138946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 31), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 149)
# Adding element type (line 149)
unicode_138947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 31), 'unicode', u'phv')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 149, 31), tuple_138946, unicode_138947)
# Adding element type (line 149)
unicode_138948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 38), 'unicode', u'\\usepackage{helvet}')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 149, 31), tuple_138946, unicode_138948)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 139, 16), dict_138908, (unicode_138945, tuple_138946))
# Adding element type (key, value) (line 139)
unicode_138949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 17), 'unicode', u'avant garde')

# Obtaining an instance of the builtin type 'tuple' (line 150)
tuple_138950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 33), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 150)
# Adding element type (line 150)
unicode_138951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 33), 'unicode', u'pag')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 150, 33), tuple_138950, unicode_138951)
# Adding element type (line 150)
unicode_138952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 40), 'unicode', u'\\usepackage{avant}')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 150, 33), tuple_138950, unicode_138952)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 139, 16), dict_138908, (unicode_138949, tuple_138950))
# Adding element type (key, value) (line 139)
unicode_138953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 17), 'unicode', u'courier')

# Obtaining an instance of the builtin type 'tuple' (line 151)
tuple_138954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 29), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 151)
# Adding element type (line 151)
unicode_138955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 29), 'unicode', u'pcr')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 151, 29), tuple_138954, unicode_138955)
# Adding element type (line 151)
unicode_138956 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 36), 'unicode', u'\\usepackage{courier}')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 151, 29), tuple_138954, unicode_138956)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 139, 16), dict_138908, (unicode_138953, tuple_138954))
# Adding element type (key, value) (line 139)
unicode_138957 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 17), 'unicode', u'monospace')

# Obtaining an instance of the builtin type 'tuple' (line 152)
tuple_138958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 31), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 152)
# Adding element type (line 152)
unicode_138959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 31), 'unicode', u'cmtt')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 31), tuple_138958, unicode_138959)
# Adding element type (line 152)
unicode_138960 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 39), 'unicode', u'')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 31), tuple_138958, unicode_138960)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 139, 16), dict_138908, (unicode_138957, tuple_138958))
# Adding element type (key, value) (line 139)
unicode_138961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 17), 'unicode', u'computer modern roman')

# Obtaining an instance of the builtin type 'tuple' (line 153)
tuple_138962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 43), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 153)
# Adding element type (line 153)
unicode_138963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 43), 'unicode', u'cmr')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 153, 43), tuple_138962, unicode_138963)
# Adding element type (line 153)
unicode_138964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 50), 'unicode', u'')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 153, 43), tuple_138962, unicode_138964)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 139, 16), dict_138908, (unicode_138961, tuple_138962))
# Adding element type (key, value) (line 139)
unicode_138965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 17), 'unicode', u'computer modern sans serif')

# Obtaining an instance of the builtin type 'tuple' (line 154)
tuple_138966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 48), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 154)
# Adding element type (line 154)
unicode_138967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 48), 'unicode', u'cmss')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 154, 48), tuple_138966, unicode_138967)
# Adding element type (line 154)
unicode_138968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 56), 'unicode', u'')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 154, 48), tuple_138966, unicode_138968)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 139, 16), dict_138908, (unicode_138965, tuple_138966))
# Adding element type (key, value) (line 139)
unicode_138969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 17), 'unicode', u'computer modern typewriter')

# Obtaining an instance of the builtin type 'tuple' (line 155)
tuple_138970 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 48), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 155)
# Adding element type (line 155)
unicode_138971 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 48), 'unicode', u'cmtt')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 155, 48), tuple_138970, unicode_138971)
# Adding element type (line 155)
unicode_138972 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 56), 'unicode', u'')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 155, 48), tuple_138970, unicode_138972)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 139, 16), dict_138908, (unicode_138969, tuple_138970))

# Getting the type of 'TexManager'
TexManager_138973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TexManager')
# Setting the type of the member 'font_info' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TexManager_138973, 'font_info', dict_138908)

# Assigning a Name to a Name (line 157):
# Getting the type of 'None' (line 157)
None_138974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 16), 'None')
# Getting the type of 'TexManager'
TexManager_138975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TexManager')
# Setting the type of the member '_rc_cache' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TexManager_138975, '_rc_cache', None_138974)

# Assigning a BinOp to a Name (line 158):

# Obtaining an instance of the builtin type 'tuple' (line 158)
tuple_138976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 23), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 158)
# Adding element type (line 158)
unicode_138977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 23), 'unicode', u'text.latex.preamble')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 158, 23), tuple_138976, unicode_138977)


# Call to tuple(...): (line 159)
# Processing the call arguments (line 159)
# Calculating list comprehension
# Calculating comprehension expression

# Obtaining an instance of the builtin type 'tuple' (line 159)
tuple_138982 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 51), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 159)
# Adding element type (line 159)
unicode_138983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 51), 'unicode', u'family')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 159, 51), tuple_138982, unicode_138983)

# Getting the type of 'TexManager'
TexManager_138984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TexManager', False)
# Obtaining the member 'font_families' of a type
font_families_138985 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TexManager_138984, 'font_families')
# Applying the binary operator '+' (line 159)
result_add_138986 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 50), '+', tuple_138982, font_families_138985)

comprehension_138987 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 159, 29), result_add_138986)
# Assigning a type to the variable 'n' (line 159)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 29), 'n', comprehension_138987)
unicode_138979 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 29), 'unicode', u'font.')
# Getting the type of 'n' (line 159)
n_138980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 39), 'n', False)
# Applying the binary operator '+' (line 159)
result_add_138981 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 29), '+', unicode_138979, n_138980)

list_138988 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 29), 'list')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 159, 29), list_138988, result_add_138981)
# Processing the call keyword arguments (line 159)
kwargs_138989 = {}
# Getting the type of 'tuple' (line 159)
tuple_138978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 22), 'tuple', False)
# Calling tuple(args, kwargs) (line 159)
tuple_call_result_138990 = invoke(stypy.reporting.localization.Localization(__file__, 159, 22), tuple_138978, *[list_138988], **kwargs_138989)

# Applying the binary operator '+' (line 158)
result_add_138991 = python_operator(stypy.reporting.localization.Localization(__file__, 158, 22), '+', tuple_138976, tuple_call_result_138990)

# Getting the type of 'TexManager'
TexManager_138992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TexManager')
# Setting the type of the member '_rc_cache_keys' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TexManager_138992, '_rc_cache_keys', result_add_138991)

# Assigning a Call to a Name (line 310):

# Call to compile(...): (line 310)
# Processing the call arguments (line 310)
unicode_138995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 8), 'unicode', u'MatplotlibBox:\\(([\\d.]+)pt\\+([\\d.]+)pt\\)x([\\d.]+)pt')
# Processing the call keyword arguments (line 310)
kwargs_138996 = {}
# Getting the type of 're' (line 310)
re_138993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 15), 're', False)
# Obtaining the member 'compile' of a type (line 310)
compile_138994 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 15), re_138993, 'compile')
# Calling compile(args, kwargs) (line 310)
compile_call_result_138997 = invoke(stypy.reporting.localization.Localization(__file__, 310, 15), compile_138994, *[unicode_138995], **kwargs_138996)

# Getting the type of 'TexManager'
TexManager_138998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TexManager')
# Setting the type of the member '_re_vbox' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TexManager_138998, '_re_vbox', compile_call_result_138997)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
