
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: A module for reading dvi files output by TeX. Several limitations make
3: this not (currently) useful as a general-purpose dvi preprocessor, but
4: it is currently used by the pdf backend for processing usetex text.
5: 
6: Interface::
7: 
8:   with Dvi(filename, 72) as dvi:
9:       # iterate over pages:
10:       for page in dvi:
11:           w, h, d = page.width, page.height, page.descent
12:           for x,y,font,glyph,width in page.text:
13:               fontname = font.texname
14:               pointsize = font.size
15:               ...
16:           for x,y,height,width in page.boxes:
17:               ...
18: 
19: '''
20: from __future__ import (absolute_import, division, print_function,
21:                         unicode_literals)
22: 
23: import six
24: from six.moves import xrange
25: 
26: from collections import namedtuple
27: import errno
28: from functools import partial, wraps
29: import matplotlib
30: import matplotlib.cbook as mpl_cbook
31: from matplotlib.compat import subprocess
32: from matplotlib import rcParams
33: import numpy as np
34: import re
35: import struct
36: import sys
37: import textwrap
38: import os
39: 
40: if six.PY3:
41:     def ord(x):
42:         return x
43: 
44: # Dvi is a bytecode format documented in
45: # http://mirrors.ctan.org/systems/knuth/dist/texware/dvitype.web
46: # http://texdoc.net/texmf-dist/doc/generic/knuth/texware/dvitype.pdf
47: #
48: # The file consists of a preamble, some number of pages, a postamble,
49: # and a finale. Different opcodes are allowed in different contexts,
50: # so the Dvi object has a parser state:
51: #
52: #   pre:       expecting the preamble
53: #   outer:     between pages (followed by a page or the postamble,
54: #              also e.g. font definitions are allowed)
55: #   page:      processing a page
56: #   post_post: state after the postamble (our current implementation
57: #              just stops reading)
58: #   finale:    the finale (unimplemented in our current implementation)
59: 
60: _dvistate = mpl_cbook.Bunch(pre=0, outer=1, inpage=2, post_post=3, finale=4)
61: 
62: # The marks on a page consist of text and boxes. A page also has dimensions.
63: Page = namedtuple('Page', 'text boxes height width descent')
64: Text = namedtuple('Text', 'x y font glyph width')
65: Box = namedtuple('Box', 'x y height width')
66: 
67: 
68: # Opcode argument parsing
69: #
70: # Each of the following functions takes a Dvi object and delta,
71: # which is the difference between the opcode and the minimum opcode
72: # with the same meaning. Dvi opcodes often encode the number of
73: # argument bytes in this delta.
74: 
75: def _arg_raw(dvi, delta):
76:     '''Return *delta* without reading anything more from the dvi file'''
77:     return delta
78: 
79: 
80: def _arg(bytes, signed, dvi, _):
81:     '''Read *bytes* bytes, returning the bytes interpreted as a
82:     signed integer if *signed* is true, unsigned otherwise.'''
83:     return dvi._arg(bytes, signed)
84: 
85: 
86: def _arg_slen(dvi, delta):
87:     '''Signed, length *delta*
88: 
89:     Read *delta* bytes, returning None if *delta* is zero, and
90:     the bytes interpreted as a signed integer otherwise.'''
91:     if delta == 0:
92:         return None
93:     return dvi._arg(delta, True)
94: 
95: 
96: def _arg_slen1(dvi, delta):
97:     '''Signed, length *delta*+1
98: 
99:     Read *delta*+1 bytes, returning the bytes interpreted as signed.'''
100:     return dvi._arg(delta+1, True)
101: 
102: 
103: def _arg_ulen1(dvi, delta):
104:     '''Unsigned length *delta*+1
105: 
106:     Read *delta*+1 bytes, returning the bytes interpreted as unsigned.'''
107:     return dvi._arg(delta+1, False)
108: 
109: 
110: def _arg_olen1(dvi, delta):
111:     '''Optionally signed, length *delta*+1
112: 
113:     Read *delta*+1 bytes, returning the bytes interpreted as
114:     unsigned integer for 0<=*delta*<3 and signed if *delta*==3.'''
115:     return dvi._arg(delta + 1, delta == 3)
116: 
117: 
118: _arg_mapping = dict(raw=_arg_raw,
119:                     u1=partial(_arg, 1, False),
120:                     u4=partial(_arg, 4, False),
121:                     s4=partial(_arg, 4, True),
122:                     slen=_arg_slen,
123:                     olen1=_arg_olen1,
124:                     slen1=_arg_slen1,
125:                     ulen1=_arg_ulen1)
126: 
127: 
128: def _dispatch(table, min, max=None, state=None, args=('raw',)):
129:     '''Decorator for dispatch by opcode. Sets the values in *table*
130:     from *min* to *max* to this method, adds a check that the Dvi state
131:     matches *state* if not None, reads arguments from the file according
132:     to *args*.
133: 
134:     *table*
135:         the dispatch table to be filled in
136: 
137:     *min*
138:         minimum opcode for calling this function
139: 
140:     *max*
141:         maximum opcode for calling this function, None if only *min* is allowed
142: 
143:     *state*
144:         state of the Dvi object in which these opcodes are allowed
145: 
146:     *args*
147:         sequence of argument specifications:
148: 
149:         ``'raw'``: opcode minus minimum
150:         ``'u1'``: read one unsigned byte
151:         ``'u4'``: read four bytes, treat as an unsigned number
152:         ``'s4'``: read four bytes, treat as a signed number
153:         ``'slen'``: read (opcode - minimum) bytes, treat as signed
154:         ``'slen1'``: read (opcode - minimum + 1) bytes, treat as signed
155:         ``'ulen1'``: read (opcode - minimum + 1) bytes, treat as unsigned
156:         ``'olen1'``: read (opcode - minimum + 1) bytes, treat as unsigned
157:                      if under four bytes, signed if four bytes
158:     '''
159:     def decorate(method):
160:         get_args = [_arg_mapping[x] for x in args]
161: 
162:         @wraps(method)
163:         def wrapper(self, byte):
164:             if state is not None and self.state != state:
165:                 raise ValueError("state precondition failed")
166:             return method(self, *[f(self, byte-min) for f in get_args])
167:         if max is None:
168:             table[min] = wrapper
169:         else:
170:             for i in xrange(min, max+1):
171:                 assert table[i] is None
172:                 table[i] = wrapper
173:         return wrapper
174:     return decorate
175: 
176: 
177: class Dvi(object):
178:     '''
179:     A reader for a dvi ("device-independent") file, as produced by TeX.
180:     The current implementation can only iterate through pages in order,
181:     and does not even attempt to verify the postamble.
182: 
183:     This class can be used as a context manager to close the underlying
184:     file upon exit. Pages can be read via iteration. Here is an overly
185:     simple way to extract text without trying to detect whitespace::
186: 
187:     >>> with matplotlib.dviread.Dvi('input.dvi', 72) as dvi:
188:     >>>     for page in dvi:
189:     >>>         print ''.join(unichr(t.glyph) for t in page.text)
190:     '''
191:     # dispatch table
192:     _dtable = [None for _ in xrange(256)]
193:     dispatch = partial(_dispatch, _dtable)
194: 
195:     def __init__(self, filename, dpi):
196:         '''
197:         Read the data from the file named *filename* and convert
198:         TeX's internal units to units of *dpi* per inch.
199:         *dpi* only sets the units and does not limit the resolution.
200:         Use None to return TeX's internal units.
201:         '''
202:         matplotlib.verbose.report('Dvi: ' + filename, 'debug')
203:         self.file = open(filename, 'rb')
204:         self.dpi = dpi
205:         self.fonts = {}
206:         self.state = _dvistate.pre
207:         self.baseline = self._get_baseline(filename)
208: 
209:     def _get_baseline(self, filename):
210:         if rcParams['text.latex.preview']:
211:             base, ext = os.path.splitext(filename)
212:             baseline_filename = base + ".baseline"
213:             if os.path.exists(baseline_filename):
214:                 with open(baseline_filename, 'rb') as fd:
215:                     l = fd.read().split()
216:                 height, depth, width = l
217:                 return float(depth)
218:         return None
219: 
220:     def __enter__(self):
221:         '''
222:         Context manager enter method, does nothing.
223:         '''
224:         return self
225: 
226:     def __exit__(self, etype, evalue, etrace):
227:         '''
228:         Context manager exit method, closes the underlying file if it is open.
229:         '''
230:         self.close()
231: 
232:     def __iter__(self):
233:         '''
234:         Iterate through the pages of the file.
235: 
236:         Yields
237:         ------
238:         Page
239:             Details of all the text and box objects on the page.
240:             The Page tuple contains lists of Text and Box tuples and
241:             the page dimensions, and the Text and Box tuples contain
242:             coordinates transformed into a standard Cartesian
243:             coordinate system at the dpi value given when initializing.
244:             The coordinates are floating point numbers, but otherwise
245:             precision is not lost and coordinate values are not clipped to
246:             integers.
247:         '''
248:         while True:
249:             have_page = self._read()
250:             if have_page:
251:                 yield self._output()
252:             else:
253:                 break
254: 
255:     def close(self):
256:         '''
257:         Close the underlying file if it is open.
258:         '''
259:         if not self.file.closed:
260:             self.file.close()
261: 
262:     def _output(self):
263:         '''
264:         Output the text and boxes belonging to the most recent page.
265:         page = dvi._output()
266:         '''
267:         minx, miny, maxx, maxy = np.inf, np.inf, -np.inf, -np.inf
268:         maxy_pure = -np.inf
269:         for elt in self.text + self.boxes:
270:             if isinstance(elt, Box):
271:                 x, y, h, w = elt
272:                 e = 0           # zero depth
273:             else:               # glyph
274:                 x, y, font, g, w = elt
275:                 h, e = font._height_depth_of(g)
276:             minx = min(minx, x)
277:             miny = min(miny, y - h)
278:             maxx = max(maxx, x + w)
279:             maxy = max(maxy, y + e)
280:             maxy_pure = max(maxy_pure, y)
281: 
282:         if self.dpi is None:
283:             # special case for ease of debugging: output raw dvi coordinates
284:             return Page(text=self.text, boxes=self.boxes,
285:                         width=maxx-minx, height=maxy_pure-miny,
286:                         descent=maxy-maxy_pure)
287: 
288:         # convert from TeX's "scaled points" to dpi units
289:         d = self.dpi / (72.27 * 2**16)
290:         if self.baseline is None:
291:             descent = (maxy - maxy_pure) * d
292:         else:
293:             descent = self.baseline
294: 
295:         text = [Text((x-minx)*d, (maxy-y)*d - descent, f, g, w*d)
296:                 for (x, y, f, g, w) in self.text]
297:         boxes = [Box((x-minx)*d, (maxy-y)*d - descent, h*d, w*d)
298:                  for (x, y, h, w) in self.boxes]
299: 
300:         return Page(text=text, boxes=boxes, width=(maxx-minx)*d,
301:                     height=(maxy_pure-miny)*d, descent=descent)
302: 
303:     def _read(self):
304:         '''
305:         Read one page from the file. Return True if successful,
306:         False if there were no more pages.
307:         '''
308:         while True:
309:             byte = ord(self.file.read(1)[0])
310:             self._dtable[byte](self, byte)
311:             if byte == 140:                         # end of page
312:                 return True
313:             if self.state == _dvistate.post_post:   # end of file
314:                 self.close()
315:                 return False
316: 
317:     def _arg(self, nbytes, signed=False):
318:         '''
319:         Read and return an integer argument *nbytes* long.
320:         Signedness is determined by the *signed* keyword.
321:         '''
322:         str = self.file.read(nbytes)
323:         value = ord(str[0])
324:         if signed and value >= 0x80:
325:             value = value - 0x100
326:         for i in range(1, nbytes):
327:             value = 0x100*value + ord(str[i])
328:         return value
329: 
330:     @dispatch(min=0, max=127, state=_dvistate.inpage)
331:     def _set_char_immediate(self, char):
332:         self._put_char_real(char)
333:         self.h += self.fonts[self.f]._width_of(char)
334: 
335:     @dispatch(min=128, max=131, state=_dvistate.inpage, args=('olen1',))
336:     def _set_char(self, char):
337:         self._put_char_real(char)
338:         self.h += self.fonts[self.f]._width_of(char)
339: 
340:     @dispatch(132, state=_dvistate.inpage, args=('s4', 's4'))
341:     def _set_rule(self, a, b):
342:         self._put_rule_real(a, b)
343:         self.h += b
344: 
345:     @dispatch(min=133, max=136, state=_dvistate.inpage, args=('olen1',))
346:     def _put_char(self, char):
347:         self._put_char_real(char)
348: 
349:     def _put_char_real(self, char):
350:         font = self.fonts[self.f]
351:         if font._vf is None:
352:             self.text.append(Text(self.h, self.v, font, char,
353:                                   font._width_of(char)))
354:         else:
355:             scale = font._scale
356:             for x, y, f, g, w in font._vf[char].text:
357:                 newf = DviFont(scale=_mul2012(scale, f._scale),
358:                                tfm=f._tfm, texname=f.texname, vf=f._vf)
359:                 self.text.append(Text(self.h + _mul2012(x, scale),
360:                                       self.v + _mul2012(y, scale),
361:                                       newf, g, newf._width_of(g)))
362:             self.boxes.extend([Box(self.h + _mul2012(x, scale),
363:                                    self.v + _mul2012(y, scale),
364:                                    _mul2012(a, scale), _mul2012(b, scale))
365:                                for x, y, a, b in font._vf[char].boxes])
366: 
367:     @dispatch(137, state=_dvistate.inpage, args=('s4', 's4'))
368:     def _put_rule(self, a, b):
369:         self._put_rule_real(a, b)
370: 
371:     def _put_rule_real(self, a, b):
372:         if a > 0 and b > 0:
373:             self.boxes.append(Box(self.h, self.v, a, b))
374: 
375:     @dispatch(138)
376:     def _nop(self, _):
377:         pass
378: 
379:     @dispatch(139, state=_dvistate.outer, args=('s4',)*11)
380:     def _bop(self, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, p):
381:         self.state = _dvistate.inpage
382:         self.h, self.v, self.w, self.x, self.y, self.z = 0, 0, 0, 0, 0, 0
383:         self.stack = []
384:         self.text = []          # list of Text objects
385:         self.boxes = []         # list of Box objects
386: 
387:     @dispatch(140, state=_dvistate.inpage)
388:     def _eop(self, _):
389:         self.state = _dvistate.outer
390:         del self.h, self.v, self.w, self.x, self.y, self.z, self.stack
391: 
392:     @dispatch(141, state=_dvistate.inpage)
393:     def _push(self, _):
394:         self.stack.append((self.h, self.v, self.w, self.x, self.y, self.z))
395: 
396:     @dispatch(142, state=_dvistate.inpage)
397:     def _pop(self, _):
398:         self.h, self.v, self.w, self.x, self.y, self.z = self.stack.pop()
399: 
400:     @dispatch(min=143, max=146, state=_dvistate.inpage, args=('slen1',))
401:     def _right(self, b):
402:         self.h += b
403: 
404:     @dispatch(min=147, max=151, state=_dvistate.inpage, args=('slen',))
405:     def _right_w(self, new_w):
406:         if new_w is not None:
407:             self.w = new_w
408:         self.h += self.w
409: 
410:     @dispatch(min=152, max=156, state=_dvistate.inpage, args=('slen',))
411:     def _right_x(self, new_x):
412:         if new_x is not None:
413:             self.x = new_x
414:         self.h += self.x
415: 
416:     @dispatch(min=157, max=160, state=_dvistate.inpage, args=('slen1',))
417:     def _down(self, a):
418:         self.v += a
419: 
420:     @dispatch(min=161, max=165, state=_dvistate.inpage, args=('slen',))
421:     def _down_y(self, new_y):
422:         if new_y is not None:
423:             self.y = new_y
424:         self.v += self.y
425: 
426:     @dispatch(min=166, max=170, state=_dvistate.inpage, args=('slen',))
427:     def _down_z(self, new_z):
428:         if new_z is not None:
429:             self.z = new_z
430:         self.v += self.z
431: 
432:     @dispatch(min=171, max=234, state=_dvistate.inpage)
433:     def _fnt_num_immediate(self, k):
434:         self.f = k
435: 
436:     @dispatch(min=235, max=238, state=_dvistate.inpage, args=('olen1',))
437:     def _fnt_num(self, new_f):
438:         self.f = new_f
439: 
440:     @dispatch(min=239, max=242, args=('ulen1',))
441:     def _xxx(self, datalen):
442:         special = self.file.read(datalen)
443:         if six.PY3:
444:             chr_ = chr
445:         else:
446:             def chr_(x):
447:                 return x
448:         matplotlib.verbose.report(
449:             'Dvi._xxx: encountered special: %s'
450:             % ''.join([(32 <= ord(ch) < 127) and chr_(ch)
451:                        or '<%02x>' % ord(ch)
452:                        for ch in special]),
453:             'debug')
454: 
455:     @dispatch(min=243, max=246, args=('olen1', 'u4', 'u4', 'u4', 'u1', 'u1'))
456:     def _fnt_def(self, k, c, s, d, a, l):
457:         self._fnt_def_real(k, c, s, d, a, l)
458: 
459:     def _fnt_def_real(self, k, c, s, d, a, l):
460:         n = self.file.read(a + l)
461:         fontname = n[-l:].decode('ascii')
462:         tfm = _tfmfile(fontname)
463:         if tfm is None:
464:             if six.PY2:
465:                 error_class = OSError
466:             else:
467:                 error_class = FileNotFoundError
468:             raise error_class("missing font metrics file: %s" % fontname)
469:         if c != 0 and tfm.checksum != 0 and c != tfm.checksum:
470:             raise ValueError('tfm checksum mismatch: %s' % n)
471: 
472:         vf = _vffile(fontname)
473: 
474:         self.fonts[k] = DviFont(scale=s, tfm=tfm, texname=n, vf=vf)
475: 
476:     @dispatch(247, state=_dvistate.pre, args=('u1', 'u4', 'u4', 'u4', 'u1'))
477:     def _pre(self, i, num, den, mag, k):
478:         comment = self.file.read(k)
479:         if i != 2:
480:             raise ValueError("Unknown dvi format %d" % i)
481:         if num != 25400000 or den != 7227 * 2**16:
482:             raise ValueError("nonstandard units in dvi file")
483:             # meaning: TeX always uses those exact values, so it
484:             # should be enough for us to support those
485:             # (There are 72.27 pt to an inch so 7227 pt =
486:             # 7227 * 2**16 sp to 100 in. The numerator is multiplied
487:             # by 10^5 to get units of 10**-7 meters.)
488:         if mag != 1000:
489:             raise ValueError("nonstandard magnification in dvi file")
490:             # meaning: LaTeX seems to frown on setting \mag, so
491:             # I think we can assume this is constant
492:         self.state = _dvistate.outer
493: 
494:     @dispatch(248, state=_dvistate.outer)
495:     def _post(self, _):
496:         self.state = _dvistate.post_post
497:         # TODO: actually read the postamble and finale?
498:         # currently post_post just triggers closing the file
499: 
500:     @dispatch(249)
501:     def _post_post(self, _):
502:         raise NotImplementedError
503: 
504:     @dispatch(min=250, max=255)
505:     def _malformed(self, offset):
506:         raise ValueError("unknown command: byte %d", 250 + offset)
507: 
508: 
509: class DviFont(object):
510:     '''
511:     Encapsulation of a font that a DVI file can refer to.
512: 
513:     This class holds a font's texname and size, supports comparison,
514:     and knows the widths of glyphs in the same units as the AFM file.
515:     There are also internal attributes (for use by dviread.py) that
516:     are *not* used for comparison.
517: 
518:     The size is in Adobe points (converted from TeX points).
519: 
520:     Parameters
521:     ----------
522: 
523:     scale : float
524:         Factor by which the font is scaled from its natural size.
525:     tfm : Tfm
526:         TeX font metrics for this font
527:     texname : bytes
528:        Name of the font as used internally by TeX and friends, as an
529:        ASCII bytestring. This is usually very different from any external
530:        font names, and :class:`dviread.PsfontsMap` can be used to find
531:        the external name of the font.
532:     vf : Vf
533:        A TeX "virtual font" file, or None if this font is not virtual.
534: 
535:     Attributes
536:     ----------
537: 
538:     texname : bytes
539:     size : float
540:        Size of the font in Adobe points, converted from the slightly
541:        smaller TeX points.
542:     widths : list
543:        Widths of glyphs in glyph-space units, typically 1/1000ths of
544:        the point size.
545: 
546:     '''
547:     __slots__ = ('texname', 'size', 'widths', '_scale', '_vf', '_tfm')
548: 
549:     def __init__(self, scale, tfm, texname, vf):
550:         if not isinstance(texname, bytes):
551:             raise ValueError("texname must be a bytestring, got %s"
552:                              % type(texname))
553:         self._scale, self._tfm, self.texname, self._vf = \
554:             scale, tfm, texname, vf
555:         self.size = scale * (72.0 / (72.27 * 2**16))
556:         try:
557:             nchars = max(tfm.width) + 1
558:         except ValueError:
559:             nchars = 0
560:         self.widths = [(1000*tfm.width.get(char, 0)) >> 20
561:                        for char in xrange(nchars)]
562: 
563:     def __eq__(self, other):
564:         return self.__class__ == other.__class__ and \
565:             self.texname == other.texname and self.size == other.size
566: 
567:     def __ne__(self, other):
568:         return not self.__eq__(other)
569: 
570:     def _width_of(self, char):
571:         '''
572:         Width of char in dvi units. For internal use by dviread.py.
573:         '''
574: 
575:         width = self._tfm.width.get(char, None)
576:         if width is not None:
577:             return _mul2012(width, self._scale)
578: 
579:         matplotlib.verbose.report(
580:             'No width for char %d in font %s' % (char, self.texname),
581:             'debug')
582:         return 0
583: 
584:     def _height_depth_of(self, char):
585:         '''
586:         Height and depth of char in dvi units. For internal use by dviread.py.
587:         '''
588: 
589:         result = []
590:         for metric, name in ((self._tfm.height, "height"),
591:                              (self._tfm.depth, "depth")):
592:             value = metric.get(char, None)
593:             if value is None:
594:                 matplotlib.verbose.report(
595:                     'No %s for char %d in font %s' % (
596:                         name, char, self.texname),
597:                     'debug')
598:                 result.append(0)
599:             else:
600:                 result.append(_mul2012(value, self._scale))
601:         return result
602: 
603: 
604: class Vf(Dvi):
605:     '''
606:     A virtual font (\\*.vf file) containing subroutines for dvi files.
607: 
608:     Usage::
609: 
610:       vf = Vf(filename)
611:       glyph = vf[code]
612:       glyph.text, glyph.boxes, glyph.width
613: 
614:     Parameters
615:     ----------
616: 
617:     filename : string or bytestring
618: 
619:     Notes
620:     -----
621: 
622:     The virtual font format is a derivative of dvi:
623:     http://mirrors.ctan.org/info/knuth/virtual-fonts
624:     This class reuses some of the machinery of `Dvi`
625:     but replaces the `_read` loop and dispatch mechanism.
626:     '''
627: 
628:     def __init__(self, filename):
629:         Dvi.__init__(self, filename, 0)
630:         try:
631:             self._first_font = None
632:             self._chars = {}
633:             self._read()
634:         finally:
635:             self.close()
636: 
637:     def __getitem__(self, code):
638:         return self._chars[code]
639: 
640:     def _read(self):
641:         '''
642:         Read one page from the file. Return True if successful,
643:         False if there were no more pages.
644:         '''
645:         packet_len, packet_char, packet_width = None, None, None
646:         while True:
647:             byte = ord(self.file.read(1)[0])
648:             # If we are in a packet, execute the dvi instructions
649:             if self.state == _dvistate.inpage:
650:                 byte_at = self.file.tell()-1
651:                 if byte_at == packet_ends:
652:                     self._finalize_packet(packet_char, packet_width)
653:                     packet_len, packet_char, packet_width = None, None, None
654:                     # fall through to out-of-packet code
655:                 elif byte_at > packet_ends:
656:                     raise ValueError("Packet length mismatch in vf file")
657:                 else:
658:                     if byte in (139, 140) or byte >= 243:
659:                         raise ValueError(
660:                             "Inappropriate opcode %d in vf file" % byte)
661:                     Dvi._dtable[byte](self, byte)
662:                     continue
663: 
664:             # We are outside a packet
665:             if byte < 242:          # a short packet (length given by byte)
666:                 packet_len = byte
667:                 packet_char, packet_width = self._arg(1), self._arg(3)
668:                 packet_ends = self._init_packet(byte)
669:                 self.state = _dvistate.inpage
670:             elif byte == 242:       # a long packet
671:                 packet_len, packet_char, packet_width = \
672:                             [self._arg(x) for x in (4, 4, 4)]
673:                 self._init_packet(packet_len)
674:             elif 243 <= byte <= 246:
675:                 k = self._arg(byte - 242, byte == 246)
676:                 c, s, d, a, l = [self._arg(x) for x in (4, 4, 4, 1, 1)]
677:                 self._fnt_def_real(k, c, s, d, a, l)
678:                 if self._first_font is None:
679:                     self._first_font = k
680:             elif byte == 247:       # preamble
681:                 i, k = self._arg(1), self._arg(1)
682:                 x = self.file.read(k)
683:                 cs, ds = self._arg(4), self._arg(4)
684:                 self._pre(i, x, cs, ds)
685:             elif byte == 248:       # postamble (just some number of 248s)
686:                 break
687:             else:
688:                 raise ValueError("unknown vf opcode %d" % byte)
689: 
690:     def _init_packet(self, pl):
691:         if self.state != _dvistate.outer:
692:             raise ValueError("Misplaced packet in vf file")
693:         self.h, self.v, self.w, self.x, self.y, self.z = 0, 0, 0, 0, 0, 0
694:         self.stack, self.text, self.boxes = [], [], []
695:         self.f = self._first_font
696:         return self.file.tell() + pl
697: 
698:     def _finalize_packet(self, packet_char, packet_width):
699:         self._chars[packet_char] = Page(
700:             text=self.text, boxes=self.boxes, width=packet_width,
701:             height=None, descent=None)
702:         self.state = _dvistate.outer
703: 
704:     def _pre(self, i, x, cs, ds):
705:         if self.state != _dvistate.pre:
706:             raise ValueError("pre command in middle of vf file")
707:         if i != 202:
708:             raise ValueError("Unknown vf format %d" % i)
709:         if len(x):
710:             matplotlib.verbose.report('vf file comment: ' + x, 'debug')
711:         self.state = _dvistate.outer
712:         # cs = checksum, ds = design size
713: 
714: 
715: def _fix2comp(num):
716:     '''
717:     Convert from two's complement to negative.
718:     '''
719:     assert 0 <= num < 2**32
720:     if num & 2**31:
721:         return num - 2**32
722:     else:
723:         return num
724: 
725: 
726: def _mul2012(num1, num2):
727:     '''
728:     Multiply two numbers in 20.12 fixed point format.
729:     '''
730:     # Separated into a function because >> has surprising precedence
731:     return (num1*num2) >> 20
732: 
733: 
734: class Tfm(object):
735:     '''
736:     A TeX Font Metric file.
737: 
738:     This implementation covers only the bare minimum needed by the Dvi class.
739: 
740:     Parameters
741:     ----------
742:     filename : string or bytestring
743: 
744:     Attributes
745:     ----------
746:     checksum : int
747:        Used for verifying against the dvi file.
748:     design_size : int
749:        Design size of the font (unknown units)
750:     width, height, depth : dict
751:        Dimensions of each character, need to be scaled by the factor
752:        specified in the dvi file. These are dicts because indexing may
753:        not start from 0.
754:     '''
755:     __slots__ = ('checksum', 'design_size', 'width', 'height', 'depth')
756: 
757:     def __init__(self, filename):
758:         matplotlib.verbose.report('opening tfm file ' + filename, 'debug')
759:         with open(filename, 'rb') as file:
760:             header1 = file.read(24)
761:             lh, bc, ec, nw, nh, nd = \
762:                 struct.unpack(str('!6H'), header1[2:14])
763:             matplotlib.verbose.report(
764:                 'lh=%d, bc=%d, ec=%d, nw=%d, nh=%d, nd=%d' % (
765:                     lh, bc, ec, nw, nh, nd), 'debug')
766:             header2 = file.read(4*lh)
767:             self.checksum, self.design_size = \
768:                 struct.unpack(str('!2I'), header2[:8])
769:             # there is also encoding information etc.
770:             char_info = file.read(4*(ec-bc+1))
771:             widths = file.read(4*nw)
772:             heights = file.read(4*nh)
773:             depths = file.read(4*nd)
774: 
775:         self.width, self.height, self.depth = {}, {}, {}
776:         widths, heights, depths = \
777:             [struct.unpack(str('!%dI') % (len(x)/4), x)
778:              for x in (widths, heights, depths)]
779:         for idx, char in enumerate(xrange(bc, ec+1)):
780:             byte0 = ord(char_info[4*idx])
781:             byte1 = ord(char_info[4*idx+1])
782:             self.width[char] = _fix2comp(widths[byte0])
783:             self.height[char] = _fix2comp(heights[byte1 >> 4])
784:             self.depth[char] = _fix2comp(depths[byte1 & 0xf])
785: 
786: 
787: PsFont = namedtuple('Font', 'texname psname effects encoding filename')
788: 
789: 
790: class PsfontsMap(object):
791:     '''
792:     A psfonts.map formatted file, mapping TeX fonts to PS fonts.
793: 
794:     Usage::
795: 
796:      >>> map = PsfontsMap(find_tex_file('pdftex.map'))
797:      >>> entry = map[b'ptmbo8r']
798:      >>> entry.texname
799:      b'ptmbo8r'
800:      >>> entry.psname
801:      b'Times-Bold'
802:      >>> entry.encoding
803:      '/usr/local/texlive/2008/texmf-dist/fonts/enc/dvips/base/8r.enc'
804:      >>> entry.effects
805:      {'slant': 0.16700000000000001}
806:      >>> entry.filename
807: 
808:     Parameters
809:     ----------
810: 
811:     filename : string or bytestring
812: 
813:     Notes
814:     -----
815: 
816:     For historical reasons, TeX knows many Type-1 fonts by different
817:     names than the outside world. (For one thing, the names have to
818:     fit in eight characters.) Also, TeX's native fonts are not Type-1
819:     but Metafont, which is nontrivial to convert to PostScript except
820:     as a bitmap. While high-quality conversions to Type-1 format exist
821:     and are shipped with modern TeX distributions, we need to know
822:     which Type-1 fonts are the counterparts of which native fonts. For
823:     these reasons a mapping is needed from internal font names to font
824:     file names.
825: 
826:     A texmf tree typically includes mapping files called e.g.
827:     :file:`psfonts.map`, :file:`pdftex.map`, or :file:`dvipdfm.map`.
828:     The file :file:`psfonts.map` is used by :program:`dvips`,
829:     :file:`pdftex.map` by :program:`pdfTeX`, and :file:`dvipdfm.map`
830:     by :program:`dvipdfm`. :file:`psfonts.map` might avoid embedding
831:     the 35 PostScript fonts (i.e., have no filename for them, as in
832:     the Times-Bold example above), while the pdf-related files perhaps
833:     only avoid the "Base 14" pdf fonts. But the user may have
834:     configured these files differently.
835:     '''
836:     __slots__ = ('_font', '_filename')
837: 
838:     def __init__(self, filename):
839:         self._font = {}
840:         self._filename = filename
841:         if six.PY3 and isinstance(filename, bytes):
842:             encoding = sys.getfilesystemencoding() or 'utf-8'
843:             self._filename = filename.decode(encoding, errors='replace')
844:         with open(filename, 'rb') as file:
845:             self._parse(file)
846: 
847:     def __getitem__(self, texname):
848:         assert isinstance(texname, bytes)
849:         try:
850:             result = self._font[texname]
851:         except KeyError:
852:             fmt = ('A PostScript file for the font whose TeX name is "{0}" '
853:                    'could not be found in the file "{1}". The dviread module '
854:                    'can only handle fonts that have an associated PostScript '
855:                    'font file. '
856:                    'This problem can often be solved by installing '
857:                    'a suitable PostScript font package in your (TeX) '
858:                    'package manager.')
859:             msg = fmt.format(texname.decode('ascii'), self._filename)
860:             msg = textwrap.fill(msg, break_on_hyphens=False,
861:                                 break_long_words=False)
862:             matplotlib.verbose.report(msg, 'helpful')
863:             raise
864:         fn, enc = result.filename, result.encoding
865:         if fn is not None and not fn.startswith(b'/'):
866:             fn = find_tex_file(fn)
867:         if enc is not None and not enc.startswith(b'/'):
868:             enc = find_tex_file(result.encoding)
869:         return result._replace(filename=fn, encoding=enc)
870: 
871:     def _parse(self, file):
872:         '''
873:         Parse the font mapping file.
874: 
875:         The format is, AFAIK: texname fontname [effects and filenames]
876:         Effects are PostScript snippets like ".177 SlantFont",
877:         filenames begin with one or two less-than signs. A filename
878:         ending in enc is an encoding file, other filenames are font
879:         files. This can be overridden with a left bracket: <[foobar
880:         indicates an encoding file named foobar.
881: 
882:         There is some difference between <foo.pfb and <<bar.pfb in
883:         subsetting, but I have no example of << in my TeX installation.
884:         '''
885:         # If the map file specifies multiple encodings for a font, we
886:         # follow pdfTeX in choosing the last one specified. Such
887:         # entries are probably mistakes but they have occurred.
888:         # http://tex.stackexchange.com/questions/10826/
889:         # http://article.gmane.org/gmane.comp.tex.pdftex/4914
890: 
891:         empty_re = re.compile(br'%|\s*$')
892:         word_re = re.compile(
893:             br'''(?x) (?:
894:                  "<\[ (?P<enc1>  [^"]+    )" | # quoted encoding marked by [
895:                  "<   (?P<enc2>  [^"]+.enc)" | # quoted encoding, ends in .enc
896:                  "<<? (?P<file1> [^"]+    )" | # quoted font file name
897:                  "    (?P<eff1>  [^"]+    )" | # quoted effects or font name
898:                  <\[  (?P<enc3>  \S+      )  | # encoding marked by [
899:                  <    (?P<enc4>  \S+  .enc)  | # encoding, ends in .enc
900:                  <<?  (?P<file2> \S+      )  | # font file name
901:                       (?P<eff2>  \S+      )    # effects or font name
902:             )''')
903:         effects_re = re.compile(
904:             br'''(?x) (?P<slant> -?[0-9]*(?:\.[0-9]+)) \s* SlantFont
905:                     | (?P<extend>-?[0-9]*(?:\.[0-9]+)) \s* ExtendFont''')
906: 
907:         lines = (line.strip()
908:                  for line in file
909:                  if not empty_re.match(line))
910:         for line in lines:
911:             effects, encoding, filename = b'', None, None
912:             words = word_re.finditer(line)
913: 
914:             # The named groups are mutually exclusive and are
915:             # referenced below at an estimated order of probability of
916:             # occurrence based on looking at my copy of pdftex.map.
917:             # The font names are probably unquoted:
918:             w = next(words)
919:             texname = w.group('eff2') or w.group('eff1')
920:             w = next(words)
921:             psname = w.group('eff2') or w.group('eff1')
922: 
923:             for w in words:
924:                 # Any effects are almost always quoted:
925:                 eff = w.group('eff1') or w.group('eff2')
926:                 if eff:
927:                     effects = eff
928:                     continue
929:                 # Encoding files usually have the .enc suffix
930:                 # and almost never need quoting:
931:                 enc = (w.group('enc4') or w.group('enc3') or
932:                        w.group('enc2') or w.group('enc1'))
933:                 if enc:
934:                     if encoding is not None:
935:                         matplotlib.verbose.report(
936:                             'Multiple encodings for %s = %s'
937:                             % (texname, psname),
938:                             'debug')
939:                     encoding = enc
940:                     continue
941:                 # File names are probably unquoted:
942:                 filename = w.group('file2') or w.group('file1')
943: 
944:             effects_dict = {}
945:             for match in effects_re.finditer(effects):
946:                 slant = match.group('slant')
947:                 if slant:
948:                     effects_dict['slant'] = float(slant)
949:                 else:
950:                     effects_dict['extend'] = float(match.group('extend'))
951: 
952:             self._font[texname] = PsFont(
953:                 texname=texname, psname=psname, effects=effects_dict,
954:                 encoding=encoding, filename=filename)
955: 
956: 
957: class Encoding(object):
958:     '''
959:     Parses a \\*.enc file referenced from a psfonts.map style file.
960:     The format this class understands is a very limited subset of
961:     PostScript.
962: 
963:     Usage (subject to change)::
964: 
965:       for name in Encoding(filename):
966:           whatever(name)
967: 
968:     Parameters
969:     ----------
970:     filename : string or bytestring
971: 
972:     Attributes
973:     ----------
974:     encoding : list
975:         List of character names
976:     '''
977:     __slots__ = ('encoding',)
978: 
979:     def __init__(self, filename):
980:         with open(filename, 'rb') as file:
981:             matplotlib.verbose.report('Parsing TeX encoding ' + filename,
982:                                       'debug-annoying')
983:             self.encoding = self._parse(file)
984:             matplotlib.verbose.report('Result: ' + repr(self.encoding),
985:                                       'debug-annoying')
986: 
987:     def __iter__(self):
988:         for name in self.encoding:
989:             yield name
990: 
991:     def _parse(self, file):
992:         result = []
993: 
994:         lines = (line.split(b'%', 1)[0].strip() for line in file)
995:         data = b''.join(lines)
996:         beginning = data.find(b'[')
997:         if beginning < 0:
998:             raise ValueError("Cannot locate beginning of encoding in {}"
999:                              .format(file))
1000:         data = data[beginning:]
1001:         end = data.find(b']')
1002:         if end < 0:
1003:             raise ValueError("Cannot locate end of encoding in {}"
1004:                              .format(file))
1005:         data = data[:end]
1006: 
1007:         return re.findall(br'/([^][{}<>\s]+)', data)
1008: 
1009: 
1010: def find_tex_file(filename, format=None):
1011:     '''
1012:     Find a file in the texmf tree.
1013: 
1014:     Calls :program:`kpsewhich` which is an interface to the kpathsea
1015:     library [1]_. Most existing TeX distributions on Unix-like systems use
1016:     kpathsea. It is also available as part of MikTeX, a popular
1017:     distribution on Windows.
1018: 
1019:     Parameters
1020:     ----------
1021:     filename : string or bytestring
1022:     format : string or bytestring
1023:         Used as the value of the `--format` option to :program:`kpsewhich`.
1024:         Could be e.g. 'tfm' or 'vf' to limit the search to that type of files.
1025: 
1026:     References
1027:     ----------
1028: 
1029:     .. [1] `Kpathsea documentation <http://www.tug.org/kpathsea/>`_
1030:         The library that :program:`kpsewhich` is part of.
1031:     '''
1032: 
1033:     cmd = [str('kpsewhich')]
1034:     if format is not None:
1035:         cmd += ['--format=' + format]
1036:     cmd += [filename]
1037: 
1038:     matplotlib.verbose.report('find_tex_file(%s): %s'
1039:                               % (filename, cmd), 'debug')
1040:     # stderr is unused, but reading it avoids a subprocess optimization
1041:     # that breaks EINTR handling in some Python versions:
1042:     # http://bugs.python.org/issue12493
1043:     # https://github.com/matplotlib/matplotlib/issues/633
1044:     pipe = subprocess.Popen(cmd, stdout=subprocess.PIPE,
1045:                             stderr=subprocess.PIPE)
1046:     result = pipe.communicate()[0].rstrip()
1047:     matplotlib.verbose.report('find_tex_file result: %s' % result,
1048:                               'debug')
1049:     return result.decode('ascii')
1050: 
1051: # With multiple text objects per figure (e.g., tick labels) we may end
1052: # up reading the same tfm and vf files many times, so we implement a
1053: # simple cache. TODO: is this worth making persistent?
1054: 
1055: _tfmcache = {}
1056: _vfcache = {}
1057: 
1058: 
1059: def _fontfile(texname, class_, suffix, cache):
1060:     try:
1061:         return cache[texname]
1062:     except KeyError:
1063:         pass
1064: 
1065:     filename = find_tex_file(texname + suffix)
1066:     if filename:
1067:         result = class_(filename)
1068:     else:
1069:         result = None
1070: 
1071:     cache[texname] = result
1072:     return result
1073: 
1074: 
1075: def _tfmfile(texname):
1076:     return _fontfile(texname, Tfm, '.tfm', _tfmcache)
1077: 
1078: 
1079: def _vffile(texname):
1080:     return _fontfile(texname, Vf, '.vf', _vfcache)
1081: 
1082: 
1083: if __name__ == '__main__':
1084:     import sys
1085:     matplotlib.verbose.set_level('debug-annoying')
1086:     fname = sys.argv[1]
1087:     try:
1088:         dpi = float(sys.argv[2])
1089:     except IndexError:
1090:         dpi = None
1091:     with Dvi(fname, dpi) as dvi:
1092:         fontmap = PsfontsMap(find_tex_file('pdftex.map'))
1093:         for page in dvi:
1094:             print('=== new page ===')
1095:             fPrev = None
1096:             for x, y, f, c, w in page.text:
1097:                 if f != fPrev:
1098:                     print('font', f.texname, 'scaled', f._scale/pow(2.0, 20))
1099:                     fPrev = f
1100:                 print(x, y, c, 32 <= c < 128 and chr(c) or '.', w)
1101:             for x, y, w, h in page.boxes:
1102:                 print(x, y, 'BOX', w, h)
1103: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

unicode_47845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, (-1)), 'unicode', u'\nA module for reading dvi files output by TeX. Several limitations make\nthis not (currently) useful as a general-purpose dvi preprocessor, but\nit is currently used by the pdf backend for processing usetex text.\n\nInterface::\n\n  with Dvi(filename, 72) as dvi:\n      # iterate over pages:\n      for page in dvi:\n          w, h, d = page.width, page.height, page.descent\n          for x,y,font,glyph,width in page.text:\n              fontname = font.texname\n              pointsize = font.size\n              ...\n          for x,y,height,width in page.boxes:\n              ...\n\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 23, 0))

# 'import six' statement (line 23)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_47846 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 23, 0), 'six')

if (type(import_47846) is not StypyTypeError):

    if (import_47846 != 'pyd_module'):
        __import__(import_47846)
        sys_modules_47847 = sys.modules[import_47846]
        import_module(stypy.reporting.localization.Localization(__file__, 23, 0), 'six', sys_modules_47847.module_type_store, module_type_store)
    else:
        import six

        import_module(stypy.reporting.localization.Localization(__file__, 23, 0), 'six', six, module_type_store)

else:
    # Assigning a type to the variable 'six' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'six', import_47846)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 24, 0))

# 'from six.moves import xrange' statement (line 24)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_47848 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'six.moves')

if (type(import_47848) is not StypyTypeError):

    if (import_47848 != 'pyd_module'):
        __import__(import_47848)
        sys_modules_47849 = sys.modules[import_47848]
        import_from_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'six.moves', sys_modules_47849.module_type_store, module_type_store, ['xrange'])
        nest_module(stypy.reporting.localization.Localization(__file__, 24, 0), __file__, sys_modules_47849, sys_modules_47849.module_type_store, module_type_store)
    else:
        from six.moves import xrange

        import_from_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'six.moves', None, module_type_store, ['xrange'], [xrange])

else:
    # Assigning a type to the variable 'six.moves' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'six.moves', import_47848)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 26, 0))

# 'from collections import namedtuple' statement (line 26)
try:
    from collections import namedtuple

except:
    namedtuple = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'collections', None, module_type_store, ['namedtuple'], [namedtuple])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 27, 0))

# 'import errno' statement (line 27)
import errno

import_module(stypy.reporting.localization.Localization(__file__, 27, 0), 'errno', errno, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 28, 0))

# 'from functools import partial, wraps' statement (line 28)
try:
    from functools import partial, wraps

except:
    partial = UndefinedType
    wraps = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 28, 0), 'functools', None, module_type_store, ['partial', 'wraps'], [partial, wraps])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 29, 0))

# 'import matplotlib' statement (line 29)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_47850 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 29, 0), 'matplotlib')

if (type(import_47850) is not StypyTypeError):

    if (import_47850 != 'pyd_module'):
        __import__(import_47850)
        sys_modules_47851 = sys.modules[import_47850]
        import_module(stypy.reporting.localization.Localization(__file__, 29, 0), 'matplotlib', sys_modules_47851.module_type_store, module_type_store)
    else:
        import matplotlib

        import_module(stypy.reporting.localization.Localization(__file__, 29, 0), 'matplotlib', matplotlib, module_type_store)

else:
    # Assigning a type to the variable 'matplotlib' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'matplotlib', import_47850)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 30, 0))

# 'import matplotlib.cbook' statement (line 30)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_47852 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 30, 0), 'matplotlib.cbook')

if (type(import_47852) is not StypyTypeError):

    if (import_47852 != 'pyd_module'):
        __import__(import_47852)
        sys_modules_47853 = sys.modules[import_47852]
        import_module(stypy.reporting.localization.Localization(__file__, 30, 0), 'mpl_cbook', sys_modules_47853.module_type_store, module_type_store)
    else:
        import matplotlib.cbook as mpl_cbook

        import_module(stypy.reporting.localization.Localization(__file__, 30, 0), 'mpl_cbook', matplotlib.cbook, module_type_store)

else:
    # Assigning a type to the variable 'matplotlib.cbook' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 0), 'matplotlib.cbook', import_47852)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 31, 0))

# 'from matplotlib.compat import subprocess' statement (line 31)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_47854 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 31, 0), 'matplotlib.compat')

if (type(import_47854) is not StypyTypeError):

    if (import_47854 != 'pyd_module'):
        __import__(import_47854)
        sys_modules_47855 = sys.modules[import_47854]
        import_from_module(stypy.reporting.localization.Localization(__file__, 31, 0), 'matplotlib.compat', sys_modules_47855.module_type_store, module_type_store, ['subprocess'])
        nest_module(stypy.reporting.localization.Localization(__file__, 31, 0), __file__, sys_modules_47855, sys_modules_47855.module_type_store, module_type_store)
    else:
        from matplotlib.compat import subprocess

        import_from_module(stypy.reporting.localization.Localization(__file__, 31, 0), 'matplotlib.compat', None, module_type_store, ['subprocess'], [subprocess])

else:
    # Assigning a type to the variable 'matplotlib.compat' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 0), 'matplotlib.compat', import_47854)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 32, 0))

# 'from matplotlib import rcParams' statement (line 32)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_47856 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 32, 0), 'matplotlib')

if (type(import_47856) is not StypyTypeError):

    if (import_47856 != 'pyd_module'):
        __import__(import_47856)
        sys_modules_47857 = sys.modules[import_47856]
        import_from_module(stypy.reporting.localization.Localization(__file__, 32, 0), 'matplotlib', sys_modules_47857.module_type_store, module_type_store, ['rcParams'])
        nest_module(stypy.reporting.localization.Localization(__file__, 32, 0), __file__, sys_modules_47857, sys_modules_47857.module_type_store, module_type_store)
    else:
        from matplotlib import rcParams

        import_from_module(stypy.reporting.localization.Localization(__file__, 32, 0), 'matplotlib', None, module_type_store, ['rcParams'], [rcParams])

else:
    # Assigning a type to the variable 'matplotlib' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 0), 'matplotlib', import_47856)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 33, 0))

# 'import numpy' statement (line 33)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_47858 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 33, 0), 'numpy')

if (type(import_47858) is not StypyTypeError):

    if (import_47858 != 'pyd_module'):
        __import__(import_47858)
        sys_modules_47859 = sys.modules[import_47858]
        import_module(stypy.reporting.localization.Localization(__file__, 33, 0), 'np', sys_modules_47859.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 33, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 0), 'numpy', import_47858)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 34, 0))

# 'import re' statement (line 34)
import re

import_module(stypy.reporting.localization.Localization(__file__, 34, 0), 're', re, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 35, 0))

# 'import struct' statement (line 35)
import struct

import_module(stypy.reporting.localization.Localization(__file__, 35, 0), 'struct', struct, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 36, 0))

# 'import sys' statement (line 36)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 36, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 37, 0))

# 'import textwrap' statement (line 37)
import textwrap

import_module(stypy.reporting.localization.Localization(__file__, 37, 0), 'textwrap', textwrap, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 38, 0))

# 'import os' statement (line 38)
import os

import_module(stypy.reporting.localization.Localization(__file__, 38, 0), 'os', os, module_type_store)


# Getting the type of 'six' (line 40)
six_47860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 3), 'six')
# Obtaining the member 'PY3' of a type (line 40)
PY3_47861 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 3), six_47860, 'PY3')
# Testing the type of an if condition (line 40)
if_condition_47862 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 40, 0), PY3_47861)
# Assigning a type to the variable 'if_condition_47862' (line 40)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 0), 'if_condition_47862', if_condition_47862)
# SSA begins for if statement (line 40)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

@norecursion
def ord(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'ord'
    module_type_store = module_type_store.open_function_context('ord', 41, 4, False)
    
    # Passed parameters checking function
    ord.stypy_localization = localization
    ord.stypy_type_of_self = None
    ord.stypy_type_store = module_type_store
    ord.stypy_function_name = 'ord'
    ord.stypy_param_names_list = ['x']
    ord.stypy_varargs_param_name = None
    ord.stypy_kwargs_param_name = None
    ord.stypy_call_defaults = defaults
    ord.stypy_call_varargs = varargs
    ord.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'ord', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'ord', localization, ['x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'ord(...)' code ##################

    # Getting the type of 'x' (line 42)
    x_47863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 15), 'x')
    # Assigning a type to the variable 'stypy_return_type' (line 42)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'stypy_return_type', x_47863)
    
    # ################# End of 'ord(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'ord' in the type store
    # Getting the type of 'stypy_return_type' (line 41)
    stypy_return_type_47864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_47864)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'ord'
    return stypy_return_type_47864

# Assigning a type to the variable 'ord' (line 41)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'ord', ord)
# SSA join for if statement (line 40)
module_type_store = module_type_store.join_ssa_context()


# Assigning a Call to a Name (line 60):

# Assigning a Call to a Name (line 60):

# Call to Bunch(...): (line 60)
# Processing the call keyword arguments (line 60)
int_47867 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 32), 'int')
keyword_47868 = int_47867
int_47869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 41), 'int')
keyword_47870 = int_47869
int_47871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 51), 'int')
keyword_47872 = int_47871
int_47873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 64), 'int')
keyword_47874 = int_47873
int_47875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 74), 'int')
keyword_47876 = int_47875
kwargs_47877 = {'pre': keyword_47868, 'inpage': keyword_47872, 'finale': keyword_47876, 'outer': keyword_47870, 'post_post': keyword_47874}
# Getting the type of 'mpl_cbook' (line 60)
mpl_cbook_47865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 12), 'mpl_cbook', False)
# Obtaining the member 'Bunch' of a type (line 60)
Bunch_47866 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 12), mpl_cbook_47865, 'Bunch')
# Calling Bunch(args, kwargs) (line 60)
Bunch_call_result_47878 = invoke(stypy.reporting.localization.Localization(__file__, 60, 12), Bunch_47866, *[], **kwargs_47877)

# Assigning a type to the variable '_dvistate' (line 60)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 0), '_dvistate', Bunch_call_result_47878)

# Assigning a Call to a Name (line 63):

# Assigning a Call to a Name (line 63):

# Call to namedtuple(...): (line 63)
# Processing the call arguments (line 63)
unicode_47880 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 18), 'unicode', u'Page')
unicode_47881 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 26), 'unicode', u'text boxes height width descent')
# Processing the call keyword arguments (line 63)
kwargs_47882 = {}
# Getting the type of 'namedtuple' (line 63)
namedtuple_47879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 7), 'namedtuple', False)
# Calling namedtuple(args, kwargs) (line 63)
namedtuple_call_result_47883 = invoke(stypy.reporting.localization.Localization(__file__, 63, 7), namedtuple_47879, *[unicode_47880, unicode_47881], **kwargs_47882)

# Assigning a type to the variable 'Page' (line 63)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 0), 'Page', namedtuple_call_result_47883)

# Assigning a Call to a Name (line 64):

# Assigning a Call to a Name (line 64):

# Call to namedtuple(...): (line 64)
# Processing the call arguments (line 64)
unicode_47885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 18), 'unicode', u'Text')
unicode_47886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 26), 'unicode', u'x y font glyph width')
# Processing the call keyword arguments (line 64)
kwargs_47887 = {}
# Getting the type of 'namedtuple' (line 64)
namedtuple_47884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 7), 'namedtuple', False)
# Calling namedtuple(args, kwargs) (line 64)
namedtuple_call_result_47888 = invoke(stypy.reporting.localization.Localization(__file__, 64, 7), namedtuple_47884, *[unicode_47885, unicode_47886], **kwargs_47887)

# Assigning a type to the variable 'Text' (line 64)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 0), 'Text', namedtuple_call_result_47888)

# Assigning a Call to a Name (line 65):

# Assigning a Call to a Name (line 65):

# Call to namedtuple(...): (line 65)
# Processing the call arguments (line 65)
unicode_47890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 17), 'unicode', u'Box')
unicode_47891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 24), 'unicode', u'x y height width')
# Processing the call keyword arguments (line 65)
kwargs_47892 = {}
# Getting the type of 'namedtuple' (line 65)
namedtuple_47889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 6), 'namedtuple', False)
# Calling namedtuple(args, kwargs) (line 65)
namedtuple_call_result_47893 = invoke(stypy.reporting.localization.Localization(__file__, 65, 6), namedtuple_47889, *[unicode_47890, unicode_47891], **kwargs_47892)

# Assigning a type to the variable 'Box' (line 65)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 0), 'Box', namedtuple_call_result_47893)

@norecursion
def _arg_raw(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_arg_raw'
    module_type_store = module_type_store.open_function_context('_arg_raw', 75, 0, False)
    
    # Passed parameters checking function
    _arg_raw.stypy_localization = localization
    _arg_raw.stypy_type_of_self = None
    _arg_raw.stypy_type_store = module_type_store
    _arg_raw.stypy_function_name = '_arg_raw'
    _arg_raw.stypy_param_names_list = ['dvi', 'delta']
    _arg_raw.stypy_varargs_param_name = None
    _arg_raw.stypy_kwargs_param_name = None
    _arg_raw.stypy_call_defaults = defaults
    _arg_raw.stypy_call_varargs = varargs
    _arg_raw.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_arg_raw', ['dvi', 'delta'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_arg_raw', localization, ['dvi', 'delta'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_arg_raw(...)' code ##################

    unicode_47894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 4), 'unicode', u'Return *delta* without reading anything more from the dvi file')
    # Getting the type of 'delta' (line 77)
    delta_47895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 11), 'delta')
    # Assigning a type to the variable 'stypy_return_type' (line 77)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 4), 'stypy_return_type', delta_47895)
    
    # ################# End of '_arg_raw(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_arg_raw' in the type store
    # Getting the type of 'stypy_return_type' (line 75)
    stypy_return_type_47896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_47896)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_arg_raw'
    return stypy_return_type_47896

# Assigning a type to the variable '_arg_raw' (line 75)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 0), '_arg_raw', _arg_raw)

@norecursion
def _arg(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_arg'
    module_type_store = module_type_store.open_function_context('_arg', 80, 0, False)
    
    # Passed parameters checking function
    _arg.stypy_localization = localization
    _arg.stypy_type_of_self = None
    _arg.stypy_type_store = module_type_store
    _arg.stypy_function_name = '_arg'
    _arg.stypy_param_names_list = ['bytes', 'signed', 'dvi', '_']
    _arg.stypy_varargs_param_name = None
    _arg.stypy_kwargs_param_name = None
    _arg.stypy_call_defaults = defaults
    _arg.stypy_call_varargs = varargs
    _arg.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_arg', ['bytes', 'signed', 'dvi', '_'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_arg', localization, ['bytes', 'signed', 'dvi', '_'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_arg(...)' code ##################

    unicode_47897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, (-1)), 'unicode', u'Read *bytes* bytes, returning the bytes interpreted as a\n    signed integer if *signed* is true, unsigned otherwise.')
    
    # Call to _arg(...): (line 83)
    # Processing the call arguments (line 83)
    # Getting the type of 'bytes' (line 83)
    bytes_47900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 20), 'bytes', False)
    # Getting the type of 'signed' (line 83)
    signed_47901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 27), 'signed', False)
    # Processing the call keyword arguments (line 83)
    kwargs_47902 = {}
    # Getting the type of 'dvi' (line 83)
    dvi_47898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 11), 'dvi', False)
    # Obtaining the member '_arg' of a type (line 83)
    _arg_47899 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 11), dvi_47898, '_arg')
    # Calling _arg(args, kwargs) (line 83)
    _arg_call_result_47903 = invoke(stypy.reporting.localization.Localization(__file__, 83, 11), _arg_47899, *[bytes_47900, signed_47901], **kwargs_47902)
    
    # Assigning a type to the variable 'stypy_return_type' (line 83)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 4), 'stypy_return_type', _arg_call_result_47903)
    
    # ################# End of '_arg(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_arg' in the type store
    # Getting the type of 'stypy_return_type' (line 80)
    stypy_return_type_47904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_47904)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_arg'
    return stypy_return_type_47904

# Assigning a type to the variable '_arg' (line 80)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 0), '_arg', _arg)

@norecursion
def _arg_slen(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_arg_slen'
    module_type_store = module_type_store.open_function_context('_arg_slen', 86, 0, False)
    
    # Passed parameters checking function
    _arg_slen.stypy_localization = localization
    _arg_slen.stypy_type_of_self = None
    _arg_slen.stypy_type_store = module_type_store
    _arg_slen.stypy_function_name = '_arg_slen'
    _arg_slen.stypy_param_names_list = ['dvi', 'delta']
    _arg_slen.stypy_varargs_param_name = None
    _arg_slen.stypy_kwargs_param_name = None
    _arg_slen.stypy_call_defaults = defaults
    _arg_slen.stypy_call_varargs = varargs
    _arg_slen.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_arg_slen', ['dvi', 'delta'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_arg_slen', localization, ['dvi', 'delta'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_arg_slen(...)' code ##################

    unicode_47905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, (-1)), 'unicode', u'Signed, length *delta*\n\n    Read *delta* bytes, returning None if *delta* is zero, and\n    the bytes interpreted as a signed integer otherwise.')
    
    
    # Getting the type of 'delta' (line 91)
    delta_47906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 7), 'delta')
    int_47907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 16), 'int')
    # Applying the binary operator '==' (line 91)
    result_eq_47908 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 7), '==', delta_47906, int_47907)
    
    # Testing the type of an if condition (line 91)
    if_condition_47909 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 91, 4), result_eq_47908)
    # Assigning a type to the variable 'if_condition_47909' (line 91)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 4), 'if_condition_47909', if_condition_47909)
    # SSA begins for if statement (line 91)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'None' (line 92)
    None_47910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 15), 'None')
    # Assigning a type to the variable 'stypy_return_type' (line 92)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'stypy_return_type', None_47910)
    # SSA join for if statement (line 91)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to _arg(...): (line 93)
    # Processing the call arguments (line 93)
    # Getting the type of 'delta' (line 93)
    delta_47913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 20), 'delta', False)
    # Getting the type of 'True' (line 93)
    True_47914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 27), 'True', False)
    # Processing the call keyword arguments (line 93)
    kwargs_47915 = {}
    # Getting the type of 'dvi' (line 93)
    dvi_47911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 11), 'dvi', False)
    # Obtaining the member '_arg' of a type (line 93)
    _arg_47912 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 11), dvi_47911, '_arg')
    # Calling _arg(args, kwargs) (line 93)
    _arg_call_result_47916 = invoke(stypy.reporting.localization.Localization(__file__, 93, 11), _arg_47912, *[delta_47913, True_47914], **kwargs_47915)
    
    # Assigning a type to the variable 'stypy_return_type' (line 93)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 4), 'stypy_return_type', _arg_call_result_47916)
    
    # ################# End of '_arg_slen(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_arg_slen' in the type store
    # Getting the type of 'stypy_return_type' (line 86)
    stypy_return_type_47917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_47917)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_arg_slen'
    return stypy_return_type_47917

# Assigning a type to the variable '_arg_slen' (line 86)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 0), '_arg_slen', _arg_slen)

@norecursion
def _arg_slen1(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_arg_slen1'
    module_type_store = module_type_store.open_function_context('_arg_slen1', 96, 0, False)
    
    # Passed parameters checking function
    _arg_slen1.stypy_localization = localization
    _arg_slen1.stypy_type_of_self = None
    _arg_slen1.stypy_type_store = module_type_store
    _arg_slen1.stypy_function_name = '_arg_slen1'
    _arg_slen1.stypy_param_names_list = ['dvi', 'delta']
    _arg_slen1.stypy_varargs_param_name = None
    _arg_slen1.stypy_kwargs_param_name = None
    _arg_slen1.stypy_call_defaults = defaults
    _arg_slen1.stypy_call_varargs = varargs
    _arg_slen1.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_arg_slen1', ['dvi', 'delta'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_arg_slen1', localization, ['dvi', 'delta'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_arg_slen1(...)' code ##################

    unicode_47918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, (-1)), 'unicode', u'Signed, length *delta*+1\n\n    Read *delta*+1 bytes, returning the bytes interpreted as signed.')
    
    # Call to _arg(...): (line 100)
    # Processing the call arguments (line 100)
    # Getting the type of 'delta' (line 100)
    delta_47921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 20), 'delta', False)
    int_47922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 26), 'int')
    # Applying the binary operator '+' (line 100)
    result_add_47923 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 20), '+', delta_47921, int_47922)
    
    # Getting the type of 'True' (line 100)
    True_47924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 29), 'True', False)
    # Processing the call keyword arguments (line 100)
    kwargs_47925 = {}
    # Getting the type of 'dvi' (line 100)
    dvi_47919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 11), 'dvi', False)
    # Obtaining the member '_arg' of a type (line 100)
    _arg_47920 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 11), dvi_47919, '_arg')
    # Calling _arg(args, kwargs) (line 100)
    _arg_call_result_47926 = invoke(stypy.reporting.localization.Localization(__file__, 100, 11), _arg_47920, *[result_add_47923, True_47924], **kwargs_47925)
    
    # Assigning a type to the variable 'stypy_return_type' (line 100)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'stypy_return_type', _arg_call_result_47926)
    
    # ################# End of '_arg_slen1(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_arg_slen1' in the type store
    # Getting the type of 'stypy_return_type' (line 96)
    stypy_return_type_47927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_47927)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_arg_slen1'
    return stypy_return_type_47927

# Assigning a type to the variable '_arg_slen1' (line 96)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 0), '_arg_slen1', _arg_slen1)

@norecursion
def _arg_ulen1(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_arg_ulen1'
    module_type_store = module_type_store.open_function_context('_arg_ulen1', 103, 0, False)
    
    # Passed parameters checking function
    _arg_ulen1.stypy_localization = localization
    _arg_ulen1.stypy_type_of_self = None
    _arg_ulen1.stypy_type_store = module_type_store
    _arg_ulen1.stypy_function_name = '_arg_ulen1'
    _arg_ulen1.stypy_param_names_list = ['dvi', 'delta']
    _arg_ulen1.stypy_varargs_param_name = None
    _arg_ulen1.stypy_kwargs_param_name = None
    _arg_ulen1.stypy_call_defaults = defaults
    _arg_ulen1.stypy_call_varargs = varargs
    _arg_ulen1.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_arg_ulen1', ['dvi', 'delta'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_arg_ulen1', localization, ['dvi', 'delta'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_arg_ulen1(...)' code ##################

    unicode_47928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, (-1)), 'unicode', u'Unsigned length *delta*+1\n\n    Read *delta*+1 bytes, returning the bytes interpreted as unsigned.')
    
    # Call to _arg(...): (line 107)
    # Processing the call arguments (line 107)
    # Getting the type of 'delta' (line 107)
    delta_47931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 20), 'delta', False)
    int_47932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 26), 'int')
    # Applying the binary operator '+' (line 107)
    result_add_47933 = python_operator(stypy.reporting.localization.Localization(__file__, 107, 20), '+', delta_47931, int_47932)
    
    # Getting the type of 'False' (line 107)
    False_47934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 29), 'False', False)
    # Processing the call keyword arguments (line 107)
    kwargs_47935 = {}
    # Getting the type of 'dvi' (line 107)
    dvi_47929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 11), 'dvi', False)
    # Obtaining the member '_arg' of a type (line 107)
    _arg_47930 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 11), dvi_47929, '_arg')
    # Calling _arg(args, kwargs) (line 107)
    _arg_call_result_47936 = invoke(stypy.reporting.localization.Localization(__file__, 107, 11), _arg_47930, *[result_add_47933, False_47934], **kwargs_47935)
    
    # Assigning a type to the variable 'stypy_return_type' (line 107)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 4), 'stypy_return_type', _arg_call_result_47936)
    
    # ################# End of '_arg_ulen1(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_arg_ulen1' in the type store
    # Getting the type of 'stypy_return_type' (line 103)
    stypy_return_type_47937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_47937)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_arg_ulen1'
    return stypy_return_type_47937

# Assigning a type to the variable '_arg_ulen1' (line 103)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 0), '_arg_ulen1', _arg_ulen1)

@norecursion
def _arg_olen1(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_arg_olen1'
    module_type_store = module_type_store.open_function_context('_arg_olen1', 110, 0, False)
    
    # Passed parameters checking function
    _arg_olen1.stypy_localization = localization
    _arg_olen1.stypy_type_of_self = None
    _arg_olen1.stypy_type_store = module_type_store
    _arg_olen1.stypy_function_name = '_arg_olen1'
    _arg_olen1.stypy_param_names_list = ['dvi', 'delta']
    _arg_olen1.stypy_varargs_param_name = None
    _arg_olen1.stypy_kwargs_param_name = None
    _arg_olen1.stypy_call_defaults = defaults
    _arg_olen1.stypy_call_varargs = varargs
    _arg_olen1.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_arg_olen1', ['dvi', 'delta'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_arg_olen1', localization, ['dvi', 'delta'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_arg_olen1(...)' code ##################

    unicode_47938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, (-1)), 'unicode', u'Optionally signed, length *delta*+1\n\n    Read *delta*+1 bytes, returning the bytes interpreted as\n    unsigned integer for 0<=*delta*<3 and signed if *delta*==3.')
    
    # Call to _arg(...): (line 115)
    # Processing the call arguments (line 115)
    # Getting the type of 'delta' (line 115)
    delta_47941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 20), 'delta', False)
    int_47942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 28), 'int')
    # Applying the binary operator '+' (line 115)
    result_add_47943 = python_operator(stypy.reporting.localization.Localization(__file__, 115, 20), '+', delta_47941, int_47942)
    
    
    # Getting the type of 'delta' (line 115)
    delta_47944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 31), 'delta', False)
    int_47945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 40), 'int')
    # Applying the binary operator '==' (line 115)
    result_eq_47946 = python_operator(stypy.reporting.localization.Localization(__file__, 115, 31), '==', delta_47944, int_47945)
    
    # Processing the call keyword arguments (line 115)
    kwargs_47947 = {}
    # Getting the type of 'dvi' (line 115)
    dvi_47939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 11), 'dvi', False)
    # Obtaining the member '_arg' of a type (line 115)
    _arg_47940 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 11), dvi_47939, '_arg')
    # Calling _arg(args, kwargs) (line 115)
    _arg_call_result_47948 = invoke(stypy.reporting.localization.Localization(__file__, 115, 11), _arg_47940, *[result_add_47943, result_eq_47946], **kwargs_47947)
    
    # Assigning a type to the variable 'stypy_return_type' (line 115)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 4), 'stypy_return_type', _arg_call_result_47948)
    
    # ################# End of '_arg_olen1(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_arg_olen1' in the type store
    # Getting the type of 'stypy_return_type' (line 110)
    stypy_return_type_47949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_47949)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_arg_olen1'
    return stypy_return_type_47949

# Assigning a type to the variable '_arg_olen1' (line 110)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 0), '_arg_olen1', _arg_olen1)

# Assigning a Call to a Name (line 118):

# Assigning a Call to a Name (line 118):

# Call to dict(...): (line 118)
# Processing the call keyword arguments (line 118)
# Getting the type of '_arg_raw' (line 118)
_arg_raw_47951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 24), '_arg_raw', False)
keyword_47952 = _arg_raw_47951

# Call to partial(...): (line 119)
# Processing the call arguments (line 119)
# Getting the type of '_arg' (line 119)
_arg_47954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 31), '_arg', False)
int_47955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 37), 'int')
# Getting the type of 'False' (line 119)
False_47956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 40), 'False', False)
# Processing the call keyword arguments (line 119)
kwargs_47957 = {}
# Getting the type of 'partial' (line 119)
partial_47953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 23), 'partial', False)
# Calling partial(args, kwargs) (line 119)
partial_call_result_47958 = invoke(stypy.reporting.localization.Localization(__file__, 119, 23), partial_47953, *[_arg_47954, int_47955, False_47956], **kwargs_47957)

keyword_47959 = partial_call_result_47958

# Call to partial(...): (line 120)
# Processing the call arguments (line 120)
# Getting the type of '_arg' (line 120)
_arg_47961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 31), '_arg', False)
int_47962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 37), 'int')
# Getting the type of 'False' (line 120)
False_47963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 40), 'False', False)
# Processing the call keyword arguments (line 120)
kwargs_47964 = {}
# Getting the type of 'partial' (line 120)
partial_47960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 23), 'partial', False)
# Calling partial(args, kwargs) (line 120)
partial_call_result_47965 = invoke(stypy.reporting.localization.Localization(__file__, 120, 23), partial_47960, *[_arg_47961, int_47962, False_47963], **kwargs_47964)

keyword_47966 = partial_call_result_47965

# Call to partial(...): (line 121)
# Processing the call arguments (line 121)
# Getting the type of '_arg' (line 121)
_arg_47968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 31), '_arg', False)
int_47969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 37), 'int')
# Getting the type of 'True' (line 121)
True_47970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 40), 'True', False)
# Processing the call keyword arguments (line 121)
kwargs_47971 = {}
# Getting the type of 'partial' (line 121)
partial_47967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 23), 'partial', False)
# Calling partial(args, kwargs) (line 121)
partial_call_result_47972 = invoke(stypy.reporting.localization.Localization(__file__, 121, 23), partial_47967, *[_arg_47968, int_47969, True_47970], **kwargs_47971)

keyword_47973 = partial_call_result_47972
# Getting the type of '_arg_slen' (line 122)
_arg_slen_47974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 25), '_arg_slen', False)
keyword_47975 = _arg_slen_47974
# Getting the type of '_arg_olen1' (line 123)
_arg_olen1_47976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 26), '_arg_olen1', False)
keyword_47977 = _arg_olen1_47976
# Getting the type of '_arg_slen1' (line 124)
_arg_slen1_47978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 26), '_arg_slen1', False)
keyword_47979 = _arg_slen1_47978
# Getting the type of '_arg_ulen1' (line 125)
_arg_ulen1_47980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 26), '_arg_ulen1', False)
keyword_47981 = _arg_ulen1_47980
kwargs_47982 = {'olen1': keyword_47977, 'slen': keyword_47975, 'u4': keyword_47966, 'u1': keyword_47959, 's4': keyword_47973, 'raw': keyword_47952, 'slen1': keyword_47979, 'ulen1': keyword_47981}
# Getting the type of 'dict' (line 118)
dict_47950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 15), 'dict', False)
# Calling dict(args, kwargs) (line 118)
dict_call_result_47983 = invoke(stypy.reporting.localization.Localization(__file__, 118, 15), dict_47950, *[], **kwargs_47982)

# Assigning a type to the variable '_arg_mapping' (line 118)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 0), '_arg_mapping', dict_call_result_47983)

@norecursion
def _dispatch(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 128)
    None_47984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 30), 'None')
    # Getting the type of 'None' (line 128)
    None_47985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 42), 'None')
    
    # Obtaining an instance of the builtin type 'tuple' (line 128)
    tuple_47986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 54), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 128)
    # Adding element type (line 128)
    unicode_47987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 54), 'unicode', u'raw')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 128, 54), tuple_47986, unicode_47987)
    
    defaults = [None_47984, None_47985, tuple_47986]
    # Create a new context for function '_dispatch'
    module_type_store = module_type_store.open_function_context('_dispatch', 128, 0, False)
    
    # Passed parameters checking function
    _dispatch.stypy_localization = localization
    _dispatch.stypy_type_of_self = None
    _dispatch.stypy_type_store = module_type_store
    _dispatch.stypy_function_name = '_dispatch'
    _dispatch.stypy_param_names_list = ['table', 'min', 'max', 'state', 'args']
    _dispatch.stypy_varargs_param_name = None
    _dispatch.stypy_kwargs_param_name = None
    _dispatch.stypy_call_defaults = defaults
    _dispatch.stypy_call_varargs = varargs
    _dispatch.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_dispatch', ['table', 'min', 'max', 'state', 'args'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_dispatch', localization, ['table', 'min', 'max', 'state', 'args'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_dispatch(...)' code ##################

    unicode_47988 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, (-1)), 'unicode', u"Decorator for dispatch by opcode. Sets the values in *table*\n    from *min* to *max* to this method, adds a check that the Dvi state\n    matches *state* if not None, reads arguments from the file according\n    to *args*.\n\n    *table*\n        the dispatch table to be filled in\n\n    *min*\n        minimum opcode for calling this function\n\n    *max*\n        maximum opcode for calling this function, None if only *min* is allowed\n\n    *state*\n        state of the Dvi object in which these opcodes are allowed\n\n    *args*\n        sequence of argument specifications:\n\n        ``'raw'``: opcode minus minimum\n        ``'u1'``: read one unsigned byte\n        ``'u4'``: read four bytes, treat as an unsigned number\n        ``'s4'``: read four bytes, treat as a signed number\n        ``'slen'``: read (opcode - minimum) bytes, treat as signed\n        ``'slen1'``: read (opcode - minimum + 1) bytes, treat as signed\n        ``'ulen1'``: read (opcode - minimum + 1) bytes, treat as unsigned\n        ``'olen1'``: read (opcode - minimum + 1) bytes, treat as unsigned\n                     if under four bytes, signed if four bytes\n    ")

    @norecursion
    def decorate(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'decorate'
        module_type_store = module_type_store.open_function_context('decorate', 159, 4, False)
        
        # Passed parameters checking function
        decorate.stypy_localization = localization
        decorate.stypy_type_of_self = None
        decorate.stypy_type_store = module_type_store
        decorate.stypy_function_name = 'decorate'
        decorate.stypy_param_names_list = ['method']
        decorate.stypy_varargs_param_name = None
        decorate.stypy_kwargs_param_name = None
        decorate.stypy_call_defaults = defaults
        decorate.stypy_call_varargs = varargs
        decorate.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'decorate', ['method'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'decorate', localization, ['method'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'decorate(...)' code ##################

        
        # Assigning a ListComp to a Name (line 160):
        
        # Assigning a ListComp to a Name (line 160):
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'args' (line 160)
        args_47993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 45), 'args')
        comprehension_47994 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 20), args_47993)
        # Assigning a type to the variable 'x' (line 160)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 20), 'x', comprehension_47994)
        
        # Obtaining the type of the subscript
        # Getting the type of 'x' (line 160)
        x_47989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 33), 'x')
        # Getting the type of '_arg_mapping' (line 160)
        _arg_mapping_47990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 20), '_arg_mapping')
        # Obtaining the member '__getitem__' of a type (line 160)
        getitem___47991 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 20), _arg_mapping_47990, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 160)
        subscript_call_result_47992 = invoke(stypy.reporting.localization.Localization(__file__, 160, 20), getitem___47991, x_47989)
        
        list_47995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 20), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 20), list_47995, subscript_call_result_47992)
        # Assigning a type to the variable 'get_args' (line 160)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'get_args', list_47995)

        @norecursion
        def wrapper(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'wrapper'
            module_type_store = module_type_store.open_function_context('wrapper', 162, 8, False)
            
            # Passed parameters checking function
            wrapper.stypy_localization = localization
            wrapper.stypy_type_of_self = None
            wrapper.stypy_type_store = module_type_store
            wrapper.stypy_function_name = 'wrapper'
            wrapper.stypy_param_names_list = ['self', 'byte']
            wrapper.stypy_varargs_param_name = None
            wrapper.stypy_kwargs_param_name = None
            wrapper.stypy_call_defaults = defaults
            wrapper.stypy_call_varargs = varargs
            wrapper.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'wrapper', ['self', 'byte'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'wrapper', localization, ['self', 'byte'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'wrapper(...)' code ##################

            
            
            # Evaluating a boolean operation
            
            # Getting the type of 'state' (line 164)
            state_47996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 15), 'state')
            # Getting the type of 'None' (line 164)
            None_47997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 28), 'None')
            # Applying the binary operator 'isnot' (line 164)
            result_is_not_47998 = python_operator(stypy.reporting.localization.Localization(__file__, 164, 15), 'isnot', state_47996, None_47997)
            
            
            # Getting the type of 'self' (line 164)
            self_47999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 37), 'self')
            # Obtaining the member 'state' of a type (line 164)
            state_48000 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 37), self_47999, 'state')
            # Getting the type of 'state' (line 164)
            state_48001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 51), 'state')
            # Applying the binary operator '!=' (line 164)
            result_ne_48002 = python_operator(stypy.reporting.localization.Localization(__file__, 164, 37), '!=', state_48000, state_48001)
            
            # Applying the binary operator 'and' (line 164)
            result_and_keyword_48003 = python_operator(stypy.reporting.localization.Localization(__file__, 164, 15), 'and', result_is_not_47998, result_ne_48002)
            
            # Testing the type of an if condition (line 164)
            if_condition_48004 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 164, 12), result_and_keyword_48003)
            # Assigning a type to the variable 'if_condition_48004' (line 164)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 12), 'if_condition_48004', if_condition_48004)
            # SSA begins for if statement (line 164)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to ValueError(...): (line 165)
            # Processing the call arguments (line 165)
            unicode_48006 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 33), 'unicode', u'state precondition failed')
            # Processing the call keyword arguments (line 165)
            kwargs_48007 = {}
            # Getting the type of 'ValueError' (line 165)
            ValueError_48005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 22), 'ValueError', False)
            # Calling ValueError(args, kwargs) (line 165)
            ValueError_call_result_48008 = invoke(stypy.reporting.localization.Localization(__file__, 165, 22), ValueError_48005, *[unicode_48006], **kwargs_48007)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 165, 16), ValueError_call_result_48008, 'raise parameter', BaseException)
            # SSA join for if statement (line 164)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Call to method(...): (line 166)
            # Processing the call arguments (line 166)
            # Getting the type of 'self' (line 166)
            self_48010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 26), 'self', False)
            # Calculating list comprehension
            # Calculating comprehension expression
            # Getting the type of 'get_args' (line 166)
            get_args_48018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 61), 'get_args', False)
            comprehension_48019 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 166, 34), get_args_48018)
            # Assigning a type to the variable 'f' (line 166)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 34), 'f', comprehension_48019)
            
            # Call to f(...): (line 166)
            # Processing the call arguments (line 166)
            # Getting the type of 'self' (line 166)
            self_48012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 36), 'self', False)
            # Getting the type of 'byte' (line 166)
            byte_48013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 42), 'byte', False)
            # Getting the type of 'min' (line 166)
            min_48014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 47), 'min', False)
            # Applying the binary operator '-' (line 166)
            result_sub_48015 = python_operator(stypy.reporting.localization.Localization(__file__, 166, 42), '-', byte_48013, min_48014)
            
            # Processing the call keyword arguments (line 166)
            kwargs_48016 = {}
            # Getting the type of 'f' (line 166)
            f_48011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 34), 'f', False)
            # Calling f(args, kwargs) (line 166)
            f_call_result_48017 = invoke(stypy.reporting.localization.Localization(__file__, 166, 34), f_48011, *[self_48012, result_sub_48015], **kwargs_48016)
            
            list_48020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 34), 'list')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 166, 34), list_48020, f_call_result_48017)
            # Processing the call keyword arguments (line 166)
            kwargs_48021 = {}
            # Getting the type of 'method' (line 166)
            method_48009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 19), 'method', False)
            # Calling method(args, kwargs) (line 166)
            method_call_result_48022 = invoke(stypy.reporting.localization.Localization(__file__, 166, 19), method_48009, *[self_48010, list_48020], **kwargs_48021)
            
            # Assigning a type to the variable 'stypy_return_type' (line 166)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 12), 'stypy_return_type', method_call_result_48022)
            
            # ################# End of 'wrapper(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'wrapper' in the type store
            # Getting the type of 'stypy_return_type' (line 162)
            stypy_return_type_48023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_48023)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'wrapper'
            return stypy_return_type_48023

        # Assigning a type to the variable 'wrapper' (line 162)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 8), 'wrapper', wrapper)
        
        # Type idiom detected: calculating its left and rigth part (line 167)
        # Getting the type of 'max' (line 167)
        max_48024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 11), 'max')
        # Getting the type of 'None' (line 167)
        None_48025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 18), 'None')
        
        (may_be_48026, more_types_in_union_48027) = may_be_none(max_48024, None_48025)

        if may_be_48026:

            if more_types_in_union_48027:
                # Runtime conditional SSA (line 167)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Name to a Subscript (line 168):
            
            # Assigning a Name to a Subscript (line 168):
            # Getting the type of 'wrapper' (line 168)
            wrapper_48028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 25), 'wrapper')
            # Getting the type of 'table' (line 168)
            table_48029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 12), 'table')
            # Getting the type of 'min' (line 168)
            min_48030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 18), 'min')
            # Storing an element on a container (line 168)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 12), table_48029, (min_48030, wrapper_48028))

            if more_types_in_union_48027:
                # Runtime conditional SSA for else branch (line 167)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_48026) or more_types_in_union_48027):
            
            
            # Call to xrange(...): (line 170)
            # Processing the call arguments (line 170)
            # Getting the type of 'min' (line 170)
            min_48032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 28), 'min', False)
            # Getting the type of 'max' (line 170)
            max_48033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 33), 'max', False)
            int_48034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 37), 'int')
            # Applying the binary operator '+' (line 170)
            result_add_48035 = python_operator(stypy.reporting.localization.Localization(__file__, 170, 33), '+', max_48033, int_48034)
            
            # Processing the call keyword arguments (line 170)
            kwargs_48036 = {}
            # Getting the type of 'xrange' (line 170)
            xrange_48031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 21), 'xrange', False)
            # Calling xrange(args, kwargs) (line 170)
            xrange_call_result_48037 = invoke(stypy.reporting.localization.Localization(__file__, 170, 21), xrange_48031, *[min_48032, result_add_48035], **kwargs_48036)
            
            # Testing the type of a for loop iterable (line 170)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 170, 12), xrange_call_result_48037)
            # Getting the type of the for loop variable (line 170)
            for_loop_var_48038 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 170, 12), xrange_call_result_48037)
            # Assigning a type to the variable 'i' (line 170)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 12), 'i', for_loop_var_48038)
            # SSA begins for a for statement (line 170)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            # Evaluating assert statement condition
            
            
            # Obtaining the type of the subscript
            # Getting the type of 'i' (line 171)
            i_48039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 29), 'i')
            # Getting the type of 'table' (line 171)
            table_48040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 23), 'table')
            # Obtaining the member '__getitem__' of a type (line 171)
            getitem___48041 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 23), table_48040, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 171)
            subscript_call_result_48042 = invoke(stypy.reporting.localization.Localization(__file__, 171, 23), getitem___48041, i_48039)
            
            # Getting the type of 'None' (line 171)
            None_48043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 35), 'None')
            # Applying the binary operator 'is' (line 171)
            result_is__48044 = python_operator(stypy.reporting.localization.Localization(__file__, 171, 23), 'is', subscript_call_result_48042, None_48043)
            
            
            # Assigning a Name to a Subscript (line 172):
            
            # Assigning a Name to a Subscript (line 172):
            # Getting the type of 'wrapper' (line 172)
            wrapper_48045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 27), 'wrapper')
            # Getting the type of 'table' (line 172)
            table_48046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 16), 'table')
            # Getting the type of 'i' (line 172)
            i_48047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 22), 'i')
            # Storing an element on a container (line 172)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 172, 16), table_48046, (i_48047, wrapper_48045))
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()
            

            if (may_be_48026 and more_types_in_union_48027):
                # SSA join for if statement (line 167)
                module_type_store = module_type_store.join_ssa_context()


        
        # Getting the type of 'wrapper' (line 173)
        wrapper_48048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 15), 'wrapper')
        # Assigning a type to the variable 'stypy_return_type' (line 173)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 8), 'stypy_return_type', wrapper_48048)
        
        # ################# End of 'decorate(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'decorate' in the type store
        # Getting the type of 'stypy_return_type' (line 159)
        stypy_return_type_48049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_48049)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'decorate'
        return stypy_return_type_48049

    # Assigning a type to the variable 'decorate' (line 159)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 4), 'decorate', decorate)
    # Getting the type of 'decorate' (line 174)
    decorate_48050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 11), 'decorate')
    # Assigning a type to the variable 'stypy_return_type' (line 174)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 4), 'stypy_return_type', decorate_48050)
    
    # ################# End of '_dispatch(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_dispatch' in the type store
    # Getting the type of 'stypy_return_type' (line 128)
    stypy_return_type_48051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_48051)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_dispatch'
    return stypy_return_type_48051

# Assigning a type to the variable '_dispatch' (line 128)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 0), '_dispatch', _dispatch)
# Declaration of the 'Dvi' class

class Dvi(object, ):
    unicode_48052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, (-1)), 'unicode', u'\n    A reader for a dvi ("device-independent") file, as produced by TeX.\n    The current implementation can only iterate through pages in order,\n    and does not even attempt to verify the postamble.\n\n    This class can be used as a context manager to close the underlying\n    file upon exit. Pages can be read via iteration. Here is an overly\n    simple way to extract text without trying to detect whitespace::\n\n    >>> with matplotlib.dviread.Dvi(\'input.dvi\', 72) as dvi:\n    >>>     for page in dvi:\n    >>>         print \'\'.join(unichr(t.glyph) for t in page.text)\n    ')
    
    # Assigning a ListComp to a Name (line 192):
    
    # Assigning a Call to a Name (line 193):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 195, 4, False)
        # Assigning a type to the variable 'self' (line 196)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Dvi.__init__', ['filename', 'dpi'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['filename', 'dpi'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        unicode_48053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, (-1)), 'unicode', u"\n        Read the data from the file named *filename* and convert\n        TeX's internal units to units of *dpi* per inch.\n        *dpi* only sets the units and does not limit the resolution.\n        Use None to return TeX's internal units.\n        ")
        
        # Call to report(...): (line 202)
        # Processing the call arguments (line 202)
        unicode_48057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 34), 'unicode', u'Dvi: ')
        # Getting the type of 'filename' (line 202)
        filename_48058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 44), 'filename', False)
        # Applying the binary operator '+' (line 202)
        result_add_48059 = python_operator(stypy.reporting.localization.Localization(__file__, 202, 34), '+', unicode_48057, filename_48058)
        
        unicode_48060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 54), 'unicode', u'debug')
        # Processing the call keyword arguments (line 202)
        kwargs_48061 = {}
        # Getting the type of 'matplotlib' (line 202)
        matplotlib_48054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 8), 'matplotlib', False)
        # Obtaining the member 'verbose' of a type (line 202)
        verbose_48055 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 8), matplotlib_48054, 'verbose')
        # Obtaining the member 'report' of a type (line 202)
        report_48056 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 8), verbose_48055, 'report')
        # Calling report(args, kwargs) (line 202)
        report_call_result_48062 = invoke(stypy.reporting.localization.Localization(__file__, 202, 8), report_48056, *[result_add_48059, unicode_48060], **kwargs_48061)
        
        
        # Assigning a Call to a Attribute (line 203):
        
        # Assigning a Call to a Attribute (line 203):
        
        # Call to open(...): (line 203)
        # Processing the call arguments (line 203)
        # Getting the type of 'filename' (line 203)
        filename_48064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 25), 'filename', False)
        unicode_48065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 35), 'unicode', u'rb')
        # Processing the call keyword arguments (line 203)
        kwargs_48066 = {}
        # Getting the type of 'open' (line 203)
        open_48063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 20), 'open', False)
        # Calling open(args, kwargs) (line 203)
        open_call_result_48067 = invoke(stypy.reporting.localization.Localization(__file__, 203, 20), open_48063, *[filename_48064, unicode_48065], **kwargs_48066)
        
        # Getting the type of 'self' (line 203)
        self_48068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 8), 'self')
        # Setting the type of the member 'file' of a type (line 203)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 8), self_48068, 'file', open_call_result_48067)
        
        # Assigning a Name to a Attribute (line 204):
        
        # Assigning a Name to a Attribute (line 204):
        # Getting the type of 'dpi' (line 204)
        dpi_48069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 19), 'dpi')
        # Getting the type of 'self' (line 204)
        self_48070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 8), 'self')
        # Setting the type of the member 'dpi' of a type (line 204)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 8), self_48070, 'dpi', dpi_48069)
        
        # Assigning a Dict to a Attribute (line 205):
        
        # Assigning a Dict to a Attribute (line 205):
        
        # Obtaining an instance of the builtin type 'dict' (line 205)
        dict_48071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 21), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 205)
        
        # Getting the type of 'self' (line 205)
        self_48072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 8), 'self')
        # Setting the type of the member 'fonts' of a type (line 205)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 8), self_48072, 'fonts', dict_48071)
        
        # Assigning a Attribute to a Attribute (line 206):
        
        # Assigning a Attribute to a Attribute (line 206):
        # Getting the type of '_dvistate' (line 206)
        _dvistate_48073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 21), '_dvistate')
        # Obtaining the member 'pre' of a type (line 206)
        pre_48074 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 21), _dvistate_48073, 'pre')
        # Getting the type of 'self' (line 206)
        self_48075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 8), 'self')
        # Setting the type of the member 'state' of a type (line 206)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 8), self_48075, 'state', pre_48074)
        
        # Assigning a Call to a Attribute (line 207):
        
        # Assigning a Call to a Attribute (line 207):
        
        # Call to _get_baseline(...): (line 207)
        # Processing the call arguments (line 207)
        # Getting the type of 'filename' (line 207)
        filename_48078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 43), 'filename', False)
        # Processing the call keyword arguments (line 207)
        kwargs_48079 = {}
        # Getting the type of 'self' (line 207)
        self_48076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 24), 'self', False)
        # Obtaining the member '_get_baseline' of a type (line 207)
        _get_baseline_48077 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 24), self_48076, '_get_baseline')
        # Calling _get_baseline(args, kwargs) (line 207)
        _get_baseline_call_result_48080 = invoke(stypy.reporting.localization.Localization(__file__, 207, 24), _get_baseline_48077, *[filename_48078], **kwargs_48079)
        
        # Getting the type of 'self' (line 207)
        self_48081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 8), 'self')
        # Setting the type of the member 'baseline' of a type (line 207)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 8), self_48081, 'baseline', _get_baseline_call_result_48080)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def _get_baseline(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_get_baseline'
        module_type_store = module_type_store.open_function_context('_get_baseline', 209, 4, False)
        # Assigning a type to the variable 'self' (line 210)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Dvi._get_baseline.__dict__.__setitem__('stypy_localization', localization)
        Dvi._get_baseline.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Dvi._get_baseline.__dict__.__setitem__('stypy_type_store', module_type_store)
        Dvi._get_baseline.__dict__.__setitem__('stypy_function_name', 'Dvi._get_baseline')
        Dvi._get_baseline.__dict__.__setitem__('stypy_param_names_list', ['filename'])
        Dvi._get_baseline.__dict__.__setitem__('stypy_varargs_param_name', None)
        Dvi._get_baseline.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Dvi._get_baseline.__dict__.__setitem__('stypy_call_defaults', defaults)
        Dvi._get_baseline.__dict__.__setitem__('stypy_call_varargs', varargs)
        Dvi._get_baseline.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Dvi._get_baseline.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Dvi._get_baseline', ['filename'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_get_baseline', localization, ['filename'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_get_baseline(...)' code ##################

        
        
        # Obtaining the type of the subscript
        unicode_48082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 20), 'unicode', u'text.latex.preview')
        # Getting the type of 'rcParams' (line 210)
        rcParams_48083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 11), 'rcParams')
        # Obtaining the member '__getitem__' of a type (line 210)
        getitem___48084 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 11), rcParams_48083, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 210)
        subscript_call_result_48085 = invoke(stypy.reporting.localization.Localization(__file__, 210, 11), getitem___48084, unicode_48082)
        
        # Testing the type of an if condition (line 210)
        if_condition_48086 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 210, 8), subscript_call_result_48085)
        # Assigning a type to the variable 'if_condition_48086' (line 210)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 8), 'if_condition_48086', if_condition_48086)
        # SSA begins for if statement (line 210)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Tuple (line 211):
        
        # Assigning a Call to a Name:
        
        # Call to splitext(...): (line 211)
        # Processing the call arguments (line 211)
        # Getting the type of 'filename' (line 211)
        filename_48090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 41), 'filename', False)
        # Processing the call keyword arguments (line 211)
        kwargs_48091 = {}
        # Getting the type of 'os' (line 211)
        os_48087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 24), 'os', False)
        # Obtaining the member 'path' of a type (line 211)
        path_48088 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 24), os_48087, 'path')
        # Obtaining the member 'splitext' of a type (line 211)
        splitext_48089 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 24), path_48088, 'splitext')
        # Calling splitext(args, kwargs) (line 211)
        splitext_call_result_48092 = invoke(stypy.reporting.localization.Localization(__file__, 211, 24), splitext_48089, *[filename_48090], **kwargs_48091)
        
        # Assigning a type to the variable 'call_assignment_47756' (line 211)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 12), 'call_assignment_47756', splitext_call_result_48092)
        
        # Assigning a Call to a Name (line 211):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_48095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 12), 'int')
        # Processing the call keyword arguments
        kwargs_48096 = {}
        # Getting the type of 'call_assignment_47756' (line 211)
        call_assignment_47756_48093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 12), 'call_assignment_47756', False)
        # Obtaining the member '__getitem__' of a type (line 211)
        getitem___48094 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 12), call_assignment_47756_48093, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_48097 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___48094, *[int_48095], **kwargs_48096)
        
        # Assigning a type to the variable 'call_assignment_47757' (line 211)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 12), 'call_assignment_47757', getitem___call_result_48097)
        
        # Assigning a Name to a Name (line 211):
        # Getting the type of 'call_assignment_47757' (line 211)
        call_assignment_47757_48098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 12), 'call_assignment_47757')
        # Assigning a type to the variable 'base' (line 211)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 12), 'base', call_assignment_47757_48098)
        
        # Assigning a Call to a Name (line 211):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_48101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 12), 'int')
        # Processing the call keyword arguments
        kwargs_48102 = {}
        # Getting the type of 'call_assignment_47756' (line 211)
        call_assignment_47756_48099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 12), 'call_assignment_47756', False)
        # Obtaining the member '__getitem__' of a type (line 211)
        getitem___48100 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 12), call_assignment_47756_48099, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_48103 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___48100, *[int_48101], **kwargs_48102)
        
        # Assigning a type to the variable 'call_assignment_47758' (line 211)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 12), 'call_assignment_47758', getitem___call_result_48103)
        
        # Assigning a Name to a Name (line 211):
        # Getting the type of 'call_assignment_47758' (line 211)
        call_assignment_47758_48104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 12), 'call_assignment_47758')
        # Assigning a type to the variable 'ext' (line 211)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 18), 'ext', call_assignment_47758_48104)
        
        # Assigning a BinOp to a Name (line 212):
        
        # Assigning a BinOp to a Name (line 212):
        # Getting the type of 'base' (line 212)
        base_48105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 32), 'base')
        unicode_48106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 39), 'unicode', u'.baseline')
        # Applying the binary operator '+' (line 212)
        result_add_48107 = python_operator(stypy.reporting.localization.Localization(__file__, 212, 32), '+', base_48105, unicode_48106)
        
        # Assigning a type to the variable 'baseline_filename' (line 212)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 12), 'baseline_filename', result_add_48107)
        
        
        # Call to exists(...): (line 213)
        # Processing the call arguments (line 213)
        # Getting the type of 'baseline_filename' (line 213)
        baseline_filename_48111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 30), 'baseline_filename', False)
        # Processing the call keyword arguments (line 213)
        kwargs_48112 = {}
        # Getting the type of 'os' (line 213)
        os_48108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 15), 'os', False)
        # Obtaining the member 'path' of a type (line 213)
        path_48109 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 15), os_48108, 'path')
        # Obtaining the member 'exists' of a type (line 213)
        exists_48110 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 15), path_48109, 'exists')
        # Calling exists(args, kwargs) (line 213)
        exists_call_result_48113 = invoke(stypy.reporting.localization.Localization(__file__, 213, 15), exists_48110, *[baseline_filename_48111], **kwargs_48112)
        
        # Testing the type of an if condition (line 213)
        if_condition_48114 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 213, 12), exists_call_result_48113)
        # Assigning a type to the variable 'if_condition_48114' (line 213)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 12), 'if_condition_48114', if_condition_48114)
        # SSA begins for if statement (line 213)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to open(...): (line 214)
        # Processing the call arguments (line 214)
        # Getting the type of 'baseline_filename' (line 214)
        baseline_filename_48116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 26), 'baseline_filename', False)
        unicode_48117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 45), 'unicode', u'rb')
        # Processing the call keyword arguments (line 214)
        kwargs_48118 = {}
        # Getting the type of 'open' (line 214)
        open_48115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 21), 'open', False)
        # Calling open(args, kwargs) (line 214)
        open_call_result_48119 = invoke(stypy.reporting.localization.Localization(__file__, 214, 21), open_48115, *[baseline_filename_48116, unicode_48117], **kwargs_48118)
        
        with_48120 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 214, 21), open_call_result_48119, 'with parameter', '__enter__', '__exit__')

        if with_48120:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 214)
            enter___48121 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 21), open_call_result_48119, '__enter__')
            with_enter_48122 = invoke(stypy.reporting.localization.Localization(__file__, 214, 21), enter___48121)
            # Assigning a type to the variable 'fd' (line 214)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 21), 'fd', with_enter_48122)
            
            # Assigning a Call to a Name (line 215):
            
            # Assigning a Call to a Name (line 215):
            
            # Call to split(...): (line 215)
            # Processing the call keyword arguments (line 215)
            kwargs_48128 = {}
            
            # Call to read(...): (line 215)
            # Processing the call keyword arguments (line 215)
            kwargs_48125 = {}
            # Getting the type of 'fd' (line 215)
            fd_48123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 24), 'fd', False)
            # Obtaining the member 'read' of a type (line 215)
            read_48124 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 24), fd_48123, 'read')
            # Calling read(args, kwargs) (line 215)
            read_call_result_48126 = invoke(stypy.reporting.localization.Localization(__file__, 215, 24), read_48124, *[], **kwargs_48125)
            
            # Obtaining the member 'split' of a type (line 215)
            split_48127 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 24), read_call_result_48126, 'split')
            # Calling split(args, kwargs) (line 215)
            split_call_result_48129 = invoke(stypy.reporting.localization.Localization(__file__, 215, 24), split_48127, *[], **kwargs_48128)
            
            # Assigning a type to the variable 'l' (line 215)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 20), 'l', split_call_result_48129)
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 214)
            exit___48130 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 21), open_call_result_48119, '__exit__')
            with_exit_48131 = invoke(stypy.reporting.localization.Localization(__file__, 214, 21), exit___48130, None, None, None)

        
        # Assigning a Name to a Tuple (line 216):
        
        # Assigning a Subscript to a Name (line 216):
        
        # Obtaining the type of the subscript
        int_48132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 16), 'int')
        # Getting the type of 'l' (line 216)
        l_48133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 39), 'l')
        # Obtaining the member '__getitem__' of a type (line 216)
        getitem___48134 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 16), l_48133, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 216)
        subscript_call_result_48135 = invoke(stypy.reporting.localization.Localization(__file__, 216, 16), getitem___48134, int_48132)
        
        # Assigning a type to the variable 'tuple_var_assignment_47759' (line 216)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 16), 'tuple_var_assignment_47759', subscript_call_result_48135)
        
        # Assigning a Subscript to a Name (line 216):
        
        # Obtaining the type of the subscript
        int_48136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 16), 'int')
        # Getting the type of 'l' (line 216)
        l_48137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 39), 'l')
        # Obtaining the member '__getitem__' of a type (line 216)
        getitem___48138 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 16), l_48137, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 216)
        subscript_call_result_48139 = invoke(stypy.reporting.localization.Localization(__file__, 216, 16), getitem___48138, int_48136)
        
        # Assigning a type to the variable 'tuple_var_assignment_47760' (line 216)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 16), 'tuple_var_assignment_47760', subscript_call_result_48139)
        
        # Assigning a Subscript to a Name (line 216):
        
        # Obtaining the type of the subscript
        int_48140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 16), 'int')
        # Getting the type of 'l' (line 216)
        l_48141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 39), 'l')
        # Obtaining the member '__getitem__' of a type (line 216)
        getitem___48142 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 16), l_48141, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 216)
        subscript_call_result_48143 = invoke(stypy.reporting.localization.Localization(__file__, 216, 16), getitem___48142, int_48140)
        
        # Assigning a type to the variable 'tuple_var_assignment_47761' (line 216)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 16), 'tuple_var_assignment_47761', subscript_call_result_48143)
        
        # Assigning a Name to a Name (line 216):
        # Getting the type of 'tuple_var_assignment_47759' (line 216)
        tuple_var_assignment_47759_48144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 16), 'tuple_var_assignment_47759')
        # Assigning a type to the variable 'height' (line 216)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 16), 'height', tuple_var_assignment_47759_48144)
        
        # Assigning a Name to a Name (line 216):
        # Getting the type of 'tuple_var_assignment_47760' (line 216)
        tuple_var_assignment_47760_48145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 16), 'tuple_var_assignment_47760')
        # Assigning a type to the variable 'depth' (line 216)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 24), 'depth', tuple_var_assignment_47760_48145)
        
        # Assigning a Name to a Name (line 216):
        # Getting the type of 'tuple_var_assignment_47761' (line 216)
        tuple_var_assignment_47761_48146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 16), 'tuple_var_assignment_47761')
        # Assigning a type to the variable 'width' (line 216)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 31), 'width', tuple_var_assignment_47761_48146)
        
        # Call to float(...): (line 217)
        # Processing the call arguments (line 217)
        # Getting the type of 'depth' (line 217)
        depth_48148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 29), 'depth', False)
        # Processing the call keyword arguments (line 217)
        kwargs_48149 = {}
        # Getting the type of 'float' (line 217)
        float_48147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 23), 'float', False)
        # Calling float(args, kwargs) (line 217)
        float_call_result_48150 = invoke(stypy.reporting.localization.Localization(__file__, 217, 23), float_48147, *[depth_48148], **kwargs_48149)
        
        # Assigning a type to the variable 'stypy_return_type' (line 217)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 16), 'stypy_return_type', float_call_result_48150)
        # SSA join for if statement (line 213)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 210)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'None' (line 218)
        None_48151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 15), 'None')
        # Assigning a type to the variable 'stypy_return_type' (line 218)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 8), 'stypy_return_type', None_48151)
        
        # ################# End of '_get_baseline(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_get_baseline' in the type store
        # Getting the type of 'stypy_return_type' (line 209)
        stypy_return_type_48152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_48152)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_get_baseline'
        return stypy_return_type_48152


    @norecursion
    def __enter__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__enter__'
        module_type_store = module_type_store.open_function_context('__enter__', 220, 4, False)
        # Assigning a type to the variable 'self' (line 221)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Dvi.__enter__.__dict__.__setitem__('stypy_localization', localization)
        Dvi.__enter__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Dvi.__enter__.__dict__.__setitem__('stypy_type_store', module_type_store)
        Dvi.__enter__.__dict__.__setitem__('stypy_function_name', 'Dvi.__enter__')
        Dvi.__enter__.__dict__.__setitem__('stypy_param_names_list', [])
        Dvi.__enter__.__dict__.__setitem__('stypy_varargs_param_name', None)
        Dvi.__enter__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Dvi.__enter__.__dict__.__setitem__('stypy_call_defaults', defaults)
        Dvi.__enter__.__dict__.__setitem__('stypy_call_varargs', varargs)
        Dvi.__enter__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Dvi.__enter__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Dvi.__enter__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__enter__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__enter__(...)' code ##################

        unicode_48153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, (-1)), 'unicode', u'\n        Context manager enter method, does nothing.\n        ')
        # Getting the type of 'self' (line 224)
        self_48154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 15), 'self')
        # Assigning a type to the variable 'stypy_return_type' (line 224)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 8), 'stypy_return_type', self_48154)
        
        # ################# End of '__enter__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__enter__' in the type store
        # Getting the type of 'stypy_return_type' (line 220)
        stypy_return_type_48155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_48155)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__enter__'
        return stypy_return_type_48155


    @norecursion
    def __exit__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__exit__'
        module_type_store = module_type_store.open_function_context('__exit__', 226, 4, False)
        # Assigning a type to the variable 'self' (line 227)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Dvi.__exit__.__dict__.__setitem__('stypy_localization', localization)
        Dvi.__exit__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Dvi.__exit__.__dict__.__setitem__('stypy_type_store', module_type_store)
        Dvi.__exit__.__dict__.__setitem__('stypy_function_name', 'Dvi.__exit__')
        Dvi.__exit__.__dict__.__setitem__('stypy_param_names_list', ['etype', 'evalue', 'etrace'])
        Dvi.__exit__.__dict__.__setitem__('stypy_varargs_param_name', None)
        Dvi.__exit__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Dvi.__exit__.__dict__.__setitem__('stypy_call_defaults', defaults)
        Dvi.__exit__.__dict__.__setitem__('stypy_call_varargs', varargs)
        Dvi.__exit__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Dvi.__exit__.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Dvi.__exit__', ['etype', 'evalue', 'etrace'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__exit__', localization, ['etype', 'evalue', 'etrace'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__exit__(...)' code ##################

        unicode_48156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, (-1)), 'unicode', u'\n        Context manager exit method, closes the underlying file if it is open.\n        ')
        
        # Call to close(...): (line 230)
        # Processing the call keyword arguments (line 230)
        kwargs_48159 = {}
        # Getting the type of 'self' (line 230)
        self_48157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 8), 'self', False)
        # Obtaining the member 'close' of a type (line 230)
        close_48158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 8), self_48157, 'close')
        # Calling close(args, kwargs) (line 230)
        close_call_result_48160 = invoke(stypy.reporting.localization.Localization(__file__, 230, 8), close_48158, *[], **kwargs_48159)
        
        
        # ################# End of '__exit__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__exit__' in the type store
        # Getting the type of 'stypy_return_type' (line 226)
        stypy_return_type_48161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_48161)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__exit__'
        return stypy_return_type_48161


    @norecursion
    def __iter__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__iter__'
        module_type_store = module_type_store.open_function_context('__iter__', 232, 4, False)
        # Assigning a type to the variable 'self' (line 233)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Dvi.__iter__.__dict__.__setitem__('stypy_localization', localization)
        Dvi.__iter__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Dvi.__iter__.__dict__.__setitem__('stypy_type_store', module_type_store)
        Dvi.__iter__.__dict__.__setitem__('stypy_function_name', 'Dvi.__iter__')
        Dvi.__iter__.__dict__.__setitem__('stypy_param_names_list', [])
        Dvi.__iter__.__dict__.__setitem__('stypy_varargs_param_name', None)
        Dvi.__iter__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Dvi.__iter__.__dict__.__setitem__('stypy_call_defaults', defaults)
        Dvi.__iter__.__dict__.__setitem__('stypy_call_varargs', varargs)
        Dvi.__iter__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Dvi.__iter__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Dvi.__iter__', [], None, None, defaults, varargs, kwargs)

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

        unicode_48162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, (-1)), 'unicode', u'\n        Iterate through the pages of the file.\n\n        Yields\n        ------\n        Page\n            Details of all the text and box objects on the page.\n            The Page tuple contains lists of Text and Box tuples and\n            the page dimensions, and the Text and Box tuples contain\n            coordinates transformed into a standard Cartesian\n            coordinate system at the dpi value given when initializing.\n            The coordinates are floating point numbers, but otherwise\n            precision is not lost and coordinate values are not clipped to\n            integers.\n        ')
        
        # Getting the type of 'True' (line 248)
        True_48163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 14), 'True')
        # Testing the type of an if condition (line 248)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 248, 8), True_48163)
        # SSA begins for while statement (line 248)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Assigning a Call to a Name (line 249):
        
        # Assigning a Call to a Name (line 249):
        
        # Call to _read(...): (line 249)
        # Processing the call keyword arguments (line 249)
        kwargs_48166 = {}
        # Getting the type of 'self' (line 249)
        self_48164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 24), 'self', False)
        # Obtaining the member '_read' of a type (line 249)
        _read_48165 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 24), self_48164, '_read')
        # Calling _read(args, kwargs) (line 249)
        _read_call_result_48167 = invoke(stypy.reporting.localization.Localization(__file__, 249, 24), _read_48165, *[], **kwargs_48166)
        
        # Assigning a type to the variable 'have_page' (line 249)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 12), 'have_page', _read_call_result_48167)
        
        # Getting the type of 'have_page' (line 250)
        have_page_48168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 15), 'have_page')
        # Testing the type of an if condition (line 250)
        if_condition_48169 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 250, 12), have_page_48168)
        # Assigning a type to the variable 'if_condition_48169' (line 250)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 12), 'if_condition_48169', if_condition_48169)
        # SSA begins for if statement (line 250)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Creating a generator
        
        # Call to _output(...): (line 251)
        # Processing the call keyword arguments (line 251)
        kwargs_48172 = {}
        # Getting the type of 'self' (line 251)
        self_48170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 22), 'self', False)
        # Obtaining the member '_output' of a type (line 251)
        _output_48171 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 22), self_48170, '_output')
        # Calling _output(args, kwargs) (line 251)
        _output_call_result_48173 = invoke(stypy.reporting.localization.Localization(__file__, 251, 22), _output_48171, *[], **kwargs_48172)
        
        GeneratorType_48174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 16), 'GeneratorType')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 16), GeneratorType_48174, _output_call_result_48173)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 16), 'stypy_return_type', GeneratorType_48174)
        # SSA branch for the else part of an if statement (line 250)
        module_type_store.open_ssa_branch('else')
        # SSA join for if statement (line 250)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for while statement (line 248)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__iter__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__iter__' in the type store
        # Getting the type of 'stypy_return_type' (line 232)
        stypy_return_type_48175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_48175)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__iter__'
        return stypy_return_type_48175


    @norecursion
    def close(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'close'
        module_type_store = module_type_store.open_function_context('close', 255, 4, False)
        # Assigning a type to the variable 'self' (line 256)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Dvi.close.__dict__.__setitem__('stypy_localization', localization)
        Dvi.close.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Dvi.close.__dict__.__setitem__('stypy_type_store', module_type_store)
        Dvi.close.__dict__.__setitem__('stypy_function_name', 'Dvi.close')
        Dvi.close.__dict__.__setitem__('stypy_param_names_list', [])
        Dvi.close.__dict__.__setitem__('stypy_varargs_param_name', None)
        Dvi.close.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Dvi.close.__dict__.__setitem__('stypy_call_defaults', defaults)
        Dvi.close.__dict__.__setitem__('stypy_call_varargs', varargs)
        Dvi.close.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Dvi.close.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Dvi.close', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'close', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'close(...)' code ##################

        unicode_48176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, (-1)), 'unicode', u'\n        Close the underlying file if it is open.\n        ')
        
        
        # Getting the type of 'self' (line 259)
        self_48177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 15), 'self')
        # Obtaining the member 'file' of a type (line 259)
        file_48178 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 15), self_48177, 'file')
        # Obtaining the member 'closed' of a type (line 259)
        closed_48179 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 15), file_48178, 'closed')
        # Applying the 'not' unary operator (line 259)
        result_not__48180 = python_operator(stypy.reporting.localization.Localization(__file__, 259, 11), 'not', closed_48179)
        
        # Testing the type of an if condition (line 259)
        if_condition_48181 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 259, 8), result_not__48180)
        # Assigning a type to the variable 'if_condition_48181' (line 259)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 8), 'if_condition_48181', if_condition_48181)
        # SSA begins for if statement (line 259)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to close(...): (line 260)
        # Processing the call keyword arguments (line 260)
        kwargs_48185 = {}
        # Getting the type of 'self' (line 260)
        self_48182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 12), 'self', False)
        # Obtaining the member 'file' of a type (line 260)
        file_48183 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 12), self_48182, 'file')
        # Obtaining the member 'close' of a type (line 260)
        close_48184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 12), file_48183, 'close')
        # Calling close(args, kwargs) (line 260)
        close_call_result_48186 = invoke(stypy.reporting.localization.Localization(__file__, 260, 12), close_48184, *[], **kwargs_48185)
        
        # SSA join for if statement (line 259)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'close(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'close' in the type store
        # Getting the type of 'stypy_return_type' (line 255)
        stypy_return_type_48187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_48187)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'close'
        return stypy_return_type_48187


    @norecursion
    def _output(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_output'
        module_type_store = module_type_store.open_function_context('_output', 262, 4, False)
        # Assigning a type to the variable 'self' (line 263)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Dvi._output.__dict__.__setitem__('stypy_localization', localization)
        Dvi._output.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Dvi._output.__dict__.__setitem__('stypy_type_store', module_type_store)
        Dvi._output.__dict__.__setitem__('stypy_function_name', 'Dvi._output')
        Dvi._output.__dict__.__setitem__('stypy_param_names_list', [])
        Dvi._output.__dict__.__setitem__('stypy_varargs_param_name', None)
        Dvi._output.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Dvi._output.__dict__.__setitem__('stypy_call_defaults', defaults)
        Dvi._output.__dict__.__setitem__('stypy_call_varargs', varargs)
        Dvi._output.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Dvi._output.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Dvi._output', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_output', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_output(...)' code ##################

        unicode_48188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, (-1)), 'unicode', u'\n        Output the text and boxes belonging to the most recent page.\n        page = dvi._output()\n        ')
        
        # Assigning a Tuple to a Tuple (line 267):
        
        # Assigning a Attribute to a Name (line 267):
        # Getting the type of 'np' (line 267)
        np_48189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 33), 'np')
        # Obtaining the member 'inf' of a type (line 267)
        inf_48190 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 267, 33), np_48189, 'inf')
        # Assigning a type to the variable 'tuple_assignment_47762' (line 267)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 8), 'tuple_assignment_47762', inf_48190)
        
        # Assigning a Attribute to a Name (line 267):
        # Getting the type of 'np' (line 267)
        np_48191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 41), 'np')
        # Obtaining the member 'inf' of a type (line 267)
        inf_48192 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 267, 41), np_48191, 'inf')
        # Assigning a type to the variable 'tuple_assignment_47763' (line 267)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 8), 'tuple_assignment_47763', inf_48192)
        
        # Assigning a UnaryOp to a Name (line 267):
        
        # Getting the type of 'np' (line 267)
        np_48193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 50), 'np')
        # Obtaining the member 'inf' of a type (line 267)
        inf_48194 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 267, 50), np_48193, 'inf')
        # Applying the 'usub' unary operator (line 267)
        result___neg___48195 = python_operator(stypy.reporting.localization.Localization(__file__, 267, 49), 'usub', inf_48194)
        
        # Assigning a type to the variable 'tuple_assignment_47764' (line 267)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 8), 'tuple_assignment_47764', result___neg___48195)
        
        # Assigning a UnaryOp to a Name (line 267):
        
        # Getting the type of 'np' (line 267)
        np_48196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 59), 'np')
        # Obtaining the member 'inf' of a type (line 267)
        inf_48197 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 267, 59), np_48196, 'inf')
        # Applying the 'usub' unary operator (line 267)
        result___neg___48198 = python_operator(stypy.reporting.localization.Localization(__file__, 267, 58), 'usub', inf_48197)
        
        # Assigning a type to the variable 'tuple_assignment_47765' (line 267)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 8), 'tuple_assignment_47765', result___neg___48198)
        
        # Assigning a Name to a Name (line 267):
        # Getting the type of 'tuple_assignment_47762' (line 267)
        tuple_assignment_47762_48199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 8), 'tuple_assignment_47762')
        # Assigning a type to the variable 'minx' (line 267)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 8), 'minx', tuple_assignment_47762_48199)
        
        # Assigning a Name to a Name (line 267):
        # Getting the type of 'tuple_assignment_47763' (line 267)
        tuple_assignment_47763_48200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 8), 'tuple_assignment_47763')
        # Assigning a type to the variable 'miny' (line 267)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 14), 'miny', tuple_assignment_47763_48200)
        
        # Assigning a Name to a Name (line 267):
        # Getting the type of 'tuple_assignment_47764' (line 267)
        tuple_assignment_47764_48201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 8), 'tuple_assignment_47764')
        # Assigning a type to the variable 'maxx' (line 267)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 20), 'maxx', tuple_assignment_47764_48201)
        
        # Assigning a Name to a Name (line 267):
        # Getting the type of 'tuple_assignment_47765' (line 267)
        tuple_assignment_47765_48202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 8), 'tuple_assignment_47765')
        # Assigning a type to the variable 'maxy' (line 267)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 26), 'maxy', tuple_assignment_47765_48202)
        
        # Assigning a UnaryOp to a Name (line 268):
        
        # Assigning a UnaryOp to a Name (line 268):
        
        # Getting the type of 'np' (line 268)
        np_48203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 21), 'np')
        # Obtaining the member 'inf' of a type (line 268)
        inf_48204 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 21), np_48203, 'inf')
        # Applying the 'usub' unary operator (line 268)
        result___neg___48205 = python_operator(stypy.reporting.localization.Localization(__file__, 268, 20), 'usub', inf_48204)
        
        # Assigning a type to the variable 'maxy_pure' (line 268)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 8), 'maxy_pure', result___neg___48205)
        
        # Getting the type of 'self' (line 269)
        self_48206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 19), 'self')
        # Obtaining the member 'text' of a type (line 269)
        text_48207 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 19), self_48206, 'text')
        # Getting the type of 'self' (line 269)
        self_48208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 31), 'self')
        # Obtaining the member 'boxes' of a type (line 269)
        boxes_48209 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 31), self_48208, 'boxes')
        # Applying the binary operator '+' (line 269)
        result_add_48210 = python_operator(stypy.reporting.localization.Localization(__file__, 269, 19), '+', text_48207, boxes_48209)
        
        # Testing the type of a for loop iterable (line 269)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 269, 8), result_add_48210)
        # Getting the type of the for loop variable (line 269)
        for_loop_var_48211 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 269, 8), result_add_48210)
        # Assigning a type to the variable 'elt' (line 269)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 8), 'elt', for_loop_var_48211)
        # SSA begins for a for statement (line 269)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Call to isinstance(...): (line 270)
        # Processing the call arguments (line 270)
        # Getting the type of 'elt' (line 270)
        elt_48213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 26), 'elt', False)
        # Getting the type of 'Box' (line 270)
        Box_48214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 31), 'Box', False)
        # Processing the call keyword arguments (line 270)
        kwargs_48215 = {}
        # Getting the type of 'isinstance' (line 270)
        isinstance_48212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 15), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 270)
        isinstance_call_result_48216 = invoke(stypy.reporting.localization.Localization(__file__, 270, 15), isinstance_48212, *[elt_48213, Box_48214], **kwargs_48215)
        
        # Testing the type of an if condition (line 270)
        if_condition_48217 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 270, 12), isinstance_call_result_48216)
        # Assigning a type to the variable 'if_condition_48217' (line 270)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 270, 12), 'if_condition_48217', if_condition_48217)
        # SSA begins for if statement (line 270)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Tuple (line 271):
        
        # Assigning a Subscript to a Name (line 271):
        
        # Obtaining the type of the subscript
        int_48218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 16), 'int')
        # Getting the type of 'elt' (line 271)
        elt_48219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 29), 'elt')
        # Obtaining the member '__getitem__' of a type (line 271)
        getitem___48220 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 16), elt_48219, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 271)
        subscript_call_result_48221 = invoke(stypy.reporting.localization.Localization(__file__, 271, 16), getitem___48220, int_48218)
        
        # Assigning a type to the variable 'tuple_var_assignment_47766' (line 271)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 16), 'tuple_var_assignment_47766', subscript_call_result_48221)
        
        # Assigning a Subscript to a Name (line 271):
        
        # Obtaining the type of the subscript
        int_48222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 16), 'int')
        # Getting the type of 'elt' (line 271)
        elt_48223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 29), 'elt')
        # Obtaining the member '__getitem__' of a type (line 271)
        getitem___48224 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 16), elt_48223, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 271)
        subscript_call_result_48225 = invoke(stypy.reporting.localization.Localization(__file__, 271, 16), getitem___48224, int_48222)
        
        # Assigning a type to the variable 'tuple_var_assignment_47767' (line 271)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 16), 'tuple_var_assignment_47767', subscript_call_result_48225)
        
        # Assigning a Subscript to a Name (line 271):
        
        # Obtaining the type of the subscript
        int_48226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 16), 'int')
        # Getting the type of 'elt' (line 271)
        elt_48227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 29), 'elt')
        # Obtaining the member '__getitem__' of a type (line 271)
        getitem___48228 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 16), elt_48227, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 271)
        subscript_call_result_48229 = invoke(stypy.reporting.localization.Localization(__file__, 271, 16), getitem___48228, int_48226)
        
        # Assigning a type to the variable 'tuple_var_assignment_47768' (line 271)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 16), 'tuple_var_assignment_47768', subscript_call_result_48229)
        
        # Assigning a Subscript to a Name (line 271):
        
        # Obtaining the type of the subscript
        int_48230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 16), 'int')
        # Getting the type of 'elt' (line 271)
        elt_48231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 29), 'elt')
        # Obtaining the member '__getitem__' of a type (line 271)
        getitem___48232 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 16), elt_48231, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 271)
        subscript_call_result_48233 = invoke(stypy.reporting.localization.Localization(__file__, 271, 16), getitem___48232, int_48230)
        
        # Assigning a type to the variable 'tuple_var_assignment_47769' (line 271)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 16), 'tuple_var_assignment_47769', subscript_call_result_48233)
        
        # Assigning a Name to a Name (line 271):
        # Getting the type of 'tuple_var_assignment_47766' (line 271)
        tuple_var_assignment_47766_48234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 16), 'tuple_var_assignment_47766')
        # Assigning a type to the variable 'x' (line 271)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 16), 'x', tuple_var_assignment_47766_48234)
        
        # Assigning a Name to a Name (line 271):
        # Getting the type of 'tuple_var_assignment_47767' (line 271)
        tuple_var_assignment_47767_48235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 16), 'tuple_var_assignment_47767')
        # Assigning a type to the variable 'y' (line 271)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 19), 'y', tuple_var_assignment_47767_48235)
        
        # Assigning a Name to a Name (line 271):
        # Getting the type of 'tuple_var_assignment_47768' (line 271)
        tuple_var_assignment_47768_48236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 16), 'tuple_var_assignment_47768')
        # Assigning a type to the variable 'h' (line 271)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 22), 'h', tuple_var_assignment_47768_48236)
        
        # Assigning a Name to a Name (line 271):
        # Getting the type of 'tuple_var_assignment_47769' (line 271)
        tuple_var_assignment_47769_48237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 16), 'tuple_var_assignment_47769')
        # Assigning a type to the variable 'w' (line 271)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 25), 'w', tuple_var_assignment_47769_48237)
        
        # Assigning a Num to a Name (line 272):
        
        # Assigning a Num to a Name (line 272):
        int_48238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 20), 'int')
        # Assigning a type to the variable 'e' (line 272)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 16), 'e', int_48238)
        # SSA branch for the else part of an if statement (line 270)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Name to a Tuple (line 274):
        
        # Assigning a Subscript to a Name (line 274):
        
        # Obtaining the type of the subscript
        int_48239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 16), 'int')
        # Getting the type of 'elt' (line 274)
        elt_48240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 35), 'elt')
        # Obtaining the member '__getitem__' of a type (line 274)
        getitem___48241 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 16), elt_48240, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 274)
        subscript_call_result_48242 = invoke(stypy.reporting.localization.Localization(__file__, 274, 16), getitem___48241, int_48239)
        
        # Assigning a type to the variable 'tuple_var_assignment_47770' (line 274)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 16), 'tuple_var_assignment_47770', subscript_call_result_48242)
        
        # Assigning a Subscript to a Name (line 274):
        
        # Obtaining the type of the subscript
        int_48243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 16), 'int')
        # Getting the type of 'elt' (line 274)
        elt_48244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 35), 'elt')
        # Obtaining the member '__getitem__' of a type (line 274)
        getitem___48245 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 16), elt_48244, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 274)
        subscript_call_result_48246 = invoke(stypy.reporting.localization.Localization(__file__, 274, 16), getitem___48245, int_48243)
        
        # Assigning a type to the variable 'tuple_var_assignment_47771' (line 274)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 16), 'tuple_var_assignment_47771', subscript_call_result_48246)
        
        # Assigning a Subscript to a Name (line 274):
        
        # Obtaining the type of the subscript
        int_48247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 16), 'int')
        # Getting the type of 'elt' (line 274)
        elt_48248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 35), 'elt')
        # Obtaining the member '__getitem__' of a type (line 274)
        getitem___48249 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 16), elt_48248, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 274)
        subscript_call_result_48250 = invoke(stypy.reporting.localization.Localization(__file__, 274, 16), getitem___48249, int_48247)
        
        # Assigning a type to the variable 'tuple_var_assignment_47772' (line 274)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 16), 'tuple_var_assignment_47772', subscript_call_result_48250)
        
        # Assigning a Subscript to a Name (line 274):
        
        # Obtaining the type of the subscript
        int_48251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 16), 'int')
        # Getting the type of 'elt' (line 274)
        elt_48252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 35), 'elt')
        # Obtaining the member '__getitem__' of a type (line 274)
        getitem___48253 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 16), elt_48252, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 274)
        subscript_call_result_48254 = invoke(stypy.reporting.localization.Localization(__file__, 274, 16), getitem___48253, int_48251)
        
        # Assigning a type to the variable 'tuple_var_assignment_47773' (line 274)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 16), 'tuple_var_assignment_47773', subscript_call_result_48254)
        
        # Assigning a Subscript to a Name (line 274):
        
        # Obtaining the type of the subscript
        int_48255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 16), 'int')
        # Getting the type of 'elt' (line 274)
        elt_48256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 35), 'elt')
        # Obtaining the member '__getitem__' of a type (line 274)
        getitem___48257 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 16), elt_48256, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 274)
        subscript_call_result_48258 = invoke(stypy.reporting.localization.Localization(__file__, 274, 16), getitem___48257, int_48255)
        
        # Assigning a type to the variable 'tuple_var_assignment_47774' (line 274)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 16), 'tuple_var_assignment_47774', subscript_call_result_48258)
        
        # Assigning a Name to a Name (line 274):
        # Getting the type of 'tuple_var_assignment_47770' (line 274)
        tuple_var_assignment_47770_48259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 16), 'tuple_var_assignment_47770')
        # Assigning a type to the variable 'x' (line 274)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 16), 'x', tuple_var_assignment_47770_48259)
        
        # Assigning a Name to a Name (line 274):
        # Getting the type of 'tuple_var_assignment_47771' (line 274)
        tuple_var_assignment_47771_48260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 16), 'tuple_var_assignment_47771')
        # Assigning a type to the variable 'y' (line 274)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 19), 'y', tuple_var_assignment_47771_48260)
        
        # Assigning a Name to a Name (line 274):
        # Getting the type of 'tuple_var_assignment_47772' (line 274)
        tuple_var_assignment_47772_48261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 16), 'tuple_var_assignment_47772')
        # Assigning a type to the variable 'font' (line 274)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 22), 'font', tuple_var_assignment_47772_48261)
        
        # Assigning a Name to a Name (line 274):
        # Getting the type of 'tuple_var_assignment_47773' (line 274)
        tuple_var_assignment_47773_48262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 16), 'tuple_var_assignment_47773')
        # Assigning a type to the variable 'g' (line 274)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 28), 'g', tuple_var_assignment_47773_48262)
        
        # Assigning a Name to a Name (line 274):
        # Getting the type of 'tuple_var_assignment_47774' (line 274)
        tuple_var_assignment_47774_48263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 16), 'tuple_var_assignment_47774')
        # Assigning a type to the variable 'w' (line 274)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 31), 'w', tuple_var_assignment_47774_48263)
        
        # Assigning a Call to a Tuple (line 275):
        
        # Assigning a Call to a Name:
        
        # Call to _height_depth_of(...): (line 275)
        # Processing the call arguments (line 275)
        # Getting the type of 'g' (line 275)
        g_48266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 45), 'g', False)
        # Processing the call keyword arguments (line 275)
        kwargs_48267 = {}
        # Getting the type of 'font' (line 275)
        font_48264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 23), 'font', False)
        # Obtaining the member '_height_depth_of' of a type (line 275)
        _height_depth_of_48265 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 23), font_48264, '_height_depth_of')
        # Calling _height_depth_of(args, kwargs) (line 275)
        _height_depth_of_call_result_48268 = invoke(stypy.reporting.localization.Localization(__file__, 275, 23), _height_depth_of_48265, *[g_48266], **kwargs_48267)
        
        # Assigning a type to the variable 'call_assignment_47775' (line 275)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 16), 'call_assignment_47775', _height_depth_of_call_result_48268)
        
        # Assigning a Call to a Name (line 275):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_48271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 16), 'int')
        # Processing the call keyword arguments
        kwargs_48272 = {}
        # Getting the type of 'call_assignment_47775' (line 275)
        call_assignment_47775_48269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 16), 'call_assignment_47775', False)
        # Obtaining the member '__getitem__' of a type (line 275)
        getitem___48270 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 16), call_assignment_47775_48269, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_48273 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___48270, *[int_48271], **kwargs_48272)
        
        # Assigning a type to the variable 'call_assignment_47776' (line 275)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 16), 'call_assignment_47776', getitem___call_result_48273)
        
        # Assigning a Name to a Name (line 275):
        # Getting the type of 'call_assignment_47776' (line 275)
        call_assignment_47776_48274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 16), 'call_assignment_47776')
        # Assigning a type to the variable 'h' (line 275)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 16), 'h', call_assignment_47776_48274)
        
        # Assigning a Call to a Name (line 275):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_48277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 16), 'int')
        # Processing the call keyword arguments
        kwargs_48278 = {}
        # Getting the type of 'call_assignment_47775' (line 275)
        call_assignment_47775_48275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 16), 'call_assignment_47775', False)
        # Obtaining the member '__getitem__' of a type (line 275)
        getitem___48276 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 16), call_assignment_47775_48275, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_48279 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___48276, *[int_48277], **kwargs_48278)
        
        # Assigning a type to the variable 'call_assignment_47777' (line 275)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 16), 'call_assignment_47777', getitem___call_result_48279)
        
        # Assigning a Name to a Name (line 275):
        # Getting the type of 'call_assignment_47777' (line 275)
        call_assignment_47777_48280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 16), 'call_assignment_47777')
        # Assigning a type to the variable 'e' (line 275)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 19), 'e', call_assignment_47777_48280)
        # SSA join for if statement (line 270)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 276):
        
        # Assigning a Call to a Name (line 276):
        
        # Call to min(...): (line 276)
        # Processing the call arguments (line 276)
        # Getting the type of 'minx' (line 276)
        minx_48282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 23), 'minx', False)
        # Getting the type of 'x' (line 276)
        x_48283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 29), 'x', False)
        # Processing the call keyword arguments (line 276)
        kwargs_48284 = {}
        # Getting the type of 'min' (line 276)
        min_48281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 19), 'min', False)
        # Calling min(args, kwargs) (line 276)
        min_call_result_48285 = invoke(stypy.reporting.localization.Localization(__file__, 276, 19), min_48281, *[minx_48282, x_48283], **kwargs_48284)
        
        # Assigning a type to the variable 'minx' (line 276)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 12), 'minx', min_call_result_48285)
        
        # Assigning a Call to a Name (line 277):
        
        # Assigning a Call to a Name (line 277):
        
        # Call to min(...): (line 277)
        # Processing the call arguments (line 277)
        # Getting the type of 'miny' (line 277)
        miny_48287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 23), 'miny', False)
        # Getting the type of 'y' (line 277)
        y_48288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 29), 'y', False)
        # Getting the type of 'h' (line 277)
        h_48289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 33), 'h', False)
        # Applying the binary operator '-' (line 277)
        result_sub_48290 = python_operator(stypy.reporting.localization.Localization(__file__, 277, 29), '-', y_48288, h_48289)
        
        # Processing the call keyword arguments (line 277)
        kwargs_48291 = {}
        # Getting the type of 'min' (line 277)
        min_48286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 19), 'min', False)
        # Calling min(args, kwargs) (line 277)
        min_call_result_48292 = invoke(stypy.reporting.localization.Localization(__file__, 277, 19), min_48286, *[miny_48287, result_sub_48290], **kwargs_48291)
        
        # Assigning a type to the variable 'miny' (line 277)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 12), 'miny', min_call_result_48292)
        
        # Assigning a Call to a Name (line 278):
        
        # Assigning a Call to a Name (line 278):
        
        # Call to max(...): (line 278)
        # Processing the call arguments (line 278)
        # Getting the type of 'maxx' (line 278)
        maxx_48294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 23), 'maxx', False)
        # Getting the type of 'x' (line 278)
        x_48295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 29), 'x', False)
        # Getting the type of 'w' (line 278)
        w_48296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 33), 'w', False)
        # Applying the binary operator '+' (line 278)
        result_add_48297 = python_operator(stypy.reporting.localization.Localization(__file__, 278, 29), '+', x_48295, w_48296)
        
        # Processing the call keyword arguments (line 278)
        kwargs_48298 = {}
        # Getting the type of 'max' (line 278)
        max_48293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 19), 'max', False)
        # Calling max(args, kwargs) (line 278)
        max_call_result_48299 = invoke(stypy.reporting.localization.Localization(__file__, 278, 19), max_48293, *[maxx_48294, result_add_48297], **kwargs_48298)
        
        # Assigning a type to the variable 'maxx' (line 278)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 12), 'maxx', max_call_result_48299)
        
        # Assigning a Call to a Name (line 279):
        
        # Assigning a Call to a Name (line 279):
        
        # Call to max(...): (line 279)
        # Processing the call arguments (line 279)
        # Getting the type of 'maxy' (line 279)
        maxy_48301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 23), 'maxy', False)
        # Getting the type of 'y' (line 279)
        y_48302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 29), 'y', False)
        # Getting the type of 'e' (line 279)
        e_48303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 33), 'e', False)
        # Applying the binary operator '+' (line 279)
        result_add_48304 = python_operator(stypy.reporting.localization.Localization(__file__, 279, 29), '+', y_48302, e_48303)
        
        # Processing the call keyword arguments (line 279)
        kwargs_48305 = {}
        # Getting the type of 'max' (line 279)
        max_48300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 19), 'max', False)
        # Calling max(args, kwargs) (line 279)
        max_call_result_48306 = invoke(stypy.reporting.localization.Localization(__file__, 279, 19), max_48300, *[maxy_48301, result_add_48304], **kwargs_48305)
        
        # Assigning a type to the variable 'maxy' (line 279)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 12), 'maxy', max_call_result_48306)
        
        # Assigning a Call to a Name (line 280):
        
        # Assigning a Call to a Name (line 280):
        
        # Call to max(...): (line 280)
        # Processing the call arguments (line 280)
        # Getting the type of 'maxy_pure' (line 280)
        maxy_pure_48308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 28), 'maxy_pure', False)
        # Getting the type of 'y' (line 280)
        y_48309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 39), 'y', False)
        # Processing the call keyword arguments (line 280)
        kwargs_48310 = {}
        # Getting the type of 'max' (line 280)
        max_48307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 24), 'max', False)
        # Calling max(args, kwargs) (line 280)
        max_call_result_48311 = invoke(stypy.reporting.localization.Localization(__file__, 280, 24), max_48307, *[maxy_pure_48308, y_48309], **kwargs_48310)
        
        # Assigning a type to the variable 'maxy_pure' (line 280)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 12), 'maxy_pure', max_call_result_48311)
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Type idiom detected: calculating its left and rigth part (line 282)
        # Getting the type of 'self' (line 282)
        self_48312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 11), 'self')
        # Obtaining the member 'dpi' of a type (line 282)
        dpi_48313 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 11), self_48312, 'dpi')
        # Getting the type of 'None' (line 282)
        None_48314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 23), 'None')
        
        (may_be_48315, more_types_in_union_48316) = may_be_none(dpi_48313, None_48314)

        if may_be_48315:

            if more_types_in_union_48316:
                # Runtime conditional SSA (line 282)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to Page(...): (line 284)
            # Processing the call keyword arguments (line 284)
            # Getting the type of 'self' (line 284)
            self_48318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 29), 'self', False)
            # Obtaining the member 'text' of a type (line 284)
            text_48319 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 29), self_48318, 'text')
            keyword_48320 = text_48319
            # Getting the type of 'self' (line 284)
            self_48321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 46), 'self', False)
            # Obtaining the member 'boxes' of a type (line 284)
            boxes_48322 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 46), self_48321, 'boxes')
            keyword_48323 = boxes_48322
            # Getting the type of 'maxx' (line 285)
            maxx_48324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 30), 'maxx', False)
            # Getting the type of 'minx' (line 285)
            minx_48325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 35), 'minx', False)
            # Applying the binary operator '-' (line 285)
            result_sub_48326 = python_operator(stypy.reporting.localization.Localization(__file__, 285, 30), '-', maxx_48324, minx_48325)
            
            keyword_48327 = result_sub_48326
            # Getting the type of 'maxy_pure' (line 285)
            maxy_pure_48328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 48), 'maxy_pure', False)
            # Getting the type of 'miny' (line 285)
            miny_48329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 58), 'miny', False)
            # Applying the binary operator '-' (line 285)
            result_sub_48330 = python_operator(stypy.reporting.localization.Localization(__file__, 285, 48), '-', maxy_pure_48328, miny_48329)
            
            keyword_48331 = result_sub_48330
            # Getting the type of 'maxy' (line 286)
            maxy_48332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 32), 'maxy', False)
            # Getting the type of 'maxy_pure' (line 286)
            maxy_pure_48333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 37), 'maxy_pure', False)
            # Applying the binary operator '-' (line 286)
            result_sub_48334 = python_operator(stypy.reporting.localization.Localization(__file__, 286, 32), '-', maxy_48332, maxy_pure_48333)
            
            keyword_48335 = result_sub_48334
            kwargs_48336 = {'text': keyword_48320, 'boxes': keyword_48323, 'height': keyword_48331, 'descent': keyword_48335, 'width': keyword_48327}
            # Getting the type of 'Page' (line 284)
            Page_48317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 19), 'Page', False)
            # Calling Page(args, kwargs) (line 284)
            Page_call_result_48337 = invoke(stypy.reporting.localization.Localization(__file__, 284, 19), Page_48317, *[], **kwargs_48336)
            
            # Assigning a type to the variable 'stypy_return_type' (line 284)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 12), 'stypy_return_type', Page_call_result_48337)

            if more_types_in_union_48316:
                # SSA join for if statement (line 282)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a BinOp to a Name (line 289):
        
        # Assigning a BinOp to a Name (line 289):
        # Getting the type of 'self' (line 289)
        self_48338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 12), 'self')
        # Obtaining the member 'dpi' of a type (line 289)
        dpi_48339 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 12), self_48338, 'dpi')
        float_48340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 24), 'float')
        int_48341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 32), 'int')
        int_48342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 35), 'int')
        # Applying the binary operator '**' (line 289)
        result_pow_48343 = python_operator(stypy.reporting.localization.Localization(__file__, 289, 32), '**', int_48341, int_48342)
        
        # Applying the binary operator '*' (line 289)
        result_mul_48344 = python_operator(stypy.reporting.localization.Localization(__file__, 289, 24), '*', float_48340, result_pow_48343)
        
        # Applying the binary operator 'div' (line 289)
        result_div_48345 = python_operator(stypy.reporting.localization.Localization(__file__, 289, 12), 'div', dpi_48339, result_mul_48344)
        
        # Assigning a type to the variable 'd' (line 289)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 8), 'd', result_div_48345)
        
        # Type idiom detected: calculating its left and rigth part (line 290)
        # Getting the type of 'self' (line 290)
        self_48346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 11), 'self')
        # Obtaining the member 'baseline' of a type (line 290)
        baseline_48347 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 11), self_48346, 'baseline')
        # Getting the type of 'None' (line 290)
        None_48348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 28), 'None')
        
        (may_be_48349, more_types_in_union_48350) = may_be_none(baseline_48347, None_48348)

        if may_be_48349:

            if more_types_in_union_48350:
                # Runtime conditional SSA (line 290)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a BinOp to a Name (line 291):
            
            # Assigning a BinOp to a Name (line 291):
            # Getting the type of 'maxy' (line 291)
            maxy_48351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 23), 'maxy')
            # Getting the type of 'maxy_pure' (line 291)
            maxy_pure_48352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 30), 'maxy_pure')
            # Applying the binary operator '-' (line 291)
            result_sub_48353 = python_operator(stypy.reporting.localization.Localization(__file__, 291, 23), '-', maxy_48351, maxy_pure_48352)
            
            # Getting the type of 'd' (line 291)
            d_48354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 43), 'd')
            # Applying the binary operator '*' (line 291)
            result_mul_48355 = python_operator(stypy.reporting.localization.Localization(__file__, 291, 22), '*', result_sub_48353, d_48354)
            
            # Assigning a type to the variable 'descent' (line 291)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 12), 'descent', result_mul_48355)

            if more_types_in_union_48350:
                # Runtime conditional SSA for else branch (line 290)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_48349) or more_types_in_union_48350):
            
            # Assigning a Attribute to a Name (line 293):
            
            # Assigning a Attribute to a Name (line 293):
            # Getting the type of 'self' (line 293)
            self_48356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 22), 'self')
            # Obtaining the member 'baseline' of a type (line 293)
            baseline_48357 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 22), self_48356, 'baseline')
            # Assigning a type to the variable 'descent' (line 293)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 12), 'descent', baseline_48357)

            if (may_be_48349 and more_types_in_union_48350):
                # SSA join for if statement (line 290)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a ListComp to a Name (line 295):
        
        # Assigning a ListComp to a Name (line 295):
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'self' (line 296)
        self_48378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 39), 'self')
        # Obtaining the member 'text' of a type (line 296)
        text_48379 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 39), self_48378, 'text')
        comprehension_48380 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 16), text_48379)
        # Assigning a type to the variable 'x' (line 295)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 16), 'x', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 16), comprehension_48380))
        # Assigning a type to the variable 'y' (line 295)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 16), 'y', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 16), comprehension_48380))
        # Assigning a type to the variable 'f' (line 295)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 16), 'f', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 16), comprehension_48380))
        # Assigning a type to the variable 'g' (line 295)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 16), 'g', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 16), comprehension_48380))
        # Assigning a type to the variable 'w' (line 295)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 16), 'w', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 16), comprehension_48380))
        
        # Call to Text(...): (line 295)
        # Processing the call arguments (line 295)
        # Getting the type of 'x' (line 295)
        x_48359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 22), 'x', False)
        # Getting the type of 'minx' (line 295)
        minx_48360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 24), 'minx', False)
        # Applying the binary operator '-' (line 295)
        result_sub_48361 = python_operator(stypy.reporting.localization.Localization(__file__, 295, 22), '-', x_48359, minx_48360)
        
        # Getting the type of 'd' (line 295)
        d_48362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 30), 'd', False)
        # Applying the binary operator '*' (line 295)
        result_mul_48363 = python_operator(stypy.reporting.localization.Localization(__file__, 295, 21), '*', result_sub_48361, d_48362)
        
        # Getting the type of 'maxy' (line 295)
        maxy_48364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 34), 'maxy', False)
        # Getting the type of 'y' (line 295)
        y_48365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 39), 'y', False)
        # Applying the binary operator '-' (line 295)
        result_sub_48366 = python_operator(stypy.reporting.localization.Localization(__file__, 295, 34), '-', maxy_48364, y_48365)
        
        # Getting the type of 'd' (line 295)
        d_48367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 42), 'd', False)
        # Applying the binary operator '*' (line 295)
        result_mul_48368 = python_operator(stypy.reporting.localization.Localization(__file__, 295, 33), '*', result_sub_48366, d_48367)
        
        # Getting the type of 'descent' (line 295)
        descent_48369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 46), 'descent', False)
        # Applying the binary operator '-' (line 295)
        result_sub_48370 = python_operator(stypy.reporting.localization.Localization(__file__, 295, 33), '-', result_mul_48368, descent_48369)
        
        # Getting the type of 'f' (line 295)
        f_48371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 55), 'f', False)
        # Getting the type of 'g' (line 295)
        g_48372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 58), 'g', False)
        # Getting the type of 'w' (line 295)
        w_48373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 61), 'w', False)
        # Getting the type of 'd' (line 295)
        d_48374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 63), 'd', False)
        # Applying the binary operator '*' (line 295)
        result_mul_48375 = python_operator(stypy.reporting.localization.Localization(__file__, 295, 61), '*', w_48373, d_48374)
        
        # Processing the call keyword arguments (line 295)
        kwargs_48376 = {}
        # Getting the type of 'Text' (line 295)
        Text_48358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 16), 'Text', False)
        # Calling Text(args, kwargs) (line 295)
        Text_call_result_48377 = invoke(stypy.reporting.localization.Localization(__file__, 295, 16), Text_48358, *[result_mul_48363, result_sub_48370, f_48371, g_48372, result_mul_48375], **kwargs_48376)
        
        list_48381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 16), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 16), list_48381, Text_call_result_48377)
        # Assigning a type to the variable 'text' (line 295)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 8), 'text', list_48381)
        
        # Assigning a ListComp to a Name (line 297):
        
        # Assigning a ListComp to a Name (line 297):
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'self' (line 298)
        self_48403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 37), 'self')
        # Obtaining the member 'boxes' of a type (line 298)
        boxes_48404 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 37), self_48403, 'boxes')
        comprehension_48405 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 297, 17), boxes_48404)
        # Assigning a type to the variable 'x' (line 297)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 17), 'x', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 297, 17), comprehension_48405))
        # Assigning a type to the variable 'y' (line 297)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 17), 'y', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 297, 17), comprehension_48405))
        # Assigning a type to the variable 'h' (line 297)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 17), 'h', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 297, 17), comprehension_48405))
        # Assigning a type to the variable 'w' (line 297)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 17), 'w', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 297, 17), comprehension_48405))
        
        # Call to Box(...): (line 297)
        # Processing the call arguments (line 297)
        # Getting the type of 'x' (line 297)
        x_48383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 22), 'x', False)
        # Getting the type of 'minx' (line 297)
        minx_48384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 24), 'minx', False)
        # Applying the binary operator '-' (line 297)
        result_sub_48385 = python_operator(stypy.reporting.localization.Localization(__file__, 297, 22), '-', x_48383, minx_48384)
        
        # Getting the type of 'd' (line 297)
        d_48386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 30), 'd', False)
        # Applying the binary operator '*' (line 297)
        result_mul_48387 = python_operator(stypy.reporting.localization.Localization(__file__, 297, 21), '*', result_sub_48385, d_48386)
        
        # Getting the type of 'maxy' (line 297)
        maxy_48388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 34), 'maxy', False)
        # Getting the type of 'y' (line 297)
        y_48389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 39), 'y', False)
        # Applying the binary operator '-' (line 297)
        result_sub_48390 = python_operator(stypy.reporting.localization.Localization(__file__, 297, 34), '-', maxy_48388, y_48389)
        
        # Getting the type of 'd' (line 297)
        d_48391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 42), 'd', False)
        # Applying the binary operator '*' (line 297)
        result_mul_48392 = python_operator(stypy.reporting.localization.Localization(__file__, 297, 33), '*', result_sub_48390, d_48391)
        
        # Getting the type of 'descent' (line 297)
        descent_48393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 46), 'descent', False)
        # Applying the binary operator '-' (line 297)
        result_sub_48394 = python_operator(stypy.reporting.localization.Localization(__file__, 297, 33), '-', result_mul_48392, descent_48393)
        
        # Getting the type of 'h' (line 297)
        h_48395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 55), 'h', False)
        # Getting the type of 'd' (line 297)
        d_48396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 57), 'd', False)
        # Applying the binary operator '*' (line 297)
        result_mul_48397 = python_operator(stypy.reporting.localization.Localization(__file__, 297, 55), '*', h_48395, d_48396)
        
        # Getting the type of 'w' (line 297)
        w_48398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 60), 'w', False)
        # Getting the type of 'd' (line 297)
        d_48399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 62), 'd', False)
        # Applying the binary operator '*' (line 297)
        result_mul_48400 = python_operator(stypy.reporting.localization.Localization(__file__, 297, 60), '*', w_48398, d_48399)
        
        # Processing the call keyword arguments (line 297)
        kwargs_48401 = {}
        # Getting the type of 'Box' (line 297)
        Box_48382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 17), 'Box', False)
        # Calling Box(args, kwargs) (line 297)
        Box_call_result_48402 = invoke(stypy.reporting.localization.Localization(__file__, 297, 17), Box_48382, *[result_mul_48387, result_sub_48394, result_mul_48397, result_mul_48400], **kwargs_48401)
        
        list_48406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, 17), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 297, 17), list_48406, Box_call_result_48402)
        # Assigning a type to the variable 'boxes' (line 297)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 8), 'boxes', list_48406)
        
        # Call to Page(...): (line 300)
        # Processing the call keyword arguments (line 300)
        # Getting the type of 'text' (line 300)
        text_48408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 25), 'text', False)
        keyword_48409 = text_48408
        # Getting the type of 'boxes' (line 300)
        boxes_48410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 37), 'boxes', False)
        keyword_48411 = boxes_48410
        # Getting the type of 'maxx' (line 300)
        maxx_48412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 51), 'maxx', False)
        # Getting the type of 'minx' (line 300)
        minx_48413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 56), 'minx', False)
        # Applying the binary operator '-' (line 300)
        result_sub_48414 = python_operator(stypy.reporting.localization.Localization(__file__, 300, 51), '-', maxx_48412, minx_48413)
        
        # Getting the type of 'd' (line 300)
        d_48415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 62), 'd', False)
        # Applying the binary operator '*' (line 300)
        result_mul_48416 = python_operator(stypy.reporting.localization.Localization(__file__, 300, 50), '*', result_sub_48414, d_48415)
        
        keyword_48417 = result_mul_48416
        # Getting the type of 'maxy_pure' (line 301)
        maxy_pure_48418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 28), 'maxy_pure', False)
        # Getting the type of 'miny' (line 301)
        miny_48419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 38), 'miny', False)
        # Applying the binary operator '-' (line 301)
        result_sub_48420 = python_operator(stypy.reporting.localization.Localization(__file__, 301, 28), '-', maxy_pure_48418, miny_48419)
        
        # Getting the type of 'd' (line 301)
        d_48421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 44), 'd', False)
        # Applying the binary operator '*' (line 301)
        result_mul_48422 = python_operator(stypy.reporting.localization.Localization(__file__, 301, 27), '*', result_sub_48420, d_48421)
        
        keyword_48423 = result_mul_48422
        # Getting the type of 'descent' (line 301)
        descent_48424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 55), 'descent', False)
        keyword_48425 = descent_48424
        kwargs_48426 = {'text': keyword_48409, 'boxes': keyword_48411, 'height': keyword_48423, 'descent': keyword_48425, 'width': keyword_48417}
        # Getting the type of 'Page' (line 300)
        Page_48407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 15), 'Page', False)
        # Calling Page(args, kwargs) (line 300)
        Page_call_result_48427 = invoke(stypy.reporting.localization.Localization(__file__, 300, 15), Page_48407, *[], **kwargs_48426)
        
        # Assigning a type to the variable 'stypy_return_type' (line 300)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 8), 'stypy_return_type', Page_call_result_48427)
        
        # ################# End of '_output(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_output' in the type store
        # Getting the type of 'stypy_return_type' (line 262)
        stypy_return_type_48428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_48428)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_output'
        return stypy_return_type_48428


    @norecursion
    def _read(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_read'
        module_type_store = module_type_store.open_function_context('_read', 303, 4, False)
        # Assigning a type to the variable 'self' (line 304)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Dvi._read.__dict__.__setitem__('stypy_localization', localization)
        Dvi._read.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Dvi._read.__dict__.__setitem__('stypy_type_store', module_type_store)
        Dvi._read.__dict__.__setitem__('stypy_function_name', 'Dvi._read')
        Dvi._read.__dict__.__setitem__('stypy_param_names_list', [])
        Dvi._read.__dict__.__setitem__('stypy_varargs_param_name', None)
        Dvi._read.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Dvi._read.__dict__.__setitem__('stypy_call_defaults', defaults)
        Dvi._read.__dict__.__setitem__('stypy_call_varargs', varargs)
        Dvi._read.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Dvi._read.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Dvi._read', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_read', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_read(...)' code ##################

        unicode_48429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, (-1)), 'unicode', u'\n        Read one page from the file. Return True if successful,\n        False if there were no more pages.\n        ')
        
        # Getting the type of 'True' (line 308)
        True_48430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 14), 'True')
        # Testing the type of an if condition (line 308)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 308, 8), True_48430)
        # SSA begins for while statement (line 308)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Assigning a Call to a Name (line 309):
        
        # Assigning a Call to a Name (line 309):
        
        # Call to ord(...): (line 309)
        # Processing the call arguments (line 309)
        
        # Obtaining the type of the subscript
        int_48432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 41), 'int')
        
        # Call to read(...): (line 309)
        # Processing the call arguments (line 309)
        int_48436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 38), 'int')
        # Processing the call keyword arguments (line 309)
        kwargs_48437 = {}
        # Getting the type of 'self' (line 309)
        self_48433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 23), 'self', False)
        # Obtaining the member 'file' of a type (line 309)
        file_48434 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 23), self_48433, 'file')
        # Obtaining the member 'read' of a type (line 309)
        read_48435 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 23), file_48434, 'read')
        # Calling read(args, kwargs) (line 309)
        read_call_result_48438 = invoke(stypy.reporting.localization.Localization(__file__, 309, 23), read_48435, *[int_48436], **kwargs_48437)
        
        # Obtaining the member '__getitem__' of a type (line 309)
        getitem___48439 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 23), read_call_result_48438, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 309)
        subscript_call_result_48440 = invoke(stypy.reporting.localization.Localization(__file__, 309, 23), getitem___48439, int_48432)
        
        # Processing the call keyword arguments (line 309)
        kwargs_48441 = {}
        # Getting the type of 'ord' (line 309)
        ord_48431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 19), 'ord', False)
        # Calling ord(args, kwargs) (line 309)
        ord_call_result_48442 = invoke(stypy.reporting.localization.Localization(__file__, 309, 19), ord_48431, *[subscript_call_result_48440], **kwargs_48441)
        
        # Assigning a type to the variable 'byte' (line 309)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 12), 'byte', ord_call_result_48442)
        
        # Call to (...): (line 310)
        # Processing the call arguments (line 310)
        # Getting the type of 'self' (line 310)
        self_48448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 31), 'self', False)
        # Getting the type of 'byte' (line 310)
        byte_48449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 37), 'byte', False)
        # Processing the call keyword arguments (line 310)
        kwargs_48450 = {}
        
        # Obtaining the type of the subscript
        # Getting the type of 'byte' (line 310)
        byte_48443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 25), 'byte', False)
        # Getting the type of 'self' (line 310)
        self_48444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 12), 'self', False)
        # Obtaining the member '_dtable' of a type (line 310)
        _dtable_48445 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 12), self_48444, '_dtable')
        # Obtaining the member '__getitem__' of a type (line 310)
        getitem___48446 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 12), _dtable_48445, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 310)
        subscript_call_result_48447 = invoke(stypy.reporting.localization.Localization(__file__, 310, 12), getitem___48446, byte_48443)
        
        # Calling (args, kwargs) (line 310)
        _call_result_48451 = invoke(stypy.reporting.localization.Localization(__file__, 310, 12), subscript_call_result_48447, *[self_48448, byte_48449], **kwargs_48450)
        
        
        
        # Getting the type of 'byte' (line 311)
        byte_48452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 15), 'byte')
        int_48453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 23), 'int')
        # Applying the binary operator '==' (line 311)
        result_eq_48454 = python_operator(stypy.reporting.localization.Localization(__file__, 311, 15), '==', byte_48452, int_48453)
        
        # Testing the type of an if condition (line 311)
        if_condition_48455 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 311, 12), result_eq_48454)
        # Assigning a type to the variable 'if_condition_48455' (line 311)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 12), 'if_condition_48455', if_condition_48455)
        # SSA begins for if statement (line 311)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'True' (line 312)
        True_48456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 23), 'True')
        # Assigning a type to the variable 'stypy_return_type' (line 312)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 16), 'stypy_return_type', True_48456)
        # SSA join for if statement (line 311)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'self' (line 313)
        self_48457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 15), 'self')
        # Obtaining the member 'state' of a type (line 313)
        state_48458 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 15), self_48457, 'state')
        # Getting the type of '_dvistate' (line 313)
        _dvistate_48459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 29), '_dvistate')
        # Obtaining the member 'post_post' of a type (line 313)
        post_post_48460 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 29), _dvistate_48459, 'post_post')
        # Applying the binary operator '==' (line 313)
        result_eq_48461 = python_operator(stypy.reporting.localization.Localization(__file__, 313, 15), '==', state_48458, post_post_48460)
        
        # Testing the type of an if condition (line 313)
        if_condition_48462 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 313, 12), result_eq_48461)
        # Assigning a type to the variable 'if_condition_48462' (line 313)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 313, 12), 'if_condition_48462', if_condition_48462)
        # SSA begins for if statement (line 313)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to close(...): (line 314)
        # Processing the call keyword arguments (line 314)
        kwargs_48465 = {}
        # Getting the type of 'self' (line 314)
        self_48463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 16), 'self', False)
        # Obtaining the member 'close' of a type (line 314)
        close_48464 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 16), self_48463, 'close')
        # Calling close(args, kwargs) (line 314)
        close_call_result_48466 = invoke(stypy.reporting.localization.Localization(__file__, 314, 16), close_48464, *[], **kwargs_48465)
        
        # Getting the type of 'False' (line 315)
        False_48467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 23), 'False')
        # Assigning a type to the variable 'stypy_return_type' (line 315)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 16), 'stypy_return_type', False_48467)
        # SSA join for if statement (line 313)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for while statement (line 308)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_read(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_read' in the type store
        # Getting the type of 'stypy_return_type' (line 303)
        stypy_return_type_48468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_48468)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_read'
        return stypy_return_type_48468


    @norecursion
    def _arg(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'False' (line 317)
        False_48469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 34), 'False')
        defaults = [False_48469]
        # Create a new context for function '_arg'
        module_type_store = module_type_store.open_function_context('_arg', 317, 4, False)
        # Assigning a type to the variable 'self' (line 318)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Dvi._arg.__dict__.__setitem__('stypy_localization', localization)
        Dvi._arg.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Dvi._arg.__dict__.__setitem__('stypy_type_store', module_type_store)
        Dvi._arg.__dict__.__setitem__('stypy_function_name', 'Dvi._arg')
        Dvi._arg.__dict__.__setitem__('stypy_param_names_list', ['nbytes', 'signed'])
        Dvi._arg.__dict__.__setitem__('stypy_varargs_param_name', None)
        Dvi._arg.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Dvi._arg.__dict__.__setitem__('stypy_call_defaults', defaults)
        Dvi._arg.__dict__.__setitem__('stypy_call_varargs', varargs)
        Dvi._arg.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Dvi._arg.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Dvi._arg', ['nbytes', 'signed'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_arg', localization, ['nbytes', 'signed'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_arg(...)' code ##################

        unicode_48470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 321, (-1)), 'unicode', u'\n        Read and return an integer argument *nbytes* long.\n        Signedness is determined by the *signed* keyword.\n        ')
        
        # Assigning a Call to a Name (line 322):
        
        # Assigning a Call to a Name (line 322):
        
        # Call to read(...): (line 322)
        # Processing the call arguments (line 322)
        # Getting the type of 'nbytes' (line 322)
        nbytes_48474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 29), 'nbytes', False)
        # Processing the call keyword arguments (line 322)
        kwargs_48475 = {}
        # Getting the type of 'self' (line 322)
        self_48471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 14), 'self', False)
        # Obtaining the member 'file' of a type (line 322)
        file_48472 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 322, 14), self_48471, 'file')
        # Obtaining the member 'read' of a type (line 322)
        read_48473 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 322, 14), file_48472, 'read')
        # Calling read(args, kwargs) (line 322)
        read_call_result_48476 = invoke(stypy.reporting.localization.Localization(__file__, 322, 14), read_48473, *[nbytes_48474], **kwargs_48475)
        
        # Assigning a type to the variable 'str' (line 322)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 322, 8), 'str', read_call_result_48476)
        
        # Assigning a Call to a Name (line 323):
        
        # Assigning a Call to a Name (line 323):
        
        # Call to ord(...): (line 323)
        # Processing the call arguments (line 323)
        
        # Obtaining the type of the subscript
        int_48478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 24), 'int')
        # Getting the type of 'str' (line 323)
        str_48479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 20), 'str', False)
        # Obtaining the member '__getitem__' of a type (line 323)
        getitem___48480 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 323, 20), str_48479, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 323)
        subscript_call_result_48481 = invoke(stypy.reporting.localization.Localization(__file__, 323, 20), getitem___48480, int_48478)
        
        # Processing the call keyword arguments (line 323)
        kwargs_48482 = {}
        # Getting the type of 'ord' (line 323)
        ord_48477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 16), 'ord', False)
        # Calling ord(args, kwargs) (line 323)
        ord_call_result_48483 = invoke(stypy.reporting.localization.Localization(__file__, 323, 16), ord_48477, *[subscript_call_result_48481], **kwargs_48482)
        
        # Assigning a type to the variable 'value' (line 323)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 8), 'value', ord_call_result_48483)
        
        
        # Evaluating a boolean operation
        # Getting the type of 'signed' (line 324)
        signed_48484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 11), 'signed')
        
        # Getting the type of 'value' (line 324)
        value_48485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 22), 'value')
        int_48486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 31), 'int')
        # Applying the binary operator '>=' (line 324)
        result_ge_48487 = python_operator(stypy.reporting.localization.Localization(__file__, 324, 22), '>=', value_48485, int_48486)
        
        # Applying the binary operator 'and' (line 324)
        result_and_keyword_48488 = python_operator(stypy.reporting.localization.Localization(__file__, 324, 11), 'and', signed_48484, result_ge_48487)
        
        # Testing the type of an if condition (line 324)
        if_condition_48489 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 324, 8), result_and_keyword_48488)
        # Assigning a type to the variable 'if_condition_48489' (line 324)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 324, 8), 'if_condition_48489', if_condition_48489)
        # SSA begins for if statement (line 324)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 325):
        
        # Assigning a BinOp to a Name (line 325):
        # Getting the type of 'value' (line 325)
        value_48490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 20), 'value')
        int_48491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 28), 'int')
        # Applying the binary operator '-' (line 325)
        result_sub_48492 = python_operator(stypy.reporting.localization.Localization(__file__, 325, 20), '-', value_48490, int_48491)
        
        # Assigning a type to the variable 'value' (line 325)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 12), 'value', result_sub_48492)
        # SSA join for if statement (line 324)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to range(...): (line 326)
        # Processing the call arguments (line 326)
        int_48494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, 23), 'int')
        # Getting the type of 'nbytes' (line 326)
        nbytes_48495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 26), 'nbytes', False)
        # Processing the call keyword arguments (line 326)
        kwargs_48496 = {}
        # Getting the type of 'range' (line 326)
        range_48493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 17), 'range', False)
        # Calling range(args, kwargs) (line 326)
        range_call_result_48497 = invoke(stypy.reporting.localization.Localization(__file__, 326, 17), range_48493, *[int_48494, nbytes_48495], **kwargs_48496)
        
        # Testing the type of a for loop iterable (line 326)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 326, 8), range_call_result_48497)
        # Getting the type of the for loop variable (line 326)
        for_loop_var_48498 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 326, 8), range_call_result_48497)
        # Assigning a type to the variable 'i' (line 326)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 8), 'i', for_loop_var_48498)
        # SSA begins for a for statement (line 326)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a BinOp to a Name (line 327):
        
        # Assigning a BinOp to a Name (line 327):
        int_48499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 20), 'int')
        # Getting the type of 'value' (line 327)
        value_48500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 26), 'value')
        # Applying the binary operator '*' (line 327)
        result_mul_48501 = python_operator(stypy.reporting.localization.Localization(__file__, 327, 20), '*', int_48499, value_48500)
        
        
        # Call to ord(...): (line 327)
        # Processing the call arguments (line 327)
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 327)
        i_48503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 42), 'i', False)
        # Getting the type of 'str' (line 327)
        str_48504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 38), 'str', False)
        # Obtaining the member '__getitem__' of a type (line 327)
        getitem___48505 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 38), str_48504, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 327)
        subscript_call_result_48506 = invoke(stypy.reporting.localization.Localization(__file__, 327, 38), getitem___48505, i_48503)
        
        # Processing the call keyword arguments (line 327)
        kwargs_48507 = {}
        # Getting the type of 'ord' (line 327)
        ord_48502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 34), 'ord', False)
        # Calling ord(args, kwargs) (line 327)
        ord_call_result_48508 = invoke(stypy.reporting.localization.Localization(__file__, 327, 34), ord_48502, *[subscript_call_result_48506], **kwargs_48507)
        
        # Applying the binary operator '+' (line 327)
        result_add_48509 = python_operator(stypy.reporting.localization.Localization(__file__, 327, 20), '+', result_mul_48501, ord_call_result_48508)
        
        # Assigning a type to the variable 'value' (line 327)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 327, 12), 'value', result_add_48509)
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'value' (line 328)
        value_48510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 15), 'value')
        # Assigning a type to the variable 'stypy_return_type' (line 328)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 328, 8), 'stypy_return_type', value_48510)
        
        # ################# End of '_arg(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_arg' in the type store
        # Getting the type of 'stypy_return_type' (line 317)
        stypy_return_type_48511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_48511)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_arg'
        return stypy_return_type_48511


    @norecursion
    def _set_char_immediate(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_set_char_immediate'
        module_type_store = module_type_store.open_function_context('_set_char_immediate', 330, 4, False)
        # Assigning a type to the variable 'self' (line 331)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Dvi._set_char_immediate.__dict__.__setitem__('stypy_localization', localization)
        Dvi._set_char_immediate.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Dvi._set_char_immediate.__dict__.__setitem__('stypy_type_store', module_type_store)
        Dvi._set_char_immediate.__dict__.__setitem__('stypy_function_name', 'Dvi._set_char_immediate')
        Dvi._set_char_immediate.__dict__.__setitem__('stypy_param_names_list', ['char'])
        Dvi._set_char_immediate.__dict__.__setitem__('stypy_varargs_param_name', None)
        Dvi._set_char_immediate.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Dvi._set_char_immediate.__dict__.__setitem__('stypy_call_defaults', defaults)
        Dvi._set_char_immediate.__dict__.__setitem__('stypy_call_varargs', varargs)
        Dvi._set_char_immediate.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Dvi._set_char_immediate.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Dvi._set_char_immediate', ['char'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_set_char_immediate', localization, ['char'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_set_char_immediate(...)' code ##################

        
        # Call to _put_char_real(...): (line 332)
        # Processing the call arguments (line 332)
        # Getting the type of 'char' (line 332)
        char_48514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 28), 'char', False)
        # Processing the call keyword arguments (line 332)
        kwargs_48515 = {}
        # Getting the type of 'self' (line 332)
        self_48512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 8), 'self', False)
        # Obtaining the member '_put_char_real' of a type (line 332)
        _put_char_real_48513 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 8), self_48512, '_put_char_real')
        # Calling _put_char_real(args, kwargs) (line 332)
        _put_char_real_call_result_48516 = invoke(stypy.reporting.localization.Localization(__file__, 332, 8), _put_char_real_48513, *[char_48514], **kwargs_48515)
        
        
        # Getting the type of 'self' (line 333)
        self_48517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 8), 'self')
        # Obtaining the member 'h' of a type (line 333)
        h_48518 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 8), self_48517, 'h')
        
        # Call to _width_of(...): (line 333)
        # Processing the call arguments (line 333)
        # Getting the type of 'char' (line 333)
        char_48526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 47), 'char', False)
        # Processing the call keyword arguments (line 333)
        kwargs_48527 = {}
        
        # Obtaining the type of the subscript
        # Getting the type of 'self' (line 333)
        self_48519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 29), 'self', False)
        # Obtaining the member 'f' of a type (line 333)
        f_48520 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 29), self_48519, 'f')
        # Getting the type of 'self' (line 333)
        self_48521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 18), 'self', False)
        # Obtaining the member 'fonts' of a type (line 333)
        fonts_48522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 18), self_48521, 'fonts')
        # Obtaining the member '__getitem__' of a type (line 333)
        getitem___48523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 18), fonts_48522, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 333)
        subscript_call_result_48524 = invoke(stypy.reporting.localization.Localization(__file__, 333, 18), getitem___48523, f_48520)
        
        # Obtaining the member '_width_of' of a type (line 333)
        _width_of_48525 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 18), subscript_call_result_48524, '_width_of')
        # Calling _width_of(args, kwargs) (line 333)
        _width_of_call_result_48528 = invoke(stypy.reporting.localization.Localization(__file__, 333, 18), _width_of_48525, *[char_48526], **kwargs_48527)
        
        # Applying the binary operator '+=' (line 333)
        result_iadd_48529 = python_operator(stypy.reporting.localization.Localization(__file__, 333, 8), '+=', h_48518, _width_of_call_result_48528)
        # Getting the type of 'self' (line 333)
        self_48530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 8), 'self')
        # Setting the type of the member 'h' of a type (line 333)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 8), self_48530, 'h', result_iadd_48529)
        
        
        # ################# End of '_set_char_immediate(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_set_char_immediate' in the type store
        # Getting the type of 'stypy_return_type' (line 330)
        stypy_return_type_48531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_48531)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_set_char_immediate'
        return stypy_return_type_48531


    @norecursion
    def _set_char(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_set_char'
        module_type_store = module_type_store.open_function_context('_set_char', 335, 4, False)
        # Assigning a type to the variable 'self' (line 336)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 336, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Dvi._set_char.__dict__.__setitem__('stypy_localization', localization)
        Dvi._set_char.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Dvi._set_char.__dict__.__setitem__('stypy_type_store', module_type_store)
        Dvi._set_char.__dict__.__setitem__('stypy_function_name', 'Dvi._set_char')
        Dvi._set_char.__dict__.__setitem__('stypy_param_names_list', ['char'])
        Dvi._set_char.__dict__.__setitem__('stypy_varargs_param_name', None)
        Dvi._set_char.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Dvi._set_char.__dict__.__setitem__('stypy_call_defaults', defaults)
        Dvi._set_char.__dict__.__setitem__('stypy_call_varargs', varargs)
        Dvi._set_char.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Dvi._set_char.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Dvi._set_char', ['char'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_set_char', localization, ['char'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_set_char(...)' code ##################

        
        # Call to _put_char_real(...): (line 337)
        # Processing the call arguments (line 337)
        # Getting the type of 'char' (line 337)
        char_48534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 28), 'char', False)
        # Processing the call keyword arguments (line 337)
        kwargs_48535 = {}
        # Getting the type of 'self' (line 337)
        self_48532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 8), 'self', False)
        # Obtaining the member '_put_char_real' of a type (line 337)
        _put_char_real_48533 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 8), self_48532, '_put_char_real')
        # Calling _put_char_real(args, kwargs) (line 337)
        _put_char_real_call_result_48536 = invoke(stypy.reporting.localization.Localization(__file__, 337, 8), _put_char_real_48533, *[char_48534], **kwargs_48535)
        
        
        # Getting the type of 'self' (line 338)
        self_48537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 8), 'self')
        # Obtaining the member 'h' of a type (line 338)
        h_48538 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 8), self_48537, 'h')
        
        # Call to _width_of(...): (line 338)
        # Processing the call arguments (line 338)
        # Getting the type of 'char' (line 338)
        char_48546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 47), 'char', False)
        # Processing the call keyword arguments (line 338)
        kwargs_48547 = {}
        
        # Obtaining the type of the subscript
        # Getting the type of 'self' (line 338)
        self_48539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 29), 'self', False)
        # Obtaining the member 'f' of a type (line 338)
        f_48540 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 29), self_48539, 'f')
        # Getting the type of 'self' (line 338)
        self_48541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 18), 'self', False)
        # Obtaining the member 'fonts' of a type (line 338)
        fonts_48542 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 18), self_48541, 'fonts')
        # Obtaining the member '__getitem__' of a type (line 338)
        getitem___48543 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 18), fonts_48542, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 338)
        subscript_call_result_48544 = invoke(stypy.reporting.localization.Localization(__file__, 338, 18), getitem___48543, f_48540)
        
        # Obtaining the member '_width_of' of a type (line 338)
        _width_of_48545 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 18), subscript_call_result_48544, '_width_of')
        # Calling _width_of(args, kwargs) (line 338)
        _width_of_call_result_48548 = invoke(stypy.reporting.localization.Localization(__file__, 338, 18), _width_of_48545, *[char_48546], **kwargs_48547)
        
        # Applying the binary operator '+=' (line 338)
        result_iadd_48549 = python_operator(stypy.reporting.localization.Localization(__file__, 338, 8), '+=', h_48538, _width_of_call_result_48548)
        # Getting the type of 'self' (line 338)
        self_48550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 8), 'self')
        # Setting the type of the member 'h' of a type (line 338)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 8), self_48550, 'h', result_iadd_48549)
        
        
        # ################# End of '_set_char(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_set_char' in the type store
        # Getting the type of 'stypy_return_type' (line 335)
        stypy_return_type_48551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_48551)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_set_char'
        return stypy_return_type_48551


    @norecursion
    def _set_rule(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_set_rule'
        module_type_store = module_type_store.open_function_context('_set_rule', 340, 4, False)
        # Assigning a type to the variable 'self' (line 341)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Dvi._set_rule.__dict__.__setitem__('stypy_localization', localization)
        Dvi._set_rule.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Dvi._set_rule.__dict__.__setitem__('stypy_type_store', module_type_store)
        Dvi._set_rule.__dict__.__setitem__('stypy_function_name', 'Dvi._set_rule')
        Dvi._set_rule.__dict__.__setitem__('stypy_param_names_list', ['a', 'b'])
        Dvi._set_rule.__dict__.__setitem__('stypy_varargs_param_name', None)
        Dvi._set_rule.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Dvi._set_rule.__dict__.__setitem__('stypy_call_defaults', defaults)
        Dvi._set_rule.__dict__.__setitem__('stypy_call_varargs', varargs)
        Dvi._set_rule.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Dvi._set_rule.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Dvi._set_rule', ['a', 'b'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_set_rule', localization, ['a', 'b'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_set_rule(...)' code ##################

        
        # Call to _put_rule_real(...): (line 342)
        # Processing the call arguments (line 342)
        # Getting the type of 'a' (line 342)
        a_48554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 28), 'a', False)
        # Getting the type of 'b' (line 342)
        b_48555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 31), 'b', False)
        # Processing the call keyword arguments (line 342)
        kwargs_48556 = {}
        # Getting the type of 'self' (line 342)
        self_48552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 8), 'self', False)
        # Obtaining the member '_put_rule_real' of a type (line 342)
        _put_rule_real_48553 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 8), self_48552, '_put_rule_real')
        # Calling _put_rule_real(args, kwargs) (line 342)
        _put_rule_real_call_result_48557 = invoke(stypy.reporting.localization.Localization(__file__, 342, 8), _put_rule_real_48553, *[a_48554, b_48555], **kwargs_48556)
        
        
        # Getting the type of 'self' (line 343)
        self_48558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 8), 'self')
        # Obtaining the member 'h' of a type (line 343)
        h_48559 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 8), self_48558, 'h')
        # Getting the type of 'b' (line 343)
        b_48560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 18), 'b')
        # Applying the binary operator '+=' (line 343)
        result_iadd_48561 = python_operator(stypy.reporting.localization.Localization(__file__, 343, 8), '+=', h_48559, b_48560)
        # Getting the type of 'self' (line 343)
        self_48562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 8), 'self')
        # Setting the type of the member 'h' of a type (line 343)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 8), self_48562, 'h', result_iadd_48561)
        
        
        # ################# End of '_set_rule(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_set_rule' in the type store
        # Getting the type of 'stypy_return_type' (line 340)
        stypy_return_type_48563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_48563)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_set_rule'
        return stypy_return_type_48563


    @norecursion
    def _put_char(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_put_char'
        module_type_store = module_type_store.open_function_context('_put_char', 345, 4, False)
        # Assigning a type to the variable 'self' (line 346)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 346, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Dvi._put_char.__dict__.__setitem__('stypy_localization', localization)
        Dvi._put_char.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Dvi._put_char.__dict__.__setitem__('stypy_type_store', module_type_store)
        Dvi._put_char.__dict__.__setitem__('stypy_function_name', 'Dvi._put_char')
        Dvi._put_char.__dict__.__setitem__('stypy_param_names_list', ['char'])
        Dvi._put_char.__dict__.__setitem__('stypy_varargs_param_name', None)
        Dvi._put_char.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Dvi._put_char.__dict__.__setitem__('stypy_call_defaults', defaults)
        Dvi._put_char.__dict__.__setitem__('stypy_call_varargs', varargs)
        Dvi._put_char.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Dvi._put_char.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Dvi._put_char', ['char'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_put_char', localization, ['char'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_put_char(...)' code ##################

        
        # Call to _put_char_real(...): (line 347)
        # Processing the call arguments (line 347)
        # Getting the type of 'char' (line 347)
        char_48566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 28), 'char', False)
        # Processing the call keyword arguments (line 347)
        kwargs_48567 = {}
        # Getting the type of 'self' (line 347)
        self_48564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 8), 'self', False)
        # Obtaining the member '_put_char_real' of a type (line 347)
        _put_char_real_48565 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 347, 8), self_48564, '_put_char_real')
        # Calling _put_char_real(args, kwargs) (line 347)
        _put_char_real_call_result_48568 = invoke(stypy.reporting.localization.Localization(__file__, 347, 8), _put_char_real_48565, *[char_48566], **kwargs_48567)
        
        
        # ################# End of '_put_char(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_put_char' in the type store
        # Getting the type of 'stypy_return_type' (line 345)
        stypy_return_type_48569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_48569)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_put_char'
        return stypy_return_type_48569


    @norecursion
    def _put_char_real(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_put_char_real'
        module_type_store = module_type_store.open_function_context('_put_char_real', 349, 4, False)
        # Assigning a type to the variable 'self' (line 350)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Dvi._put_char_real.__dict__.__setitem__('stypy_localization', localization)
        Dvi._put_char_real.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Dvi._put_char_real.__dict__.__setitem__('stypy_type_store', module_type_store)
        Dvi._put_char_real.__dict__.__setitem__('stypy_function_name', 'Dvi._put_char_real')
        Dvi._put_char_real.__dict__.__setitem__('stypy_param_names_list', ['char'])
        Dvi._put_char_real.__dict__.__setitem__('stypy_varargs_param_name', None)
        Dvi._put_char_real.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Dvi._put_char_real.__dict__.__setitem__('stypy_call_defaults', defaults)
        Dvi._put_char_real.__dict__.__setitem__('stypy_call_varargs', varargs)
        Dvi._put_char_real.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Dvi._put_char_real.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Dvi._put_char_real', ['char'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_put_char_real', localization, ['char'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_put_char_real(...)' code ##################

        
        # Assigning a Subscript to a Name (line 350):
        
        # Assigning a Subscript to a Name (line 350):
        
        # Obtaining the type of the subscript
        # Getting the type of 'self' (line 350)
        self_48570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 26), 'self')
        # Obtaining the member 'f' of a type (line 350)
        f_48571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 26), self_48570, 'f')
        # Getting the type of 'self' (line 350)
        self_48572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 15), 'self')
        # Obtaining the member 'fonts' of a type (line 350)
        fonts_48573 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 15), self_48572, 'fonts')
        # Obtaining the member '__getitem__' of a type (line 350)
        getitem___48574 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 15), fonts_48573, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 350)
        subscript_call_result_48575 = invoke(stypy.reporting.localization.Localization(__file__, 350, 15), getitem___48574, f_48571)
        
        # Assigning a type to the variable 'font' (line 350)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 8), 'font', subscript_call_result_48575)
        
        # Type idiom detected: calculating its left and rigth part (line 351)
        # Getting the type of 'font' (line 351)
        font_48576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 11), 'font')
        # Obtaining the member '_vf' of a type (line 351)
        _vf_48577 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 351, 11), font_48576, '_vf')
        # Getting the type of 'None' (line 351)
        None_48578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 23), 'None')
        
        (may_be_48579, more_types_in_union_48580) = may_be_none(_vf_48577, None_48578)

        if may_be_48579:

            if more_types_in_union_48580:
                # Runtime conditional SSA (line 351)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to append(...): (line 352)
            # Processing the call arguments (line 352)
            
            # Call to Text(...): (line 352)
            # Processing the call arguments (line 352)
            # Getting the type of 'self' (line 352)
            self_48585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 34), 'self', False)
            # Obtaining the member 'h' of a type (line 352)
            h_48586 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 352, 34), self_48585, 'h')
            # Getting the type of 'self' (line 352)
            self_48587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 42), 'self', False)
            # Obtaining the member 'v' of a type (line 352)
            v_48588 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 352, 42), self_48587, 'v')
            # Getting the type of 'font' (line 352)
            font_48589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 50), 'font', False)
            # Getting the type of 'char' (line 352)
            char_48590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 56), 'char', False)
            
            # Call to _width_of(...): (line 353)
            # Processing the call arguments (line 353)
            # Getting the type of 'char' (line 353)
            char_48593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 49), 'char', False)
            # Processing the call keyword arguments (line 353)
            kwargs_48594 = {}
            # Getting the type of 'font' (line 353)
            font_48591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 34), 'font', False)
            # Obtaining the member '_width_of' of a type (line 353)
            _width_of_48592 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 353, 34), font_48591, '_width_of')
            # Calling _width_of(args, kwargs) (line 353)
            _width_of_call_result_48595 = invoke(stypy.reporting.localization.Localization(__file__, 353, 34), _width_of_48592, *[char_48593], **kwargs_48594)
            
            # Processing the call keyword arguments (line 352)
            kwargs_48596 = {}
            # Getting the type of 'Text' (line 352)
            Text_48584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 29), 'Text', False)
            # Calling Text(args, kwargs) (line 352)
            Text_call_result_48597 = invoke(stypy.reporting.localization.Localization(__file__, 352, 29), Text_48584, *[h_48586, v_48588, font_48589, char_48590, _width_of_call_result_48595], **kwargs_48596)
            
            # Processing the call keyword arguments (line 352)
            kwargs_48598 = {}
            # Getting the type of 'self' (line 352)
            self_48581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 12), 'self', False)
            # Obtaining the member 'text' of a type (line 352)
            text_48582 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 352, 12), self_48581, 'text')
            # Obtaining the member 'append' of a type (line 352)
            append_48583 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 352, 12), text_48582, 'append')
            # Calling append(args, kwargs) (line 352)
            append_call_result_48599 = invoke(stypy.reporting.localization.Localization(__file__, 352, 12), append_48583, *[Text_call_result_48597], **kwargs_48598)
            

            if more_types_in_union_48580:
                # Runtime conditional SSA for else branch (line 351)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_48579) or more_types_in_union_48580):
            
            # Assigning a Attribute to a Name (line 355):
            
            # Assigning a Attribute to a Name (line 355):
            # Getting the type of 'font' (line 355)
            font_48600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 20), 'font')
            # Obtaining the member '_scale' of a type (line 355)
            _scale_48601 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 355, 20), font_48600, '_scale')
            # Assigning a type to the variable 'scale' (line 355)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 355, 12), 'scale', _scale_48601)
            
            
            # Obtaining the type of the subscript
            # Getting the type of 'char' (line 356)
            char_48602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 42), 'char')
            # Getting the type of 'font' (line 356)
            font_48603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 33), 'font')
            # Obtaining the member '_vf' of a type (line 356)
            _vf_48604 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 356, 33), font_48603, '_vf')
            # Obtaining the member '__getitem__' of a type (line 356)
            getitem___48605 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 356, 33), _vf_48604, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 356)
            subscript_call_result_48606 = invoke(stypy.reporting.localization.Localization(__file__, 356, 33), getitem___48605, char_48602)
            
            # Obtaining the member 'text' of a type (line 356)
            text_48607 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 356, 33), subscript_call_result_48606, 'text')
            # Testing the type of a for loop iterable (line 356)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 356, 12), text_48607)
            # Getting the type of the for loop variable (line 356)
            for_loop_var_48608 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 356, 12), text_48607)
            # Assigning a type to the variable 'x' (line 356)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 356, 12), 'x', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 356, 12), for_loop_var_48608))
            # Assigning a type to the variable 'y' (line 356)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 356, 12), 'y', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 356, 12), for_loop_var_48608))
            # Assigning a type to the variable 'f' (line 356)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 356, 12), 'f', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 356, 12), for_loop_var_48608))
            # Assigning a type to the variable 'g' (line 356)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 356, 12), 'g', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 356, 12), for_loop_var_48608))
            # Assigning a type to the variable 'w' (line 356)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 356, 12), 'w', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 356, 12), for_loop_var_48608))
            # SSA begins for a for statement (line 356)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Call to a Name (line 357):
            
            # Assigning a Call to a Name (line 357):
            
            # Call to DviFont(...): (line 357)
            # Processing the call keyword arguments (line 357)
            
            # Call to _mul2012(...): (line 357)
            # Processing the call arguments (line 357)
            # Getting the type of 'scale' (line 357)
            scale_48611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 46), 'scale', False)
            # Getting the type of 'f' (line 357)
            f_48612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 53), 'f', False)
            # Obtaining the member '_scale' of a type (line 357)
            _scale_48613 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 357, 53), f_48612, '_scale')
            # Processing the call keyword arguments (line 357)
            kwargs_48614 = {}
            # Getting the type of '_mul2012' (line 357)
            _mul2012_48610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 37), '_mul2012', False)
            # Calling _mul2012(args, kwargs) (line 357)
            _mul2012_call_result_48615 = invoke(stypy.reporting.localization.Localization(__file__, 357, 37), _mul2012_48610, *[scale_48611, _scale_48613], **kwargs_48614)
            
            keyword_48616 = _mul2012_call_result_48615
            # Getting the type of 'f' (line 358)
            f_48617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 35), 'f', False)
            # Obtaining the member '_tfm' of a type (line 358)
            _tfm_48618 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 35), f_48617, '_tfm')
            keyword_48619 = _tfm_48618
            # Getting the type of 'f' (line 358)
            f_48620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 51), 'f', False)
            # Obtaining the member 'texname' of a type (line 358)
            texname_48621 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 51), f_48620, 'texname')
            keyword_48622 = texname_48621
            # Getting the type of 'f' (line 358)
            f_48623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 65), 'f', False)
            # Obtaining the member '_vf' of a type (line 358)
            _vf_48624 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 65), f_48623, '_vf')
            keyword_48625 = _vf_48624
            kwargs_48626 = {'tfm': keyword_48619, 'texname': keyword_48622, 'scale': keyword_48616, 'vf': keyword_48625}
            # Getting the type of 'DviFont' (line 357)
            DviFont_48609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 23), 'DviFont', False)
            # Calling DviFont(args, kwargs) (line 357)
            DviFont_call_result_48627 = invoke(stypy.reporting.localization.Localization(__file__, 357, 23), DviFont_48609, *[], **kwargs_48626)
            
            # Assigning a type to the variable 'newf' (line 357)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 357, 16), 'newf', DviFont_call_result_48627)
            
            # Call to append(...): (line 359)
            # Processing the call arguments (line 359)
            
            # Call to Text(...): (line 359)
            # Processing the call arguments (line 359)
            # Getting the type of 'self' (line 359)
            self_48632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 38), 'self', False)
            # Obtaining the member 'h' of a type (line 359)
            h_48633 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 38), self_48632, 'h')
            
            # Call to _mul2012(...): (line 359)
            # Processing the call arguments (line 359)
            # Getting the type of 'x' (line 359)
            x_48635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 56), 'x', False)
            # Getting the type of 'scale' (line 359)
            scale_48636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 59), 'scale', False)
            # Processing the call keyword arguments (line 359)
            kwargs_48637 = {}
            # Getting the type of '_mul2012' (line 359)
            _mul2012_48634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 47), '_mul2012', False)
            # Calling _mul2012(args, kwargs) (line 359)
            _mul2012_call_result_48638 = invoke(stypy.reporting.localization.Localization(__file__, 359, 47), _mul2012_48634, *[x_48635, scale_48636], **kwargs_48637)
            
            # Applying the binary operator '+' (line 359)
            result_add_48639 = python_operator(stypy.reporting.localization.Localization(__file__, 359, 38), '+', h_48633, _mul2012_call_result_48638)
            
            # Getting the type of 'self' (line 360)
            self_48640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 38), 'self', False)
            # Obtaining the member 'v' of a type (line 360)
            v_48641 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 360, 38), self_48640, 'v')
            
            # Call to _mul2012(...): (line 360)
            # Processing the call arguments (line 360)
            # Getting the type of 'y' (line 360)
            y_48643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 56), 'y', False)
            # Getting the type of 'scale' (line 360)
            scale_48644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 59), 'scale', False)
            # Processing the call keyword arguments (line 360)
            kwargs_48645 = {}
            # Getting the type of '_mul2012' (line 360)
            _mul2012_48642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 47), '_mul2012', False)
            # Calling _mul2012(args, kwargs) (line 360)
            _mul2012_call_result_48646 = invoke(stypy.reporting.localization.Localization(__file__, 360, 47), _mul2012_48642, *[y_48643, scale_48644], **kwargs_48645)
            
            # Applying the binary operator '+' (line 360)
            result_add_48647 = python_operator(stypy.reporting.localization.Localization(__file__, 360, 38), '+', v_48641, _mul2012_call_result_48646)
            
            # Getting the type of 'newf' (line 361)
            newf_48648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 38), 'newf', False)
            # Getting the type of 'g' (line 361)
            g_48649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 44), 'g', False)
            
            # Call to _width_of(...): (line 361)
            # Processing the call arguments (line 361)
            # Getting the type of 'g' (line 361)
            g_48652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 62), 'g', False)
            # Processing the call keyword arguments (line 361)
            kwargs_48653 = {}
            # Getting the type of 'newf' (line 361)
            newf_48650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 47), 'newf', False)
            # Obtaining the member '_width_of' of a type (line 361)
            _width_of_48651 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 361, 47), newf_48650, '_width_of')
            # Calling _width_of(args, kwargs) (line 361)
            _width_of_call_result_48654 = invoke(stypy.reporting.localization.Localization(__file__, 361, 47), _width_of_48651, *[g_48652], **kwargs_48653)
            
            # Processing the call keyword arguments (line 359)
            kwargs_48655 = {}
            # Getting the type of 'Text' (line 359)
            Text_48631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 33), 'Text', False)
            # Calling Text(args, kwargs) (line 359)
            Text_call_result_48656 = invoke(stypy.reporting.localization.Localization(__file__, 359, 33), Text_48631, *[result_add_48639, result_add_48647, newf_48648, g_48649, _width_of_call_result_48654], **kwargs_48655)
            
            # Processing the call keyword arguments (line 359)
            kwargs_48657 = {}
            # Getting the type of 'self' (line 359)
            self_48628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 16), 'self', False)
            # Obtaining the member 'text' of a type (line 359)
            text_48629 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 16), self_48628, 'text')
            # Obtaining the member 'append' of a type (line 359)
            append_48630 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 16), text_48629, 'append')
            # Calling append(args, kwargs) (line 359)
            append_call_result_48658 = invoke(stypy.reporting.localization.Localization(__file__, 359, 16), append_48630, *[Text_call_result_48656], **kwargs_48657)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Call to extend(...): (line 362)
            # Processing the call arguments (line 362)
            # Calculating list comprehension
            # Calculating comprehension expression
            
            # Obtaining the type of the subscript
            # Getting the type of 'char' (line 365)
            char_48691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 58), 'char', False)
            # Getting the type of 'font' (line 365)
            font_48692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 49), 'font', False)
            # Obtaining the member '_vf' of a type (line 365)
            _vf_48693 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 365, 49), font_48692, '_vf')
            # Obtaining the member '__getitem__' of a type (line 365)
            getitem___48694 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 365, 49), _vf_48693, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 365)
            subscript_call_result_48695 = invoke(stypy.reporting.localization.Localization(__file__, 365, 49), getitem___48694, char_48691)
            
            # Obtaining the member 'boxes' of a type (line 365)
            boxes_48696 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 365, 49), subscript_call_result_48695, 'boxes')
            comprehension_48697 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 362, 31), boxes_48696)
            # Assigning a type to the variable 'x' (line 362)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 31), 'x', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 362, 31), comprehension_48697))
            # Assigning a type to the variable 'y' (line 362)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 31), 'y', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 362, 31), comprehension_48697))
            # Assigning a type to the variable 'a' (line 362)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 31), 'a', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 362, 31), comprehension_48697))
            # Assigning a type to the variable 'b' (line 362)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 31), 'b', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 362, 31), comprehension_48697))
            
            # Call to Box(...): (line 362)
            # Processing the call arguments (line 362)
            # Getting the type of 'self' (line 362)
            self_48663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 35), 'self', False)
            # Obtaining the member 'h' of a type (line 362)
            h_48664 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 35), self_48663, 'h')
            
            # Call to _mul2012(...): (line 362)
            # Processing the call arguments (line 362)
            # Getting the type of 'x' (line 362)
            x_48666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 53), 'x', False)
            # Getting the type of 'scale' (line 362)
            scale_48667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 56), 'scale', False)
            # Processing the call keyword arguments (line 362)
            kwargs_48668 = {}
            # Getting the type of '_mul2012' (line 362)
            _mul2012_48665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 44), '_mul2012', False)
            # Calling _mul2012(args, kwargs) (line 362)
            _mul2012_call_result_48669 = invoke(stypy.reporting.localization.Localization(__file__, 362, 44), _mul2012_48665, *[x_48666, scale_48667], **kwargs_48668)
            
            # Applying the binary operator '+' (line 362)
            result_add_48670 = python_operator(stypy.reporting.localization.Localization(__file__, 362, 35), '+', h_48664, _mul2012_call_result_48669)
            
            # Getting the type of 'self' (line 363)
            self_48671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 35), 'self', False)
            # Obtaining the member 'v' of a type (line 363)
            v_48672 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 363, 35), self_48671, 'v')
            
            # Call to _mul2012(...): (line 363)
            # Processing the call arguments (line 363)
            # Getting the type of 'y' (line 363)
            y_48674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 53), 'y', False)
            # Getting the type of 'scale' (line 363)
            scale_48675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 56), 'scale', False)
            # Processing the call keyword arguments (line 363)
            kwargs_48676 = {}
            # Getting the type of '_mul2012' (line 363)
            _mul2012_48673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 44), '_mul2012', False)
            # Calling _mul2012(args, kwargs) (line 363)
            _mul2012_call_result_48677 = invoke(stypy.reporting.localization.Localization(__file__, 363, 44), _mul2012_48673, *[y_48674, scale_48675], **kwargs_48676)
            
            # Applying the binary operator '+' (line 363)
            result_add_48678 = python_operator(stypy.reporting.localization.Localization(__file__, 363, 35), '+', v_48672, _mul2012_call_result_48677)
            
            
            # Call to _mul2012(...): (line 364)
            # Processing the call arguments (line 364)
            # Getting the type of 'a' (line 364)
            a_48680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 44), 'a', False)
            # Getting the type of 'scale' (line 364)
            scale_48681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 47), 'scale', False)
            # Processing the call keyword arguments (line 364)
            kwargs_48682 = {}
            # Getting the type of '_mul2012' (line 364)
            _mul2012_48679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 35), '_mul2012', False)
            # Calling _mul2012(args, kwargs) (line 364)
            _mul2012_call_result_48683 = invoke(stypy.reporting.localization.Localization(__file__, 364, 35), _mul2012_48679, *[a_48680, scale_48681], **kwargs_48682)
            
            
            # Call to _mul2012(...): (line 364)
            # Processing the call arguments (line 364)
            # Getting the type of 'b' (line 364)
            b_48685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 64), 'b', False)
            # Getting the type of 'scale' (line 364)
            scale_48686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 67), 'scale', False)
            # Processing the call keyword arguments (line 364)
            kwargs_48687 = {}
            # Getting the type of '_mul2012' (line 364)
            _mul2012_48684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 55), '_mul2012', False)
            # Calling _mul2012(args, kwargs) (line 364)
            _mul2012_call_result_48688 = invoke(stypy.reporting.localization.Localization(__file__, 364, 55), _mul2012_48684, *[b_48685, scale_48686], **kwargs_48687)
            
            # Processing the call keyword arguments (line 362)
            kwargs_48689 = {}
            # Getting the type of 'Box' (line 362)
            Box_48662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 31), 'Box', False)
            # Calling Box(args, kwargs) (line 362)
            Box_call_result_48690 = invoke(stypy.reporting.localization.Localization(__file__, 362, 31), Box_48662, *[result_add_48670, result_add_48678, _mul2012_call_result_48683, _mul2012_call_result_48688], **kwargs_48689)
            
            list_48698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 31), 'list')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 362, 31), list_48698, Box_call_result_48690)
            # Processing the call keyword arguments (line 362)
            kwargs_48699 = {}
            # Getting the type of 'self' (line 362)
            self_48659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 12), 'self', False)
            # Obtaining the member 'boxes' of a type (line 362)
            boxes_48660 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 12), self_48659, 'boxes')
            # Obtaining the member 'extend' of a type (line 362)
            extend_48661 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 12), boxes_48660, 'extend')
            # Calling extend(args, kwargs) (line 362)
            extend_call_result_48700 = invoke(stypy.reporting.localization.Localization(__file__, 362, 12), extend_48661, *[list_48698], **kwargs_48699)
            

            if (may_be_48579 and more_types_in_union_48580):
                # SSA join for if statement (line 351)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # ################# End of '_put_char_real(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_put_char_real' in the type store
        # Getting the type of 'stypy_return_type' (line 349)
        stypy_return_type_48701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_48701)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_put_char_real'
        return stypy_return_type_48701


    @norecursion
    def _put_rule(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_put_rule'
        module_type_store = module_type_store.open_function_context('_put_rule', 367, 4, False)
        # Assigning a type to the variable 'self' (line 368)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 368, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Dvi._put_rule.__dict__.__setitem__('stypy_localization', localization)
        Dvi._put_rule.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Dvi._put_rule.__dict__.__setitem__('stypy_type_store', module_type_store)
        Dvi._put_rule.__dict__.__setitem__('stypy_function_name', 'Dvi._put_rule')
        Dvi._put_rule.__dict__.__setitem__('stypy_param_names_list', ['a', 'b'])
        Dvi._put_rule.__dict__.__setitem__('stypy_varargs_param_name', None)
        Dvi._put_rule.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Dvi._put_rule.__dict__.__setitem__('stypy_call_defaults', defaults)
        Dvi._put_rule.__dict__.__setitem__('stypy_call_varargs', varargs)
        Dvi._put_rule.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Dvi._put_rule.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Dvi._put_rule', ['a', 'b'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_put_rule', localization, ['a', 'b'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_put_rule(...)' code ##################

        
        # Call to _put_rule_real(...): (line 369)
        # Processing the call arguments (line 369)
        # Getting the type of 'a' (line 369)
        a_48704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 28), 'a', False)
        # Getting the type of 'b' (line 369)
        b_48705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 31), 'b', False)
        # Processing the call keyword arguments (line 369)
        kwargs_48706 = {}
        # Getting the type of 'self' (line 369)
        self_48702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 8), 'self', False)
        # Obtaining the member '_put_rule_real' of a type (line 369)
        _put_rule_real_48703 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 369, 8), self_48702, '_put_rule_real')
        # Calling _put_rule_real(args, kwargs) (line 369)
        _put_rule_real_call_result_48707 = invoke(stypy.reporting.localization.Localization(__file__, 369, 8), _put_rule_real_48703, *[a_48704, b_48705], **kwargs_48706)
        
        
        # ################# End of '_put_rule(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_put_rule' in the type store
        # Getting the type of 'stypy_return_type' (line 367)
        stypy_return_type_48708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_48708)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_put_rule'
        return stypy_return_type_48708


    @norecursion
    def _put_rule_real(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_put_rule_real'
        module_type_store = module_type_store.open_function_context('_put_rule_real', 371, 4, False)
        # Assigning a type to the variable 'self' (line 372)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 372, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Dvi._put_rule_real.__dict__.__setitem__('stypy_localization', localization)
        Dvi._put_rule_real.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Dvi._put_rule_real.__dict__.__setitem__('stypy_type_store', module_type_store)
        Dvi._put_rule_real.__dict__.__setitem__('stypy_function_name', 'Dvi._put_rule_real')
        Dvi._put_rule_real.__dict__.__setitem__('stypy_param_names_list', ['a', 'b'])
        Dvi._put_rule_real.__dict__.__setitem__('stypy_varargs_param_name', None)
        Dvi._put_rule_real.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Dvi._put_rule_real.__dict__.__setitem__('stypy_call_defaults', defaults)
        Dvi._put_rule_real.__dict__.__setitem__('stypy_call_varargs', varargs)
        Dvi._put_rule_real.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Dvi._put_rule_real.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Dvi._put_rule_real', ['a', 'b'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_put_rule_real', localization, ['a', 'b'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_put_rule_real(...)' code ##################

        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'a' (line 372)
        a_48709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 11), 'a')
        int_48710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 372, 15), 'int')
        # Applying the binary operator '>' (line 372)
        result_gt_48711 = python_operator(stypy.reporting.localization.Localization(__file__, 372, 11), '>', a_48709, int_48710)
        
        
        # Getting the type of 'b' (line 372)
        b_48712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 21), 'b')
        int_48713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 372, 25), 'int')
        # Applying the binary operator '>' (line 372)
        result_gt_48714 = python_operator(stypy.reporting.localization.Localization(__file__, 372, 21), '>', b_48712, int_48713)
        
        # Applying the binary operator 'and' (line 372)
        result_and_keyword_48715 = python_operator(stypy.reporting.localization.Localization(__file__, 372, 11), 'and', result_gt_48711, result_gt_48714)
        
        # Testing the type of an if condition (line 372)
        if_condition_48716 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 372, 8), result_and_keyword_48715)
        # Assigning a type to the variable 'if_condition_48716' (line 372)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 372, 8), 'if_condition_48716', if_condition_48716)
        # SSA begins for if statement (line 372)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 373)
        # Processing the call arguments (line 373)
        
        # Call to Box(...): (line 373)
        # Processing the call arguments (line 373)
        # Getting the type of 'self' (line 373)
        self_48721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 34), 'self', False)
        # Obtaining the member 'h' of a type (line 373)
        h_48722 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 34), self_48721, 'h')
        # Getting the type of 'self' (line 373)
        self_48723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 42), 'self', False)
        # Obtaining the member 'v' of a type (line 373)
        v_48724 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 42), self_48723, 'v')
        # Getting the type of 'a' (line 373)
        a_48725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 50), 'a', False)
        # Getting the type of 'b' (line 373)
        b_48726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 53), 'b', False)
        # Processing the call keyword arguments (line 373)
        kwargs_48727 = {}
        # Getting the type of 'Box' (line 373)
        Box_48720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 30), 'Box', False)
        # Calling Box(args, kwargs) (line 373)
        Box_call_result_48728 = invoke(stypy.reporting.localization.Localization(__file__, 373, 30), Box_48720, *[h_48722, v_48724, a_48725, b_48726], **kwargs_48727)
        
        # Processing the call keyword arguments (line 373)
        kwargs_48729 = {}
        # Getting the type of 'self' (line 373)
        self_48717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 12), 'self', False)
        # Obtaining the member 'boxes' of a type (line 373)
        boxes_48718 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 12), self_48717, 'boxes')
        # Obtaining the member 'append' of a type (line 373)
        append_48719 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 12), boxes_48718, 'append')
        # Calling append(args, kwargs) (line 373)
        append_call_result_48730 = invoke(stypy.reporting.localization.Localization(__file__, 373, 12), append_48719, *[Box_call_result_48728], **kwargs_48729)
        
        # SSA join for if statement (line 372)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_put_rule_real(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_put_rule_real' in the type store
        # Getting the type of 'stypy_return_type' (line 371)
        stypy_return_type_48731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_48731)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_put_rule_real'
        return stypy_return_type_48731


    @norecursion
    def _nop(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_nop'
        module_type_store = module_type_store.open_function_context('_nop', 375, 4, False)
        # Assigning a type to the variable 'self' (line 376)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 376, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Dvi._nop.__dict__.__setitem__('stypy_localization', localization)
        Dvi._nop.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Dvi._nop.__dict__.__setitem__('stypy_type_store', module_type_store)
        Dvi._nop.__dict__.__setitem__('stypy_function_name', 'Dvi._nop')
        Dvi._nop.__dict__.__setitem__('stypy_param_names_list', ['_'])
        Dvi._nop.__dict__.__setitem__('stypy_varargs_param_name', None)
        Dvi._nop.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Dvi._nop.__dict__.__setitem__('stypy_call_defaults', defaults)
        Dvi._nop.__dict__.__setitem__('stypy_call_varargs', varargs)
        Dvi._nop.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Dvi._nop.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Dvi._nop', ['_'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_nop', localization, ['_'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_nop(...)' code ##################

        pass
        
        # ################# End of '_nop(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_nop' in the type store
        # Getting the type of 'stypy_return_type' (line 375)
        stypy_return_type_48732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_48732)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_nop'
        return stypy_return_type_48732


    @norecursion
    def _bop(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_bop'
        module_type_store = module_type_store.open_function_context('_bop', 379, 4, False)
        # Assigning a type to the variable 'self' (line 380)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 380, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Dvi._bop.__dict__.__setitem__('stypy_localization', localization)
        Dvi._bop.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Dvi._bop.__dict__.__setitem__('stypy_type_store', module_type_store)
        Dvi._bop.__dict__.__setitem__('stypy_function_name', 'Dvi._bop')
        Dvi._bop.__dict__.__setitem__('stypy_param_names_list', ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'p'])
        Dvi._bop.__dict__.__setitem__('stypy_varargs_param_name', None)
        Dvi._bop.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Dvi._bop.__dict__.__setitem__('stypy_call_defaults', defaults)
        Dvi._bop.__dict__.__setitem__('stypy_call_varargs', varargs)
        Dvi._bop.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Dvi._bop.__dict__.__setitem__('stypy_declared_arg_number', 12)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Dvi._bop', ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'p'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_bop', localization, ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'p'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_bop(...)' code ##################

        
        # Assigning a Attribute to a Attribute (line 381):
        
        # Assigning a Attribute to a Attribute (line 381):
        # Getting the type of '_dvistate' (line 381)
        _dvistate_48733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 21), '_dvistate')
        # Obtaining the member 'inpage' of a type (line 381)
        inpage_48734 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 381, 21), _dvistate_48733, 'inpage')
        # Getting the type of 'self' (line 381)
        self_48735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 8), 'self')
        # Setting the type of the member 'state' of a type (line 381)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 381, 8), self_48735, 'state', inpage_48734)
        
        # Assigning a Tuple to a Tuple (line 382):
        
        # Assigning a Num to a Name (line 382):
        int_48736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 382, 57), 'int')
        # Assigning a type to the variable 'tuple_assignment_47778' (line 382)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 382, 8), 'tuple_assignment_47778', int_48736)
        
        # Assigning a Num to a Name (line 382):
        int_48737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 382, 60), 'int')
        # Assigning a type to the variable 'tuple_assignment_47779' (line 382)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 382, 8), 'tuple_assignment_47779', int_48737)
        
        # Assigning a Num to a Name (line 382):
        int_48738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 382, 63), 'int')
        # Assigning a type to the variable 'tuple_assignment_47780' (line 382)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 382, 8), 'tuple_assignment_47780', int_48738)
        
        # Assigning a Num to a Name (line 382):
        int_48739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 382, 66), 'int')
        # Assigning a type to the variable 'tuple_assignment_47781' (line 382)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 382, 8), 'tuple_assignment_47781', int_48739)
        
        # Assigning a Num to a Name (line 382):
        int_48740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 382, 69), 'int')
        # Assigning a type to the variable 'tuple_assignment_47782' (line 382)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 382, 8), 'tuple_assignment_47782', int_48740)
        
        # Assigning a Num to a Name (line 382):
        int_48741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 382, 72), 'int')
        # Assigning a type to the variable 'tuple_assignment_47783' (line 382)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 382, 8), 'tuple_assignment_47783', int_48741)
        
        # Assigning a Name to a Attribute (line 382):
        # Getting the type of 'tuple_assignment_47778' (line 382)
        tuple_assignment_47778_48742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 8), 'tuple_assignment_47778')
        # Getting the type of 'self' (line 382)
        self_48743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 8), 'self')
        # Setting the type of the member 'h' of a type (line 382)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 382, 8), self_48743, 'h', tuple_assignment_47778_48742)
        
        # Assigning a Name to a Attribute (line 382):
        # Getting the type of 'tuple_assignment_47779' (line 382)
        tuple_assignment_47779_48744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 8), 'tuple_assignment_47779')
        # Getting the type of 'self' (line 382)
        self_48745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 16), 'self')
        # Setting the type of the member 'v' of a type (line 382)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 382, 16), self_48745, 'v', tuple_assignment_47779_48744)
        
        # Assigning a Name to a Attribute (line 382):
        # Getting the type of 'tuple_assignment_47780' (line 382)
        tuple_assignment_47780_48746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 8), 'tuple_assignment_47780')
        # Getting the type of 'self' (line 382)
        self_48747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 24), 'self')
        # Setting the type of the member 'w' of a type (line 382)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 382, 24), self_48747, 'w', tuple_assignment_47780_48746)
        
        # Assigning a Name to a Attribute (line 382):
        # Getting the type of 'tuple_assignment_47781' (line 382)
        tuple_assignment_47781_48748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 8), 'tuple_assignment_47781')
        # Getting the type of 'self' (line 382)
        self_48749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 32), 'self')
        # Setting the type of the member 'x' of a type (line 382)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 382, 32), self_48749, 'x', tuple_assignment_47781_48748)
        
        # Assigning a Name to a Attribute (line 382):
        # Getting the type of 'tuple_assignment_47782' (line 382)
        tuple_assignment_47782_48750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 8), 'tuple_assignment_47782')
        # Getting the type of 'self' (line 382)
        self_48751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 40), 'self')
        # Setting the type of the member 'y' of a type (line 382)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 382, 40), self_48751, 'y', tuple_assignment_47782_48750)
        
        # Assigning a Name to a Attribute (line 382):
        # Getting the type of 'tuple_assignment_47783' (line 382)
        tuple_assignment_47783_48752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 8), 'tuple_assignment_47783')
        # Getting the type of 'self' (line 382)
        self_48753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 48), 'self')
        # Setting the type of the member 'z' of a type (line 382)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 382, 48), self_48753, 'z', tuple_assignment_47783_48752)
        
        # Assigning a List to a Attribute (line 383):
        
        # Assigning a List to a Attribute (line 383):
        
        # Obtaining an instance of the builtin type 'list' (line 383)
        list_48754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 383)
        
        # Getting the type of 'self' (line 383)
        self_48755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 8), 'self')
        # Setting the type of the member 'stack' of a type (line 383)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 383, 8), self_48755, 'stack', list_48754)
        
        # Assigning a List to a Attribute (line 384):
        
        # Assigning a List to a Attribute (line 384):
        
        # Obtaining an instance of the builtin type 'list' (line 384)
        list_48756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 384)
        
        # Getting the type of 'self' (line 384)
        self_48757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 8), 'self')
        # Setting the type of the member 'text' of a type (line 384)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 384, 8), self_48757, 'text', list_48756)
        
        # Assigning a List to a Attribute (line 385):
        
        # Assigning a List to a Attribute (line 385):
        
        # Obtaining an instance of the builtin type 'list' (line 385)
        list_48758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 385, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 385)
        
        # Getting the type of 'self' (line 385)
        self_48759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 8), 'self')
        # Setting the type of the member 'boxes' of a type (line 385)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 385, 8), self_48759, 'boxes', list_48758)
        
        # ################# End of '_bop(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_bop' in the type store
        # Getting the type of 'stypy_return_type' (line 379)
        stypy_return_type_48760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_48760)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_bop'
        return stypy_return_type_48760


    @norecursion
    def _eop(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_eop'
        module_type_store = module_type_store.open_function_context('_eop', 387, 4, False)
        # Assigning a type to the variable 'self' (line 388)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 388, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Dvi._eop.__dict__.__setitem__('stypy_localization', localization)
        Dvi._eop.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Dvi._eop.__dict__.__setitem__('stypy_type_store', module_type_store)
        Dvi._eop.__dict__.__setitem__('stypy_function_name', 'Dvi._eop')
        Dvi._eop.__dict__.__setitem__('stypy_param_names_list', ['_'])
        Dvi._eop.__dict__.__setitem__('stypy_varargs_param_name', None)
        Dvi._eop.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Dvi._eop.__dict__.__setitem__('stypy_call_defaults', defaults)
        Dvi._eop.__dict__.__setitem__('stypy_call_varargs', varargs)
        Dvi._eop.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Dvi._eop.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Dvi._eop', ['_'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_eop', localization, ['_'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_eop(...)' code ##################

        
        # Assigning a Attribute to a Attribute (line 389):
        
        # Assigning a Attribute to a Attribute (line 389):
        # Getting the type of '_dvistate' (line 389)
        _dvistate_48761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 21), '_dvistate')
        # Obtaining the member 'outer' of a type (line 389)
        outer_48762 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 21), _dvistate_48761, 'outer')
        # Getting the type of 'self' (line 389)
        self_48763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 8), 'self')
        # Setting the type of the member 'state' of a type (line 389)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 8), self_48763, 'state', outer_48762)
        # Deleting a member
        # Getting the type of 'self' (line 390)
        self_48764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 8), 'self')
        module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 390, 8), self_48764, 'h')
        # Deleting a member
        # Getting the type of 'self' (line 390)
        self_48765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 8), 'self')
        module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 390, 8), self_48765, 'v')
        # Deleting a member
        # Getting the type of 'self' (line 390)
        self_48766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 8), 'self')
        module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 390, 8), self_48766, 'w')
        # Deleting a member
        # Getting the type of 'self' (line 390)
        self_48767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 8), 'self')
        module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 390, 8), self_48767, 'x')
        # Deleting a member
        # Getting the type of 'self' (line 390)
        self_48768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 8), 'self')
        module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 390, 8), self_48768, 'y')
        # Deleting a member
        # Getting the type of 'self' (line 390)
        self_48769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 8), 'self')
        module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 390, 8), self_48769, 'z')
        # Deleting a member
        # Getting the type of 'self' (line 390)
        self_48770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 8), 'self')
        module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 390, 8), self_48770, 'stack')
        
        # ################# End of '_eop(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_eop' in the type store
        # Getting the type of 'stypy_return_type' (line 387)
        stypy_return_type_48771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_48771)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_eop'
        return stypy_return_type_48771


    @norecursion
    def _push(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_push'
        module_type_store = module_type_store.open_function_context('_push', 392, 4, False)
        # Assigning a type to the variable 'self' (line 393)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 393, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Dvi._push.__dict__.__setitem__('stypy_localization', localization)
        Dvi._push.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Dvi._push.__dict__.__setitem__('stypy_type_store', module_type_store)
        Dvi._push.__dict__.__setitem__('stypy_function_name', 'Dvi._push')
        Dvi._push.__dict__.__setitem__('stypy_param_names_list', ['_'])
        Dvi._push.__dict__.__setitem__('stypy_varargs_param_name', None)
        Dvi._push.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Dvi._push.__dict__.__setitem__('stypy_call_defaults', defaults)
        Dvi._push.__dict__.__setitem__('stypy_call_varargs', varargs)
        Dvi._push.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Dvi._push.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Dvi._push', ['_'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_push', localization, ['_'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_push(...)' code ##################

        
        # Call to append(...): (line 394)
        # Processing the call arguments (line 394)
        
        # Obtaining an instance of the builtin type 'tuple' (line 394)
        tuple_48775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 27), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 394)
        # Adding element type (line 394)
        # Getting the type of 'self' (line 394)
        self_48776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 27), 'self', False)
        # Obtaining the member 'h' of a type (line 394)
        h_48777 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 27), self_48776, 'h')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 394, 27), tuple_48775, h_48777)
        # Adding element type (line 394)
        # Getting the type of 'self' (line 394)
        self_48778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 35), 'self', False)
        # Obtaining the member 'v' of a type (line 394)
        v_48779 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 35), self_48778, 'v')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 394, 27), tuple_48775, v_48779)
        # Adding element type (line 394)
        # Getting the type of 'self' (line 394)
        self_48780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 43), 'self', False)
        # Obtaining the member 'w' of a type (line 394)
        w_48781 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 43), self_48780, 'w')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 394, 27), tuple_48775, w_48781)
        # Adding element type (line 394)
        # Getting the type of 'self' (line 394)
        self_48782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 51), 'self', False)
        # Obtaining the member 'x' of a type (line 394)
        x_48783 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 51), self_48782, 'x')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 394, 27), tuple_48775, x_48783)
        # Adding element type (line 394)
        # Getting the type of 'self' (line 394)
        self_48784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 59), 'self', False)
        # Obtaining the member 'y' of a type (line 394)
        y_48785 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 59), self_48784, 'y')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 394, 27), tuple_48775, y_48785)
        # Adding element type (line 394)
        # Getting the type of 'self' (line 394)
        self_48786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 67), 'self', False)
        # Obtaining the member 'z' of a type (line 394)
        z_48787 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 67), self_48786, 'z')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 394, 27), tuple_48775, z_48787)
        
        # Processing the call keyword arguments (line 394)
        kwargs_48788 = {}
        # Getting the type of 'self' (line 394)
        self_48772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 8), 'self', False)
        # Obtaining the member 'stack' of a type (line 394)
        stack_48773 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 8), self_48772, 'stack')
        # Obtaining the member 'append' of a type (line 394)
        append_48774 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 8), stack_48773, 'append')
        # Calling append(args, kwargs) (line 394)
        append_call_result_48789 = invoke(stypy.reporting.localization.Localization(__file__, 394, 8), append_48774, *[tuple_48775], **kwargs_48788)
        
        
        # ################# End of '_push(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_push' in the type store
        # Getting the type of 'stypy_return_type' (line 392)
        stypy_return_type_48790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_48790)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_push'
        return stypy_return_type_48790


    @norecursion
    def _pop(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_pop'
        module_type_store = module_type_store.open_function_context('_pop', 396, 4, False)
        # Assigning a type to the variable 'self' (line 397)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 397, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Dvi._pop.__dict__.__setitem__('stypy_localization', localization)
        Dvi._pop.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Dvi._pop.__dict__.__setitem__('stypy_type_store', module_type_store)
        Dvi._pop.__dict__.__setitem__('stypy_function_name', 'Dvi._pop')
        Dvi._pop.__dict__.__setitem__('stypy_param_names_list', ['_'])
        Dvi._pop.__dict__.__setitem__('stypy_varargs_param_name', None)
        Dvi._pop.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Dvi._pop.__dict__.__setitem__('stypy_call_defaults', defaults)
        Dvi._pop.__dict__.__setitem__('stypy_call_varargs', varargs)
        Dvi._pop.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Dvi._pop.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Dvi._pop', ['_'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_pop', localization, ['_'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_pop(...)' code ##################

        
        # Assigning a Call to a Tuple (line 398):
        
        # Assigning a Call to a Name:
        
        # Call to pop(...): (line 398)
        # Processing the call keyword arguments (line 398)
        kwargs_48794 = {}
        # Getting the type of 'self' (line 398)
        self_48791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 57), 'self', False)
        # Obtaining the member 'stack' of a type (line 398)
        stack_48792 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 57), self_48791, 'stack')
        # Obtaining the member 'pop' of a type (line 398)
        pop_48793 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 57), stack_48792, 'pop')
        # Calling pop(args, kwargs) (line 398)
        pop_call_result_48795 = invoke(stypy.reporting.localization.Localization(__file__, 398, 57), pop_48793, *[], **kwargs_48794)
        
        # Assigning a type to the variable 'call_assignment_47784' (line 398)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 398, 8), 'call_assignment_47784', pop_call_result_48795)
        
        # Assigning a Call to a Name (line 398):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_48798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 8), 'int')
        # Processing the call keyword arguments
        kwargs_48799 = {}
        # Getting the type of 'call_assignment_47784' (line 398)
        call_assignment_47784_48796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 8), 'call_assignment_47784', False)
        # Obtaining the member '__getitem__' of a type (line 398)
        getitem___48797 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 8), call_assignment_47784_48796, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_48800 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___48797, *[int_48798], **kwargs_48799)
        
        # Assigning a type to the variable 'call_assignment_47785' (line 398)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 398, 8), 'call_assignment_47785', getitem___call_result_48800)
        
        # Assigning a Name to a Attribute (line 398):
        # Getting the type of 'call_assignment_47785' (line 398)
        call_assignment_47785_48801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 8), 'call_assignment_47785')
        # Getting the type of 'self' (line 398)
        self_48802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 8), 'self')
        # Setting the type of the member 'h' of a type (line 398)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 8), self_48802, 'h', call_assignment_47785_48801)
        
        # Assigning a Call to a Name (line 398):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_48805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 8), 'int')
        # Processing the call keyword arguments
        kwargs_48806 = {}
        # Getting the type of 'call_assignment_47784' (line 398)
        call_assignment_47784_48803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 8), 'call_assignment_47784', False)
        # Obtaining the member '__getitem__' of a type (line 398)
        getitem___48804 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 8), call_assignment_47784_48803, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_48807 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___48804, *[int_48805], **kwargs_48806)
        
        # Assigning a type to the variable 'call_assignment_47786' (line 398)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 398, 8), 'call_assignment_47786', getitem___call_result_48807)
        
        # Assigning a Name to a Attribute (line 398):
        # Getting the type of 'call_assignment_47786' (line 398)
        call_assignment_47786_48808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 8), 'call_assignment_47786')
        # Getting the type of 'self' (line 398)
        self_48809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 16), 'self')
        # Setting the type of the member 'v' of a type (line 398)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 16), self_48809, 'v', call_assignment_47786_48808)
        
        # Assigning a Call to a Name (line 398):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_48812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 8), 'int')
        # Processing the call keyword arguments
        kwargs_48813 = {}
        # Getting the type of 'call_assignment_47784' (line 398)
        call_assignment_47784_48810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 8), 'call_assignment_47784', False)
        # Obtaining the member '__getitem__' of a type (line 398)
        getitem___48811 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 8), call_assignment_47784_48810, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_48814 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___48811, *[int_48812], **kwargs_48813)
        
        # Assigning a type to the variable 'call_assignment_47787' (line 398)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 398, 8), 'call_assignment_47787', getitem___call_result_48814)
        
        # Assigning a Name to a Attribute (line 398):
        # Getting the type of 'call_assignment_47787' (line 398)
        call_assignment_47787_48815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 8), 'call_assignment_47787')
        # Getting the type of 'self' (line 398)
        self_48816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 24), 'self')
        # Setting the type of the member 'w' of a type (line 398)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 24), self_48816, 'w', call_assignment_47787_48815)
        
        # Assigning a Call to a Name (line 398):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_48819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 8), 'int')
        # Processing the call keyword arguments
        kwargs_48820 = {}
        # Getting the type of 'call_assignment_47784' (line 398)
        call_assignment_47784_48817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 8), 'call_assignment_47784', False)
        # Obtaining the member '__getitem__' of a type (line 398)
        getitem___48818 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 8), call_assignment_47784_48817, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_48821 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___48818, *[int_48819], **kwargs_48820)
        
        # Assigning a type to the variable 'call_assignment_47788' (line 398)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 398, 8), 'call_assignment_47788', getitem___call_result_48821)
        
        # Assigning a Name to a Attribute (line 398):
        # Getting the type of 'call_assignment_47788' (line 398)
        call_assignment_47788_48822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 8), 'call_assignment_47788')
        # Getting the type of 'self' (line 398)
        self_48823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 32), 'self')
        # Setting the type of the member 'x' of a type (line 398)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 32), self_48823, 'x', call_assignment_47788_48822)
        
        # Assigning a Call to a Name (line 398):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_48826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 8), 'int')
        # Processing the call keyword arguments
        kwargs_48827 = {}
        # Getting the type of 'call_assignment_47784' (line 398)
        call_assignment_47784_48824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 8), 'call_assignment_47784', False)
        # Obtaining the member '__getitem__' of a type (line 398)
        getitem___48825 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 8), call_assignment_47784_48824, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_48828 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___48825, *[int_48826], **kwargs_48827)
        
        # Assigning a type to the variable 'call_assignment_47789' (line 398)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 398, 8), 'call_assignment_47789', getitem___call_result_48828)
        
        # Assigning a Name to a Attribute (line 398):
        # Getting the type of 'call_assignment_47789' (line 398)
        call_assignment_47789_48829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 8), 'call_assignment_47789')
        # Getting the type of 'self' (line 398)
        self_48830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 40), 'self')
        # Setting the type of the member 'y' of a type (line 398)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 40), self_48830, 'y', call_assignment_47789_48829)
        
        # Assigning a Call to a Name (line 398):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_48833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 8), 'int')
        # Processing the call keyword arguments
        kwargs_48834 = {}
        # Getting the type of 'call_assignment_47784' (line 398)
        call_assignment_47784_48831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 8), 'call_assignment_47784', False)
        # Obtaining the member '__getitem__' of a type (line 398)
        getitem___48832 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 8), call_assignment_47784_48831, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_48835 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___48832, *[int_48833], **kwargs_48834)
        
        # Assigning a type to the variable 'call_assignment_47790' (line 398)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 398, 8), 'call_assignment_47790', getitem___call_result_48835)
        
        # Assigning a Name to a Attribute (line 398):
        # Getting the type of 'call_assignment_47790' (line 398)
        call_assignment_47790_48836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 8), 'call_assignment_47790')
        # Getting the type of 'self' (line 398)
        self_48837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 48), 'self')
        # Setting the type of the member 'z' of a type (line 398)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 48), self_48837, 'z', call_assignment_47790_48836)
        
        # ################# End of '_pop(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_pop' in the type store
        # Getting the type of 'stypy_return_type' (line 396)
        stypy_return_type_48838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_48838)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_pop'
        return stypy_return_type_48838


    @norecursion
    def _right(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_right'
        module_type_store = module_type_store.open_function_context('_right', 400, 4, False)
        # Assigning a type to the variable 'self' (line 401)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 401, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Dvi._right.__dict__.__setitem__('stypy_localization', localization)
        Dvi._right.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Dvi._right.__dict__.__setitem__('stypy_type_store', module_type_store)
        Dvi._right.__dict__.__setitem__('stypy_function_name', 'Dvi._right')
        Dvi._right.__dict__.__setitem__('stypy_param_names_list', ['b'])
        Dvi._right.__dict__.__setitem__('stypy_varargs_param_name', None)
        Dvi._right.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Dvi._right.__dict__.__setitem__('stypy_call_defaults', defaults)
        Dvi._right.__dict__.__setitem__('stypy_call_varargs', varargs)
        Dvi._right.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Dvi._right.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Dvi._right', ['b'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_right', localization, ['b'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_right(...)' code ##################

        
        # Getting the type of 'self' (line 402)
        self_48839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 8), 'self')
        # Obtaining the member 'h' of a type (line 402)
        h_48840 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 402, 8), self_48839, 'h')
        # Getting the type of 'b' (line 402)
        b_48841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 18), 'b')
        # Applying the binary operator '+=' (line 402)
        result_iadd_48842 = python_operator(stypy.reporting.localization.Localization(__file__, 402, 8), '+=', h_48840, b_48841)
        # Getting the type of 'self' (line 402)
        self_48843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 8), 'self')
        # Setting the type of the member 'h' of a type (line 402)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 402, 8), self_48843, 'h', result_iadd_48842)
        
        
        # ################# End of '_right(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_right' in the type store
        # Getting the type of 'stypy_return_type' (line 400)
        stypy_return_type_48844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_48844)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_right'
        return stypy_return_type_48844


    @norecursion
    def _right_w(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_right_w'
        module_type_store = module_type_store.open_function_context('_right_w', 404, 4, False)
        # Assigning a type to the variable 'self' (line 405)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 405, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Dvi._right_w.__dict__.__setitem__('stypy_localization', localization)
        Dvi._right_w.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Dvi._right_w.__dict__.__setitem__('stypy_type_store', module_type_store)
        Dvi._right_w.__dict__.__setitem__('stypy_function_name', 'Dvi._right_w')
        Dvi._right_w.__dict__.__setitem__('stypy_param_names_list', ['new_w'])
        Dvi._right_w.__dict__.__setitem__('stypy_varargs_param_name', None)
        Dvi._right_w.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Dvi._right_w.__dict__.__setitem__('stypy_call_defaults', defaults)
        Dvi._right_w.__dict__.__setitem__('stypy_call_varargs', varargs)
        Dvi._right_w.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Dvi._right_w.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Dvi._right_w', ['new_w'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_right_w', localization, ['new_w'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_right_w(...)' code ##################

        
        # Type idiom detected: calculating its left and rigth part (line 406)
        # Getting the type of 'new_w' (line 406)
        new_w_48845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 8), 'new_w')
        # Getting the type of 'None' (line 406)
        None_48846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 24), 'None')
        
        (may_be_48847, more_types_in_union_48848) = may_not_be_none(new_w_48845, None_48846)

        if may_be_48847:

            if more_types_in_union_48848:
                # Runtime conditional SSA (line 406)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Name to a Attribute (line 407):
            
            # Assigning a Name to a Attribute (line 407):
            # Getting the type of 'new_w' (line 407)
            new_w_48849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 21), 'new_w')
            # Getting the type of 'self' (line 407)
            self_48850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 12), 'self')
            # Setting the type of the member 'w' of a type (line 407)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 407, 12), self_48850, 'w', new_w_48849)

            if more_types_in_union_48848:
                # SSA join for if statement (line 406)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Getting the type of 'self' (line 408)
        self_48851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 8), 'self')
        # Obtaining the member 'h' of a type (line 408)
        h_48852 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 408, 8), self_48851, 'h')
        # Getting the type of 'self' (line 408)
        self_48853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 18), 'self')
        # Obtaining the member 'w' of a type (line 408)
        w_48854 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 408, 18), self_48853, 'w')
        # Applying the binary operator '+=' (line 408)
        result_iadd_48855 = python_operator(stypy.reporting.localization.Localization(__file__, 408, 8), '+=', h_48852, w_48854)
        # Getting the type of 'self' (line 408)
        self_48856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 8), 'self')
        # Setting the type of the member 'h' of a type (line 408)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 408, 8), self_48856, 'h', result_iadd_48855)
        
        
        # ################# End of '_right_w(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_right_w' in the type store
        # Getting the type of 'stypy_return_type' (line 404)
        stypy_return_type_48857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_48857)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_right_w'
        return stypy_return_type_48857


    @norecursion
    def _right_x(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_right_x'
        module_type_store = module_type_store.open_function_context('_right_x', 410, 4, False)
        # Assigning a type to the variable 'self' (line 411)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 411, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Dvi._right_x.__dict__.__setitem__('stypy_localization', localization)
        Dvi._right_x.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Dvi._right_x.__dict__.__setitem__('stypy_type_store', module_type_store)
        Dvi._right_x.__dict__.__setitem__('stypy_function_name', 'Dvi._right_x')
        Dvi._right_x.__dict__.__setitem__('stypy_param_names_list', ['new_x'])
        Dvi._right_x.__dict__.__setitem__('stypy_varargs_param_name', None)
        Dvi._right_x.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Dvi._right_x.__dict__.__setitem__('stypy_call_defaults', defaults)
        Dvi._right_x.__dict__.__setitem__('stypy_call_varargs', varargs)
        Dvi._right_x.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Dvi._right_x.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Dvi._right_x', ['new_x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_right_x', localization, ['new_x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_right_x(...)' code ##################

        
        # Type idiom detected: calculating its left and rigth part (line 412)
        # Getting the type of 'new_x' (line 412)
        new_x_48858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 8), 'new_x')
        # Getting the type of 'None' (line 412)
        None_48859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 24), 'None')
        
        (may_be_48860, more_types_in_union_48861) = may_not_be_none(new_x_48858, None_48859)

        if may_be_48860:

            if more_types_in_union_48861:
                # Runtime conditional SSA (line 412)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Name to a Attribute (line 413):
            
            # Assigning a Name to a Attribute (line 413):
            # Getting the type of 'new_x' (line 413)
            new_x_48862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 21), 'new_x')
            # Getting the type of 'self' (line 413)
            self_48863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 12), 'self')
            # Setting the type of the member 'x' of a type (line 413)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 413, 12), self_48863, 'x', new_x_48862)

            if more_types_in_union_48861:
                # SSA join for if statement (line 412)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Getting the type of 'self' (line 414)
        self_48864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 8), 'self')
        # Obtaining the member 'h' of a type (line 414)
        h_48865 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 414, 8), self_48864, 'h')
        # Getting the type of 'self' (line 414)
        self_48866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 18), 'self')
        # Obtaining the member 'x' of a type (line 414)
        x_48867 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 414, 18), self_48866, 'x')
        # Applying the binary operator '+=' (line 414)
        result_iadd_48868 = python_operator(stypy.reporting.localization.Localization(__file__, 414, 8), '+=', h_48865, x_48867)
        # Getting the type of 'self' (line 414)
        self_48869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 8), 'self')
        # Setting the type of the member 'h' of a type (line 414)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 414, 8), self_48869, 'h', result_iadd_48868)
        
        
        # ################# End of '_right_x(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_right_x' in the type store
        # Getting the type of 'stypy_return_type' (line 410)
        stypy_return_type_48870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_48870)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_right_x'
        return stypy_return_type_48870


    @norecursion
    def _down(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_down'
        module_type_store = module_type_store.open_function_context('_down', 416, 4, False)
        # Assigning a type to the variable 'self' (line 417)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 417, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Dvi._down.__dict__.__setitem__('stypy_localization', localization)
        Dvi._down.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Dvi._down.__dict__.__setitem__('stypy_type_store', module_type_store)
        Dvi._down.__dict__.__setitem__('stypy_function_name', 'Dvi._down')
        Dvi._down.__dict__.__setitem__('stypy_param_names_list', ['a'])
        Dvi._down.__dict__.__setitem__('stypy_varargs_param_name', None)
        Dvi._down.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Dvi._down.__dict__.__setitem__('stypy_call_defaults', defaults)
        Dvi._down.__dict__.__setitem__('stypy_call_varargs', varargs)
        Dvi._down.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Dvi._down.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Dvi._down', ['a'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_down', localization, ['a'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_down(...)' code ##################

        
        # Getting the type of 'self' (line 418)
        self_48871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 8), 'self')
        # Obtaining the member 'v' of a type (line 418)
        v_48872 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 418, 8), self_48871, 'v')
        # Getting the type of 'a' (line 418)
        a_48873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 18), 'a')
        # Applying the binary operator '+=' (line 418)
        result_iadd_48874 = python_operator(stypy.reporting.localization.Localization(__file__, 418, 8), '+=', v_48872, a_48873)
        # Getting the type of 'self' (line 418)
        self_48875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 8), 'self')
        # Setting the type of the member 'v' of a type (line 418)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 418, 8), self_48875, 'v', result_iadd_48874)
        
        
        # ################# End of '_down(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_down' in the type store
        # Getting the type of 'stypy_return_type' (line 416)
        stypy_return_type_48876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_48876)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_down'
        return stypy_return_type_48876


    @norecursion
    def _down_y(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_down_y'
        module_type_store = module_type_store.open_function_context('_down_y', 420, 4, False)
        # Assigning a type to the variable 'self' (line 421)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 421, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Dvi._down_y.__dict__.__setitem__('stypy_localization', localization)
        Dvi._down_y.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Dvi._down_y.__dict__.__setitem__('stypy_type_store', module_type_store)
        Dvi._down_y.__dict__.__setitem__('stypy_function_name', 'Dvi._down_y')
        Dvi._down_y.__dict__.__setitem__('stypy_param_names_list', ['new_y'])
        Dvi._down_y.__dict__.__setitem__('stypy_varargs_param_name', None)
        Dvi._down_y.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Dvi._down_y.__dict__.__setitem__('stypy_call_defaults', defaults)
        Dvi._down_y.__dict__.__setitem__('stypy_call_varargs', varargs)
        Dvi._down_y.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Dvi._down_y.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Dvi._down_y', ['new_y'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_down_y', localization, ['new_y'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_down_y(...)' code ##################

        
        # Type idiom detected: calculating its left and rigth part (line 422)
        # Getting the type of 'new_y' (line 422)
        new_y_48877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 8), 'new_y')
        # Getting the type of 'None' (line 422)
        None_48878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 24), 'None')
        
        (may_be_48879, more_types_in_union_48880) = may_not_be_none(new_y_48877, None_48878)

        if may_be_48879:

            if more_types_in_union_48880:
                # Runtime conditional SSA (line 422)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Name to a Attribute (line 423):
            
            # Assigning a Name to a Attribute (line 423):
            # Getting the type of 'new_y' (line 423)
            new_y_48881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 21), 'new_y')
            # Getting the type of 'self' (line 423)
            self_48882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 12), 'self')
            # Setting the type of the member 'y' of a type (line 423)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 423, 12), self_48882, 'y', new_y_48881)

            if more_types_in_union_48880:
                # SSA join for if statement (line 422)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Getting the type of 'self' (line 424)
        self_48883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 8), 'self')
        # Obtaining the member 'v' of a type (line 424)
        v_48884 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 424, 8), self_48883, 'v')
        # Getting the type of 'self' (line 424)
        self_48885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 18), 'self')
        # Obtaining the member 'y' of a type (line 424)
        y_48886 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 424, 18), self_48885, 'y')
        # Applying the binary operator '+=' (line 424)
        result_iadd_48887 = python_operator(stypy.reporting.localization.Localization(__file__, 424, 8), '+=', v_48884, y_48886)
        # Getting the type of 'self' (line 424)
        self_48888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 8), 'self')
        # Setting the type of the member 'v' of a type (line 424)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 424, 8), self_48888, 'v', result_iadd_48887)
        
        
        # ################# End of '_down_y(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_down_y' in the type store
        # Getting the type of 'stypy_return_type' (line 420)
        stypy_return_type_48889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_48889)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_down_y'
        return stypy_return_type_48889


    @norecursion
    def _down_z(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_down_z'
        module_type_store = module_type_store.open_function_context('_down_z', 426, 4, False)
        # Assigning a type to the variable 'self' (line 427)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 427, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Dvi._down_z.__dict__.__setitem__('stypy_localization', localization)
        Dvi._down_z.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Dvi._down_z.__dict__.__setitem__('stypy_type_store', module_type_store)
        Dvi._down_z.__dict__.__setitem__('stypy_function_name', 'Dvi._down_z')
        Dvi._down_z.__dict__.__setitem__('stypy_param_names_list', ['new_z'])
        Dvi._down_z.__dict__.__setitem__('stypy_varargs_param_name', None)
        Dvi._down_z.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Dvi._down_z.__dict__.__setitem__('stypy_call_defaults', defaults)
        Dvi._down_z.__dict__.__setitem__('stypy_call_varargs', varargs)
        Dvi._down_z.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Dvi._down_z.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Dvi._down_z', ['new_z'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_down_z', localization, ['new_z'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_down_z(...)' code ##################

        
        # Type idiom detected: calculating its left and rigth part (line 428)
        # Getting the type of 'new_z' (line 428)
        new_z_48890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 8), 'new_z')
        # Getting the type of 'None' (line 428)
        None_48891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 24), 'None')
        
        (may_be_48892, more_types_in_union_48893) = may_not_be_none(new_z_48890, None_48891)

        if may_be_48892:

            if more_types_in_union_48893:
                # Runtime conditional SSA (line 428)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Name to a Attribute (line 429):
            
            # Assigning a Name to a Attribute (line 429):
            # Getting the type of 'new_z' (line 429)
            new_z_48894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 21), 'new_z')
            # Getting the type of 'self' (line 429)
            self_48895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 12), 'self')
            # Setting the type of the member 'z' of a type (line 429)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 429, 12), self_48895, 'z', new_z_48894)

            if more_types_in_union_48893:
                # SSA join for if statement (line 428)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Getting the type of 'self' (line 430)
        self_48896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 8), 'self')
        # Obtaining the member 'v' of a type (line 430)
        v_48897 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 8), self_48896, 'v')
        # Getting the type of 'self' (line 430)
        self_48898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 18), 'self')
        # Obtaining the member 'z' of a type (line 430)
        z_48899 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 18), self_48898, 'z')
        # Applying the binary operator '+=' (line 430)
        result_iadd_48900 = python_operator(stypy.reporting.localization.Localization(__file__, 430, 8), '+=', v_48897, z_48899)
        # Getting the type of 'self' (line 430)
        self_48901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 8), 'self')
        # Setting the type of the member 'v' of a type (line 430)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 8), self_48901, 'v', result_iadd_48900)
        
        
        # ################# End of '_down_z(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_down_z' in the type store
        # Getting the type of 'stypy_return_type' (line 426)
        stypy_return_type_48902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_48902)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_down_z'
        return stypy_return_type_48902


    @norecursion
    def _fnt_num_immediate(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_fnt_num_immediate'
        module_type_store = module_type_store.open_function_context('_fnt_num_immediate', 432, 4, False)
        # Assigning a type to the variable 'self' (line 433)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 433, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Dvi._fnt_num_immediate.__dict__.__setitem__('stypy_localization', localization)
        Dvi._fnt_num_immediate.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Dvi._fnt_num_immediate.__dict__.__setitem__('stypy_type_store', module_type_store)
        Dvi._fnt_num_immediate.__dict__.__setitem__('stypy_function_name', 'Dvi._fnt_num_immediate')
        Dvi._fnt_num_immediate.__dict__.__setitem__('stypy_param_names_list', ['k'])
        Dvi._fnt_num_immediate.__dict__.__setitem__('stypy_varargs_param_name', None)
        Dvi._fnt_num_immediate.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Dvi._fnt_num_immediate.__dict__.__setitem__('stypy_call_defaults', defaults)
        Dvi._fnt_num_immediate.__dict__.__setitem__('stypy_call_varargs', varargs)
        Dvi._fnt_num_immediate.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Dvi._fnt_num_immediate.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Dvi._fnt_num_immediate', ['k'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_fnt_num_immediate', localization, ['k'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_fnt_num_immediate(...)' code ##################

        
        # Assigning a Name to a Attribute (line 434):
        
        # Assigning a Name to a Attribute (line 434):
        # Getting the type of 'k' (line 434)
        k_48903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 17), 'k')
        # Getting the type of 'self' (line 434)
        self_48904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 8), 'self')
        # Setting the type of the member 'f' of a type (line 434)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 434, 8), self_48904, 'f', k_48903)
        
        # ################# End of '_fnt_num_immediate(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_fnt_num_immediate' in the type store
        # Getting the type of 'stypy_return_type' (line 432)
        stypy_return_type_48905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_48905)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_fnt_num_immediate'
        return stypy_return_type_48905


    @norecursion
    def _fnt_num(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_fnt_num'
        module_type_store = module_type_store.open_function_context('_fnt_num', 436, 4, False)
        # Assigning a type to the variable 'self' (line 437)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 437, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Dvi._fnt_num.__dict__.__setitem__('stypy_localization', localization)
        Dvi._fnt_num.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Dvi._fnt_num.__dict__.__setitem__('stypy_type_store', module_type_store)
        Dvi._fnt_num.__dict__.__setitem__('stypy_function_name', 'Dvi._fnt_num')
        Dvi._fnt_num.__dict__.__setitem__('stypy_param_names_list', ['new_f'])
        Dvi._fnt_num.__dict__.__setitem__('stypy_varargs_param_name', None)
        Dvi._fnt_num.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Dvi._fnt_num.__dict__.__setitem__('stypy_call_defaults', defaults)
        Dvi._fnt_num.__dict__.__setitem__('stypy_call_varargs', varargs)
        Dvi._fnt_num.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Dvi._fnt_num.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Dvi._fnt_num', ['new_f'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_fnt_num', localization, ['new_f'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_fnt_num(...)' code ##################

        
        # Assigning a Name to a Attribute (line 438):
        
        # Assigning a Name to a Attribute (line 438):
        # Getting the type of 'new_f' (line 438)
        new_f_48906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 17), 'new_f')
        # Getting the type of 'self' (line 438)
        self_48907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 8), 'self')
        # Setting the type of the member 'f' of a type (line 438)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 438, 8), self_48907, 'f', new_f_48906)
        
        # ################# End of '_fnt_num(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_fnt_num' in the type store
        # Getting the type of 'stypy_return_type' (line 436)
        stypy_return_type_48908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_48908)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_fnt_num'
        return stypy_return_type_48908


    @norecursion
    def _xxx(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_xxx'
        module_type_store = module_type_store.open_function_context('_xxx', 440, 4, False)
        # Assigning a type to the variable 'self' (line 441)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 441, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Dvi._xxx.__dict__.__setitem__('stypy_localization', localization)
        Dvi._xxx.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Dvi._xxx.__dict__.__setitem__('stypy_type_store', module_type_store)
        Dvi._xxx.__dict__.__setitem__('stypy_function_name', 'Dvi._xxx')
        Dvi._xxx.__dict__.__setitem__('stypy_param_names_list', ['datalen'])
        Dvi._xxx.__dict__.__setitem__('stypy_varargs_param_name', None)
        Dvi._xxx.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Dvi._xxx.__dict__.__setitem__('stypy_call_defaults', defaults)
        Dvi._xxx.__dict__.__setitem__('stypy_call_varargs', varargs)
        Dvi._xxx.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Dvi._xxx.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Dvi._xxx', ['datalen'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_xxx', localization, ['datalen'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_xxx(...)' code ##################

        
        # Assigning a Call to a Name (line 442):
        
        # Assigning a Call to a Name (line 442):
        
        # Call to read(...): (line 442)
        # Processing the call arguments (line 442)
        # Getting the type of 'datalen' (line 442)
        datalen_48912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 33), 'datalen', False)
        # Processing the call keyword arguments (line 442)
        kwargs_48913 = {}
        # Getting the type of 'self' (line 442)
        self_48909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 18), 'self', False)
        # Obtaining the member 'file' of a type (line 442)
        file_48910 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 442, 18), self_48909, 'file')
        # Obtaining the member 'read' of a type (line 442)
        read_48911 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 442, 18), file_48910, 'read')
        # Calling read(args, kwargs) (line 442)
        read_call_result_48914 = invoke(stypy.reporting.localization.Localization(__file__, 442, 18), read_48911, *[datalen_48912], **kwargs_48913)
        
        # Assigning a type to the variable 'special' (line 442)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 442, 8), 'special', read_call_result_48914)
        
        # Getting the type of 'six' (line 443)
        six_48915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 11), 'six')
        # Obtaining the member 'PY3' of a type (line 443)
        PY3_48916 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 443, 11), six_48915, 'PY3')
        # Testing the type of an if condition (line 443)
        if_condition_48917 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 443, 8), PY3_48916)
        # Assigning a type to the variable 'if_condition_48917' (line 443)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 443, 8), 'if_condition_48917', if_condition_48917)
        # SSA begins for if statement (line 443)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 444):
        
        # Assigning a Name to a Name (line 444):
        # Getting the type of 'chr' (line 444)
        chr_48918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 19), 'chr')
        # Assigning a type to the variable 'chr_' (line 444)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 12), 'chr_', chr_48918)
        # SSA branch for the else part of an if statement (line 443)
        module_type_store.open_ssa_branch('else')

        @norecursion
        def chr_(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'chr_'
            module_type_store = module_type_store.open_function_context('chr_', 446, 12, False)
            
            # Passed parameters checking function
            chr_.stypy_localization = localization
            chr_.stypy_type_of_self = None
            chr_.stypy_type_store = module_type_store
            chr_.stypy_function_name = 'chr_'
            chr_.stypy_param_names_list = ['x']
            chr_.stypy_varargs_param_name = None
            chr_.stypy_kwargs_param_name = None
            chr_.stypy_call_defaults = defaults
            chr_.stypy_call_varargs = varargs
            chr_.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'chr_', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'chr_', localization, ['x'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'chr_(...)' code ##################

            # Getting the type of 'x' (line 447)
            x_48919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 23), 'x')
            # Assigning a type to the variable 'stypy_return_type' (line 447)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 447, 16), 'stypy_return_type', x_48919)
            
            # ################# End of 'chr_(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'chr_' in the type store
            # Getting the type of 'stypy_return_type' (line 446)
            stypy_return_type_48920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 12), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_48920)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'chr_'
            return stypy_return_type_48920

        # Assigning a type to the variable 'chr_' (line 446)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 446, 12), 'chr_', chr_)
        # SSA join for if statement (line 443)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to report(...): (line 448)
        # Processing the call arguments (line 448)
        unicode_48924 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 449, 12), 'unicode', u'Dvi._xxx: encountered special: %s')
        
        # Call to join(...): (line 450)
        # Processing the call arguments (line 450)
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'special' (line 452)
        special_48948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 33), 'special', False)
        comprehension_48949 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 450, 23), special_48948)
        # Assigning a type to the variable 'ch' (line 450)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 450, 23), 'ch', comprehension_48949)
        
        # Evaluating a boolean operation
        
        # Evaluating a boolean operation
        
        int_48927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 450, 24), 'int')
        
        # Call to ord(...): (line 450)
        # Processing the call arguments (line 450)
        # Getting the type of 'ch' (line 450)
        ch_48929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 34), 'ch', False)
        # Processing the call keyword arguments (line 450)
        kwargs_48930 = {}
        # Getting the type of 'ord' (line 450)
        ord_48928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 30), 'ord', False)
        # Calling ord(args, kwargs) (line 450)
        ord_call_result_48931 = invoke(stypy.reporting.localization.Localization(__file__, 450, 30), ord_48928, *[ch_48929], **kwargs_48930)
        
        # Applying the binary operator '<=' (line 450)
        result_le_48932 = python_operator(stypy.reporting.localization.Localization(__file__, 450, 24), '<=', int_48927, ord_call_result_48931)
        int_48933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 450, 40), 'int')
        # Applying the binary operator '<' (line 450)
        result_lt_48934 = python_operator(stypy.reporting.localization.Localization(__file__, 450, 24), '<', ord_call_result_48931, int_48933)
        # Applying the binary operator '&' (line 450)
        result_and__48935 = python_operator(stypy.reporting.localization.Localization(__file__, 450, 24), '&', result_le_48932, result_lt_48934)
        
        
        # Call to chr_(...): (line 450)
        # Processing the call arguments (line 450)
        # Getting the type of 'ch' (line 450)
        ch_48937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 54), 'ch', False)
        # Processing the call keyword arguments (line 450)
        kwargs_48938 = {}
        # Getting the type of 'chr_' (line 450)
        chr__48936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 49), 'chr_', False)
        # Calling chr_(args, kwargs) (line 450)
        chr__call_result_48939 = invoke(stypy.reporting.localization.Localization(__file__, 450, 49), chr__48936, *[ch_48937], **kwargs_48938)
        
        # Applying the binary operator 'and' (line 450)
        result_and_keyword_48940 = python_operator(stypy.reporting.localization.Localization(__file__, 450, 23), 'and', result_and__48935, chr__call_result_48939)
        
        unicode_48941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 451, 26), 'unicode', u'<%02x>')
        
        # Call to ord(...): (line 451)
        # Processing the call arguments (line 451)
        # Getting the type of 'ch' (line 451)
        ch_48943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 41), 'ch', False)
        # Processing the call keyword arguments (line 451)
        kwargs_48944 = {}
        # Getting the type of 'ord' (line 451)
        ord_48942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 37), 'ord', False)
        # Calling ord(args, kwargs) (line 451)
        ord_call_result_48945 = invoke(stypy.reporting.localization.Localization(__file__, 451, 37), ord_48942, *[ch_48943], **kwargs_48944)
        
        # Applying the binary operator '%' (line 451)
        result_mod_48946 = python_operator(stypy.reporting.localization.Localization(__file__, 451, 26), '%', unicode_48941, ord_call_result_48945)
        
        # Applying the binary operator 'or' (line 450)
        result_or_keyword_48947 = python_operator(stypy.reporting.localization.Localization(__file__, 450, 23), 'or', result_and_keyword_48940, result_mod_48946)
        
        list_48950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 450, 23), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 450, 23), list_48950, result_or_keyword_48947)
        # Processing the call keyword arguments (line 450)
        kwargs_48951 = {}
        unicode_48925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 450, 14), 'unicode', u'')
        # Obtaining the member 'join' of a type (line 450)
        join_48926 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 450, 14), unicode_48925, 'join')
        # Calling join(args, kwargs) (line 450)
        join_call_result_48952 = invoke(stypy.reporting.localization.Localization(__file__, 450, 14), join_48926, *[list_48950], **kwargs_48951)
        
        # Applying the binary operator '%' (line 449)
        result_mod_48953 = python_operator(stypy.reporting.localization.Localization(__file__, 449, 12), '%', unicode_48924, join_call_result_48952)
        
        unicode_48954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 453, 12), 'unicode', u'debug')
        # Processing the call keyword arguments (line 448)
        kwargs_48955 = {}
        # Getting the type of 'matplotlib' (line 448)
        matplotlib_48921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 8), 'matplotlib', False)
        # Obtaining the member 'verbose' of a type (line 448)
        verbose_48922 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 448, 8), matplotlib_48921, 'verbose')
        # Obtaining the member 'report' of a type (line 448)
        report_48923 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 448, 8), verbose_48922, 'report')
        # Calling report(args, kwargs) (line 448)
        report_call_result_48956 = invoke(stypy.reporting.localization.Localization(__file__, 448, 8), report_48923, *[result_mod_48953, unicode_48954], **kwargs_48955)
        
        
        # ################# End of '_xxx(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_xxx' in the type store
        # Getting the type of 'stypy_return_type' (line 440)
        stypy_return_type_48957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_48957)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_xxx'
        return stypy_return_type_48957


    @norecursion
    def _fnt_def(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_fnt_def'
        module_type_store = module_type_store.open_function_context('_fnt_def', 455, 4, False)
        # Assigning a type to the variable 'self' (line 456)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 456, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Dvi._fnt_def.__dict__.__setitem__('stypy_localization', localization)
        Dvi._fnt_def.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Dvi._fnt_def.__dict__.__setitem__('stypy_type_store', module_type_store)
        Dvi._fnt_def.__dict__.__setitem__('stypy_function_name', 'Dvi._fnt_def')
        Dvi._fnt_def.__dict__.__setitem__('stypy_param_names_list', ['k', 'c', 's', 'd', 'a', 'l'])
        Dvi._fnt_def.__dict__.__setitem__('stypy_varargs_param_name', None)
        Dvi._fnt_def.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Dvi._fnt_def.__dict__.__setitem__('stypy_call_defaults', defaults)
        Dvi._fnt_def.__dict__.__setitem__('stypy_call_varargs', varargs)
        Dvi._fnt_def.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Dvi._fnt_def.__dict__.__setitem__('stypy_declared_arg_number', 7)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Dvi._fnt_def', ['k', 'c', 's', 'd', 'a', 'l'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_fnt_def', localization, ['k', 'c', 's', 'd', 'a', 'l'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_fnt_def(...)' code ##################

        
        # Call to _fnt_def_real(...): (line 457)
        # Processing the call arguments (line 457)
        # Getting the type of 'k' (line 457)
        k_48960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 27), 'k', False)
        # Getting the type of 'c' (line 457)
        c_48961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 30), 'c', False)
        # Getting the type of 's' (line 457)
        s_48962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 33), 's', False)
        # Getting the type of 'd' (line 457)
        d_48963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 36), 'd', False)
        # Getting the type of 'a' (line 457)
        a_48964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 39), 'a', False)
        # Getting the type of 'l' (line 457)
        l_48965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 42), 'l', False)
        # Processing the call keyword arguments (line 457)
        kwargs_48966 = {}
        # Getting the type of 'self' (line 457)
        self_48958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 8), 'self', False)
        # Obtaining the member '_fnt_def_real' of a type (line 457)
        _fnt_def_real_48959 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 457, 8), self_48958, '_fnt_def_real')
        # Calling _fnt_def_real(args, kwargs) (line 457)
        _fnt_def_real_call_result_48967 = invoke(stypy.reporting.localization.Localization(__file__, 457, 8), _fnt_def_real_48959, *[k_48960, c_48961, s_48962, d_48963, a_48964, l_48965], **kwargs_48966)
        
        
        # ################# End of '_fnt_def(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_fnt_def' in the type store
        # Getting the type of 'stypy_return_type' (line 455)
        stypy_return_type_48968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_48968)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_fnt_def'
        return stypy_return_type_48968


    @norecursion
    def _fnt_def_real(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_fnt_def_real'
        module_type_store = module_type_store.open_function_context('_fnt_def_real', 459, 4, False)
        # Assigning a type to the variable 'self' (line 460)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 460, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Dvi._fnt_def_real.__dict__.__setitem__('stypy_localization', localization)
        Dvi._fnt_def_real.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Dvi._fnt_def_real.__dict__.__setitem__('stypy_type_store', module_type_store)
        Dvi._fnt_def_real.__dict__.__setitem__('stypy_function_name', 'Dvi._fnt_def_real')
        Dvi._fnt_def_real.__dict__.__setitem__('stypy_param_names_list', ['k', 'c', 's', 'd', 'a', 'l'])
        Dvi._fnt_def_real.__dict__.__setitem__('stypy_varargs_param_name', None)
        Dvi._fnt_def_real.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Dvi._fnt_def_real.__dict__.__setitem__('stypy_call_defaults', defaults)
        Dvi._fnt_def_real.__dict__.__setitem__('stypy_call_varargs', varargs)
        Dvi._fnt_def_real.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Dvi._fnt_def_real.__dict__.__setitem__('stypy_declared_arg_number', 7)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Dvi._fnt_def_real', ['k', 'c', 's', 'd', 'a', 'l'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_fnt_def_real', localization, ['k', 'c', 's', 'd', 'a', 'l'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_fnt_def_real(...)' code ##################

        
        # Assigning a Call to a Name (line 460):
        
        # Assigning a Call to a Name (line 460):
        
        # Call to read(...): (line 460)
        # Processing the call arguments (line 460)
        # Getting the type of 'a' (line 460)
        a_48972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 27), 'a', False)
        # Getting the type of 'l' (line 460)
        l_48973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 31), 'l', False)
        # Applying the binary operator '+' (line 460)
        result_add_48974 = python_operator(stypy.reporting.localization.Localization(__file__, 460, 27), '+', a_48972, l_48973)
        
        # Processing the call keyword arguments (line 460)
        kwargs_48975 = {}
        # Getting the type of 'self' (line 460)
        self_48969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 12), 'self', False)
        # Obtaining the member 'file' of a type (line 460)
        file_48970 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 460, 12), self_48969, 'file')
        # Obtaining the member 'read' of a type (line 460)
        read_48971 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 460, 12), file_48970, 'read')
        # Calling read(args, kwargs) (line 460)
        read_call_result_48976 = invoke(stypy.reporting.localization.Localization(__file__, 460, 12), read_48971, *[result_add_48974], **kwargs_48975)
        
        # Assigning a type to the variable 'n' (line 460)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 460, 8), 'n', read_call_result_48976)
        
        # Assigning a Call to a Name (line 461):
        
        # Assigning a Call to a Name (line 461):
        
        # Call to decode(...): (line 461)
        # Processing the call arguments (line 461)
        unicode_48984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 461, 33), 'unicode', u'ascii')
        # Processing the call keyword arguments (line 461)
        kwargs_48985 = {}
        
        # Obtaining the type of the subscript
        
        # Getting the type of 'l' (line 461)
        l_48977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 22), 'l', False)
        # Applying the 'usub' unary operator (line 461)
        result___neg___48978 = python_operator(stypy.reporting.localization.Localization(__file__, 461, 21), 'usub', l_48977)
        
        slice_48979 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 461, 19), result___neg___48978, None, None)
        # Getting the type of 'n' (line 461)
        n_48980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 19), 'n', False)
        # Obtaining the member '__getitem__' of a type (line 461)
        getitem___48981 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 461, 19), n_48980, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 461)
        subscript_call_result_48982 = invoke(stypy.reporting.localization.Localization(__file__, 461, 19), getitem___48981, slice_48979)
        
        # Obtaining the member 'decode' of a type (line 461)
        decode_48983 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 461, 19), subscript_call_result_48982, 'decode')
        # Calling decode(args, kwargs) (line 461)
        decode_call_result_48986 = invoke(stypy.reporting.localization.Localization(__file__, 461, 19), decode_48983, *[unicode_48984], **kwargs_48985)
        
        # Assigning a type to the variable 'fontname' (line 461)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 461, 8), 'fontname', decode_call_result_48986)
        
        # Assigning a Call to a Name (line 462):
        
        # Assigning a Call to a Name (line 462):
        
        # Call to _tfmfile(...): (line 462)
        # Processing the call arguments (line 462)
        # Getting the type of 'fontname' (line 462)
        fontname_48988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 23), 'fontname', False)
        # Processing the call keyword arguments (line 462)
        kwargs_48989 = {}
        # Getting the type of '_tfmfile' (line 462)
        _tfmfile_48987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 14), '_tfmfile', False)
        # Calling _tfmfile(args, kwargs) (line 462)
        _tfmfile_call_result_48990 = invoke(stypy.reporting.localization.Localization(__file__, 462, 14), _tfmfile_48987, *[fontname_48988], **kwargs_48989)
        
        # Assigning a type to the variable 'tfm' (line 462)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 462, 8), 'tfm', _tfmfile_call_result_48990)
        
        # Type idiom detected: calculating its left and rigth part (line 463)
        # Getting the type of 'tfm' (line 463)
        tfm_48991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 11), 'tfm')
        # Getting the type of 'None' (line 463)
        None_48992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 18), 'None')
        
        (may_be_48993, more_types_in_union_48994) = may_be_none(tfm_48991, None_48992)

        if may_be_48993:

            if more_types_in_union_48994:
                # Runtime conditional SSA (line 463)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Getting the type of 'six' (line 464)
            six_48995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 15), 'six')
            # Obtaining the member 'PY2' of a type (line 464)
            PY2_48996 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 464, 15), six_48995, 'PY2')
            # Testing the type of an if condition (line 464)
            if_condition_48997 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 464, 12), PY2_48996)
            # Assigning a type to the variable 'if_condition_48997' (line 464)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 464, 12), 'if_condition_48997', if_condition_48997)
            # SSA begins for if statement (line 464)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Name to a Name (line 465):
            
            # Assigning a Name to a Name (line 465):
            # Getting the type of 'OSError' (line 465)
            OSError_48998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 30), 'OSError')
            # Assigning a type to the variable 'error_class' (line 465)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 465, 16), 'error_class', OSError_48998)
            # SSA branch for the else part of an if statement (line 464)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Name to a Name (line 467):
            
            # Assigning a Name to a Name (line 467):
            # Getting the type of 'FileNotFoundError' (line 467)
            FileNotFoundError_48999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 30), 'FileNotFoundError')
            # Assigning a type to the variable 'error_class' (line 467)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 467, 16), 'error_class', FileNotFoundError_48999)
            # SSA join for if statement (line 464)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Call to error_class(...): (line 468)
            # Processing the call arguments (line 468)
            unicode_49001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 468, 30), 'unicode', u'missing font metrics file: %s')
            # Getting the type of 'fontname' (line 468)
            fontname_49002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 64), 'fontname', False)
            # Applying the binary operator '%' (line 468)
            result_mod_49003 = python_operator(stypy.reporting.localization.Localization(__file__, 468, 30), '%', unicode_49001, fontname_49002)
            
            # Processing the call keyword arguments (line 468)
            kwargs_49004 = {}
            # Getting the type of 'error_class' (line 468)
            error_class_49000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 18), 'error_class', False)
            # Calling error_class(args, kwargs) (line 468)
            error_class_call_result_49005 = invoke(stypy.reporting.localization.Localization(__file__, 468, 18), error_class_49000, *[result_mod_49003], **kwargs_49004)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 468, 12), error_class_call_result_49005, 'raise parameter', BaseException)

            if more_types_in_union_48994:
                # SSA join for if statement (line 463)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'c' (line 469)
        c_49006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 11), 'c')
        int_49007 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 469, 16), 'int')
        # Applying the binary operator '!=' (line 469)
        result_ne_49008 = python_operator(stypy.reporting.localization.Localization(__file__, 469, 11), '!=', c_49006, int_49007)
        
        
        # Getting the type of 'tfm' (line 469)
        tfm_49009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 22), 'tfm')
        # Obtaining the member 'checksum' of a type (line 469)
        checksum_49010 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 469, 22), tfm_49009, 'checksum')
        int_49011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 469, 38), 'int')
        # Applying the binary operator '!=' (line 469)
        result_ne_49012 = python_operator(stypy.reporting.localization.Localization(__file__, 469, 22), '!=', checksum_49010, int_49011)
        
        # Applying the binary operator 'and' (line 469)
        result_and_keyword_49013 = python_operator(stypy.reporting.localization.Localization(__file__, 469, 11), 'and', result_ne_49008, result_ne_49012)
        
        # Getting the type of 'c' (line 469)
        c_49014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 44), 'c')
        # Getting the type of 'tfm' (line 469)
        tfm_49015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 49), 'tfm')
        # Obtaining the member 'checksum' of a type (line 469)
        checksum_49016 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 469, 49), tfm_49015, 'checksum')
        # Applying the binary operator '!=' (line 469)
        result_ne_49017 = python_operator(stypy.reporting.localization.Localization(__file__, 469, 44), '!=', c_49014, checksum_49016)
        
        # Applying the binary operator 'and' (line 469)
        result_and_keyword_49018 = python_operator(stypy.reporting.localization.Localization(__file__, 469, 11), 'and', result_and_keyword_49013, result_ne_49017)
        
        # Testing the type of an if condition (line 469)
        if_condition_49019 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 469, 8), result_and_keyword_49018)
        # Assigning a type to the variable 'if_condition_49019' (line 469)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 469, 8), 'if_condition_49019', if_condition_49019)
        # SSA begins for if statement (line 469)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 470)
        # Processing the call arguments (line 470)
        unicode_49021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 470, 29), 'unicode', u'tfm checksum mismatch: %s')
        # Getting the type of 'n' (line 470)
        n_49022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 59), 'n', False)
        # Applying the binary operator '%' (line 470)
        result_mod_49023 = python_operator(stypy.reporting.localization.Localization(__file__, 470, 29), '%', unicode_49021, n_49022)
        
        # Processing the call keyword arguments (line 470)
        kwargs_49024 = {}
        # Getting the type of 'ValueError' (line 470)
        ValueError_49020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 470)
        ValueError_call_result_49025 = invoke(stypy.reporting.localization.Localization(__file__, 470, 18), ValueError_49020, *[result_mod_49023], **kwargs_49024)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 470, 12), ValueError_call_result_49025, 'raise parameter', BaseException)
        # SSA join for if statement (line 469)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 472):
        
        # Assigning a Call to a Name (line 472):
        
        # Call to _vffile(...): (line 472)
        # Processing the call arguments (line 472)
        # Getting the type of 'fontname' (line 472)
        fontname_49027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 21), 'fontname', False)
        # Processing the call keyword arguments (line 472)
        kwargs_49028 = {}
        # Getting the type of '_vffile' (line 472)
        _vffile_49026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 13), '_vffile', False)
        # Calling _vffile(args, kwargs) (line 472)
        _vffile_call_result_49029 = invoke(stypy.reporting.localization.Localization(__file__, 472, 13), _vffile_49026, *[fontname_49027], **kwargs_49028)
        
        # Assigning a type to the variable 'vf' (line 472)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 472, 8), 'vf', _vffile_call_result_49029)
        
        # Assigning a Call to a Subscript (line 474):
        
        # Assigning a Call to a Subscript (line 474):
        
        # Call to DviFont(...): (line 474)
        # Processing the call keyword arguments (line 474)
        # Getting the type of 's' (line 474)
        s_49031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 38), 's', False)
        keyword_49032 = s_49031
        # Getting the type of 'tfm' (line 474)
        tfm_49033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 45), 'tfm', False)
        keyword_49034 = tfm_49033
        # Getting the type of 'n' (line 474)
        n_49035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 58), 'n', False)
        keyword_49036 = n_49035
        # Getting the type of 'vf' (line 474)
        vf_49037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 64), 'vf', False)
        keyword_49038 = vf_49037
        kwargs_49039 = {'tfm': keyword_49034, 'texname': keyword_49036, 'scale': keyword_49032, 'vf': keyword_49038}
        # Getting the type of 'DviFont' (line 474)
        DviFont_49030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 24), 'DviFont', False)
        # Calling DviFont(args, kwargs) (line 474)
        DviFont_call_result_49040 = invoke(stypy.reporting.localization.Localization(__file__, 474, 24), DviFont_49030, *[], **kwargs_49039)
        
        # Getting the type of 'self' (line 474)
        self_49041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 8), 'self')
        # Obtaining the member 'fonts' of a type (line 474)
        fonts_49042 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 474, 8), self_49041, 'fonts')
        # Getting the type of 'k' (line 474)
        k_49043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 19), 'k')
        # Storing an element on a container (line 474)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 474, 8), fonts_49042, (k_49043, DviFont_call_result_49040))
        
        # ################# End of '_fnt_def_real(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_fnt_def_real' in the type store
        # Getting the type of 'stypy_return_type' (line 459)
        stypy_return_type_49044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_49044)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_fnt_def_real'
        return stypy_return_type_49044


    @norecursion
    def _pre(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_pre'
        module_type_store = module_type_store.open_function_context('_pre', 476, 4, False)
        # Assigning a type to the variable 'self' (line 477)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 477, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Dvi._pre.__dict__.__setitem__('stypy_localization', localization)
        Dvi._pre.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Dvi._pre.__dict__.__setitem__('stypy_type_store', module_type_store)
        Dvi._pre.__dict__.__setitem__('stypy_function_name', 'Dvi._pre')
        Dvi._pre.__dict__.__setitem__('stypy_param_names_list', ['i', 'num', 'den', 'mag', 'k'])
        Dvi._pre.__dict__.__setitem__('stypy_varargs_param_name', None)
        Dvi._pre.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Dvi._pre.__dict__.__setitem__('stypy_call_defaults', defaults)
        Dvi._pre.__dict__.__setitem__('stypy_call_varargs', varargs)
        Dvi._pre.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Dvi._pre.__dict__.__setitem__('stypy_declared_arg_number', 6)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Dvi._pre', ['i', 'num', 'den', 'mag', 'k'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_pre', localization, ['i', 'num', 'den', 'mag', 'k'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_pre(...)' code ##################

        
        # Assigning a Call to a Name (line 478):
        
        # Assigning a Call to a Name (line 478):
        
        # Call to read(...): (line 478)
        # Processing the call arguments (line 478)
        # Getting the type of 'k' (line 478)
        k_49048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 33), 'k', False)
        # Processing the call keyword arguments (line 478)
        kwargs_49049 = {}
        # Getting the type of 'self' (line 478)
        self_49045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 18), 'self', False)
        # Obtaining the member 'file' of a type (line 478)
        file_49046 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 478, 18), self_49045, 'file')
        # Obtaining the member 'read' of a type (line 478)
        read_49047 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 478, 18), file_49046, 'read')
        # Calling read(args, kwargs) (line 478)
        read_call_result_49050 = invoke(stypy.reporting.localization.Localization(__file__, 478, 18), read_49047, *[k_49048], **kwargs_49049)
        
        # Assigning a type to the variable 'comment' (line 478)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 478, 8), 'comment', read_call_result_49050)
        
        
        # Getting the type of 'i' (line 479)
        i_49051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 11), 'i')
        int_49052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 479, 16), 'int')
        # Applying the binary operator '!=' (line 479)
        result_ne_49053 = python_operator(stypy.reporting.localization.Localization(__file__, 479, 11), '!=', i_49051, int_49052)
        
        # Testing the type of an if condition (line 479)
        if_condition_49054 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 479, 8), result_ne_49053)
        # Assigning a type to the variable 'if_condition_49054' (line 479)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 479, 8), 'if_condition_49054', if_condition_49054)
        # SSA begins for if statement (line 479)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 480)
        # Processing the call arguments (line 480)
        unicode_49056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 480, 29), 'unicode', u'Unknown dvi format %d')
        # Getting the type of 'i' (line 480)
        i_49057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 55), 'i', False)
        # Applying the binary operator '%' (line 480)
        result_mod_49058 = python_operator(stypy.reporting.localization.Localization(__file__, 480, 29), '%', unicode_49056, i_49057)
        
        # Processing the call keyword arguments (line 480)
        kwargs_49059 = {}
        # Getting the type of 'ValueError' (line 480)
        ValueError_49055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 480)
        ValueError_call_result_49060 = invoke(stypy.reporting.localization.Localization(__file__, 480, 18), ValueError_49055, *[result_mod_49058], **kwargs_49059)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 480, 12), ValueError_call_result_49060, 'raise parameter', BaseException)
        # SSA join for if statement (line 479)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'num' (line 481)
        num_49061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 11), 'num')
        int_49062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 481, 18), 'int')
        # Applying the binary operator '!=' (line 481)
        result_ne_49063 = python_operator(stypy.reporting.localization.Localization(__file__, 481, 11), '!=', num_49061, int_49062)
        
        
        # Getting the type of 'den' (line 481)
        den_49064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 30), 'den')
        int_49065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 481, 37), 'int')
        int_49066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 481, 44), 'int')
        int_49067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 481, 47), 'int')
        # Applying the binary operator '**' (line 481)
        result_pow_49068 = python_operator(stypy.reporting.localization.Localization(__file__, 481, 44), '**', int_49066, int_49067)
        
        # Applying the binary operator '*' (line 481)
        result_mul_49069 = python_operator(stypy.reporting.localization.Localization(__file__, 481, 37), '*', int_49065, result_pow_49068)
        
        # Applying the binary operator '!=' (line 481)
        result_ne_49070 = python_operator(stypy.reporting.localization.Localization(__file__, 481, 30), '!=', den_49064, result_mul_49069)
        
        # Applying the binary operator 'or' (line 481)
        result_or_keyword_49071 = python_operator(stypy.reporting.localization.Localization(__file__, 481, 11), 'or', result_ne_49063, result_ne_49070)
        
        # Testing the type of an if condition (line 481)
        if_condition_49072 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 481, 8), result_or_keyword_49071)
        # Assigning a type to the variable 'if_condition_49072' (line 481)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 481, 8), 'if_condition_49072', if_condition_49072)
        # SSA begins for if statement (line 481)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 482)
        # Processing the call arguments (line 482)
        unicode_49074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 482, 29), 'unicode', u'nonstandard units in dvi file')
        # Processing the call keyword arguments (line 482)
        kwargs_49075 = {}
        # Getting the type of 'ValueError' (line 482)
        ValueError_49073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 482)
        ValueError_call_result_49076 = invoke(stypy.reporting.localization.Localization(__file__, 482, 18), ValueError_49073, *[unicode_49074], **kwargs_49075)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 482, 12), ValueError_call_result_49076, 'raise parameter', BaseException)
        # SSA join for if statement (line 481)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'mag' (line 488)
        mag_49077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 11), 'mag')
        int_49078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 488, 18), 'int')
        # Applying the binary operator '!=' (line 488)
        result_ne_49079 = python_operator(stypy.reporting.localization.Localization(__file__, 488, 11), '!=', mag_49077, int_49078)
        
        # Testing the type of an if condition (line 488)
        if_condition_49080 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 488, 8), result_ne_49079)
        # Assigning a type to the variable 'if_condition_49080' (line 488)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 488, 8), 'if_condition_49080', if_condition_49080)
        # SSA begins for if statement (line 488)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 489)
        # Processing the call arguments (line 489)
        unicode_49082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 489, 29), 'unicode', u'nonstandard magnification in dvi file')
        # Processing the call keyword arguments (line 489)
        kwargs_49083 = {}
        # Getting the type of 'ValueError' (line 489)
        ValueError_49081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 489)
        ValueError_call_result_49084 = invoke(stypy.reporting.localization.Localization(__file__, 489, 18), ValueError_49081, *[unicode_49082], **kwargs_49083)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 489, 12), ValueError_call_result_49084, 'raise parameter', BaseException)
        # SSA join for if statement (line 488)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Attribute to a Attribute (line 492):
        
        # Assigning a Attribute to a Attribute (line 492):
        # Getting the type of '_dvistate' (line 492)
        _dvistate_49085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 21), '_dvistate')
        # Obtaining the member 'outer' of a type (line 492)
        outer_49086 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 492, 21), _dvistate_49085, 'outer')
        # Getting the type of 'self' (line 492)
        self_49087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 8), 'self')
        # Setting the type of the member 'state' of a type (line 492)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 492, 8), self_49087, 'state', outer_49086)
        
        # ################# End of '_pre(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_pre' in the type store
        # Getting the type of 'stypy_return_type' (line 476)
        stypy_return_type_49088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_49088)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_pre'
        return stypy_return_type_49088


    @norecursion
    def _post(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_post'
        module_type_store = module_type_store.open_function_context('_post', 494, 4, False)
        # Assigning a type to the variable 'self' (line 495)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 495, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Dvi._post.__dict__.__setitem__('stypy_localization', localization)
        Dvi._post.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Dvi._post.__dict__.__setitem__('stypy_type_store', module_type_store)
        Dvi._post.__dict__.__setitem__('stypy_function_name', 'Dvi._post')
        Dvi._post.__dict__.__setitem__('stypy_param_names_list', ['_'])
        Dvi._post.__dict__.__setitem__('stypy_varargs_param_name', None)
        Dvi._post.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Dvi._post.__dict__.__setitem__('stypy_call_defaults', defaults)
        Dvi._post.__dict__.__setitem__('stypy_call_varargs', varargs)
        Dvi._post.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Dvi._post.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Dvi._post', ['_'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_post', localization, ['_'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_post(...)' code ##################

        
        # Assigning a Attribute to a Attribute (line 496):
        
        # Assigning a Attribute to a Attribute (line 496):
        # Getting the type of '_dvistate' (line 496)
        _dvistate_49089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 21), '_dvistate')
        # Obtaining the member 'post_post' of a type (line 496)
        post_post_49090 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 496, 21), _dvistate_49089, 'post_post')
        # Getting the type of 'self' (line 496)
        self_49091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 8), 'self')
        # Setting the type of the member 'state' of a type (line 496)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 496, 8), self_49091, 'state', post_post_49090)
        
        # ################# End of '_post(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_post' in the type store
        # Getting the type of 'stypy_return_type' (line 494)
        stypy_return_type_49092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_49092)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_post'
        return stypy_return_type_49092


    @norecursion
    def _post_post(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_post_post'
        module_type_store = module_type_store.open_function_context('_post_post', 500, 4, False)
        # Assigning a type to the variable 'self' (line 501)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 501, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Dvi._post_post.__dict__.__setitem__('stypy_localization', localization)
        Dvi._post_post.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Dvi._post_post.__dict__.__setitem__('stypy_type_store', module_type_store)
        Dvi._post_post.__dict__.__setitem__('stypy_function_name', 'Dvi._post_post')
        Dvi._post_post.__dict__.__setitem__('stypy_param_names_list', ['_'])
        Dvi._post_post.__dict__.__setitem__('stypy_varargs_param_name', None)
        Dvi._post_post.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Dvi._post_post.__dict__.__setitem__('stypy_call_defaults', defaults)
        Dvi._post_post.__dict__.__setitem__('stypy_call_varargs', varargs)
        Dvi._post_post.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Dvi._post_post.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Dvi._post_post', ['_'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_post_post', localization, ['_'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_post_post(...)' code ##################

        # Getting the type of 'NotImplementedError' (line 502)
        NotImplementedError_49093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 14), 'NotImplementedError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 502, 8), NotImplementedError_49093, 'raise parameter', BaseException)
        
        # ################# End of '_post_post(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_post_post' in the type store
        # Getting the type of 'stypy_return_type' (line 500)
        stypy_return_type_49094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_49094)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_post_post'
        return stypy_return_type_49094


    @norecursion
    def _malformed(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_malformed'
        module_type_store = module_type_store.open_function_context('_malformed', 504, 4, False)
        # Assigning a type to the variable 'self' (line 505)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 505, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Dvi._malformed.__dict__.__setitem__('stypy_localization', localization)
        Dvi._malformed.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Dvi._malformed.__dict__.__setitem__('stypy_type_store', module_type_store)
        Dvi._malformed.__dict__.__setitem__('stypy_function_name', 'Dvi._malformed')
        Dvi._malformed.__dict__.__setitem__('stypy_param_names_list', ['offset'])
        Dvi._malformed.__dict__.__setitem__('stypy_varargs_param_name', None)
        Dvi._malformed.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Dvi._malformed.__dict__.__setitem__('stypy_call_defaults', defaults)
        Dvi._malformed.__dict__.__setitem__('stypy_call_varargs', varargs)
        Dvi._malformed.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Dvi._malformed.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Dvi._malformed', ['offset'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_malformed', localization, ['offset'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_malformed(...)' code ##################

        
        # Call to ValueError(...): (line 506)
        # Processing the call arguments (line 506)
        unicode_49096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 506, 25), 'unicode', u'unknown command: byte %d')
        int_49097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 506, 53), 'int')
        # Getting the type of 'offset' (line 506)
        offset_49098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 59), 'offset', False)
        # Applying the binary operator '+' (line 506)
        result_add_49099 = python_operator(stypy.reporting.localization.Localization(__file__, 506, 53), '+', int_49097, offset_49098)
        
        # Processing the call keyword arguments (line 506)
        kwargs_49100 = {}
        # Getting the type of 'ValueError' (line 506)
        ValueError_49095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 14), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 506)
        ValueError_call_result_49101 = invoke(stypy.reporting.localization.Localization(__file__, 506, 14), ValueError_49095, *[unicode_49096, result_add_49099], **kwargs_49100)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 506, 8), ValueError_call_result_49101, 'raise parameter', BaseException)
        
        # ################# End of '_malformed(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_malformed' in the type store
        # Getting the type of 'stypy_return_type' (line 504)
        stypy_return_type_49102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_49102)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_malformed'
        return stypy_return_type_49102


# Assigning a type to the variable 'Dvi' (line 177)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 0), 'Dvi', Dvi)

# Assigning a ListComp to a Name (line 192):
# Calculating list comprehension
# Calculating comprehension expression

# Call to xrange(...): (line 192)
# Processing the call arguments (line 192)
int_49105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 36), 'int')
# Processing the call keyword arguments (line 192)
kwargs_49106 = {}
# Getting the type of 'xrange' (line 192)
xrange_49104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 29), 'xrange', False)
# Calling xrange(args, kwargs) (line 192)
xrange_call_result_49107 = invoke(stypy.reporting.localization.Localization(__file__, 192, 29), xrange_49104, *[int_49105], **kwargs_49106)

comprehension_49108 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 192, 15), xrange_call_result_49107)
# Assigning a type to the variable '_' (line 192)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 15), '_', comprehension_49108)
# Getting the type of 'None' (line 192)
None_49103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 15), 'None')
list_49109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 15), 'list')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 192, 15), list_49109, None_49103)
# Getting the type of 'Dvi'
Dvi_49110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Dvi')
# Setting the type of the member '_dtable' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Dvi_49110, '_dtable', list_49109)

# Assigning a Call to a Name (line 193):

# Call to partial(...): (line 193)
# Processing the call arguments (line 193)
# Getting the type of '_dispatch' (line 193)
_dispatch_49112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 23), '_dispatch', False)
# Getting the type of 'Dvi'
Dvi_49113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Dvi', False)
# Obtaining the member '_dtable' of a type
_dtable_49114 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Dvi_49113, '_dtable')
# Processing the call keyword arguments (line 193)
kwargs_49115 = {}
# Getting the type of 'partial' (line 193)
partial_49111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 15), 'partial', False)
# Calling partial(args, kwargs) (line 193)
partial_call_result_49116 = invoke(stypy.reporting.localization.Localization(__file__, 193, 15), partial_49111, *[_dispatch_49112, _dtable_49114], **kwargs_49115)

# Getting the type of 'Dvi'
Dvi_49117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Dvi')
# Setting the type of the member 'dispatch' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Dvi_49117, 'dispatch', partial_call_result_49116)
# Declaration of the 'DviFont' class

class DviFont(object, ):
    unicode_49118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 546, (-1)), 'unicode', u'\n    Encapsulation of a font that a DVI file can refer to.\n\n    This class holds a font\'s texname and size, supports comparison,\n    and knows the widths of glyphs in the same units as the AFM file.\n    There are also internal attributes (for use by dviread.py) that\n    are *not* used for comparison.\n\n    The size is in Adobe points (converted from TeX points).\n\n    Parameters\n    ----------\n\n    scale : float\n        Factor by which the font is scaled from its natural size.\n    tfm : Tfm\n        TeX font metrics for this font\n    texname : bytes\n       Name of the font as used internally by TeX and friends, as an\n       ASCII bytestring. This is usually very different from any external\n       font names, and :class:`dviread.PsfontsMap` can be used to find\n       the external name of the font.\n    vf : Vf\n       A TeX "virtual font" file, or None if this font is not virtual.\n\n    Attributes\n    ----------\n\n    texname : bytes\n    size : float\n       Size of the font in Adobe points, converted from the slightly\n       smaller TeX points.\n    widths : list\n       Widths of glyphs in glyph-space units, typically 1/1000ths of\n       the point size.\n\n    ')
    
    # Assigning a Tuple to a Name (line 547):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 549, 4, False)
        # Assigning a type to the variable 'self' (line 550)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 550, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DviFont.__init__', ['scale', 'tfm', 'texname', 'vf'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['scale', 'tfm', 'texname', 'vf'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Type idiom detected: calculating its left and rigth part (line 550)
        # Getting the type of 'bytes' (line 550)
        bytes_49119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 35), 'bytes')
        # Getting the type of 'texname' (line 550)
        texname_49120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 26), 'texname')
        
        (may_be_49121, more_types_in_union_49122) = may_not_be_subtype(bytes_49119, texname_49120)

        if may_be_49121:

            if more_types_in_union_49122:
                # Runtime conditional SSA (line 550)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'texname' (line 550)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 550, 8), 'texname', remove_subtype_from_union(texname_49120, bytes))
            
            # Call to ValueError(...): (line 551)
            # Processing the call arguments (line 551)
            unicode_49124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 551, 29), 'unicode', u'texname must be a bytestring, got %s')
            
            # Call to type(...): (line 552)
            # Processing the call arguments (line 552)
            # Getting the type of 'texname' (line 552)
            texname_49126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 36), 'texname', False)
            # Processing the call keyword arguments (line 552)
            kwargs_49127 = {}
            # Getting the type of 'type' (line 552)
            type_49125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 31), 'type', False)
            # Calling type(args, kwargs) (line 552)
            type_call_result_49128 = invoke(stypy.reporting.localization.Localization(__file__, 552, 31), type_49125, *[texname_49126], **kwargs_49127)
            
            # Applying the binary operator '%' (line 551)
            result_mod_49129 = python_operator(stypy.reporting.localization.Localization(__file__, 551, 29), '%', unicode_49124, type_call_result_49128)
            
            # Processing the call keyword arguments (line 551)
            kwargs_49130 = {}
            # Getting the type of 'ValueError' (line 551)
            ValueError_49123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 18), 'ValueError', False)
            # Calling ValueError(args, kwargs) (line 551)
            ValueError_call_result_49131 = invoke(stypy.reporting.localization.Localization(__file__, 551, 18), ValueError_49123, *[result_mod_49129], **kwargs_49130)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 551, 12), ValueError_call_result_49131, 'raise parameter', BaseException)

            if more_types_in_union_49122:
                # SSA join for if statement (line 550)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Tuple to a Tuple (line 553):
        
        # Assigning a Name to a Name (line 553):
        # Getting the type of 'scale' (line 554)
        scale_49132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 12), 'scale')
        # Assigning a type to the variable 'tuple_assignment_47791' (line 553)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 553, 8), 'tuple_assignment_47791', scale_49132)
        
        # Assigning a Name to a Name (line 553):
        # Getting the type of 'tfm' (line 554)
        tfm_49133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 19), 'tfm')
        # Assigning a type to the variable 'tuple_assignment_47792' (line 553)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 553, 8), 'tuple_assignment_47792', tfm_49133)
        
        # Assigning a Name to a Name (line 553):
        # Getting the type of 'texname' (line 554)
        texname_49134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 24), 'texname')
        # Assigning a type to the variable 'tuple_assignment_47793' (line 553)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 553, 8), 'tuple_assignment_47793', texname_49134)
        
        # Assigning a Name to a Name (line 553):
        # Getting the type of 'vf' (line 554)
        vf_49135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 33), 'vf')
        # Assigning a type to the variable 'tuple_assignment_47794' (line 553)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 553, 8), 'tuple_assignment_47794', vf_49135)
        
        # Assigning a Name to a Attribute (line 553):
        # Getting the type of 'tuple_assignment_47791' (line 553)
        tuple_assignment_47791_49136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 8), 'tuple_assignment_47791')
        # Getting the type of 'self' (line 553)
        self_49137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 8), 'self')
        # Setting the type of the member '_scale' of a type (line 553)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 553, 8), self_49137, '_scale', tuple_assignment_47791_49136)
        
        # Assigning a Name to a Attribute (line 553):
        # Getting the type of 'tuple_assignment_47792' (line 553)
        tuple_assignment_47792_49138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 8), 'tuple_assignment_47792')
        # Getting the type of 'self' (line 553)
        self_49139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 21), 'self')
        # Setting the type of the member '_tfm' of a type (line 553)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 553, 21), self_49139, '_tfm', tuple_assignment_47792_49138)
        
        # Assigning a Name to a Attribute (line 553):
        # Getting the type of 'tuple_assignment_47793' (line 553)
        tuple_assignment_47793_49140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 8), 'tuple_assignment_47793')
        # Getting the type of 'self' (line 553)
        self_49141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 32), 'self')
        # Setting the type of the member 'texname' of a type (line 553)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 553, 32), self_49141, 'texname', tuple_assignment_47793_49140)
        
        # Assigning a Name to a Attribute (line 553):
        # Getting the type of 'tuple_assignment_47794' (line 553)
        tuple_assignment_47794_49142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 8), 'tuple_assignment_47794')
        # Getting the type of 'self' (line 553)
        self_49143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 46), 'self')
        # Setting the type of the member '_vf' of a type (line 553)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 553, 46), self_49143, '_vf', tuple_assignment_47794_49142)
        
        # Assigning a BinOp to a Attribute (line 555):
        
        # Assigning a BinOp to a Attribute (line 555):
        # Getting the type of 'scale' (line 555)
        scale_49144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 20), 'scale')
        float_49145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 555, 29), 'float')
        float_49146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 555, 37), 'float')
        int_49147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 555, 45), 'int')
        int_49148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 555, 48), 'int')
        # Applying the binary operator '**' (line 555)
        result_pow_49149 = python_operator(stypy.reporting.localization.Localization(__file__, 555, 45), '**', int_49147, int_49148)
        
        # Applying the binary operator '*' (line 555)
        result_mul_49150 = python_operator(stypy.reporting.localization.Localization(__file__, 555, 37), '*', float_49146, result_pow_49149)
        
        # Applying the binary operator 'div' (line 555)
        result_div_49151 = python_operator(stypy.reporting.localization.Localization(__file__, 555, 29), 'div', float_49145, result_mul_49150)
        
        # Applying the binary operator '*' (line 555)
        result_mul_49152 = python_operator(stypy.reporting.localization.Localization(__file__, 555, 20), '*', scale_49144, result_div_49151)
        
        # Getting the type of 'self' (line 555)
        self_49153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 8), 'self')
        # Setting the type of the member 'size' of a type (line 555)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 555, 8), self_49153, 'size', result_mul_49152)
        
        
        # SSA begins for try-except statement (line 556)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a BinOp to a Name (line 557):
        
        # Assigning a BinOp to a Name (line 557):
        
        # Call to max(...): (line 557)
        # Processing the call arguments (line 557)
        # Getting the type of 'tfm' (line 557)
        tfm_49155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 25), 'tfm', False)
        # Obtaining the member 'width' of a type (line 557)
        width_49156 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 557, 25), tfm_49155, 'width')
        # Processing the call keyword arguments (line 557)
        kwargs_49157 = {}
        # Getting the type of 'max' (line 557)
        max_49154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 21), 'max', False)
        # Calling max(args, kwargs) (line 557)
        max_call_result_49158 = invoke(stypy.reporting.localization.Localization(__file__, 557, 21), max_49154, *[width_49156], **kwargs_49157)
        
        int_49159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 557, 38), 'int')
        # Applying the binary operator '+' (line 557)
        result_add_49160 = python_operator(stypy.reporting.localization.Localization(__file__, 557, 21), '+', max_call_result_49158, int_49159)
        
        # Assigning a type to the variable 'nchars' (line 557)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 557, 12), 'nchars', result_add_49160)
        # SSA branch for the except part of a try statement (line 556)
        # SSA branch for the except 'ValueError' branch of a try statement (line 556)
        module_type_store.open_ssa_branch('except')
        
        # Assigning a Num to a Name (line 559):
        
        # Assigning a Num to a Name (line 559):
        int_49161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 559, 21), 'int')
        # Assigning a type to the variable 'nchars' (line 559)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 559, 12), 'nchars', int_49161)
        # SSA join for try-except statement (line 556)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a ListComp to a Attribute (line 560):
        
        # Assigning a ListComp to a Attribute (line 560):
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to xrange(...): (line 561)
        # Processing the call arguments (line 561)
        # Getting the type of 'nchars' (line 561)
        nchars_49174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 42), 'nchars', False)
        # Processing the call keyword arguments (line 561)
        kwargs_49175 = {}
        # Getting the type of 'xrange' (line 561)
        xrange_49173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 35), 'xrange', False)
        # Calling xrange(args, kwargs) (line 561)
        xrange_call_result_49176 = invoke(stypy.reporting.localization.Localization(__file__, 561, 35), xrange_49173, *[nchars_49174], **kwargs_49175)
        
        comprehension_49177 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 560, 23), xrange_call_result_49176)
        # Assigning a type to the variable 'char' (line 560)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 560, 23), 'char', comprehension_49177)
        int_49162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 560, 24), 'int')
        
        # Call to get(...): (line 560)
        # Processing the call arguments (line 560)
        # Getting the type of 'char' (line 560)
        char_49166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 43), 'char', False)
        int_49167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 560, 49), 'int')
        # Processing the call keyword arguments (line 560)
        kwargs_49168 = {}
        # Getting the type of 'tfm' (line 560)
        tfm_49163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 29), 'tfm', False)
        # Obtaining the member 'width' of a type (line 560)
        width_49164 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 560, 29), tfm_49163, 'width')
        # Obtaining the member 'get' of a type (line 560)
        get_49165 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 560, 29), width_49164, 'get')
        # Calling get(args, kwargs) (line 560)
        get_call_result_49169 = invoke(stypy.reporting.localization.Localization(__file__, 560, 29), get_49165, *[char_49166, int_49167], **kwargs_49168)
        
        # Applying the binary operator '*' (line 560)
        result_mul_49170 = python_operator(stypy.reporting.localization.Localization(__file__, 560, 24), '*', int_49162, get_call_result_49169)
        
        int_49171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 560, 56), 'int')
        # Applying the binary operator '>>' (line 560)
        result_rshift_49172 = python_operator(stypy.reporting.localization.Localization(__file__, 560, 23), '>>', result_mul_49170, int_49171)
        
        list_49178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 560, 23), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 560, 23), list_49178, result_rshift_49172)
        # Getting the type of 'self' (line 560)
        self_49179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 8), 'self')
        # Setting the type of the member 'widths' of a type (line 560)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 560, 8), self_49179, 'widths', list_49178)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def stypy__eq__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__eq__'
        module_type_store = module_type_store.open_function_context('__eq__', 563, 4, False)
        # Assigning a type to the variable 'self' (line 564)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 564, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DviFont.stypy__eq__.__dict__.__setitem__('stypy_localization', localization)
        DviFont.stypy__eq__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DviFont.stypy__eq__.__dict__.__setitem__('stypy_type_store', module_type_store)
        DviFont.stypy__eq__.__dict__.__setitem__('stypy_function_name', 'DviFont.stypy__eq__')
        DviFont.stypy__eq__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        DviFont.stypy__eq__.__dict__.__setitem__('stypy_varargs_param_name', None)
        DviFont.stypy__eq__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DviFont.stypy__eq__.__dict__.__setitem__('stypy_call_defaults', defaults)
        DviFont.stypy__eq__.__dict__.__setitem__('stypy_call_varargs', varargs)
        DviFont.stypy__eq__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DviFont.stypy__eq__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DviFont.stypy__eq__', ['other'], None, None, defaults, varargs, kwargs)

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

        
        # Evaluating a boolean operation
        
        # Getting the type of 'self' (line 564)
        self_49180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 15), 'self')
        # Obtaining the member '__class__' of a type (line 564)
        class___49181 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 564, 15), self_49180, '__class__')
        # Getting the type of 'other' (line 564)
        other_49182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 33), 'other')
        # Obtaining the member '__class__' of a type (line 564)
        class___49183 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 564, 33), other_49182, '__class__')
        # Applying the binary operator '==' (line 564)
        result_eq_49184 = python_operator(stypy.reporting.localization.Localization(__file__, 564, 15), '==', class___49181, class___49183)
        
        
        # Getting the type of 'self' (line 565)
        self_49185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 12), 'self')
        # Obtaining the member 'texname' of a type (line 565)
        texname_49186 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 565, 12), self_49185, 'texname')
        # Getting the type of 'other' (line 565)
        other_49187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 28), 'other')
        # Obtaining the member 'texname' of a type (line 565)
        texname_49188 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 565, 28), other_49187, 'texname')
        # Applying the binary operator '==' (line 565)
        result_eq_49189 = python_operator(stypy.reporting.localization.Localization(__file__, 565, 12), '==', texname_49186, texname_49188)
        
        # Applying the binary operator 'and' (line 564)
        result_and_keyword_49190 = python_operator(stypy.reporting.localization.Localization(__file__, 564, 15), 'and', result_eq_49184, result_eq_49189)
        
        # Getting the type of 'self' (line 565)
        self_49191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 46), 'self')
        # Obtaining the member 'size' of a type (line 565)
        size_49192 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 565, 46), self_49191, 'size')
        # Getting the type of 'other' (line 565)
        other_49193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 59), 'other')
        # Obtaining the member 'size' of a type (line 565)
        size_49194 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 565, 59), other_49193, 'size')
        # Applying the binary operator '==' (line 565)
        result_eq_49195 = python_operator(stypy.reporting.localization.Localization(__file__, 565, 46), '==', size_49192, size_49194)
        
        # Applying the binary operator 'and' (line 564)
        result_and_keyword_49196 = python_operator(stypy.reporting.localization.Localization(__file__, 564, 15), 'and', result_and_keyword_49190, result_eq_49195)
        
        # Assigning a type to the variable 'stypy_return_type' (line 564)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 564, 8), 'stypy_return_type', result_and_keyword_49196)
        
        # ################# End of '__eq__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__eq__' in the type store
        # Getting the type of 'stypy_return_type' (line 563)
        stypy_return_type_49197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_49197)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__eq__'
        return stypy_return_type_49197


    @norecursion
    def __ne__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__ne__'
        module_type_store = module_type_store.open_function_context('__ne__', 567, 4, False)
        # Assigning a type to the variable 'self' (line 568)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 568, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DviFont.__ne__.__dict__.__setitem__('stypy_localization', localization)
        DviFont.__ne__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DviFont.__ne__.__dict__.__setitem__('stypy_type_store', module_type_store)
        DviFont.__ne__.__dict__.__setitem__('stypy_function_name', 'DviFont.__ne__')
        DviFont.__ne__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        DviFont.__ne__.__dict__.__setitem__('stypy_varargs_param_name', None)
        DviFont.__ne__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DviFont.__ne__.__dict__.__setitem__('stypy_call_defaults', defaults)
        DviFont.__ne__.__dict__.__setitem__('stypy_call_varargs', varargs)
        DviFont.__ne__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DviFont.__ne__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DviFont.__ne__', ['other'], None, None, defaults, varargs, kwargs)

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

        
        
        # Call to __eq__(...): (line 568)
        # Processing the call arguments (line 568)
        # Getting the type of 'other' (line 568)
        other_49200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 31), 'other', False)
        # Processing the call keyword arguments (line 568)
        kwargs_49201 = {}
        # Getting the type of 'self' (line 568)
        self_49198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 19), 'self', False)
        # Obtaining the member '__eq__' of a type (line 568)
        eq___49199 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 568, 19), self_49198, '__eq__')
        # Calling __eq__(args, kwargs) (line 568)
        eq___call_result_49202 = invoke(stypy.reporting.localization.Localization(__file__, 568, 19), eq___49199, *[other_49200], **kwargs_49201)
        
        # Applying the 'not' unary operator (line 568)
        result_not__49203 = python_operator(stypy.reporting.localization.Localization(__file__, 568, 15), 'not', eq___call_result_49202)
        
        # Assigning a type to the variable 'stypy_return_type' (line 568)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 568, 8), 'stypy_return_type', result_not__49203)
        
        # ################# End of '__ne__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__ne__' in the type store
        # Getting the type of 'stypy_return_type' (line 567)
        stypy_return_type_49204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_49204)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__ne__'
        return stypy_return_type_49204


    @norecursion
    def _width_of(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_width_of'
        module_type_store = module_type_store.open_function_context('_width_of', 570, 4, False)
        # Assigning a type to the variable 'self' (line 571)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 571, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DviFont._width_of.__dict__.__setitem__('stypy_localization', localization)
        DviFont._width_of.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DviFont._width_of.__dict__.__setitem__('stypy_type_store', module_type_store)
        DviFont._width_of.__dict__.__setitem__('stypy_function_name', 'DviFont._width_of')
        DviFont._width_of.__dict__.__setitem__('stypy_param_names_list', ['char'])
        DviFont._width_of.__dict__.__setitem__('stypy_varargs_param_name', None)
        DviFont._width_of.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DviFont._width_of.__dict__.__setitem__('stypy_call_defaults', defaults)
        DviFont._width_of.__dict__.__setitem__('stypy_call_varargs', varargs)
        DviFont._width_of.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DviFont._width_of.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DviFont._width_of', ['char'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_width_of', localization, ['char'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_width_of(...)' code ##################

        unicode_49205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 573, (-1)), 'unicode', u'\n        Width of char in dvi units. For internal use by dviread.py.\n        ')
        
        # Assigning a Call to a Name (line 575):
        
        # Assigning a Call to a Name (line 575):
        
        # Call to get(...): (line 575)
        # Processing the call arguments (line 575)
        # Getting the type of 'char' (line 575)
        char_49210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 36), 'char', False)
        # Getting the type of 'None' (line 575)
        None_49211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 42), 'None', False)
        # Processing the call keyword arguments (line 575)
        kwargs_49212 = {}
        # Getting the type of 'self' (line 575)
        self_49206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 16), 'self', False)
        # Obtaining the member '_tfm' of a type (line 575)
        _tfm_49207 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 575, 16), self_49206, '_tfm')
        # Obtaining the member 'width' of a type (line 575)
        width_49208 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 575, 16), _tfm_49207, 'width')
        # Obtaining the member 'get' of a type (line 575)
        get_49209 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 575, 16), width_49208, 'get')
        # Calling get(args, kwargs) (line 575)
        get_call_result_49213 = invoke(stypy.reporting.localization.Localization(__file__, 575, 16), get_49209, *[char_49210, None_49211], **kwargs_49212)
        
        # Assigning a type to the variable 'width' (line 575)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 575, 8), 'width', get_call_result_49213)
        
        # Type idiom detected: calculating its left and rigth part (line 576)
        # Getting the type of 'width' (line 576)
        width_49214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 8), 'width')
        # Getting the type of 'None' (line 576)
        None_49215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 24), 'None')
        
        (may_be_49216, more_types_in_union_49217) = may_not_be_none(width_49214, None_49215)

        if may_be_49216:

            if more_types_in_union_49217:
                # Runtime conditional SSA (line 576)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to _mul2012(...): (line 577)
            # Processing the call arguments (line 577)
            # Getting the type of 'width' (line 577)
            width_49219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 28), 'width', False)
            # Getting the type of 'self' (line 577)
            self_49220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 35), 'self', False)
            # Obtaining the member '_scale' of a type (line 577)
            _scale_49221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 577, 35), self_49220, '_scale')
            # Processing the call keyword arguments (line 577)
            kwargs_49222 = {}
            # Getting the type of '_mul2012' (line 577)
            _mul2012_49218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 19), '_mul2012', False)
            # Calling _mul2012(args, kwargs) (line 577)
            _mul2012_call_result_49223 = invoke(stypy.reporting.localization.Localization(__file__, 577, 19), _mul2012_49218, *[width_49219, _scale_49221], **kwargs_49222)
            
            # Assigning a type to the variable 'stypy_return_type' (line 577)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 577, 12), 'stypy_return_type', _mul2012_call_result_49223)

            if more_types_in_union_49217:
                # SSA join for if statement (line 576)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Call to report(...): (line 579)
        # Processing the call arguments (line 579)
        unicode_49227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 580, 12), 'unicode', u'No width for char %d in font %s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 580)
        tuple_49228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 580, 49), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 580)
        # Adding element type (line 580)
        # Getting the type of 'char' (line 580)
        char_49229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 49), 'char', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 580, 49), tuple_49228, char_49229)
        # Adding element type (line 580)
        # Getting the type of 'self' (line 580)
        self_49230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 55), 'self', False)
        # Obtaining the member 'texname' of a type (line 580)
        texname_49231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 580, 55), self_49230, 'texname')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 580, 49), tuple_49228, texname_49231)
        
        # Applying the binary operator '%' (line 580)
        result_mod_49232 = python_operator(stypy.reporting.localization.Localization(__file__, 580, 12), '%', unicode_49227, tuple_49228)
        
        unicode_49233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 581, 12), 'unicode', u'debug')
        # Processing the call keyword arguments (line 579)
        kwargs_49234 = {}
        # Getting the type of 'matplotlib' (line 579)
        matplotlib_49224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 8), 'matplotlib', False)
        # Obtaining the member 'verbose' of a type (line 579)
        verbose_49225 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 579, 8), matplotlib_49224, 'verbose')
        # Obtaining the member 'report' of a type (line 579)
        report_49226 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 579, 8), verbose_49225, 'report')
        # Calling report(args, kwargs) (line 579)
        report_call_result_49235 = invoke(stypy.reporting.localization.Localization(__file__, 579, 8), report_49226, *[result_mod_49232, unicode_49233], **kwargs_49234)
        
        int_49236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 582, 15), 'int')
        # Assigning a type to the variable 'stypy_return_type' (line 582)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 582, 8), 'stypy_return_type', int_49236)
        
        # ################# End of '_width_of(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_width_of' in the type store
        # Getting the type of 'stypy_return_type' (line 570)
        stypy_return_type_49237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_49237)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_width_of'
        return stypy_return_type_49237


    @norecursion
    def _height_depth_of(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_height_depth_of'
        module_type_store = module_type_store.open_function_context('_height_depth_of', 584, 4, False)
        # Assigning a type to the variable 'self' (line 585)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 585, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DviFont._height_depth_of.__dict__.__setitem__('stypy_localization', localization)
        DviFont._height_depth_of.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DviFont._height_depth_of.__dict__.__setitem__('stypy_type_store', module_type_store)
        DviFont._height_depth_of.__dict__.__setitem__('stypy_function_name', 'DviFont._height_depth_of')
        DviFont._height_depth_of.__dict__.__setitem__('stypy_param_names_list', ['char'])
        DviFont._height_depth_of.__dict__.__setitem__('stypy_varargs_param_name', None)
        DviFont._height_depth_of.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DviFont._height_depth_of.__dict__.__setitem__('stypy_call_defaults', defaults)
        DviFont._height_depth_of.__dict__.__setitem__('stypy_call_varargs', varargs)
        DviFont._height_depth_of.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DviFont._height_depth_of.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DviFont._height_depth_of', ['char'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_height_depth_of', localization, ['char'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_height_depth_of(...)' code ##################

        unicode_49238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 587, (-1)), 'unicode', u'\n        Height and depth of char in dvi units. For internal use by dviread.py.\n        ')
        
        # Assigning a List to a Name (line 589):
        
        # Assigning a List to a Name (line 589):
        
        # Obtaining an instance of the builtin type 'list' (line 589)
        list_49239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 589, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 589)
        
        # Assigning a type to the variable 'result' (line 589)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 589, 8), 'result', list_49239)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 590)
        tuple_49240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 590, 29), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 590)
        # Adding element type (line 590)
        
        # Obtaining an instance of the builtin type 'tuple' (line 590)
        tuple_49241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 590, 30), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 590)
        # Adding element type (line 590)
        # Getting the type of 'self' (line 590)
        self_49242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 30), 'self')
        # Obtaining the member '_tfm' of a type (line 590)
        _tfm_49243 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 590, 30), self_49242, '_tfm')
        # Obtaining the member 'height' of a type (line 590)
        height_49244 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 590, 30), _tfm_49243, 'height')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 590, 30), tuple_49241, height_49244)
        # Adding element type (line 590)
        unicode_49245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 590, 48), 'unicode', u'height')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 590, 30), tuple_49241, unicode_49245)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 590, 29), tuple_49240, tuple_49241)
        # Adding element type (line 590)
        
        # Obtaining an instance of the builtin type 'tuple' (line 591)
        tuple_49246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 591, 30), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 591)
        # Adding element type (line 591)
        # Getting the type of 'self' (line 591)
        self_49247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 30), 'self')
        # Obtaining the member '_tfm' of a type (line 591)
        _tfm_49248 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 591, 30), self_49247, '_tfm')
        # Obtaining the member 'depth' of a type (line 591)
        depth_49249 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 591, 30), _tfm_49248, 'depth')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 591, 30), tuple_49246, depth_49249)
        # Adding element type (line 591)
        unicode_49250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 591, 47), 'unicode', u'depth')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 591, 30), tuple_49246, unicode_49250)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 590, 29), tuple_49240, tuple_49246)
        
        # Testing the type of a for loop iterable (line 590)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 590, 8), tuple_49240)
        # Getting the type of the for loop variable (line 590)
        for_loop_var_49251 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 590, 8), tuple_49240)
        # Assigning a type to the variable 'metric' (line 590)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 590, 8), 'metric', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 590, 8), for_loop_var_49251))
        # Assigning a type to the variable 'name' (line 590)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 590, 8), 'name', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 590, 8), for_loop_var_49251))
        # SSA begins for a for statement (line 590)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 592):
        
        # Assigning a Call to a Name (line 592):
        
        # Call to get(...): (line 592)
        # Processing the call arguments (line 592)
        # Getting the type of 'char' (line 592)
        char_49254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 31), 'char', False)
        # Getting the type of 'None' (line 592)
        None_49255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 37), 'None', False)
        # Processing the call keyword arguments (line 592)
        kwargs_49256 = {}
        # Getting the type of 'metric' (line 592)
        metric_49252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 20), 'metric', False)
        # Obtaining the member 'get' of a type (line 592)
        get_49253 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 592, 20), metric_49252, 'get')
        # Calling get(args, kwargs) (line 592)
        get_call_result_49257 = invoke(stypy.reporting.localization.Localization(__file__, 592, 20), get_49253, *[char_49254, None_49255], **kwargs_49256)
        
        # Assigning a type to the variable 'value' (line 592)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 592, 12), 'value', get_call_result_49257)
        
        # Type idiom detected: calculating its left and rigth part (line 593)
        # Getting the type of 'value' (line 593)
        value_49258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 15), 'value')
        # Getting the type of 'None' (line 593)
        None_49259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 24), 'None')
        
        (may_be_49260, more_types_in_union_49261) = may_be_none(value_49258, None_49259)

        if may_be_49260:

            if more_types_in_union_49261:
                # Runtime conditional SSA (line 593)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to report(...): (line 594)
            # Processing the call arguments (line 594)
            unicode_49265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 595, 20), 'unicode', u'No %s for char %d in font %s')
            
            # Obtaining an instance of the builtin type 'tuple' (line 596)
            tuple_49266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 596, 24), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 596)
            # Adding element type (line 596)
            # Getting the type of 'name' (line 596)
            name_49267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 24), 'name', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 596, 24), tuple_49266, name_49267)
            # Adding element type (line 596)
            # Getting the type of 'char' (line 596)
            char_49268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 30), 'char', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 596, 24), tuple_49266, char_49268)
            # Adding element type (line 596)
            # Getting the type of 'self' (line 596)
            self_49269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 36), 'self', False)
            # Obtaining the member 'texname' of a type (line 596)
            texname_49270 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 596, 36), self_49269, 'texname')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 596, 24), tuple_49266, texname_49270)
            
            # Applying the binary operator '%' (line 595)
            result_mod_49271 = python_operator(stypy.reporting.localization.Localization(__file__, 595, 20), '%', unicode_49265, tuple_49266)
            
            unicode_49272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 597, 20), 'unicode', u'debug')
            # Processing the call keyword arguments (line 594)
            kwargs_49273 = {}
            # Getting the type of 'matplotlib' (line 594)
            matplotlib_49262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 16), 'matplotlib', False)
            # Obtaining the member 'verbose' of a type (line 594)
            verbose_49263 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 594, 16), matplotlib_49262, 'verbose')
            # Obtaining the member 'report' of a type (line 594)
            report_49264 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 594, 16), verbose_49263, 'report')
            # Calling report(args, kwargs) (line 594)
            report_call_result_49274 = invoke(stypy.reporting.localization.Localization(__file__, 594, 16), report_49264, *[result_mod_49271, unicode_49272], **kwargs_49273)
            
            
            # Call to append(...): (line 598)
            # Processing the call arguments (line 598)
            int_49277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 598, 30), 'int')
            # Processing the call keyword arguments (line 598)
            kwargs_49278 = {}
            # Getting the type of 'result' (line 598)
            result_49275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 16), 'result', False)
            # Obtaining the member 'append' of a type (line 598)
            append_49276 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 598, 16), result_49275, 'append')
            # Calling append(args, kwargs) (line 598)
            append_call_result_49279 = invoke(stypy.reporting.localization.Localization(__file__, 598, 16), append_49276, *[int_49277], **kwargs_49278)
            

            if more_types_in_union_49261:
                # Runtime conditional SSA for else branch (line 593)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_49260) or more_types_in_union_49261):
            
            # Call to append(...): (line 600)
            # Processing the call arguments (line 600)
            
            # Call to _mul2012(...): (line 600)
            # Processing the call arguments (line 600)
            # Getting the type of 'value' (line 600)
            value_49283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 39), 'value', False)
            # Getting the type of 'self' (line 600)
            self_49284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 46), 'self', False)
            # Obtaining the member '_scale' of a type (line 600)
            _scale_49285 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 600, 46), self_49284, '_scale')
            # Processing the call keyword arguments (line 600)
            kwargs_49286 = {}
            # Getting the type of '_mul2012' (line 600)
            _mul2012_49282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 30), '_mul2012', False)
            # Calling _mul2012(args, kwargs) (line 600)
            _mul2012_call_result_49287 = invoke(stypy.reporting.localization.Localization(__file__, 600, 30), _mul2012_49282, *[value_49283, _scale_49285], **kwargs_49286)
            
            # Processing the call keyword arguments (line 600)
            kwargs_49288 = {}
            # Getting the type of 'result' (line 600)
            result_49280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 16), 'result', False)
            # Obtaining the member 'append' of a type (line 600)
            append_49281 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 600, 16), result_49280, 'append')
            # Calling append(args, kwargs) (line 600)
            append_call_result_49289 = invoke(stypy.reporting.localization.Localization(__file__, 600, 16), append_49281, *[_mul2012_call_result_49287], **kwargs_49288)
            

            if (may_be_49260 and more_types_in_union_49261):
                # SSA join for if statement (line 593)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'result' (line 601)
        result_49290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 15), 'result')
        # Assigning a type to the variable 'stypy_return_type' (line 601)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 601, 8), 'stypy_return_type', result_49290)
        
        # ################# End of '_height_depth_of(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_height_depth_of' in the type store
        # Getting the type of 'stypy_return_type' (line 584)
        stypy_return_type_49291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_49291)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_height_depth_of'
        return stypy_return_type_49291


# Assigning a type to the variable 'DviFont' (line 509)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 509, 0), 'DviFont', DviFont)

# Assigning a Tuple to a Name (line 547):

# Obtaining an instance of the builtin type 'tuple' (line 547)
tuple_49292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 547, 17), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 547)
# Adding element type (line 547)
unicode_49293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 547, 17), 'unicode', u'texname')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 547, 17), tuple_49292, unicode_49293)
# Adding element type (line 547)
unicode_49294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 547, 28), 'unicode', u'size')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 547, 17), tuple_49292, unicode_49294)
# Adding element type (line 547)
unicode_49295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 547, 36), 'unicode', u'widths')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 547, 17), tuple_49292, unicode_49295)
# Adding element type (line 547)
unicode_49296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 547, 46), 'unicode', u'_scale')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 547, 17), tuple_49292, unicode_49296)
# Adding element type (line 547)
unicode_49297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 547, 56), 'unicode', u'_vf')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 547, 17), tuple_49292, unicode_49297)
# Adding element type (line 547)
unicode_49298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 547, 63), 'unicode', u'_tfm')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 547, 17), tuple_49292, unicode_49298)

# Getting the type of 'DviFont'
DviFont_49299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'DviFont')
# Setting the type of the member '__slots__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), DviFont_49299, '__slots__', tuple_49292)
# Declaration of the 'Vf' class
# Getting the type of 'Dvi' (line 604)
Dvi_49300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 9), 'Dvi')

class Vf(Dvi_49300, ):
    unicode_49301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 626, (-1)), 'unicode', u'\n    A virtual font (\\*.vf file) containing subroutines for dvi files.\n\n    Usage::\n\n      vf = Vf(filename)\n      glyph = vf[code]\n      glyph.text, glyph.boxes, glyph.width\n\n    Parameters\n    ----------\n\n    filename : string or bytestring\n\n    Notes\n    -----\n\n    The virtual font format is a derivative of dvi:\n    http://mirrors.ctan.org/info/knuth/virtual-fonts\n    This class reuses some of the machinery of `Dvi`\n    but replaces the `_read` loop and dispatch mechanism.\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 628, 4, False)
        # Assigning a type to the variable 'self' (line 629)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 629, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Vf.__init__', ['filename'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['filename'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Call to __init__(...): (line 629)
        # Processing the call arguments (line 629)
        # Getting the type of 'self' (line 629)
        self_49304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 629, 21), 'self', False)
        # Getting the type of 'filename' (line 629)
        filename_49305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 629, 27), 'filename', False)
        int_49306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 629, 37), 'int')
        # Processing the call keyword arguments (line 629)
        kwargs_49307 = {}
        # Getting the type of 'Dvi' (line 629)
        Dvi_49302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 629, 8), 'Dvi', False)
        # Obtaining the member '__init__' of a type (line 629)
        init___49303 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 629, 8), Dvi_49302, '__init__')
        # Calling __init__(args, kwargs) (line 629)
        init___call_result_49308 = invoke(stypy.reporting.localization.Localization(__file__, 629, 8), init___49303, *[self_49304, filename_49305, int_49306], **kwargs_49307)
        
        
        # Try-finally block (line 630)
        
        # Assigning a Name to a Attribute (line 631):
        
        # Assigning a Name to a Attribute (line 631):
        # Getting the type of 'None' (line 631)
        None_49309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 631, 31), 'None')
        # Getting the type of 'self' (line 631)
        self_49310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 631, 12), 'self')
        # Setting the type of the member '_first_font' of a type (line 631)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 631, 12), self_49310, '_first_font', None_49309)
        
        # Assigning a Dict to a Attribute (line 632):
        
        # Assigning a Dict to a Attribute (line 632):
        
        # Obtaining an instance of the builtin type 'dict' (line 632)
        dict_49311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 632, 26), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 632)
        
        # Getting the type of 'self' (line 632)
        self_49312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 632, 12), 'self')
        # Setting the type of the member '_chars' of a type (line 632)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 632, 12), self_49312, '_chars', dict_49311)
        
        # Call to _read(...): (line 633)
        # Processing the call keyword arguments (line 633)
        kwargs_49315 = {}
        # Getting the type of 'self' (line 633)
        self_49313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 633, 12), 'self', False)
        # Obtaining the member '_read' of a type (line 633)
        _read_49314 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 633, 12), self_49313, '_read')
        # Calling _read(args, kwargs) (line 633)
        _read_call_result_49316 = invoke(stypy.reporting.localization.Localization(__file__, 633, 12), _read_49314, *[], **kwargs_49315)
        
        
        # finally branch of the try-finally block (line 630)
        
        # Call to close(...): (line 635)
        # Processing the call keyword arguments (line 635)
        kwargs_49319 = {}
        # Getting the type of 'self' (line 635)
        self_49317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 12), 'self', False)
        # Obtaining the member 'close' of a type (line 635)
        close_49318 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 635, 12), self_49317, 'close')
        # Calling close(args, kwargs) (line 635)
        close_call_result_49320 = invoke(stypy.reporting.localization.Localization(__file__, 635, 12), close_49318, *[], **kwargs_49319)
        
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def __getitem__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__getitem__'
        module_type_store = module_type_store.open_function_context('__getitem__', 637, 4, False)
        # Assigning a type to the variable 'self' (line 638)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 638, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Vf.__getitem__.__dict__.__setitem__('stypy_localization', localization)
        Vf.__getitem__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Vf.__getitem__.__dict__.__setitem__('stypy_type_store', module_type_store)
        Vf.__getitem__.__dict__.__setitem__('stypy_function_name', 'Vf.__getitem__')
        Vf.__getitem__.__dict__.__setitem__('stypy_param_names_list', ['code'])
        Vf.__getitem__.__dict__.__setitem__('stypy_varargs_param_name', None)
        Vf.__getitem__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Vf.__getitem__.__dict__.__setitem__('stypy_call_defaults', defaults)
        Vf.__getitem__.__dict__.__setitem__('stypy_call_varargs', varargs)
        Vf.__getitem__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Vf.__getitem__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Vf.__getitem__', ['code'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__getitem__', localization, ['code'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__getitem__(...)' code ##################

        
        # Obtaining the type of the subscript
        # Getting the type of 'code' (line 638)
        code_49321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 638, 27), 'code')
        # Getting the type of 'self' (line 638)
        self_49322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 638, 15), 'self')
        # Obtaining the member '_chars' of a type (line 638)
        _chars_49323 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 638, 15), self_49322, '_chars')
        # Obtaining the member '__getitem__' of a type (line 638)
        getitem___49324 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 638, 15), _chars_49323, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 638)
        subscript_call_result_49325 = invoke(stypy.reporting.localization.Localization(__file__, 638, 15), getitem___49324, code_49321)
        
        # Assigning a type to the variable 'stypy_return_type' (line 638)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 638, 8), 'stypy_return_type', subscript_call_result_49325)
        
        # ################# End of '__getitem__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__getitem__' in the type store
        # Getting the type of 'stypy_return_type' (line 637)
        stypy_return_type_49326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_49326)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__getitem__'
        return stypy_return_type_49326


    @norecursion
    def _read(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_read'
        module_type_store = module_type_store.open_function_context('_read', 640, 4, False)
        # Assigning a type to the variable 'self' (line 641)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 641, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Vf._read.__dict__.__setitem__('stypy_localization', localization)
        Vf._read.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Vf._read.__dict__.__setitem__('stypy_type_store', module_type_store)
        Vf._read.__dict__.__setitem__('stypy_function_name', 'Vf._read')
        Vf._read.__dict__.__setitem__('stypy_param_names_list', [])
        Vf._read.__dict__.__setitem__('stypy_varargs_param_name', None)
        Vf._read.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Vf._read.__dict__.__setitem__('stypy_call_defaults', defaults)
        Vf._read.__dict__.__setitem__('stypy_call_varargs', varargs)
        Vf._read.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Vf._read.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Vf._read', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_read', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_read(...)' code ##################

        unicode_49327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 644, (-1)), 'unicode', u'\n        Read one page from the file. Return True if successful,\n        False if there were no more pages.\n        ')
        
        # Assigning a Tuple to a Tuple (line 645):
        
        # Assigning a Name to a Name (line 645):
        # Getting the type of 'None' (line 645)
        None_49328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 645, 48), 'None')
        # Assigning a type to the variable 'tuple_assignment_47795' (line 645)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 645, 8), 'tuple_assignment_47795', None_49328)
        
        # Assigning a Name to a Name (line 645):
        # Getting the type of 'None' (line 645)
        None_49329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 645, 54), 'None')
        # Assigning a type to the variable 'tuple_assignment_47796' (line 645)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 645, 8), 'tuple_assignment_47796', None_49329)
        
        # Assigning a Name to a Name (line 645):
        # Getting the type of 'None' (line 645)
        None_49330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 645, 60), 'None')
        # Assigning a type to the variable 'tuple_assignment_47797' (line 645)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 645, 8), 'tuple_assignment_47797', None_49330)
        
        # Assigning a Name to a Name (line 645):
        # Getting the type of 'tuple_assignment_47795' (line 645)
        tuple_assignment_47795_49331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 645, 8), 'tuple_assignment_47795')
        # Assigning a type to the variable 'packet_len' (line 645)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 645, 8), 'packet_len', tuple_assignment_47795_49331)
        
        # Assigning a Name to a Name (line 645):
        # Getting the type of 'tuple_assignment_47796' (line 645)
        tuple_assignment_47796_49332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 645, 8), 'tuple_assignment_47796')
        # Assigning a type to the variable 'packet_char' (line 645)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 645, 20), 'packet_char', tuple_assignment_47796_49332)
        
        # Assigning a Name to a Name (line 645):
        # Getting the type of 'tuple_assignment_47797' (line 645)
        tuple_assignment_47797_49333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 645, 8), 'tuple_assignment_47797')
        # Assigning a type to the variable 'packet_width' (line 645)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 645, 33), 'packet_width', tuple_assignment_47797_49333)
        
        # Getting the type of 'True' (line 646)
        True_49334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 646, 14), 'True')
        # Testing the type of an if condition (line 646)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 646, 8), True_49334)
        # SSA begins for while statement (line 646)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Assigning a Call to a Name (line 647):
        
        # Assigning a Call to a Name (line 647):
        
        # Call to ord(...): (line 647)
        # Processing the call arguments (line 647)
        
        # Obtaining the type of the subscript
        int_49336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 647, 41), 'int')
        
        # Call to read(...): (line 647)
        # Processing the call arguments (line 647)
        int_49340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 647, 38), 'int')
        # Processing the call keyword arguments (line 647)
        kwargs_49341 = {}
        # Getting the type of 'self' (line 647)
        self_49337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 647, 23), 'self', False)
        # Obtaining the member 'file' of a type (line 647)
        file_49338 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 647, 23), self_49337, 'file')
        # Obtaining the member 'read' of a type (line 647)
        read_49339 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 647, 23), file_49338, 'read')
        # Calling read(args, kwargs) (line 647)
        read_call_result_49342 = invoke(stypy.reporting.localization.Localization(__file__, 647, 23), read_49339, *[int_49340], **kwargs_49341)
        
        # Obtaining the member '__getitem__' of a type (line 647)
        getitem___49343 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 647, 23), read_call_result_49342, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 647)
        subscript_call_result_49344 = invoke(stypy.reporting.localization.Localization(__file__, 647, 23), getitem___49343, int_49336)
        
        # Processing the call keyword arguments (line 647)
        kwargs_49345 = {}
        # Getting the type of 'ord' (line 647)
        ord_49335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 647, 19), 'ord', False)
        # Calling ord(args, kwargs) (line 647)
        ord_call_result_49346 = invoke(stypy.reporting.localization.Localization(__file__, 647, 19), ord_49335, *[subscript_call_result_49344], **kwargs_49345)
        
        # Assigning a type to the variable 'byte' (line 647)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 647, 12), 'byte', ord_call_result_49346)
        
        
        # Getting the type of 'self' (line 649)
        self_49347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 15), 'self')
        # Obtaining the member 'state' of a type (line 649)
        state_49348 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 649, 15), self_49347, 'state')
        # Getting the type of '_dvistate' (line 649)
        _dvistate_49349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 29), '_dvistate')
        # Obtaining the member 'inpage' of a type (line 649)
        inpage_49350 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 649, 29), _dvistate_49349, 'inpage')
        # Applying the binary operator '==' (line 649)
        result_eq_49351 = python_operator(stypy.reporting.localization.Localization(__file__, 649, 15), '==', state_49348, inpage_49350)
        
        # Testing the type of an if condition (line 649)
        if_condition_49352 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 649, 12), result_eq_49351)
        # Assigning a type to the variable 'if_condition_49352' (line 649)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 649, 12), 'if_condition_49352', if_condition_49352)
        # SSA begins for if statement (line 649)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 650):
        
        # Assigning a BinOp to a Name (line 650):
        
        # Call to tell(...): (line 650)
        # Processing the call keyword arguments (line 650)
        kwargs_49356 = {}
        # Getting the type of 'self' (line 650)
        self_49353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 26), 'self', False)
        # Obtaining the member 'file' of a type (line 650)
        file_49354 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 650, 26), self_49353, 'file')
        # Obtaining the member 'tell' of a type (line 650)
        tell_49355 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 650, 26), file_49354, 'tell')
        # Calling tell(args, kwargs) (line 650)
        tell_call_result_49357 = invoke(stypy.reporting.localization.Localization(__file__, 650, 26), tell_49355, *[], **kwargs_49356)
        
        int_49358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 650, 43), 'int')
        # Applying the binary operator '-' (line 650)
        result_sub_49359 = python_operator(stypy.reporting.localization.Localization(__file__, 650, 26), '-', tell_call_result_49357, int_49358)
        
        # Assigning a type to the variable 'byte_at' (line 650)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 650, 16), 'byte_at', result_sub_49359)
        
        
        # Getting the type of 'byte_at' (line 651)
        byte_at_49360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 19), 'byte_at')
        # Getting the type of 'packet_ends' (line 651)
        packet_ends_49361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 30), 'packet_ends')
        # Applying the binary operator '==' (line 651)
        result_eq_49362 = python_operator(stypy.reporting.localization.Localization(__file__, 651, 19), '==', byte_at_49360, packet_ends_49361)
        
        # Testing the type of an if condition (line 651)
        if_condition_49363 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 651, 16), result_eq_49362)
        # Assigning a type to the variable 'if_condition_49363' (line 651)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 651, 16), 'if_condition_49363', if_condition_49363)
        # SSA begins for if statement (line 651)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to _finalize_packet(...): (line 652)
        # Processing the call arguments (line 652)
        # Getting the type of 'packet_char' (line 652)
        packet_char_49366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 652, 42), 'packet_char', False)
        # Getting the type of 'packet_width' (line 652)
        packet_width_49367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 652, 55), 'packet_width', False)
        # Processing the call keyword arguments (line 652)
        kwargs_49368 = {}
        # Getting the type of 'self' (line 652)
        self_49364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 652, 20), 'self', False)
        # Obtaining the member '_finalize_packet' of a type (line 652)
        _finalize_packet_49365 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 652, 20), self_49364, '_finalize_packet')
        # Calling _finalize_packet(args, kwargs) (line 652)
        _finalize_packet_call_result_49369 = invoke(stypy.reporting.localization.Localization(__file__, 652, 20), _finalize_packet_49365, *[packet_char_49366, packet_width_49367], **kwargs_49368)
        
        
        # Assigning a Tuple to a Tuple (line 653):
        
        # Assigning a Name to a Name (line 653):
        # Getting the type of 'None' (line 653)
        None_49370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 60), 'None')
        # Assigning a type to the variable 'tuple_assignment_47798' (line 653)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 653, 20), 'tuple_assignment_47798', None_49370)
        
        # Assigning a Name to a Name (line 653):
        # Getting the type of 'None' (line 653)
        None_49371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 66), 'None')
        # Assigning a type to the variable 'tuple_assignment_47799' (line 653)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 653, 20), 'tuple_assignment_47799', None_49371)
        
        # Assigning a Name to a Name (line 653):
        # Getting the type of 'None' (line 653)
        None_49372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 72), 'None')
        # Assigning a type to the variable 'tuple_assignment_47800' (line 653)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 653, 20), 'tuple_assignment_47800', None_49372)
        
        # Assigning a Name to a Name (line 653):
        # Getting the type of 'tuple_assignment_47798' (line 653)
        tuple_assignment_47798_49373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 20), 'tuple_assignment_47798')
        # Assigning a type to the variable 'packet_len' (line 653)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 653, 20), 'packet_len', tuple_assignment_47798_49373)
        
        # Assigning a Name to a Name (line 653):
        # Getting the type of 'tuple_assignment_47799' (line 653)
        tuple_assignment_47799_49374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 20), 'tuple_assignment_47799')
        # Assigning a type to the variable 'packet_char' (line 653)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 653, 32), 'packet_char', tuple_assignment_47799_49374)
        
        # Assigning a Name to a Name (line 653):
        # Getting the type of 'tuple_assignment_47800' (line 653)
        tuple_assignment_47800_49375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 20), 'tuple_assignment_47800')
        # Assigning a type to the variable 'packet_width' (line 653)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 653, 45), 'packet_width', tuple_assignment_47800_49375)
        # SSA branch for the else part of an if statement (line 651)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'byte_at' (line 655)
        byte_at_49376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 21), 'byte_at')
        # Getting the type of 'packet_ends' (line 655)
        packet_ends_49377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 31), 'packet_ends')
        # Applying the binary operator '>' (line 655)
        result_gt_49378 = python_operator(stypy.reporting.localization.Localization(__file__, 655, 21), '>', byte_at_49376, packet_ends_49377)
        
        # Testing the type of an if condition (line 655)
        if_condition_49379 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 655, 21), result_gt_49378)
        # Assigning a type to the variable 'if_condition_49379' (line 655)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 655, 21), 'if_condition_49379', if_condition_49379)
        # SSA begins for if statement (line 655)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 656)
        # Processing the call arguments (line 656)
        unicode_49381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 656, 37), 'unicode', u'Packet length mismatch in vf file')
        # Processing the call keyword arguments (line 656)
        kwargs_49382 = {}
        # Getting the type of 'ValueError' (line 656)
        ValueError_49380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 26), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 656)
        ValueError_call_result_49383 = invoke(stypy.reporting.localization.Localization(__file__, 656, 26), ValueError_49380, *[unicode_49381], **kwargs_49382)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 656, 20), ValueError_call_result_49383, 'raise parameter', BaseException)
        # SSA branch for the else part of an if statement (line 655)
        module_type_store.open_ssa_branch('else')
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'byte' (line 658)
        byte_49384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 658, 23), 'byte')
        
        # Obtaining an instance of the builtin type 'tuple' (line 658)
        tuple_49385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 658, 32), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 658)
        # Adding element type (line 658)
        int_49386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 658, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 658, 32), tuple_49385, int_49386)
        # Adding element type (line 658)
        int_49387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 658, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 658, 32), tuple_49385, int_49387)
        
        # Applying the binary operator 'in' (line 658)
        result_contains_49388 = python_operator(stypy.reporting.localization.Localization(__file__, 658, 23), 'in', byte_49384, tuple_49385)
        
        
        # Getting the type of 'byte' (line 658)
        byte_49389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 658, 45), 'byte')
        int_49390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 658, 53), 'int')
        # Applying the binary operator '>=' (line 658)
        result_ge_49391 = python_operator(stypy.reporting.localization.Localization(__file__, 658, 45), '>=', byte_49389, int_49390)
        
        # Applying the binary operator 'or' (line 658)
        result_or_keyword_49392 = python_operator(stypy.reporting.localization.Localization(__file__, 658, 23), 'or', result_contains_49388, result_ge_49391)
        
        # Testing the type of an if condition (line 658)
        if_condition_49393 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 658, 20), result_or_keyword_49392)
        # Assigning a type to the variable 'if_condition_49393' (line 658)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 658, 20), 'if_condition_49393', if_condition_49393)
        # SSA begins for if statement (line 658)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 659)
        # Processing the call arguments (line 659)
        unicode_49395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 660, 28), 'unicode', u'Inappropriate opcode %d in vf file')
        # Getting the type of 'byte' (line 660)
        byte_49396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 67), 'byte', False)
        # Applying the binary operator '%' (line 660)
        result_mod_49397 = python_operator(stypy.reporting.localization.Localization(__file__, 660, 28), '%', unicode_49395, byte_49396)
        
        # Processing the call keyword arguments (line 659)
        kwargs_49398 = {}
        # Getting the type of 'ValueError' (line 659)
        ValueError_49394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 30), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 659)
        ValueError_call_result_49399 = invoke(stypy.reporting.localization.Localization(__file__, 659, 30), ValueError_49394, *[result_mod_49397], **kwargs_49398)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 659, 24), ValueError_call_result_49399, 'raise parameter', BaseException)
        # SSA join for if statement (line 658)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to (...): (line 661)
        # Processing the call arguments (line 661)
        # Getting the type of 'self' (line 661)
        self_49405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 38), 'self', False)
        # Getting the type of 'byte' (line 661)
        byte_49406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 44), 'byte', False)
        # Processing the call keyword arguments (line 661)
        kwargs_49407 = {}
        
        # Obtaining the type of the subscript
        # Getting the type of 'byte' (line 661)
        byte_49400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 32), 'byte', False)
        # Getting the type of 'Dvi' (line 661)
        Dvi_49401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 20), 'Dvi', False)
        # Obtaining the member '_dtable' of a type (line 661)
        _dtable_49402 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 661, 20), Dvi_49401, '_dtable')
        # Obtaining the member '__getitem__' of a type (line 661)
        getitem___49403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 661, 20), _dtable_49402, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 661)
        subscript_call_result_49404 = invoke(stypy.reporting.localization.Localization(__file__, 661, 20), getitem___49403, byte_49400)
        
        # Calling (args, kwargs) (line 661)
        _call_result_49408 = invoke(stypy.reporting.localization.Localization(__file__, 661, 20), subscript_call_result_49404, *[self_49405, byte_49406], **kwargs_49407)
        
        # SSA join for if statement (line 655)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 651)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 649)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'byte' (line 665)
        byte_49409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 665, 15), 'byte')
        int_49410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 665, 22), 'int')
        # Applying the binary operator '<' (line 665)
        result_lt_49411 = python_operator(stypy.reporting.localization.Localization(__file__, 665, 15), '<', byte_49409, int_49410)
        
        # Testing the type of an if condition (line 665)
        if_condition_49412 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 665, 12), result_lt_49411)
        # Assigning a type to the variable 'if_condition_49412' (line 665)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 665, 12), 'if_condition_49412', if_condition_49412)
        # SSA begins for if statement (line 665)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 666):
        
        # Assigning a Name to a Name (line 666):
        # Getting the type of 'byte' (line 666)
        byte_49413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 29), 'byte')
        # Assigning a type to the variable 'packet_len' (line 666)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 666, 16), 'packet_len', byte_49413)
        
        # Assigning a Tuple to a Tuple (line 667):
        
        # Assigning a Call to a Name (line 667):
        
        # Call to _arg(...): (line 667)
        # Processing the call arguments (line 667)
        int_49416 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 667, 54), 'int')
        # Processing the call keyword arguments (line 667)
        kwargs_49417 = {}
        # Getting the type of 'self' (line 667)
        self_49414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 44), 'self', False)
        # Obtaining the member '_arg' of a type (line 667)
        _arg_49415 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 667, 44), self_49414, '_arg')
        # Calling _arg(args, kwargs) (line 667)
        _arg_call_result_49418 = invoke(stypy.reporting.localization.Localization(__file__, 667, 44), _arg_49415, *[int_49416], **kwargs_49417)
        
        # Assigning a type to the variable 'tuple_assignment_47801' (line 667)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 667, 16), 'tuple_assignment_47801', _arg_call_result_49418)
        
        # Assigning a Call to a Name (line 667):
        
        # Call to _arg(...): (line 667)
        # Processing the call arguments (line 667)
        int_49421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 667, 68), 'int')
        # Processing the call keyword arguments (line 667)
        kwargs_49422 = {}
        # Getting the type of 'self' (line 667)
        self_49419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 58), 'self', False)
        # Obtaining the member '_arg' of a type (line 667)
        _arg_49420 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 667, 58), self_49419, '_arg')
        # Calling _arg(args, kwargs) (line 667)
        _arg_call_result_49423 = invoke(stypy.reporting.localization.Localization(__file__, 667, 58), _arg_49420, *[int_49421], **kwargs_49422)
        
        # Assigning a type to the variable 'tuple_assignment_47802' (line 667)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 667, 16), 'tuple_assignment_47802', _arg_call_result_49423)
        
        # Assigning a Name to a Name (line 667):
        # Getting the type of 'tuple_assignment_47801' (line 667)
        tuple_assignment_47801_49424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 16), 'tuple_assignment_47801')
        # Assigning a type to the variable 'packet_char' (line 667)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 667, 16), 'packet_char', tuple_assignment_47801_49424)
        
        # Assigning a Name to a Name (line 667):
        # Getting the type of 'tuple_assignment_47802' (line 667)
        tuple_assignment_47802_49425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 16), 'tuple_assignment_47802')
        # Assigning a type to the variable 'packet_width' (line 667)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 667, 29), 'packet_width', tuple_assignment_47802_49425)
        
        # Assigning a Call to a Name (line 668):
        
        # Assigning a Call to a Name (line 668):
        
        # Call to _init_packet(...): (line 668)
        # Processing the call arguments (line 668)
        # Getting the type of 'byte' (line 668)
        byte_49428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 48), 'byte', False)
        # Processing the call keyword arguments (line 668)
        kwargs_49429 = {}
        # Getting the type of 'self' (line 668)
        self_49426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 30), 'self', False)
        # Obtaining the member '_init_packet' of a type (line 668)
        _init_packet_49427 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 668, 30), self_49426, '_init_packet')
        # Calling _init_packet(args, kwargs) (line 668)
        _init_packet_call_result_49430 = invoke(stypy.reporting.localization.Localization(__file__, 668, 30), _init_packet_49427, *[byte_49428], **kwargs_49429)
        
        # Assigning a type to the variable 'packet_ends' (line 668)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 668, 16), 'packet_ends', _init_packet_call_result_49430)
        
        # Assigning a Attribute to a Attribute (line 669):
        
        # Assigning a Attribute to a Attribute (line 669):
        # Getting the type of '_dvistate' (line 669)
        _dvistate_49431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 669, 29), '_dvistate')
        # Obtaining the member 'inpage' of a type (line 669)
        inpage_49432 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 669, 29), _dvistate_49431, 'inpage')
        # Getting the type of 'self' (line 669)
        self_49433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 669, 16), 'self')
        # Setting the type of the member 'state' of a type (line 669)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 669, 16), self_49433, 'state', inpage_49432)
        # SSA branch for the else part of an if statement (line 665)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'byte' (line 670)
        byte_49434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 670, 17), 'byte')
        int_49435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 670, 25), 'int')
        # Applying the binary operator '==' (line 670)
        result_eq_49436 = python_operator(stypy.reporting.localization.Localization(__file__, 670, 17), '==', byte_49434, int_49435)
        
        # Testing the type of an if condition (line 670)
        if_condition_49437 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 670, 17), result_eq_49436)
        # Assigning a type to the variable 'if_condition_49437' (line 670)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 670, 17), 'if_condition_49437', if_condition_49437)
        # SSA begins for if statement (line 670)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a ListComp to a Tuple (line 671):
        
        # Assigning a Subscript to a Name (line 671):
        
        # Obtaining the type of the subscript
        int_49438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 671, 16), 'int')
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Obtaining an instance of the builtin type 'tuple' (line 672)
        tuple_49444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 672, 52), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 672)
        # Adding element type (line 672)
        int_49445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 672, 52), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 672, 52), tuple_49444, int_49445)
        # Adding element type (line 672)
        int_49446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 672, 55), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 672, 52), tuple_49444, int_49446)
        # Adding element type (line 672)
        int_49447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 672, 58), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 672, 52), tuple_49444, int_49447)
        
        comprehension_49448 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 672, 29), tuple_49444)
        # Assigning a type to the variable 'x' (line 672)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 672, 29), 'x', comprehension_49448)
        
        # Call to _arg(...): (line 672)
        # Processing the call arguments (line 672)
        # Getting the type of 'x' (line 672)
        x_49441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 672, 39), 'x', False)
        # Processing the call keyword arguments (line 672)
        kwargs_49442 = {}
        # Getting the type of 'self' (line 672)
        self_49439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 672, 29), 'self', False)
        # Obtaining the member '_arg' of a type (line 672)
        _arg_49440 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 672, 29), self_49439, '_arg')
        # Calling _arg(args, kwargs) (line 672)
        _arg_call_result_49443 = invoke(stypy.reporting.localization.Localization(__file__, 672, 29), _arg_49440, *[x_49441], **kwargs_49442)
        
        list_49449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 672, 29), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 672, 29), list_49449, _arg_call_result_49443)
        # Obtaining the member '__getitem__' of a type (line 671)
        getitem___49450 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 671, 16), list_49449, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 671)
        subscript_call_result_49451 = invoke(stypy.reporting.localization.Localization(__file__, 671, 16), getitem___49450, int_49438)
        
        # Assigning a type to the variable 'tuple_var_assignment_47803' (line 671)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 671, 16), 'tuple_var_assignment_47803', subscript_call_result_49451)
        
        # Assigning a Subscript to a Name (line 671):
        
        # Obtaining the type of the subscript
        int_49452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 671, 16), 'int')
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Obtaining an instance of the builtin type 'tuple' (line 672)
        tuple_49458 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 672, 52), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 672)
        # Adding element type (line 672)
        int_49459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 672, 52), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 672, 52), tuple_49458, int_49459)
        # Adding element type (line 672)
        int_49460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 672, 55), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 672, 52), tuple_49458, int_49460)
        # Adding element type (line 672)
        int_49461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 672, 58), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 672, 52), tuple_49458, int_49461)
        
        comprehension_49462 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 672, 29), tuple_49458)
        # Assigning a type to the variable 'x' (line 672)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 672, 29), 'x', comprehension_49462)
        
        # Call to _arg(...): (line 672)
        # Processing the call arguments (line 672)
        # Getting the type of 'x' (line 672)
        x_49455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 672, 39), 'x', False)
        # Processing the call keyword arguments (line 672)
        kwargs_49456 = {}
        # Getting the type of 'self' (line 672)
        self_49453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 672, 29), 'self', False)
        # Obtaining the member '_arg' of a type (line 672)
        _arg_49454 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 672, 29), self_49453, '_arg')
        # Calling _arg(args, kwargs) (line 672)
        _arg_call_result_49457 = invoke(stypy.reporting.localization.Localization(__file__, 672, 29), _arg_49454, *[x_49455], **kwargs_49456)
        
        list_49463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 672, 29), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 672, 29), list_49463, _arg_call_result_49457)
        # Obtaining the member '__getitem__' of a type (line 671)
        getitem___49464 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 671, 16), list_49463, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 671)
        subscript_call_result_49465 = invoke(stypy.reporting.localization.Localization(__file__, 671, 16), getitem___49464, int_49452)
        
        # Assigning a type to the variable 'tuple_var_assignment_47804' (line 671)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 671, 16), 'tuple_var_assignment_47804', subscript_call_result_49465)
        
        # Assigning a Subscript to a Name (line 671):
        
        # Obtaining the type of the subscript
        int_49466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 671, 16), 'int')
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Obtaining an instance of the builtin type 'tuple' (line 672)
        tuple_49472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 672, 52), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 672)
        # Adding element type (line 672)
        int_49473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 672, 52), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 672, 52), tuple_49472, int_49473)
        # Adding element type (line 672)
        int_49474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 672, 55), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 672, 52), tuple_49472, int_49474)
        # Adding element type (line 672)
        int_49475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 672, 58), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 672, 52), tuple_49472, int_49475)
        
        comprehension_49476 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 672, 29), tuple_49472)
        # Assigning a type to the variable 'x' (line 672)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 672, 29), 'x', comprehension_49476)
        
        # Call to _arg(...): (line 672)
        # Processing the call arguments (line 672)
        # Getting the type of 'x' (line 672)
        x_49469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 672, 39), 'x', False)
        # Processing the call keyword arguments (line 672)
        kwargs_49470 = {}
        # Getting the type of 'self' (line 672)
        self_49467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 672, 29), 'self', False)
        # Obtaining the member '_arg' of a type (line 672)
        _arg_49468 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 672, 29), self_49467, '_arg')
        # Calling _arg(args, kwargs) (line 672)
        _arg_call_result_49471 = invoke(stypy.reporting.localization.Localization(__file__, 672, 29), _arg_49468, *[x_49469], **kwargs_49470)
        
        list_49477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 672, 29), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 672, 29), list_49477, _arg_call_result_49471)
        # Obtaining the member '__getitem__' of a type (line 671)
        getitem___49478 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 671, 16), list_49477, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 671)
        subscript_call_result_49479 = invoke(stypy.reporting.localization.Localization(__file__, 671, 16), getitem___49478, int_49466)
        
        # Assigning a type to the variable 'tuple_var_assignment_47805' (line 671)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 671, 16), 'tuple_var_assignment_47805', subscript_call_result_49479)
        
        # Assigning a Name to a Name (line 671):
        # Getting the type of 'tuple_var_assignment_47803' (line 671)
        tuple_var_assignment_47803_49480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 671, 16), 'tuple_var_assignment_47803')
        # Assigning a type to the variable 'packet_len' (line 671)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 671, 16), 'packet_len', tuple_var_assignment_47803_49480)
        
        # Assigning a Name to a Name (line 671):
        # Getting the type of 'tuple_var_assignment_47804' (line 671)
        tuple_var_assignment_47804_49481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 671, 16), 'tuple_var_assignment_47804')
        # Assigning a type to the variable 'packet_char' (line 671)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 671, 28), 'packet_char', tuple_var_assignment_47804_49481)
        
        # Assigning a Name to a Name (line 671):
        # Getting the type of 'tuple_var_assignment_47805' (line 671)
        tuple_var_assignment_47805_49482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 671, 16), 'tuple_var_assignment_47805')
        # Assigning a type to the variable 'packet_width' (line 671)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 671, 41), 'packet_width', tuple_var_assignment_47805_49482)
        
        # Call to _init_packet(...): (line 673)
        # Processing the call arguments (line 673)
        # Getting the type of 'packet_len' (line 673)
        packet_len_49485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 673, 34), 'packet_len', False)
        # Processing the call keyword arguments (line 673)
        kwargs_49486 = {}
        # Getting the type of 'self' (line 673)
        self_49483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 673, 16), 'self', False)
        # Obtaining the member '_init_packet' of a type (line 673)
        _init_packet_49484 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 673, 16), self_49483, '_init_packet')
        # Calling _init_packet(args, kwargs) (line 673)
        _init_packet_call_result_49487 = invoke(stypy.reporting.localization.Localization(__file__, 673, 16), _init_packet_49484, *[packet_len_49485], **kwargs_49486)
        
        # SSA branch for the else part of an if statement (line 670)
        module_type_store.open_ssa_branch('else')
        
        
        int_49488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 674, 17), 'int')
        # Getting the type of 'byte' (line 674)
        byte_49489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 674, 24), 'byte')
        # Applying the binary operator '<=' (line 674)
        result_le_49490 = python_operator(stypy.reporting.localization.Localization(__file__, 674, 17), '<=', int_49488, byte_49489)
        int_49491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 674, 32), 'int')
        # Applying the binary operator '<=' (line 674)
        result_le_49492 = python_operator(stypy.reporting.localization.Localization(__file__, 674, 17), '<=', byte_49489, int_49491)
        # Applying the binary operator '&' (line 674)
        result_and__49493 = python_operator(stypy.reporting.localization.Localization(__file__, 674, 17), '&', result_le_49490, result_le_49492)
        
        # Testing the type of an if condition (line 674)
        if_condition_49494 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 674, 17), result_and__49493)
        # Assigning a type to the variable 'if_condition_49494' (line 674)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 674, 17), 'if_condition_49494', if_condition_49494)
        # SSA begins for if statement (line 674)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 675):
        
        # Assigning a Call to a Name (line 675):
        
        # Call to _arg(...): (line 675)
        # Processing the call arguments (line 675)
        # Getting the type of 'byte' (line 675)
        byte_49497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 30), 'byte', False)
        int_49498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 675, 37), 'int')
        # Applying the binary operator '-' (line 675)
        result_sub_49499 = python_operator(stypy.reporting.localization.Localization(__file__, 675, 30), '-', byte_49497, int_49498)
        
        
        # Getting the type of 'byte' (line 675)
        byte_49500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 42), 'byte', False)
        int_49501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 675, 50), 'int')
        # Applying the binary operator '==' (line 675)
        result_eq_49502 = python_operator(stypy.reporting.localization.Localization(__file__, 675, 42), '==', byte_49500, int_49501)
        
        # Processing the call keyword arguments (line 675)
        kwargs_49503 = {}
        # Getting the type of 'self' (line 675)
        self_49495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 20), 'self', False)
        # Obtaining the member '_arg' of a type (line 675)
        _arg_49496 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 675, 20), self_49495, '_arg')
        # Calling _arg(args, kwargs) (line 675)
        _arg_call_result_49504 = invoke(stypy.reporting.localization.Localization(__file__, 675, 20), _arg_49496, *[result_sub_49499, result_eq_49502], **kwargs_49503)
        
        # Assigning a type to the variable 'k' (line 675)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 675, 16), 'k', _arg_call_result_49504)
        
        # Assigning a ListComp to a Tuple (line 676):
        
        # Assigning a Subscript to a Name (line 676):
        
        # Obtaining the type of the subscript
        int_49505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 676, 16), 'int')
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Obtaining an instance of the builtin type 'tuple' (line 676)
        tuple_49511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 676, 56), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 676)
        # Adding element type (line 676)
        int_49512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 676, 56), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 676, 56), tuple_49511, int_49512)
        # Adding element type (line 676)
        int_49513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 676, 59), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 676, 56), tuple_49511, int_49513)
        # Adding element type (line 676)
        int_49514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 676, 62), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 676, 56), tuple_49511, int_49514)
        # Adding element type (line 676)
        int_49515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 676, 65), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 676, 56), tuple_49511, int_49515)
        # Adding element type (line 676)
        int_49516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 676, 68), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 676, 56), tuple_49511, int_49516)
        
        comprehension_49517 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 676, 33), tuple_49511)
        # Assigning a type to the variable 'x' (line 676)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 676, 33), 'x', comprehension_49517)
        
        # Call to _arg(...): (line 676)
        # Processing the call arguments (line 676)
        # Getting the type of 'x' (line 676)
        x_49508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 676, 43), 'x', False)
        # Processing the call keyword arguments (line 676)
        kwargs_49509 = {}
        # Getting the type of 'self' (line 676)
        self_49506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 676, 33), 'self', False)
        # Obtaining the member '_arg' of a type (line 676)
        _arg_49507 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 676, 33), self_49506, '_arg')
        # Calling _arg(args, kwargs) (line 676)
        _arg_call_result_49510 = invoke(stypy.reporting.localization.Localization(__file__, 676, 33), _arg_49507, *[x_49508], **kwargs_49509)
        
        list_49518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 676, 33), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 676, 33), list_49518, _arg_call_result_49510)
        # Obtaining the member '__getitem__' of a type (line 676)
        getitem___49519 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 676, 16), list_49518, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 676)
        subscript_call_result_49520 = invoke(stypy.reporting.localization.Localization(__file__, 676, 16), getitem___49519, int_49505)
        
        # Assigning a type to the variable 'tuple_var_assignment_47806' (line 676)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 676, 16), 'tuple_var_assignment_47806', subscript_call_result_49520)
        
        # Assigning a Subscript to a Name (line 676):
        
        # Obtaining the type of the subscript
        int_49521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 676, 16), 'int')
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Obtaining an instance of the builtin type 'tuple' (line 676)
        tuple_49527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 676, 56), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 676)
        # Adding element type (line 676)
        int_49528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 676, 56), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 676, 56), tuple_49527, int_49528)
        # Adding element type (line 676)
        int_49529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 676, 59), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 676, 56), tuple_49527, int_49529)
        # Adding element type (line 676)
        int_49530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 676, 62), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 676, 56), tuple_49527, int_49530)
        # Adding element type (line 676)
        int_49531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 676, 65), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 676, 56), tuple_49527, int_49531)
        # Adding element type (line 676)
        int_49532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 676, 68), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 676, 56), tuple_49527, int_49532)
        
        comprehension_49533 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 676, 33), tuple_49527)
        # Assigning a type to the variable 'x' (line 676)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 676, 33), 'x', comprehension_49533)
        
        # Call to _arg(...): (line 676)
        # Processing the call arguments (line 676)
        # Getting the type of 'x' (line 676)
        x_49524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 676, 43), 'x', False)
        # Processing the call keyword arguments (line 676)
        kwargs_49525 = {}
        # Getting the type of 'self' (line 676)
        self_49522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 676, 33), 'self', False)
        # Obtaining the member '_arg' of a type (line 676)
        _arg_49523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 676, 33), self_49522, '_arg')
        # Calling _arg(args, kwargs) (line 676)
        _arg_call_result_49526 = invoke(stypy.reporting.localization.Localization(__file__, 676, 33), _arg_49523, *[x_49524], **kwargs_49525)
        
        list_49534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 676, 33), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 676, 33), list_49534, _arg_call_result_49526)
        # Obtaining the member '__getitem__' of a type (line 676)
        getitem___49535 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 676, 16), list_49534, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 676)
        subscript_call_result_49536 = invoke(stypy.reporting.localization.Localization(__file__, 676, 16), getitem___49535, int_49521)
        
        # Assigning a type to the variable 'tuple_var_assignment_47807' (line 676)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 676, 16), 'tuple_var_assignment_47807', subscript_call_result_49536)
        
        # Assigning a Subscript to a Name (line 676):
        
        # Obtaining the type of the subscript
        int_49537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 676, 16), 'int')
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Obtaining an instance of the builtin type 'tuple' (line 676)
        tuple_49543 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 676, 56), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 676)
        # Adding element type (line 676)
        int_49544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 676, 56), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 676, 56), tuple_49543, int_49544)
        # Adding element type (line 676)
        int_49545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 676, 59), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 676, 56), tuple_49543, int_49545)
        # Adding element type (line 676)
        int_49546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 676, 62), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 676, 56), tuple_49543, int_49546)
        # Adding element type (line 676)
        int_49547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 676, 65), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 676, 56), tuple_49543, int_49547)
        # Adding element type (line 676)
        int_49548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 676, 68), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 676, 56), tuple_49543, int_49548)
        
        comprehension_49549 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 676, 33), tuple_49543)
        # Assigning a type to the variable 'x' (line 676)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 676, 33), 'x', comprehension_49549)
        
        # Call to _arg(...): (line 676)
        # Processing the call arguments (line 676)
        # Getting the type of 'x' (line 676)
        x_49540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 676, 43), 'x', False)
        # Processing the call keyword arguments (line 676)
        kwargs_49541 = {}
        # Getting the type of 'self' (line 676)
        self_49538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 676, 33), 'self', False)
        # Obtaining the member '_arg' of a type (line 676)
        _arg_49539 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 676, 33), self_49538, '_arg')
        # Calling _arg(args, kwargs) (line 676)
        _arg_call_result_49542 = invoke(stypy.reporting.localization.Localization(__file__, 676, 33), _arg_49539, *[x_49540], **kwargs_49541)
        
        list_49550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 676, 33), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 676, 33), list_49550, _arg_call_result_49542)
        # Obtaining the member '__getitem__' of a type (line 676)
        getitem___49551 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 676, 16), list_49550, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 676)
        subscript_call_result_49552 = invoke(stypy.reporting.localization.Localization(__file__, 676, 16), getitem___49551, int_49537)
        
        # Assigning a type to the variable 'tuple_var_assignment_47808' (line 676)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 676, 16), 'tuple_var_assignment_47808', subscript_call_result_49552)
        
        # Assigning a Subscript to a Name (line 676):
        
        # Obtaining the type of the subscript
        int_49553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 676, 16), 'int')
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Obtaining an instance of the builtin type 'tuple' (line 676)
        tuple_49559 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 676, 56), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 676)
        # Adding element type (line 676)
        int_49560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 676, 56), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 676, 56), tuple_49559, int_49560)
        # Adding element type (line 676)
        int_49561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 676, 59), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 676, 56), tuple_49559, int_49561)
        # Adding element type (line 676)
        int_49562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 676, 62), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 676, 56), tuple_49559, int_49562)
        # Adding element type (line 676)
        int_49563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 676, 65), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 676, 56), tuple_49559, int_49563)
        # Adding element type (line 676)
        int_49564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 676, 68), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 676, 56), tuple_49559, int_49564)
        
        comprehension_49565 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 676, 33), tuple_49559)
        # Assigning a type to the variable 'x' (line 676)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 676, 33), 'x', comprehension_49565)
        
        # Call to _arg(...): (line 676)
        # Processing the call arguments (line 676)
        # Getting the type of 'x' (line 676)
        x_49556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 676, 43), 'x', False)
        # Processing the call keyword arguments (line 676)
        kwargs_49557 = {}
        # Getting the type of 'self' (line 676)
        self_49554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 676, 33), 'self', False)
        # Obtaining the member '_arg' of a type (line 676)
        _arg_49555 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 676, 33), self_49554, '_arg')
        # Calling _arg(args, kwargs) (line 676)
        _arg_call_result_49558 = invoke(stypy.reporting.localization.Localization(__file__, 676, 33), _arg_49555, *[x_49556], **kwargs_49557)
        
        list_49566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 676, 33), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 676, 33), list_49566, _arg_call_result_49558)
        # Obtaining the member '__getitem__' of a type (line 676)
        getitem___49567 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 676, 16), list_49566, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 676)
        subscript_call_result_49568 = invoke(stypy.reporting.localization.Localization(__file__, 676, 16), getitem___49567, int_49553)
        
        # Assigning a type to the variable 'tuple_var_assignment_47809' (line 676)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 676, 16), 'tuple_var_assignment_47809', subscript_call_result_49568)
        
        # Assigning a Subscript to a Name (line 676):
        
        # Obtaining the type of the subscript
        int_49569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 676, 16), 'int')
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Obtaining an instance of the builtin type 'tuple' (line 676)
        tuple_49575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 676, 56), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 676)
        # Adding element type (line 676)
        int_49576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 676, 56), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 676, 56), tuple_49575, int_49576)
        # Adding element type (line 676)
        int_49577 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 676, 59), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 676, 56), tuple_49575, int_49577)
        # Adding element type (line 676)
        int_49578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 676, 62), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 676, 56), tuple_49575, int_49578)
        # Adding element type (line 676)
        int_49579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 676, 65), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 676, 56), tuple_49575, int_49579)
        # Adding element type (line 676)
        int_49580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 676, 68), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 676, 56), tuple_49575, int_49580)
        
        comprehension_49581 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 676, 33), tuple_49575)
        # Assigning a type to the variable 'x' (line 676)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 676, 33), 'x', comprehension_49581)
        
        # Call to _arg(...): (line 676)
        # Processing the call arguments (line 676)
        # Getting the type of 'x' (line 676)
        x_49572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 676, 43), 'x', False)
        # Processing the call keyword arguments (line 676)
        kwargs_49573 = {}
        # Getting the type of 'self' (line 676)
        self_49570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 676, 33), 'self', False)
        # Obtaining the member '_arg' of a type (line 676)
        _arg_49571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 676, 33), self_49570, '_arg')
        # Calling _arg(args, kwargs) (line 676)
        _arg_call_result_49574 = invoke(stypy.reporting.localization.Localization(__file__, 676, 33), _arg_49571, *[x_49572], **kwargs_49573)
        
        list_49582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 676, 33), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 676, 33), list_49582, _arg_call_result_49574)
        # Obtaining the member '__getitem__' of a type (line 676)
        getitem___49583 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 676, 16), list_49582, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 676)
        subscript_call_result_49584 = invoke(stypy.reporting.localization.Localization(__file__, 676, 16), getitem___49583, int_49569)
        
        # Assigning a type to the variable 'tuple_var_assignment_47810' (line 676)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 676, 16), 'tuple_var_assignment_47810', subscript_call_result_49584)
        
        # Assigning a Name to a Name (line 676):
        # Getting the type of 'tuple_var_assignment_47806' (line 676)
        tuple_var_assignment_47806_49585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 676, 16), 'tuple_var_assignment_47806')
        # Assigning a type to the variable 'c' (line 676)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 676, 16), 'c', tuple_var_assignment_47806_49585)
        
        # Assigning a Name to a Name (line 676):
        # Getting the type of 'tuple_var_assignment_47807' (line 676)
        tuple_var_assignment_47807_49586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 676, 16), 'tuple_var_assignment_47807')
        # Assigning a type to the variable 's' (line 676)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 676, 19), 's', tuple_var_assignment_47807_49586)
        
        # Assigning a Name to a Name (line 676):
        # Getting the type of 'tuple_var_assignment_47808' (line 676)
        tuple_var_assignment_47808_49587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 676, 16), 'tuple_var_assignment_47808')
        # Assigning a type to the variable 'd' (line 676)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 676, 22), 'd', tuple_var_assignment_47808_49587)
        
        # Assigning a Name to a Name (line 676):
        # Getting the type of 'tuple_var_assignment_47809' (line 676)
        tuple_var_assignment_47809_49588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 676, 16), 'tuple_var_assignment_47809')
        # Assigning a type to the variable 'a' (line 676)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 676, 25), 'a', tuple_var_assignment_47809_49588)
        
        # Assigning a Name to a Name (line 676):
        # Getting the type of 'tuple_var_assignment_47810' (line 676)
        tuple_var_assignment_47810_49589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 676, 16), 'tuple_var_assignment_47810')
        # Assigning a type to the variable 'l' (line 676)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 676, 28), 'l', tuple_var_assignment_47810_49589)
        
        # Call to _fnt_def_real(...): (line 677)
        # Processing the call arguments (line 677)
        # Getting the type of 'k' (line 677)
        k_49592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 677, 35), 'k', False)
        # Getting the type of 'c' (line 677)
        c_49593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 677, 38), 'c', False)
        # Getting the type of 's' (line 677)
        s_49594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 677, 41), 's', False)
        # Getting the type of 'd' (line 677)
        d_49595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 677, 44), 'd', False)
        # Getting the type of 'a' (line 677)
        a_49596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 677, 47), 'a', False)
        # Getting the type of 'l' (line 677)
        l_49597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 677, 50), 'l', False)
        # Processing the call keyword arguments (line 677)
        kwargs_49598 = {}
        # Getting the type of 'self' (line 677)
        self_49590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 677, 16), 'self', False)
        # Obtaining the member '_fnt_def_real' of a type (line 677)
        _fnt_def_real_49591 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 677, 16), self_49590, '_fnt_def_real')
        # Calling _fnt_def_real(args, kwargs) (line 677)
        _fnt_def_real_call_result_49599 = invoke(stypy.reporting.localization.Localization(__file__, 677, 16), _fnt_def_real_49591, *[k_49592, c_49593, s_49594, d_49595, a_49596, l_49597], **kwargs_49598)
        
        
        # Type idiom detected: calculating its left and rigth part (line 678)
        # Getting the type of 'self' (line 678)
        self_49600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 678, 19), 'self')
        # Obtaining the member '_first_font' of a type (line 678)
        _first_font_49601 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 678, 19), self_49600, '_first_font')
        # Getting the type of 'None' (line 678)
        None_49602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 678, 39), 'None')
        
        (may_be_49603, more_types_in_union_49604) = may_be_none(_first_font_49601, None_49602)

        if may_be_49603:

            if more_types_in_union_49604:
                # Runtime conditional SSA (line 678)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Name to a Attribute (line 679):
            
            # Assigning a Name to a Attribute (line 679):
            # Getting the type of 'k' (line 679)
            k_49605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 679, 39), 'k')
            # Getting the type of 'self' (line 679)
            self_49606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 679, 20), 'self')
            # Setting the type of the member '_first_font' of a type (line 679)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 679, 20), self_49606, '_first_font', k_49605)

            if more_types_in_union_49604:
                # SSA join for if statement (line 678)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA branch for the else part of an if statement (line 674)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'byte' (line 680)
        byte_49607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 680, 17), 'byte')
        int_49608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 680, 25), 'int')
        # Applying the binary operator '==' (line 680)
        result_eq_49609 = python_operator(stypy.reporting.localization.Localization(__file__, 680, 17), '==', byte_49607, int_49608)
        
        # Testing the type of an if condition (line 680)
        if_condition_49610 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 680, 17), result_eq_49609)
        # Assigning a type to the variable 'if_condition_49610' (line 680)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 680, 17), 'if_condition_49610', if_condition_49610)
        # SSA begins for if statement (line 680)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Tuple to a Tuple (line 681):
        
        # Assigning a Call to a Name (line 681):
        
        # Call to _arg(...): (line 681)
        # Processing the call arguments (line 681)
        int_49613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 681, 33), 'int')
        # Processing the call keyword arguments (line 681)
        kwargs_49614 = {}
        # Getting the type of 'self' (line 681)
        self_49611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 23), 'self', False)
        # Obtaining the member '_arg' of a type (line 681)
        _arg_49612 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 681, 23), self_49611, '_arg')
        # Calling _arg(args, kwargs) (line 681)
        _arg_call_result_49615 = invoke(stypy.reporting.localization.Localization(__file__, 681, 23), _arg_49612, *[int_49613], **kwargs_49614)
        
        # Assigning a type to the variable 'tuple_assignment_47811' (line 681)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 681, 16), 'tuple_assignment_47811', _arg_call_result_49615)
        
        # Assigning a Call to a Name (line 681):
        
        # Call to _arg(...): (line 681)
        # Processing the call arguments (line 681)
        int_49618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 681, 47), 'int')
        # Processing the call keyword arguments (line 681)
        kwargs_49619 = {}
        # Getting the type of 'self' (line 681)
        self_49616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 37), 'self', False)
        # Obtaining the member '_arg' of a type (line 681)
        _arg_49617 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 681, 37), self_49616, '_arg')
        # Calling _arg(args, kwargs) (line 681)
        _arg_call_result_49620 = invoke(stypy.reporting.localization.Localization(__file__, 681, 37), _arg_49617, *[int_49618], **kwargs_49619)
        
        # Assigning a type to the variable 'tuple_assignment_47812' (line 681)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 681, 16), 'tuple_assignment_47812', _arg_call_result_49620)
        
        # Assigning a Name to a Name (line 681):
        # Getting the type of 'tuple_assignment_47811' (line 681)
        tuple_assignment_47811_49621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 16), 'tuple_assignment_47811')
        # Assigning a type to the variable 'i' (line 681)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 681, 16), 'i', tuple_assignment_47811_49621)
        
        # Assigning a Name to a Name (line 681):
        # Getting the type of 'tuple_assignment_47812' (line 681)
        tuple_assignment_47812_49622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 16), 'tuple_assignment_47812')
        # Assigning a type to the variable 'k' (line 681)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 681, 19), 'k', tuple_assignment_47812_49622)
        
        # Assigning a Call to a Name (line 682):
        
        # Assigning a Call to a Name (line 682):
        
        # Call to read(...): (line 682)
        # Processing the call arguments (line 682)
        # Getting the type of 'k' (line 682)
        k_49626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 682, 35), 'k', False)
        # Processing the call keyword arguments (line 682)
        kwargs_49627 = {}
        # Getting the type of 'self' (line 682)
        self_49623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 682, 20), 'self', False)
        # Obtaining the member 'file' of a type (line 682)
        file_49624 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 682, 20), self_49623, 'file')
        # Obtaining the member 'read' of a type (line 682)
        read_49625 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 682, 20), file_49624, 'read')
        # Calling read(args, kwargs) (line 682)
        read_call_result_49628 = invoke(stypy.reporting.localization.Localization(__file__, 682, 20), read_49625, *[k_49626], **kwargs_49627)
        
        # Assigning a type to the variable 'x' (line 682)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 682, 16), 'x', read_call_result_49628)
        
        # Assigning a Tuple to a Tuple (line 683):
        
        # Assigning a Call to a Name (line 683):
        
        # Call to _arg(...): (line 683)
        # Processing the call arguments (line 683)
        int_49631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 683, 35), 'int')
        # Processing the call keyword arguments (line 683)
        kwargs_49632 = {}
        # Getting the type of 'self' (line 683)
        self_49629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 683, 25), 'self', False)
        # Obtaining the member '_arg' of a type (line 683)
        _arg_49630 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 683, 25), self_49629, '_arg')
        # Calling _arg(args, kwargs) (line 683)
        _arg_call_result_49633 = invoke(stypy.reporting.localization.Localization(__file__, 683, 25), _arg_49630, *[int_49631], **kwargs_49632)
        
        # Assigning a type to the variable 'tuple_assignment_47813' (line 683)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 683, 16), 'tuple_assignment_47813', _arg_call_result_49633)
        
        # Assigning a Call to a Name (line 683):
        
        # Call to _arg(...): (line 683)
        # Processing the call arguments (line 683)
        int_49636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 683, 49), 'int')
        # Processing the call keyword arguments (line 683)
        kwargs_49637 = {}
        # Getting the type of 'self' (line 683)
        self_49634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 683, 39), 'self', False)
        # Obtaining the member '_arg' of a type (line 683)
        _arg_49635 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 683, 39), self_49634, '_arg')
        # Calling _arg(args, kwargs) (line 683)
        _arg_call_result_49638 = invoke(stypy.reporting.localization.Localization(__file__, 683, 39), _arg_49635, *[int_49636], **kwargs_49637)
        
        # Assigning a type to the variable 'tuple_assignment_47814' (line 683)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 683, 16), 'tuple_assignment_47814', _arg_call_result_49638)
        
        # Assigning a Name to a Name (line 683):
        # Getting the type of 'tuple_assignment_47813' (line 683)
        tuple_assignment_47813_49639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 683, 16), 'tuple_assignment_47813')
        # Assigning a type to the variable 'cs' (line 683)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 683, 16), 'cs', tuple_assignment_47813_49639)
        
        # Assigning a Name to a Name (line 683):
        # Getting the type of 'tuple_assignment_47814' (line 683)
        tuple_assignment_47814_49640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 683, 16), 'tuple_assignment_47814')
        # Assigning a type to the variable 'ds' (line 683)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 683, 20), 'ds', tuple_assignment_47814_49640)
        
        # Call to _pre(...): (line 684)
        # Processing the call arguments (line 684)
        # Getting the type of 'i' (line 684)
        i_49643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 684, 26), 'i', False)
        # Getting the type of 'x' (line 684)
        x_49644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 684, 29), 'x', False)
        # Getting the type of 'cs' (line 684)
        cs_49645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 684, 32), 'cs', False)
        # Getting the type of 'ds' (line 684)
        ds_49646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 684, 36), 'ds', False)
        # Processing the call keyword arguments (line 684)
        kwargs_49647 = {}
        # Getting the type of 'self' (line 684)
        self_49641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 684, 16), 'self', False)
        # Obtaining the member '_pre' of a type (line 684)
        _pre_49642 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 684, 16), self_49641, '_pre')
        # Calling _pre(args, kwargs) (line 684)
        _pre_call_result_49648 = invoke(stypy.reporting.localization.Localization(__file__, 684, 16), _pre_49642, *[i_49643, x_49644, cs_49645, ds_49646], **kwargs_49647)
        
        # SSA branch for the else part of an if statement (line 680)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'byte' (line 685)
        byte_49649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 685, 17), 'byte')
        int_49650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 685, 25), 'int')
        # Applying the binary operator '==' (line 685)
        result_eq_49651 = python_operator(stypy.reporting.localization.Localization(__file__, 685, 17), '==', byte_49649, int_49650)
        
        # Testing the type of an if condition (line 685)
        if_condition_49652 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 685, 17), result_eq_49651)
        # Assigning a type to the variable 'if_condition_49652' (line 685)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 685, 17), 'if_condition_49652', if_condition_49652)
        # SSA begins for if statement (line 685)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # SSA branch for the else part of an if statement (line 685)
        module_type_store.open_ssa_branch('else')
        
        # Call to ValueError(...): (line 688)
        # Processing the call arguments (line 688)
        unicode_49654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 688, 33), 'unicode', u'unknown vf opcode %d')
        # Getting the type of 'byte' (line 688)
        byte_49655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 688, 58), 'byte', False)
        # Applying the binary operator '%' (line 688)
        result_mod_49656 = python_operator(stypy.reporting.localization.Localization(__file__, 688, 33), '%', unicode_49654, byte_49655)
        
        # Processing the call keyword arguments (line 688)
        kwargs_49657 = {}
        # Getting the type of 'ValueError' (line 688)
        ValueError_49653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 688, 22), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 688)
        ValueError_call_result_49658 = invoke(stypy.reporting.localization.Localization(__file__, 688, 22), ValueError_49653, *[result_mod_49656], **kwargs_49657)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 688, 16), ValueError_call_result_49658, 'raise parameter', BaseException)
        # SSA join for if statement (line 685)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 680)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 674)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 670)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 665)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for while statement (line 646)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_read(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_read' in the type store
        # Getting the type of 'stypy_return_type' (line 640)
        stypy_return_type_49659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 640, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_49659)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_read'
        return stypy_return_type_49659


    @norecursion
    def _init_packet(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_init_packet'
        module_type_store = module_type_store.open_function_context('_init_packet', 690, 4, False)
        # Assigning a type to the variable 'self' (line 691)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 691, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Vf._init_packet.__dict__.__setitem__('stypy_localization', localization)
        Vf._init_packet.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Vf._init_packet.__dict__.__setitem__('stypy_type_store', module_type_store)
        Vf._init_packet.__dict__.__setitem__('stypy_function_name', 'Vf._init_packet')
        Vf._init_packet.__dict__.__setitem__('stypy_param_names_list', ['pl'])
        Vf._init_packet.__dict__.__setitem__('stypy_varargs_param_name', None)
        Vf._init_packet.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Vf._init_packet.__dict__.__setitem__('stypy_call_defaults', defaults)
        Vf._init_packet.__dict__.__setitem__('stypy_call_varargs', varargs)
        Vf._init_packet.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Vf._init_packet.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Vf._init_packet', ['pl'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_init_packet', localization, ['pl'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_init_packet(...)' code ##################

        
        
        # Getting the type of 'self' (line 691)
        self_49660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 691, 11), 'self')
        # Obtaining the member 'state' of a type (line 691)
        state_49661 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 691, 11), self_49660, 'state')
        # Getting the type of '_dvistate' (line 691)
        _dvistate_49662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 691, 25), '_dvistate')
        # Obtaining the member 'outer' of a type (line 691)
        outer_49663 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 691, 25), _dvistate_49662, 'outer')
        # Applying the binary operator '!=' (line 691)
        result_ne_49664 = python_operator(stypy.reporting.localization.Localization(__file__, 691, 11), '!=', state_49661, outer_49663)
        
        # Testing the type of an if condition (line 691)
        if_condition_49665 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 691, 8), result_ne_49664)
        # Assigning a type to the variable 'if_condition_49665' (line 691)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 691, 8), 'if_condition_49665', if_condition_49665)
        # SSA begins for if statement (line 691)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 692)
        # Processing the call arguments (line 692)
        unicode_49667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 692, 29), 'unicode', u'Misplaced packet in vf file')
        # Processing the call keyword arguments (line 692)
        kwargs_49668 = {}
        # Getting the type of 'ValueError' (line 692)
        ValueError_49666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 692)
        ValueError_call_result_49669 = invoke(stypy.reporting.localization.Localization(__file__, 692, 18), ValueError_49666, *[unicode_49667], **kwargs_49668)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 692, 12), ValueError_call_result_49669, 'raise parameter', BaseException)
        # SSA join for if statement (line 691)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Tuple to a Tuple (line 693):
        
        # Assigning a Num to a Name (line 693):
        int_49670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 693, 57), 'int')
        # Assigning a type to the variable 'tuple_assignment_47815' (line 693)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 693, 8), 'tuple_assignment_47815', int_49670)
        
        # Assigning a Num to a Name (line 693):
        int_49671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 693, 60), 'int')
        # Assigning a type to the variable 'tuple_assignment_47816' (line 693)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 693, 8), 'tuple_assignment_47816', int_49671)
        
        # Assigning a Num to a Name (line 693):
        int_49672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 693, 63), 'int')
        # Assigning a type to the variable 'tuple_assignment_47817' (line 693)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 693, 8), 'tuple_assignment_47817', int_49672)
        
        # Assigning a Num to a Name (line 693):
        int_49673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 693, 66), 'int')
        # Assigning a type to the variable 'tuple_assignment_47818' (line 693)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 693, 8), 'tuple_assignment_47818', int_49673)
        
        # Assigning a Num to a Name (line 693):
        int_49674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 693, 69), 'int')
        # Assigning a type to the variable 'tuple_assignment_47819' (line 693)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 693, 8), 'tuple_assignment_47819', int_49674)
        
        # Assigning a Num to a Name (line 693):
        int_49675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 693, 72), 'int')
        # Assigning a type to the variable 'tuple_assignment_47820' (line 693)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 693, 8), 'tuple_assignment_47820', int_49675)
        
        # Assigning a Name to a Attribute (line 693):
        # Getting the type of 'tuple_assignment_47815' (line 693)
        tuple_assignment_47815_49676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 693, 8), 'tuple_assignment_47815')
        # Getting the type of 'self' (line 693)
        self_49677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 693, 8), 'self')
        # Setting the type of the member 'h' of a type (line 693)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 693, 8), self_49677, 'h', tuple_assignment_47815_49676)
        
        # Assigning a Name to a Attribute (line 693):
        # Getting the type of 'tuple_assignment_47816' (line 693)
        tuple_assignment_47816_49678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 693, 8), 'tuple_assignment_47816')
        # Getting the type of 'self' (line 693)
        self_49679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 693, 16), 'self')
        # Setting the type of the member 'v' of a type (line 693)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 693, 16), self_49679, 'v', tuple_assignment_47816_49678)
        
        # Assigning a Name to a Attribute (line 693):
        # Getting the type of 'tuple_assignment_47817' (line 693)
        tuple_assignment_47817_49680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 693, 8), 'tuple_assignment_47817')
        # Getting the type of 'self' (line 693)
        self_49681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 693, 24), 'self')
        # Setting the type of the member 'w' of a type (line 693)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 693, 24), self_49681, 'w', tuple_assignment_47817_49680)
        
        # Assigning a Name to a Attribute (line 693):
        # Getting the type of 'tuple_assignment_47818' (line 693)
        tuple_assignment_47818_49682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 693, 8), 'tuple_assignment_47818')
        # Getting the type of 'self' (line 693)
        self_49683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 693, 32), 'self')
        # Setting the type of the member 'x' of a type (line 693)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 693, 32), self_49683, 'x', tuple_assignment_47818_49682)
        
        # Assigning a Name to a Attribute (line 693):
        # Getting the type of 'tuple_assignment_47819' (line 693)
        tuple_assignment_47819_49684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 693, 8), 'tuple_assignment_47819')
        # Getting the type of 'self' (line 693)
        self_49685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 693, 40), 'self')
        # Setting the type of the member 'y' of a type (line 693)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 693, 40), self_49685, 'y', tuple_assignment_47819_49684)
        
        # Assigning a Name to a Attribute (line 693):
        # Getting the type of 'tuple_assignment_47820' (line 693)
        tuple_assignment_47820_49686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 693, 8), 'tuple_assignment_47820')
        # Getting the type of 'self' (line 693)
        self_49687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 693, 48), 'self')
        # Setting the type of the member 'z' of a type (line 693)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 693, 48), self_49687, 'z', tuple_assignment_47820_49686)
        
        # Assigning a Tuple to a Tuple (line 694):
        
        # Assigning a List to a Name (line 694):
        
        # Obtaining an instance of the builtin type 'list' (line 694)
        list_49688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 694, 44), 'list')
        # Adding type elements to the builtin type 'list' instance (line 694)
        
        # Assigning a type to the variable 'tuple_assignment_47821' (line 694)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 694, 8), 'tuple_assignment_47821', list_49688)
        
        # Assigning a List to a Name (line 694):
        
        # Obtaining an instance of the builtin type 'list' (line 694)
        list_49689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 694, 48), 'list')
        # Adding type elements to the builtin type 'list' instance (line 694)
        
        # Assigning a type to the variable 'tuple_assignment_47822' (line 694)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 694, 8), 'tuple_assignment_47822', list_49689)
        
        # Assigning a List to a Name (line 694):
        
        # Obtaining an instance of the builtin type 'list' (line 694)
        list_49690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 694, 52), 'list')
        # Adding type elements to the builtin type 'list' instance (line 694)
        
        # Assigning a type to the variable 'tuple_assignment_47823' (line 694)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 694, 8), 'tuple_assignment_47823', list_49690)
        
        # Assigning a Name to a Attribute (line 694):
        # Getting the type of 'tuple_assignment_47821' (line 694)
        tuple_assignment_47821_49691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 8), 'tuple_assignment_47821')
        # Getting the type of 'self' (line 694)
        self_49692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 8), 'self')
        # Setting the type of the member 'stack' of a type (line 694)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 694, 8), self_49692, 'stack', tuple_assignment_47821_49691)
        
        # Assigning a Name to a Attribute (line 694):
        # Getting the type of 'tuple_assignment_47822' (line 694)
        tuple_assignment_47822_49693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 8), 'tuple_assignment_47822')
        # Getting the type of 'self' (line 694)
        self_49694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 20), 'self')
        # Setting the type of the member 'text' of a type (line 694)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 694, 20), self_49694, 'text', tuple_assignment_47822_49693)
        
        # Assigning a Name to a Attribute (line 694):
        # Getting the type of 'tuple_assignment_47823' (line 694)
        tuple_assignment_47823_49695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 8), 'tuple_assignment_47823')
        # Getting the type of 'self' (line 694)
        self_49696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 31), 'self')
        # Setting the type of the member 'boxes' of a type (line 694)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 694, 31), self_49696, 'boxes', tuple_assignment_47823_49695)
        
        # Assigning a Attribute to a Attribute (line 695):
        
        # Assigning a Attribute to a Attribute (line 695):
        # Getting the type of 'self' (line 695)
        self_49697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 17), 'self')
        # Obtaining the member '_first_font' of a type (line 695)
        _first_font_49698 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 695, 17), self_49697, '_first_font')
        # Getting the type of 'self' (line 695)
        self_49699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 8), 'self')
        # Setting the type of the member 'f' of a type (line 695)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 695, 8), self_49699, 'f', _first_font_49698)
        
        # Call to tell(...): (line 696)
        # Processing the call keyword arguments (line 696)
        kwargs_49703 = {}
        # Getting the type of 'self' (line 696)
        self_49700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 696, 15), 'self', False)
        # Obtaining the member 'file' of a type (line 696)
        file_49701 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 696, 15), self_49700, 'file')
        # Obtaining the member 'tell' of a type (line 696)
        tell_49702 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 696, 15), file_49701, 'tell')
        # Calling tell(args, kwargs) (line 696)
        tell_call_result_49704 = invoke(stypy.reporting.localization.Localization(__file__, 696, 15), tell_49702, *[], **kwargs_49703)
        
        # Getting the type of 'pl' (line 696)
        pl_49705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 696, 34), 'pl')
        # Applying the binary operator '+' (line 696)
        result_add_49706 = python_operator(stypy.reporting.localization.Localization(__file__, 696, 15), '+', tell_call_result_49704, pl_49705)
        
        # Assigning a type to the variable 'stypy_return_type' (line 696)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 696, 8), 'stypy_return_type', result_add_49706)
        
        # ################# End of '_init_packet(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_init_packet' in the type store
        # Getting the type of 'stypy_return_type' (line 690)
        stypy_return_type_49707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_49707)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_init_packet'
        return stypy_return_type_49707


    @norecursion
    def _finalize_packet(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_finalize_packet'
        module_type_store = module_type_store.open_function_context('_finalize_packet', 698, 4, False)
        # Assigning a type to the variable 'self' (line 699)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 699, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Vf._finalize_packet.__dict__.__setitem__('stypy_localization', localization)
        Vf._finalize_packet.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Vf._finalize_packet.__dict__.__setitem__('stypy_type_store', module_type_store)
        Vf._finalize_packet.__dict__.__setitem__('stypy_function_name', 'Vf._finalize_packet')
        Vf._finalize_packet.__dict__.__setitem__('stypy_param_names_list', ['packet_char', 'packet_width'])
        Vf._finalize_packet.__dict__.__setitem__('stypy_varargs_param_name', None)
        Vf._finalize_packet.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Vf._finalize_packet.__dict__.__setitem__('stypy_call_defaults', defaults)
        Vf._finalize_packet.__dict__.__setitem__('stypy_call_varargs', varargs)
        Vf._finalize_packet.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Vf._finalize_packet.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Vf._finalize_packet', ['packet_char', 'packet_width'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_finalize_packet', localization, ['packet_char', 'packet_width'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_finalize_packet(...)' code ##################

        
        # Assigning a Call to a Subscript (line 699):
        
        # Assigning a Call to a Subscript (line 699):
        
        # Call to Page(...): (line 699)
        # Processing the call keyword arguments (line 699)
        # Getting the type of 'self' (line 700)
        self_49709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 700, 17), 'self', False)
        # Obtaining the member 'text' of a type (line 700)
        text_49710 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 700, 17), self_49709, 'text')
        keyword_49711 = text_49710
        # Getting the type of 'self' (line 700)
        self_49712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 700, 34), 'self', False)
        # Obtaining the member 'boxes' of a type (line 700)
        boxes_49713 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 700, 34), self_49712, 'boxes')
        keyword_49714 = boxes_49713
        # Getting the type of 'packet_width' (line 700)
        packet_width_49715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 700, 52), 'packet_width', False)
        keyword_49716 = packet_width_49715
        # Getting the type of 'None' (line 701)
        None_49717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 701, 19), 'None', False)
        keyword_49718 = None_49717
        # Getting the type of 'None' (line 701)
        None_49719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 701, 33), 'None', False)
        keyword_49720 = None_49719
        kwargs_49721 = {'text': keyword_49711, 'boxes': keyword_49714, 'height': keyword_49718, 'descent': keyword_49720, 'width': keyword_49716}
        # Getting the type of 'Page' (line 699)
        Page_49708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 699, 35), 'Page', False)
        # Calling Page(args, kwargs) (line 699)
        Page_call_result_49722 = invoke(stypy.reporting.localization.Localization(__file__, 699, 35), Page_49708, *[], **kwargs_49721)
        
        # Getting the type of 'self' (line 699)
        self_49723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 699, 8), 'self')
        # Obtaining the member '_chars' of a type (line 699)
        _chars_49724 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 699, 8), self_49723, '_chars')
        # Getting the type of 'packet_char' (line 699)
        packet_char_49725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 699, 20), 'packet_char')
        # Storing an element on a container (line 699)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 699, 8), _chars_49724, (packet_char_49725, Page_call_result_49722))
        
        # Assigning a Attribute to a Attribute (line 702):
        
        # Assigning a Attribute to a Attribute (line 702):
        # Getting the type of '_dvistate' (line 702)
        _dvistate_49726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 21), '_dvistate')
        # Obtaining the member 'outer' of a type (line 702)
        outer_49727 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 702, 21), _dvistate_49726, 'outer')
        # Getting the type of 'self' (line 702)
        self_49728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 8), 'self')
        # Setting the type of the member 'state' of a type (line 702)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 702, 8), self_49728, 'state', outer_49727)
        
        # ################# End of '_finalize_packet(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_finalize_packet' in the type store
        # Getting the type of 'stypy_return_type' (line 698)
        stypy_return_type_49729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_49729)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_finalize_packet'
        return stypy_return_type_49729


    @norecursion
    def _pre(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_pre'
        module_type_store = module_type_store.open_function_context('_pre', 704, 4, False)
        # Assigning a type to the variable 'self' (line 705)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 705, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Vf._pre.__dict__.__setitem__('stypy_localization', localization)
        Vf._pre.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Vf._pre.__dict__.__setitem__('stypy_type_store', module_type_store)
        Vf._pre.__dict__.__setitem__('stypy_function_name', 'Vf._pre')
        Vf._pre.__dict__.__setitem__('stypy_param_names_list', ['i', 'x', 'cs', 'ds'])
        Vf._pre.__dict__.__setitem__('stypy_varargs_param_name', None)
        Vf._pre.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Vf._pre.__dict__.__setitem__('stypy_call_defaults', defaults)
        Vf._pre.__dict__.__setitem__('stypy_call_varargs', varargs)
        Vf._pre.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Vf._pre.__dict__.__setitem__('stypy_declared_arg_number', 5)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Vf._pre', ['i', 'x', 'cs', 'ds'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_pre', localization, ['i', 'x', 'cs', 'ds'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_pre(...)' code ##################

        
        
        # Getting the type of 'self' (line 705)
        self_49730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 11), 'self')
        # Obtaining the member 'state' of a type (line 705)
        state_49731 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 705, 11), self_49730, 'state')
        # Getting the type of '_dvistate' (line 705)
        _dvistate_49732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 25), '_dvistate')
        # Obtaining the member 'pre' of a type (line 705)
        pre_49733 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 705, 25), _dvistate_49732, 'pre')
        # Applying the binary operator '!=' (line 705)
        result_ne_49734 = python_operator(stypy.reporting.localization.Localization(__file__, 705, 11), '!=', state_49731, pre_49733)
        
        # Testing the type of an if condition (line 705)
        if_condition_49735 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 705, 8), result_ne_49734)
        # Assigning a type to the variable 'if_condition_49735' (line 705)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 705, 8), 'if_condition_49735', if_condition_49735)
        # SSA begins for if statement (line 705)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 706)
        # Processing the call arguments (line 706)
        unicode_49737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 706, 29), 'unicode', u'pre command in middle of vf file')
        # Processing the call keyword arguments (line 706)
        kwargs_49738 = {}
        # Getting the type of 'ValueError' (line 706)
        ValueError_49736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 706, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 706)
        ValueError_call_result_49739 = invoke(stypy.reporting.localization.Localization(__file__, 706, 18), ValueError_49736, *[unicode_49737], **kwargs_49738)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 706, 12), ValueError_call_result_49739, 'raise parameter', BaseException)
        # SSA join for if statement (line 705)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'i' (line 707)
        i_49740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 707, 11), 'i')
        int_49741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 707, 16), 'int')
        # Applying the binary operator '!=' (line 707)
        result_ne_49742 = python_operator(stypy.reporting.localization.Localization(__file__, 707, 11), '!=', i_49740, int_49741)
        
        # Testing the type of an if condition (line 707)
        if_condition_49743 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 707, 8), result_ne_49742)
        # Assigning a type to the variable 'if_condition_49743' (line 707)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 707, 8), 'if_condition_49743', if_condition_49743)
        # SSA begins for if statement (line 707)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 708)
        # Processing the call arguments (line 708)
        unicode_49745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 708, 29), 'unicode', u'Unknown vf format %d')
        # Getting the type of 'i' (line 708)
        i_49746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 708, 54), 'i', False)
        # Applying the binary operator '%' (line 708)
        result_mod_49747 = python_operator(stypy.reporting.localization.Localization(__file__, 708, 29), '%', unicode_49745, i_49746)
        
        # Processing the call keyword arguments (line 708)
        kwargs_49748 = {}
        # Getting the type of 'ValueError' (line 708)
        ValueError_49744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 708, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 708)
        ValueError_call_result_49749 = invoke(stypy.reporting.localization.Localization(__file__, 708, 18), ValueError_49744, *[result_mod_49747], **kwargs_49748)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 708, 12), ValueError_call_result_49749, 'raise parameter', BaseException)
        # SSA join for if statement (line 707)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to len(...): (line 709)
        # Processing the call arguments (line 709)
        # Getting the type of 'x' (line 709)
        x_49751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 709, 15), 'x', False)
        # Processing the call keyword arguments (line 709)
        kwargs_49752 = {}
        # Getting the type of 'len' (line 709)
        len_49750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 709, 11), 'len', False)
        # Calling len(args, kwargs) (line 709)
        len_call_result_49753 = invoke(stypy.reporting.localization.Localization(__file__, 709, 11), len_49750, *[x_49751], **kwargs_49752)
        
        # Testing the type of an if condition (line 709)
        if_condition_49754 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 709, 8), len_call_result_49753)
        # Assigning a type to the variable 'if_condition_49754' (line 709)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 709, 8), 'if_condition_49754', if_condition_49754)
        # SSA begins for if statement (line 709)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to report(...): (line 710)
        # Processing the call arguments (line 710)
        unicode_49758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 710, 38), 'unicode', u'vf file comment: ')
        # Getting the type of 'x' (line 710)
        x_49759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 710, 60), 'x', False)
        # Applying the binary operator '+' (line 710)
        result_add_49760 = python_operator(stypy.reporting.localization.Localization(__file__, 710, 38), '+', unicode_49758, x_49759)
        
        unicode_49761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 710, 63), 'unicode', u'debug')
        # Processing the call keyword arguments (line 710)
        kwargs_49762 = {}
        # Getting the type of 'matplotlib' (line 710)
        matplotlib_49755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 710, 12), 'matplotlib', False)
        # Obtaining the member 'verbose' of a type (line 710)
        verbose_49756 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 710, 12), matplotlib_49755, 'verbose')
        # Obtaining the member 'report' of a type (line 710)
        report_49757 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 710, 12), verbose_49756, 'report')
        # Calling report(args, kwargs) (line 710)
        report_call_result_49763 = invoke(stypy.reporting.localization.Localization(__file__, 710, 12), report_49757, *[result_add_49760, unicode_49761], **kwargs_49762)
        
        # SSA join for if statement (line 709)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Attribute to a Attribute (line 711):
        
        # Assigning a Attribute to a Attribute (line 711):
        # Getting the type of '_dvistate' (line 711)
        _dvistate_49764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 711, 21), '_dvistate')
        # Obtaining the member 'outer' of a type (line 711)
        outer_49765 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 711, 21), _dvistate_49764, 'outer')
        # Getting the type of 'self' (line 711)
        self_49766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 711, 8), 'self')
        # Setting the type of the member 'state' of a type (line 711)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 711, 8), self_49766, 'state', outer_49765)
        
        # ################# End of '_pre(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_pre' in the type store
        # Getting the type of 'stypy_return_type' (line 704)
        stypy_return_type_49767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 704, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_49767)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_pre'
        return stypy_return_type_49767


# Assigning a type to the variable 'Vf' (line 604)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 604, 0), 'Vf', Vf)

@norecursion
def _fix2comp(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_fix2comp'
    module_type_store = module_type_store.open_function_context('_fix2comp', 715, 0, False)
    
    # Passed parameters checking function
    _fix2comp.stypy_localization = localization
    _fix2comp.stypy_type_of_self = None
    _fix2comp.stypy_type_store = module_type_store
    _fix2comp.stypy_function_name = '_fix2comp'
    _fix2comp.stypy_param_names_list = ['num']
    _fix2comp.stypy_varargs_param_name = None
    _fix2comp.stypy_kwargs_param_name = None
    _fix2comp.stypy_call_defaults = defaults
    _fix2comp.stypy_call_varargs = varargs
    _fix2comp.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_fix2comp', ['num'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_fix2comp', localization, ['num'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_fix2comp(...)' code ##################

    unicode_49768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 718, (-1)), 'unicode', u"\n    Convert from two's complement to negative.\n    ")
    # Evaluating assert statement condition
    
    int_49769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 719, 11), 'int')
    # Getting the type of 'num' (line 719)
    num_49770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 719, 16), 'num')
    # Applying the binary operator '<=' (line 719)
    result_le_49771 = python_operator(stypy.reporting.localization.Localization(__file__, 719, 11), '<=', int_49769, num_49770)
    int_49772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 719, 22), 'int')
    int_49773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 719, 25), 'int')
    # Applying the binary operator '**' (line 719)
    result_pow_49774 = python_operator(stypy.reporting.localization.Localization(__file__, 719, 22), '**', int_49772, int_49773)
    
    # Applying the binary operator '<' (line 719)
    result_lt_49775 = python_operator(stypy.reporting.localization.Localization(__file__, 719, 11), '<', num_49770, result_pow_49774)
    # Applying the binary operator '&' (line 719)
    result_and__49776 = python_operator(stypy.reporting.localization.Localization(__file__, 719, 11), '&', result_le_49771, result_lt_49775)
    
    
    # Getting the type of 'num' (line 720)
    num_49777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 720, 7), 'num')
    int_49778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 720, 13), 'int')
    int_49779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 720, 16), 'int')
    # Applying the binary operator '**' (line 720)
    result_pow_49780 = python_operator(stypy.reporting.localization.Localization(__file__, 720, 13), '**', int_49778, int_49779)
    
    # Applying the binary operator '&' (line 720)
    result_and__49781 = python_operator(stypy.reporting.localization.Localization(__file__, 720, 7), '&', num_49777, result_pow_49780)
    
    # Testing the type of an if condition (line 720)
    if_condition_49782 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 720, 4), result_and__49781)
    # Assigning a type to the variable 'if_condition_49782' (line 720)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 720, 4), 'if_condition_49782', if_condition_49782)
    # SSA begins for if statement (line 720)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'num' (line 721)
    num_49783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 721, 15), 'num')
    int_49784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 721, 21), 'int')
    int_49785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 721, 24), 'int')
    # Applying the binary operator '**' (line 721)
    result_pow_49786 = python_operator(stypy.reporting.localization.Localization(__file__, 721, 21), '**', int_49784, int_49785)
    
    # Applying the binary operator '-' (line 721)
    result_sub_49787 = python_operator(stypy.reporting.localization.Localization(__file__, 721, 15), '-', num_49783, result_pow_49786)
    
    # Assigning a type to the variable 'stypy_return_type' (line 721)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 721, 8), 'stypy_return_type', result_sub_49787)
    # SSA branch for the else part of an if statement (line 720)
    module_type_store.open_ssa_branch('else')
    # Getting the type of 'num' (line 723)
    num_49788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 723, 15), 'num')
    # Assigning a type to the variable 'stypy_return_type' (line 723)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 723, 8), 'stypy_return_type', num_49788)
    # SSA join for if statement (line 720)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of '_fix2comp(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_fix2comp' in the type store
    # Getting the type of 'stypy_return_type' (line 715)
    stypy_return_type_49789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 715, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_49789)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_fix2comp'
    return stypy_return_type_49789

# Assigning a type to the variable '_fix2comp' (line 715)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 715, 0), '_fix2comp', _fix2comp)

@norecursion
def _mul2012(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_mul2012'
    module_type_store = module_type_store.open_function_context('_mul2012', 726, 0, False)
    
    # Passed parameters checking function
    _mul2012.stypy_localization = localization
    _mul2012.stypy_type_of_self = None
    _mul2012.stypy_type_store = module_type_store
    _mul2012.stypy_function_name = '_mul2012'
    _mul2012.stypy_param_names_list = ['num1', 'num2']
    _mul2012.stypy_varargs_param_name = None
    _mul2012.stypy_kwargs_param_name = None
    _mul2012.stypy_call_defaults = defaults
    _mul2012.stypy_call_varargs = varargs
    _mul2012.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_mul2012', ['num1', 'num2'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_mul2012', localization, ['num1', 'num2'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_mul2012(...)' code ##################

    unicode_49790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 729, (-1)), 'unicode', u'\n    Multiply two numbers in 20.12 fixed point format.\n    ')
    # Getting the type of 'num1' (line 731)
    num1_49791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 731, 12), 'num1')
    # Getting the type of 'num2' (line 731)
    num2_49792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 731, 17), 'num2')
    # Applying the binary operator '*' (line 731)
    result_mul_49793 = python_operator(stypy.reporting.localization.Localization(__file__, 731, 12), '*', num1_49791, num2_49792)
    
    int_49794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 731, 26), 'int')
    # Applying the binary operator '>>' (line 731)
    result_rshift_49795 = python_operator(stypy.reporting.localization.Localization(__file__, 731, 11), '>>', result_mul_49793, int_49794)
    
    # Assigning a type to the variable 'stypy_return_type' (line 731)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 731, 4), 'stypy_return_type', result_rshift_49795)
    
    # ################# End of '_mul2012(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_mul2012' in the type store
    # Getting the type of 'stypy_return_type' (line 726)
    stypy_return_type_49796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 726, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_49796)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_mul2012'
    return stypy_return_type_49796

# Assigning a type to the variable '_mul2012' (line 726)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 726, 0), '_mul2012', _mul2012)
# Declaration of the 'Tfm' class

class Tfm(object, ):
    unicode_49797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 754, (-1)), 'unicode', u'\n    A TeX Font Metric file.\n\n    This implementation covers only the bare minimum needed by the Dvi class.\n\n    Parameters\n    ----------\n    filename : string or bytestring\n\n    Attributes\n    ----------\n    checksum : int\n       Used for verifying against the dvi file.\n    design_size : int\n       Design size of the font (unknown units)\n    width, height, depth : dict\n       Dimensions of each character, need to be scaled by the factor\n       specified in the dvi file. These are dicts because indexing may\n       not start from 0.\n    ')
    
    # Assigning a Tuple to a Name (line 755):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 757, 4, False)
        # Assigning a type to the variable 'self' (line 758)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 758, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Tfm.__init__', ['filename'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['filename'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Call to report(...): (line 758)
        # Processing the call arguments (line 758)
        unicode_49801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 758, 34), 'unicode', u'opening tfm file ')
        # Getting the type of 'filename' (line 758)
        filename_49802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 758, 56), 'filename', False)
        # Applying the binary operator '+' (line 758)
        result_add_49803 = python_operator(stypy.reporting.localization.Localization(__file__, 758, 34), '+', unicode_49801, filename_49802)
        
        unicode_49804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 758, 66), 'unicode', u'debug')
        # Processing the call keyword arguments (line 758)
        kwargs_49805 = {}
        # Getting the type of 'matplotlib' (line 758)
        matplotlib_49798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 758, 8), 'matplotlib', False)
        # Obtaining the member 'verbose' of a type (line 758)
        verbose_49799 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 758, 8), matplotlib_49798, 'verbose')
        # Obtaining the member 'report' of a type (line 758)
        report_49800 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 758, 8), verbose_49799, 'report')
        # Calling report(args, kwargs) (line 758)
        report_call_result_49806 = invoke(stypy.reporting.localization.Localization(__file__, 758, 8), report_49800, *[result_add_49803, unicode_49804], **kwargs_49805)
        
        
        # Call to open(...): (line 759)
        # Processing the call arguments (line 759)
        # Getting the type of 'filename' (line 759)
        filename_49808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 759, 18), 'filename', False)
        unicode_49809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 759, 28), 'unicode', u'rb')
        # Processing the call keyword arguments (line 759)
        kwargs_49810 = {}
        # Getting the type of 'open' (line 759)
        open_49807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 759, 13), 'open', False)
        # Calling open(args, kwargs) (line 759)
        open_call_result_49811 = invoke(stypy.reporting.localization.Localization(__file__, 759, 13), open_49807, *[filename_49808, unicode_49809], **kwargs_49810)
        
        with_49812 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 759, 13), open_call_result_49811, 'with parameter', '__enter__', '__exit__')

        if with_49812:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 759)
            enter___49813 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 759, 13), open_call_result_49811, '__enter__')
            with_enter_49814 = invoke(stypy.reporting.localization.Localization(__file__, 759, 13), enter___49813)
            # Assigning a type to the variable 'file' (line 759)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 759, 13), 'file', with_enter_49814)
            
            # Assigning a Call to a Name (line 760):
            
            # Assigning a Call to a Name (line 760):
            
            # Call to read(...): (line 760)
            # Processing the call arguments (line 760)
            int_49817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 760, 32), 'int')
            # Processing the call keyword arguments (line 760)
            kwargs_49818 = {}
            # Getting the type of 'file' (line 760)
            file_49815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 760, 22), 'file', False)
            # Obtaining the member 'read' of a type (line 760)
            read_49816 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 760, 22), file_49815, 'read')
            # Calling read(args, kwargs) (line 760)
            read_call_result_49819 = invoke(stypy.reporting.localization.Localization(__file__, 760, 22), read_49816, *[int_49817], **kwargs_49818)
            
            # Assigning a type to the variable 'header1' (line 760)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 760, 12), 'header1', read_call_result_49819)
            
            # Assigning a Call to a Tuple (line 761):
            
            # Assigning a Call to a Name:
            
            # Call to unpack(...): (line 762)
            # Processing the call arguments (line 762)
            
            # Call to str(...): (line 762)
            # Processing the call arguments (line 762)
            unicode_49823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 762, 34), 'unicode', u'!6H')
            # Processing the call keyword arguments (line 762)
            kwargs_49824 = {}
            # Getting the type of 'str' (line 762)
            str_49822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 762, 30), 'str', False)
            # Calling str(args, kwargs) (line 762)
            str_call_result_49825 = invoke(stypy.reporting.localization.Localization(__file__, 762, 30), str_49822, *[unicode_49823], **kwargs_49824)
            
            
            # Obtaining the type of the subscript
            int_49826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 762, 50), 'int')
            int_49827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 762, 52), 'int')
            slice_49828 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 762, 42), int_49826, int_49827, None)
            # Getting the type of 'header1' (line 762)
            header1_49829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 762, 42), 'header1', False)
            # Obtaining the member '__getitem__' of a type (line 762)
            getitem___49830 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 762, 42), header1_49829, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 762)
            subscript_call_result_49831 = invoke(stypy.reporting.localization.Localization(__file__, 762, 42), getitem___49830, slice_49828)
            
            # Processing the call keyword arguments (line 762)
            kwargs_49832 = {}
            # Getting the type of 'struct' (line 762)
            struct_49820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 762, 16), 'struct', False)
            # Obtaining the member 'unpack' of a type (line 762)
            unpack_49821 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 762, 16), struct_49820, 'unpack')
            # Calling unpack(args, kwargs) (line 762)
            unpack_call_result_49833 = invoke(stypy.reporting.localization.Localization(__file__, 762, 16), unpack_49821, *[str_call_result_49825, subscript_call_result_49831], **kwargs_49832)
            
            # Assigning a type to the variable 'call_assignment_47824' (line 761)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 761, 12), 'call_assignment_47824', unpack_call_result_49833)
            
            # Assigning a Call to a Name (line 761):
            
            # Call to __getitem__(...):
            # Processing the call arguments
            int_49836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 761, 12), 'int')
            # Processing the call keyword arguments
            kwargs_49837 = {}
            # Getting the type of 'call_assignment_47824' (line 761)
            call_assignment_47824_49834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 761, 12), 'call_assignment_47824', False)
            # Obtaining the member '__getitem__' of a type (line 761)
            getitem___49835 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 761, 12), call_assignment_47824_49834, '__getitem__')
            # Calling __getitem__(args, kwargs)
            getitem___call_result_49838 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___49835, *[int_49836], **kwargs_49837)
            
            # Assigning a type to the variable 'call_assignment_47825' (line 761)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 761, 12), 'call_assignment_47825', getitem___call_result_49838)
            
            # Assigning a Name to a Name (line 761):
            # Getting the type of 'call_assignment_47825' (line 761)
            call_assignment_47825_49839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 761, 12), 'call_assignment_47825')
            # Assigning a type to the variable 'lh' (line 761)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 761, 12), 'lh', call_assignment_47825_49839)
            
            # Assigning a Call to a Name (line 761):
            
            # Call to __getitem__(...):
            # Processing the call arguments
            int_49842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 761, 12), 'int')
            # Processing the call keyword arguments
            kwargs_49843 = {}
            # Getting the type of 'call_assignment_47824' (line 761)
            call_assignment_47824_49840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 761, 12), 'call_assignment_47824', False)
            # Obtaining the member '__getitem__' of a type (line 761)
            getitem___49841 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 761, 12), call_assignment_47824_49840, '__getitem__')
            # Calling __getitem__(args, kwargs)
            getitem___call_result_49844 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___49841, *[int_49842], **kwargs_49843)
            
            # Assigning a type to the variable 'call_assignment_47826' (line 761)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 761, 12), 'call_assignment_47826', getitem___call_result_49844)
            
            # Assigning a Name to a Name (line 761):
            # Getting the type of 'call_assignment_47826' (line 761)
            call_assignment_47826_49845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 761, 12), 'call_assignment_47826')
            # Assigning a type to the variable 'bc' (line 761)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 761, 16), 'bc', call_assignment_47826_49845)
            
            # Assigning a Call to a Name (line 761):
            
            # Call to __getitem__(...):
            # Processing the call arguments
            int_49848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 761, 12), 'int')
            # Processing the call keyword arguments
            kwargs_49849 = {}
            # Getting the type of 'call_assignment_47824' (line 761)
            call_assignment_47824_49846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 761, 12), 'call_assignment_47824', False)
            # Obtaining the member '__getitem__' of a type (line 761)
            getitem___49847 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 761, 12), call_assignment_47824_49846, '__getitem__')
            # Calling __getitem__(args, kwargs)
            getitem___call_result_49850 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___49847, *[int_49848], **kwargs_49849)
            
            # Assigning a type to the variable 'call_assignment_47827' (line 761)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 761, 12), 'call_assignment_47827', getitem___call_result_49850)
            
            # Assigning a Name to a Name (line 761):
            # Getting the type of 'call_assignment_47827' (line 761)
            call_assignment_47827_49851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 761, 12), 'call_assignment_47827')
            # Assigning a type to the variable 'ec' (line 761)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 761, 20), 'ec', call_assignment_47827_49851)
            
            # Assigning a Call to a Name (line 761):
            
            # Call to __getitem__(...):
            # Processing the call arguments
            int_49854 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 761, 12), 'int')
            # Processing the call keyword arguments
            kwargs_49855 = {}
            # Getting the type of 'call_assignment_47824' (line 761)
            call_assignment_47824_49852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 761, 12), 'call_assignment_47824', False)
            # Obtaining the member '__getitem__' of a type (line 761)
            getitem___49853 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 761, 12), call_assignment_47824_49852, '__getitem__')
            # Calling __getitem__(args, kwargs)
            getitem___call_result_49856 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___49853, *[int_49854], **kwargs_49855)
            
            # Assigning a type to the variable 'call_assignment_47828' (line 761)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 761, 12), 'call_assignment_47828', getitem___call_result_49856)
            
            # Assigning a Name to a Name (line 761):
            # Getting the type of 'call_assignment_47828' (line 761)
            call_assignment_47828_49857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 761, 12), 'call_assignment_47828')
            # Assigning a type to the variable 'nw' (line 761)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 761, 24), 'nw', call_assignment_47828_49857)
            
            # Assigning a Call to a Name (line 761):
            
            # Call to __getitem__(...):
            # Processing the call arguments
            int_49860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 761, 12), 'int')
            # Processing the call keyword arguments
            kwargs_49861 = {}
            # Getting the type of 'call_assignment_47824' (line 761)
            call_assignment_47824_49858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 761, 12), 'call_assignment_47824', False)
            # Obtaining the member '__getitem__' of a type (line 761)
            getitem___49859 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 761, 12), call_assignment_47824_49858, '__getitem__')
            # Calling __getitem__(args, kwargs)
            getitem___call_result_49862 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___49859, *[int_49860], **kwargs_49861)
            
            # Assigning a type to the variable 'call_assignment_47829' (line 761)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 761, 12), 'call_assignment_47829', getitem___call_result_49862)
            
            # Assigning a Name to a Name (line 761):
            # Getting the type of 'call_assignment_47829' (line 761)
            call_assignment_47829_49863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 761, 12), 'call_assignment_47829')
            # Assigning a type to the variable 'nh' (line 761)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 761, 28), 'nh', call_assignment_47829_49863)
            
            # Assigning a Call to a Name (line 761):
            
            # Call to __getitem__(...):
            # Processing the call arguments
            int_49866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 761, 12), 'int')
            # Processing the call keyword arguments
            kwargs_49867 = {}
            # Getting the type of 'call_assignment_47824' (line 761)
            call_assignment_47824_49864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 761, 12), 'call_assignment_47824', False)
            # Obtaining the member '__getitem__' of a type (line 761)
            getitem___49865 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 761, 12), call_assignment_47824_49864, '__getitem__')
            # Calling __getitem__(args, kwargs)
            getitem___call_result_49868 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___49865, *[int_49866], **kwargs_49867)
            
            # Assigning a type to the variable 'call_assignment_47830' (line 761)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 761, 12), 'call_assignment_47830', getitem___call_result_49868)
            
            # Assigning a Name to a Name (line 761):
            # Getting the type of 'call_assignment_47830' (line 761)
            call_assignment_47830_49869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 761, 12), 'call_assignment_47830')
            # Assigning a type to the variable 'nd' (line 761)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 761, 32), 'nd', call_assignment_47830_49869)
            
            # Call to report(...): (line 763)
            # Processing the call arguments (line 763)
            unicode_49873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 764, 16), 'unicode', u'lh=%d, bc=%d, ec=%d, nw=%d, nh=%d, nd=%d')
            
            # Obtaining an instance of the builtin type 'tuple' (line 765)
            tuple_49874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 765, 20), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 765)
            # Adding element type (line 765)
            # Getting the type of 'lh' (line 765)
            lh_49875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 765, 20), 'lh', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 765, 20), tuple_49874, lh_49875)
            # Adding element type (line 765)
            # Getting the type of 'bc' (line 765)
            bc_49876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 765, 24), 'bc', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 765, 20), tuple_49874, bc_49876)
            # Adding element type (line 765)
            # Getting the type of 'ec' (line 765)
            ec_49877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 765, 28), 'ec', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 765, 20), tuple_49874, ec_49877)
            # Adding element type (line 765)
            # Getting the type of 'nw' (line 765)
            nw_49878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 765, 32), 'nw', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 765, 20), tuple_49874, nw_49878)
            # Adding element type (line 765)
            # Getting the type of 'nh' (line 765)
            nh_49879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 765, 36), 'nh', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 765, 20), tuple_49874, nh_49879)
            # Adding element type (line 765)
            # Getting the type of 'nd' (line 765)
            nd_49880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 765, 40), 'nd', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 765, 20), tuple_49874, nd_49880)
            
            # Applying the binary operator '%' (line 764)
            result_mod_49881 = python_operator(stypy.reporting.localization.Localization(__file__, 764, 16), '%', unicode_49873, tuple_49874)
            
            unicode_49882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 765, 45), 'unicode', u'debug')
            # Processing the call keyword arguments (line 763)
            kwargs_49883 = {}
            # Getting the type of 'matplotlib' (line 763)
            matplotlib_49870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 763, 12), 'matplotlib', False)
            # Obtaining the member 'verbose' of a type (line 763)
            verbose_49871 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 763, 12), matplotlib_49870, 'verbose')
            # Obtaining the member 'report' of a type (line 763)
            report_49872 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 763, 12), verbose_49871, 'report')
            # Calling report(args, kwargs) (line 763)
            report_call_result_49884 = invoke(stypy.reporting.localization.Localization(__file__, 763, 12), report_49872, *[result_mod_49881, unicode_49882], **kwargs_49883)
            
            
            # Assigning a Call to a Name (line 766):
            
            # Assigning a Call to a Name (line 766):
            
            # Call to read(...): (line 766)
            # Processing the call arguments (line 766)
            int_49887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 766, 32), 'int')
            # Getting the type of 'lh' (line 766)
            lh_49888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 766, 34), 'lh', False)
            # Applying the binary operator '*' (line 766)
            result_mul_49889 = python_operator(stypy.reporting.localization.Localization(__file__, 766, 32), '*', int_49887, lh_49888)
            
            # Processing the call keyword arguments (line 766)
            kwargs_49890 = {}
            # Getting the type of 'file' (line 766)
            file_49885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 766, 22), 'file', False)
            # Obtaining the member 'read' of a type (line 766)
            read_49886 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 766, 22), file_49885, 'read')
            # Calling read(args, kwargs) (line 766)
            read_call_result_49891 = invoke(stypy.reporting.localization.Localization(__file__, 766, 22), read_49886, *[result_mul_49889], **kwargs_49890)
            
            # Assigning a type to the variable 'header2' (line 766)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 766, 12), 'header2', read_call_result_49891)
            
            # Assigning a Call to a Tuple (line 767):
            
            # Assigning a Call to a Name:
            
            # Call to unpack(...): (line 768)
            # Processing the call arguments (line 768)
            
            # Call to str(...): (line 768)
            # Processing the call arguments (line 768)
            unicode_49895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 768, 34), 'unicode', u'!2I')
            # Processing the call keyword arguments (line 768)
            kwargs_49896 = {}
            # Getting the type of 'str' (line 768)
            str_49894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 768, 30), 'str', False)
            # Calling str(args, kwargs) (line 768)
            str_call_result_49897 = invoke(stypy.reporting.localization.Localization(__file__, 768, 30), str_49894, *[unicode_49895], **kwargs_49896)
            
            
            # Obtaining the type of the subscript
            int_49898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 768, 51), 'int')
            slice_49899 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 768, 42), None, int_49898, None)
            # Getting the type of 'header2' (line 768)
            header2_49900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 768, 42), 'header2', False)
            # Obtaining the member '__getitem__' of a type (line 768)
            getitem___49901 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 768, 42), header2_49900, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 768)
            subscript_call_result_49902 = invoke(stypy.reporting.localization.Localization(__file__, 768, 42), getitem___49901, slice_49899)
            
            # Processing the call keyword arguments (line 768)
            kwargs_49903 = {}
            # Getting the type of 'struct' (line 768)
            struct_49892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 768, 16), 'struct', False)
            # Obtaining the member 'unpack' of a type (line 768)
            unpack_49893 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 768, 16), struct_49892, 'unpack')
            # Calling unpack(args, kwargs) (line 768)
            unpack_call_result_49904 = invoke(stypy.reporting.localization.Localization(__file__, 768, 16), unpack_49893, *[str_call_result_49897, subscript_call_result_49902], **kwargs_49903)
            
            # Assigning a type to the variable 'call_assignment_47831' (line 767)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 767, 12), 'call_assignment_47831', unpack_call_result_49904)
            
            # Assigning a Call to a Name (line 767):
            
            # Call to __getitem__(...):
            # Processing the call arguments
            int_49907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 767, 12), 'int')
            # Processing the call keyword arguments
            kwargs_49908 = {}
            # Getting the type of 'call_assignment_47831' (line 767)
            call_assignment_47831_49905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 767, 12), 'call_assignment_47831', False)
            # Obtaining the member '__getitem__' of a type (line 767)
            getitem___49906 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 767, 12), call_assignment_47831_49905, '__getitem__')
            # Calling __getitem__(args, kwargs)
            getitem___call_result_49909 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___49906, *[int_49907], **kwargs_49908)
            
            # Assigning a type to the variable 'call_assignment_47832' (line 767)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 767, 12), 'call_assignment_47832', getitem___call_result_49909)
            
            # Assigning a Name to a Attribute (line 767):
            # Getting the type of 'call_assignment_47832' (line 767)
            call_assignment_47832_49910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 767, 12), 'call_assignment_47832')
            # Getting the type of 'self' (line 767)
            self_49911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 767, 12), 'self')
            # Setting the type of the member 'checksum' of a type (line 767)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 767, 12), self_49911, 'checksum', call_assignment_47832_49910)
            
            # Assigning a Call to a Name (line 767):
            
            # Call to __getitem__(...):
            # Processing the call arguments
            int_49914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 767, 12), 'int')
            # Processing the call keyword arguments
            kwargs_49915 = {}
            # Getting the type of 'call_assignment_47831' (line 767)
            call_assignment_47831_49912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 767, 12), 'call_assignment_47831', False)
            # Obtaining the member '__getitem__' of a type (line 767)
            getitem___49913 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 767, 12), call_assignment_47831_49912, '__getitem__')
            # Calling __getitem__(args, kwargs)
            getitem___call_result_49916 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___49913, *[int_49914], **kwargs_49915)
            
            # Assigning a type to the variable 'call_assignment_47833' (line 767)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 767, 12), 'call_assignment_47833', getitem___call_result_49916)
            
            # Assigning a Name to a Attribute (line 767):
            # Getting the type of 'call_assignment_47833' (line 767)
            call_assignment_47833_49917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 767, 12), 'call_assignment_47833')
            # Getting the type of 'self' (line 767)
            self_49918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 767, 27), 'self')
            # Setting the type of the member 'design_size' of a type (line 767)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 767, 27), self_49918, 'design_size', call_assignment_47833_49917)
            
            # Assigning a Call to a Name (line 770):
            
            # Assigning a Call to a Name (line 770):
            
            # Call to read(...): (line 770)
            # Processing the call arguments (line 770)
            int_49921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 770, 34), 'int')
            # Getting the type of 'ec' (line 770)
            ec_49922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 770, 37), 'ec', False)
            # Getting the type of 'bc' (line 770)
            bc_49923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 770, 40), 'bc', False)
            # Applying the binary operator '-' (line 770)
            result_sub_49924 = python_operator(stypy.reporting.localization.Localization(__file__, 770, 37), '-', ec_49922, bc_49923)
            
            int_49925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 770, 43), 'int')
            # Applying the binary operator '+' (line 770)
            result_add_49926 = python_operator(stypy.reporting.localization.Localization(__file__, 770, 42), '+', result_sub_49924, int_49925)
            
            # Applying the binary operator '*' (line 770)
            result_mul_49927 = python_operator(stypy.reporting.localization.Localization(__file__, 770, 34), '*', int_49921, result_add_49926)
            
            # Processing the call keyword arguments (line 770)
            kwargs_49928 = {}
            # Getting the type of 'file' (line 770)
            file_49919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 770, 24), 'file', False)
            # Obtaining the member 'read' of a type (line 770)
            read_49920 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 770, 24), file_49919, 'read')
            # Calling read(args, kwargs) (line 770)
            read_call_result_49929 = invoke(stypy.reporting.localization.Localization(__file__, 770, 24), read_49920, *[result_mul_49927], **kwargs_49928)
            
            # Assigning a type to the variable 'char_info' (line 770)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 770, 12), 'char_info', read_call_result_49929)
            
            # Assigning a Call to a Name (line 771):
            
            # Assigning a Call to a Name (line 771):
            
            # Call to read(...): (line 771)
            # Processing the call arguments (line 771)
            int_49932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 771, 31), 'int')
            # Getting the type of 'nw' (line 771)
            nw_49933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 771, 33), 'nw', False)
            # Applying the binary operator '*' (line 771)
            result_mul_49934 = python_operator(stypy.reporting.localization.Localization(__file__, 771, 31), '*', int_49932, nw_49933)
            
            # Processing the call keyword arguments (line 771)
            kwargs_49935 = {}
            # Getting the type of 'file' (line 771)
            file_49930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 771, 21), 'file', False)
            # Obtaining the member 'read' of a type (line 771)
            read_49931 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 771, 21), file_49930, 'read')
            # Calling read(args, kwargs) (line 771)
            read_call_result_49936 = invoke(stypy.reporting.localization.Localization(__file__, 771, 21), read_49931, *[result_mul_49934], **kwargs_49935)
            
            # Assigning a type to the variable 'widths' (line 771)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 771, 12), 'widths', read_call_result_49936)
            
            # Assigning a Call to a Name (line 772):
            
            # Assigning a Call to a Name (line 772):
            
            # Call to read(...): (line 772)
            # Processing the call arguments (line 772)
            int_49939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 772, 32), 'int')
            # Getting the type of 'nh' (line 772)
            nh_49940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 772, 34), 'nh', False)
            # Applying the binary operator '*' (line 772)
            result_mul_49941 = python_operator(stypy.reporting.localization.Localization(__file__, 772, 32), '*', int_49939, nh_49940)
            
            # Processing the call keyword arguments (line 772)
            kwargs_49942 = {}
            # Getting the type of 'file' (line 772)
            file_49937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 772, 22), 'file', False)
            # Obtaining the member 'read' of a type (line 772)
            read_49938 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 772, 22), file_49937, 'read')
            # Calling read(args, kwargs) (line 772)
            read_call_result_49943 = invoke(stypy.reporting.localization.Localization(__file__, 772, 22), read_49938, *[result_mul_49941], **kwargs_49942)
            
            # Assigning a type to the variable 'heights' (line 772)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 772, 12), 'heights', read_call_result_49943)
            
            # Assigning a Call to a Name (line 773):
            
            # Assigning a Call to a Name (line 773):
            
            # Call to read(...): (line 773)
            # Processing the call arguments (line 773)
            int_49946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 773, 31), 'int')
            # Getting the type of 'nd' (line 773)
            nd_49947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 773, 33), 'nd', False)
            # Applying the binary operator '*' (line 773)
            result_mul_49948 = python_operator(stypy.reporting.localization.Localization(__file__, 773, 31), '*', int_49946, nd_49947)
            
            # Processing the call keyword arguments (line 773)
            kwargs_49949 = {}
            # Getting the type of 'file' (line 773)
            file_49944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 773, 21), 'file', False)
            # Obtaining the member 'read' of a type (line 773)
            read_49945 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 773, 21), file_49944, 'read')
            # Calling read(args, kwargs) (line 773)
            read_call_result_49950 = invoke(stypy.reporting.localization.Localization(__file__, 773, 21), read_49945, *[result_mul_49948], **kwargs_49949)
            
            # Assigning a type to the variable 'depths' (line 773)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 773, 12), 'depths', read_call_result_49950)
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 759)
            exit___49951 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 759, 13), open_call_result_49811, '__exit__')
            with_exit_49952 = invoke(stypy.reporting.localization.Localization(__file__, 759, 13), exit___49951, None, None, None)

        
        # Assigning a Tuple to a Tuple (line 775):
        
        # Assigning a Dict to a Name (line 775):
        
        # Obtaining an instance of the builtin type 'dict' (line 775)
        dict_49953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 775, 46), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 775)
        
        # Assigning a type to the variable 'tuple_assignment_47834' (line 775)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 775, 8), 'tuple_assignment_47834', dict_49953)
        
        # Assigning a Dict to a Name (line 775):
        
        # Obtaining an instance of the builtin type 'dict' (line 775)
        dict_49954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 775, 50), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 775)
        
        # Assigning a type to the variable 'tuple_assignment_47835' (line 775)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 775, 8), 'tuple_assignment_47835', dict_49954)
        
        # Assigning a Dict to a Name (line 775):
        
        # Obtaining an instance of the builtin type 'dict' (line 775)
        dict_49955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 775, 54), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 775)
        
        # Assigning a type to the variable 'tuple_assignment_47836' (line 775)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 775, 8), 'tuple_assignment_47836', dict_49955)
        
        # Assigning a Name to a Attribute (line 775):
        # Getting the type of 'tuple_assignment_47834' (line 775)
        tuple_assignment_47834_49956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 775, 8), 'tuple_assignment_47834')
        # Getting the type of 'self' (line 775)
        self_49957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 775, 8), 'self')
        # Setting the type of the member 'width' of a type (line 775)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 775, 8), self_49957, 'width', tuple_assignment_47834_49956)
        
        # Assigning a Name to a Attribute (line 775):
        # Getting the type of 'tuple_assignment_47835' (line 775)
        tuple_assignment_47835_49958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 775, 8), 'tuple_assignment_47835')
        # Getting the type of 'self' (line 775)
        self_49959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 775, 20), 'self')
        # Setting the type of the member 'height' of a type (line 775)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 775, 20), self_49959, 'height', tuple_assignment_47835_49958)
        
        # Assigning a Name to a Attribute (line 775):
        # Getting the type of 'tuple_assignment_47836' (line 775)
        tuple_assignment_47836_49960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 775, 8), 'tuple_assignment_47836')
        # Getting the type of 'self' (line 775)
        self_49961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 775, 33), 'self')
        # Setting the type of the member 'depth' of a type (line 775)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 775, 33), self_49961, 'depth', tuple_assignment_47836_49960)
        
        # Assigning a ListComp to a Tuple (line 776):
        
        # Assigning a Subscript to a Name (line 776):
        
        # Obtaining the type of the subscript
        int_49962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 776, 8), 'int')
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Obtaining an instance of the builtin type 'tuple' (line 778)
        tuple_49979 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 778, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 778)
        # Adding element type (line 778)
        # Getting the type of 'widths' (line 778)
        widths_49980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 778, 23), 'widths')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 778, 23), tuple_49979, widths_49980)
        # Adding element type (line 778)
        # Getting the type of 'heights' (line 778)
        heights_49981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 778, 31), 'heights')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 778, 23), tuple_49979, heights_49981)
        # Adding element type (line 778)
        # Getting the type of 'depths' (line 778)
        depths_49982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 778, 40), 'depths')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 778, 23), tuple_49979, depths_49982)
        
        comprehension_49983 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 777, 13), tuple_49979)
        # Assigning a type to the variable 'x' (line 777)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 777, 13), 'x', comprehension_49983)
        
        # Call to unpack(...): (line 777)
        # Processing the call arguments (line 777)
        
        # Call to str(...): (line 777)
        # Processing the call arguments (line 777)
        unicode_49966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 777, 31), 'unicode', u'!%dI')
        # Processing the call keyword arguments (line 777)
        kwargs_49967 = {}
        # Getting the type of 'str' (line 777)
        str_49965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 777, 27), 'str', False)
        # Calling str(args, kwargs) (line 777)
        str_call_result_49968 = invoke(stypy.reporting.localization.Localization(__file__, 777, 27), str_49965, *[unicode_49966], **kwargs_49967)
        
        
        # Call to len(...): (line 777)
        # Processing the call arguments (line 777)
        # Getting the type of 'x' (line 777)
        x_49970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 777, 46), 'x', False)
        # Processing the call keyword arguments (line 777)
        kwargs_49971 = {}
        # Getting the type of 'len' (line 777)
        len_49969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 777, 42), 'len', False)
        # Calling len(args, kwargs) (line 777)
        len_call_result_49972 = invoke(stypy.reporting.localization.Localization(__file__, 777, 42), len_49969, *[x_49970], **kwargs_49971)
        
        int_49973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 777, 49), 'int')
        # Applying the binary operator 'div' (line 777)
        result_div_49974 = python_operator(stypy.reporting.localization.Localization(__file__, 777, 42), 'div', len_call_result_49972, int_49973)
        
        # Applying the binary operator '%' (line 777)
        result_mod_49975 = python_operator(stypy.reporting.localization.Localization(__file__, 777, 27), '%', str_call_result_49968, result_div_49974)
        
        # Getting the type of 'x' (line 777)
        x_49976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 777, 53), 'x', False)
        # Processing the call keyword arguments (line 777)
        kwargs_49977 = {}
        # Getting the type of 'struct' (line 777)
        struct_49963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 777, 13), 'struct', False)
        # Obtaining the member 'unpack' of a type (line 777)
        unpack_49964 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 777, 13), struct_49963, 'unpack')
        # Calling unpack(args, kwargs) (line 777)
        unpack_call_result_49978 = invoke(stypy.reporting.localization.Localization(__file__, 777, 13), unpack_49964, *[result_mod_49975, x_49976], **kwargs_49977)
        
        list_49984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 777, 13), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 777, 13), list_49984, unpack_call_result_49978)
        # Obtaining the member '__getitem__' of a type (line 776)
        getitem___49985 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 776, 8), list_49984, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 776)
        subscript_call_result_49986 = invoke(stypy.reporting.localization.Localization(__file__, 776, 8), getitem___49985, int_49962)
        
        # Assigning a type to the variable 'tuple_var_assignment_47837' (line 776)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 776, 8), 'tuple_var_assignment_47837', subscript_call_result_49986)
        
        # Assigning a Subscript to a Name (line 776):
        
        # Obtaining the type of the subscript
        int_49987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 776, 8), 'int')
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Obtaining an instance of the builtin type 'tuple' (line 778)
        tuple_50004 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 778, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 778)
        # Adding element type (line 778)
        # Getting the type of 'widths' (line 778)
        widths_50005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 778, 23), 'widths')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 778, 23), tuple_50004, widths_50005)
        # Adding element type (line 778)
        # Getting the type of 'heights' (line 778)
        heights_50006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 778, 31), 'heights')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 778, 23), tuple_50004, heights_50006)
        # Adding element type (line 778)
        # Getting the type of 'depths' (line 778)
        depths_50007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 778, 40), 'depths')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 778, 23), tuple_50004, depths_50007)
        
        comprehension_50008 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 777, 13), tuple_50004)
        # Assigning a type to the variable 'x' (line 777)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 777, 13), 'x', comprehension_50008)
        
        # Call to unpack(...): (line 777)
        # Processing the call arguments (line 777)
        
        # Call to str(...): (line 777)
        # Processing the call arguments (line 777)
        unicode_49991 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 777, 31), 'unicode', u'!%dI')
        # Processing the call keyword arguments (line 777)
        kwargs_49992 = {}
        # Getting the type of 'str' (line 777)
        str_49990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 777, 27), 'str', False)
        # Calling str(args, kwargs) (line 777)
        str_call_result_49993 = invoke(stypy.reporting.localization.Localization(__file__, 777, 27), str_49990, *[unicode_49991], **kwargs_49992)
        
        
        # Call to len(...): (line 777)
        # Processing the call arguments (line 777)
        # Getting the type of 'x' (line 777)
        x_49995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 777, 46), 'x', False)
        # Processing the call keyword arguments (line 777)
        kwargs_49996 = {}
        # Getting the type of 'len' (line 777)
        len_49994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 777, 42), 'len', False)
        # Calling len(args, kwargs) (line 777)
        len_call_result_49997 = invoke(stypy.reporting.localization.Localization(__file__, 777, 42), len_49994, *[x_49995], **kwargs_49996)
        
        int_49998 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 777, 49), 'int')
        # Applying the binary operator 'div' (line 777)
        result_div_49999 = python_operator(stypy.reporting.localization.Localization(__file__, 777, 42), 'div', len_call_result_49997, int_49998)
        
        # Applying the binary operator '%' (line 777)
        result_mod_50000 = python_operator(stypy.reporting.localization.Localization(__file__, 777, 27), '%', str_call_result_49993, result_div_49999)
        
        # Getting the type of 'x' (line 777)
        x_50001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 777, 53), 'x', False)
        # Processing the call keyword arguments (line 777)
        kwargs_50002 = {}
        # Getting the type of 'struct' (line 777)
        struct_49988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 777, 13), 'struct', False)
        # Obtaining the member 'unpack' of a type (line 777)
        unpack_49989 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 777, 13), struct_49988, 'unpack')
        # Calling unpack(args, kwargs) (line 777)
        unpack_call_result_50003 = invoke(stypy.reporting.localization.Localization(__file__, 777, 13), unpack_49989, *[result_mod_50000, x_50001], **kwargs_50002)
        
        list_50009 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 777, 13), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 777, 13), list_50009, unpack_call_result_50003)
        # Obtaining the member '__getitem__' of a type (line 776)
        getitem___50010 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 776, 8), list_50009, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 776)
        subscript_call_result_50011 = invoke(stypy.reporting.localization.Localization(__file__, 776, 8), getitem___50010, int_49987)
        
        # Assigning a type to the variable 'tuple_var_assignment_47838' (line 776)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 776, 8), 'tuple_var_assignment_47838', subscript_call_result_50011)
        
        # Assigning a Subscript to a Name (line 776):
        
        # Obtaining the type of the subscript
        int_50012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 776, 8), 'int')
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Obtaining an instance of the builtin type 'tuple' (line 778)
        tuple_50029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 778, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 778)
        # Adding element type (line 778)
        # Getting the type of 'widths' (line 778)
        widths_50030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 778, 23), 'widths')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 778, 23), tuple_50029, widths_50030)
        # Adding element type (line 778)
        # Getting the type of 'heights' (line 778)
        heights_50031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 778, 31), 'heights')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 778, 23), tuple_50029, heights_50031)
        # Adding element type (line 778)
        # Getting the type of 'depths' (line 778)
        depths_50032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 778, 40), 'depths')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 778, 23), tuple_50029, depths_50032)
        
        comprehension_50033 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 777, 13), tuple_50029)
        # Assigning a type to the variable 'x' (line 777)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 777, 13), 'x', comprehension_50033)
        
        # Call to unpack(...): (line 777)
        # Processing the call arguments (line 777)
        
        # Call to str(...): (line 777)
        # Processing the call arguments (line 777)
        unicode_50016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 777, 31), 'unicode', u'!%dI')
        # Processing the call keyword arguments (line 777)
        kwargs_50017 = {}
        # Getting the type of 'str' (line 777)
        str_50015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 777, 27), 'str', False)
        # Calling str(args, kwargs) (line 777)
        str_call_result_50018 = invoke(stypy.reporting.localization.Localization(__file__, 777, 27), str_50015, *[unicode_50016], **kwargs_50017)
        
        
        # Call to len(...): (line 777)
        # Processing the call arguments (line 777)
        # Getting the type of 'x' (line 777)
        x_50020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 777, 46), 'x', False)
        # Processing the call keyword arguments (line 777)
        kwargs_50021 = {}
        # Getting the type of 'len' (line 777)
        len_50019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 777, 42), 'len', False)
        # Calling len(args, kwargs) (line 777)
        len_call_result_50022 = invoke(stypy.reporting.localization.Localization(__file__, 777, 42), len_50019, *[x_50020], **kwargs_50021)
        
        int_50023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 777, 49), 'int')
        # Applying the binary operator 'div' (line 777)
        result_div_50024 = python_operator(stypy.reporting.localization.Localization(__file__, 777, 42), 'div', len_call_result_50022, int_50023)
        
        # Applying the binary operator '%' (line 777)
        result_mod_50025 = python_operator(stypy.reporting.localization.Localization(__file__, 777, 27), '%', str_call_result_50018, result_div_50024)
        
        # Getting the type of 'x' (line 777)
        x_50026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 777, 53), 'x', False)
        # Processing the call keyword arguments (line 777)
        kwargs_50027 = {}
        # Getting the type of 'struct' (line 777)
        struct_50013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 777, 13), 'struct', False)
        # Obtaining the member 'unpack' of a type (line 777)
        unpack_50014 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 777, 13), struct_50013, 'unpack')
        # Calling unpack(args, kwargs) (line 777)
        unpack_call_result_50028 = invoke(stypy.reporting.localization.Localization(__file__, 777, 13), unpack_50014, *[result_mod_50025, x_50026], **kwargs_50027)
        
        list_50034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 777, 13), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 777, 13), list_50034, unpack_call_result_50028)
        # Obtaining the member '__getitem__' of a type (line 776)
        getitem___50035 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 776, 8), list_50034, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 776)
        subscript_call_result_50036 = invoke(stypy.reporting.localization.Localization(__file__, 776, 8), getitem___50035, int_50012)
        
        # Assigning a type to the variable 'tuple_var_assignment_47839' (line 776)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 776, 8), 'tuple_var_assignment_47839', subscript_call_result_50036)
        
        # Assigning a Name to a Name (line 776):
        # Getting the type of 'tuple_var_assignment_47837' (line 776)
        tuple_var_assignment_47837_50037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 776, 8), 'tuple_var_assignment_47837')
        # Assigning a type to the variable 'widths' (line 776)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 776, 8), 'widths', tuple_var_assignment_47837_50037)
        
        # Assigning a Name to a Name (line 776):
        # Getting the type of 'tuple_var_assignment_47838' (line 776)
        tuple_var_assignment_47838_50038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 776, 8), 'tuple_var_assignment_47838')
        # Assigning a type to the variable 'heights' (line 776)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 776, 16), 'heights', tuple_var_assignment_47838_50038)
        
        # Assigning a Name to a Name (line 776):
        # Getting the type of 'tuple_var_assignment_47839' (line 776)
        tuple_var_assignment_47839_50039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 776, 8), 'tuple_var_assignment_47839')
        # Assigning a type to the variable 'depths' (line 776)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 776, 25), 'depths', tuple_var_assignment_47839_50039)
        
        
        # Call to enumerate(...): (line 779)
        # Processing the call arguments (line 779)
        
        # Call to xrange(...): (line 779)
        # Processing the call arguments (line 779)
        # Getting the type of 'bc' (line 779)
        bc_50042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 779, 42), 'bc', False)
        # Getting the type of 'ec' (line 779)
        ec_50043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 779, 46), 'ec', False)
        int_50044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 779, 49), 'int')
        # Applying the binary operator '+' (line 779)
        result_add_50045 = python_operator(stypy.reporting.localization.Localization(__file__, 779, 46), '+', ec_50043, int_50044)
        
        # Processing the call keyword arguments (line 779)
        kwargs_50046 = {}
        # Getting the type of 'xrange' (line 779)
        xrange_50041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 779, 35), 'xrange', False)
        # Calling xrange(args, kwargs) (line 779)
        xrange_call_result_50047 = invoke(stypy.reporting.localization.Localization(__file__, 779, 35), xrange_50041, *[bc_50042, result_add_50045], **kwargs_50046)
        
        # Processing the call keyword arguments (line 779)
        kwargs_50048 = {}
        # Getting the type of 'enumerate' (line 779)
        enumerate_50040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 779, 25), 'enumerate', False)
        # Calling enumerate(args, kwargs) (line 779)
        enumerate_call_result_50049 = invoke(stypy.reporting.localization.Localization(__file__, 779, 25), enumerate_50040, *[xrange_call_result_50047], **kwargs_50048)
        
        # Testing the type of a for loop iterable (line 779)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 779, 8), enumerate_call_result_50049)
        # Getting the type of the for loop variable (line 779)
        for_loop_var_50050 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 779, 8), enumerate_call_result_50049)
        # Assigning a type to the variable 'idx' (line 779)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 779, 8), 'idx', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 779, 8), for_loop_var_50050))
        # Assigning a type to the variable 'char' (line 779)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 779, 8), 'char', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 779, 8), for_loop_var_50050))
        # SSA begins for a for statement (line 779)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 780):
        
        # Assigning a Call to a Name (line 780):
        
        # Call to ord(...): (line 780)
        # Processing the call arguments (line 780)
        
        # Obtaining the type of the subscript
        int_50052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 780, 34), 'int')
        # Getting the type of 'idx' (line 780)
        idx_50053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 780, 36), 'idx', False)
        # Applying the binary operator '*' (line 780)
        result_mul_50054 = python_operator(stypy.reporting.localization.Localization(__file__, 780, 34), '*', int_50052, idx_50053)
        
        # Getting the type of 'char_info' (line 780)
        char_info_50055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 780, 24), 'char_info', False)
        # Obtaining the member '__getitem__' of a type (line 780)
        getitem___50056 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 780, 24), char_info_50055, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 780)
        subscript_call_result_50057 = invoke(stypy.reporting.localization.Localization(__file__, 780, 24), getitem___50056, result_mul_50054)
        
        # Processing the call keyword arguments (line 780)
        kwargs_50058 = {}
        # Getting the type of 'ord' (line 780)
        ord_50051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 780, 20), 'ord', False)
        # Calling ord(args, kwargs) (line 780)
        ord_call_result_50059 = invoke(stypy.reporting.localization.Localization(__file__, 780, 20), ord_50051, *[subscript_call_result_50057], **kwargs_50058)
        
        # Assigning a type to the variable 'byte0' (line 780)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 780, 12), 'byte0', ord_call_result_50059)
        
        # Assigning a Call to a Name (line 781):
        
        # Assigning a Call to a Name (line 781):
        
        # Call to ord(...): (line 781)
        # Processing the call arguments (line 781)
        
        # Obtaining the type of the subscript
        int_50061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 781, 34), 'int')
        # Getting the type of 'idx' (line 781)
        idx_50062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 781, 36), 'idx', False)
        # Applying the binary operator '*' (line 781)
        result_mul_50063 = python_operator(stypy.reporting.localization.Localization(__file__, 781, 34), '*', int_50061, idx_50062)
        
        int_50064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 781, 40), 'int')
        # Applying the binary operator '+' (line 781)
        result_add_50065 = python_operator(stypy.reporting.localization.Localization(__file__, 781, 34), '+', result_mul_50063, int_50064)
        
        # Getting the type of 'char_info' (line 781)
        char_info_50066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 781, 24), 'char_info', False)
        # Obtaining the member '__getitem__' of a type (line 781)
        getitem___50067 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 781, 24), char_info_50066, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 781)
        subscript_call_result_50068 = invoke(stypy.reporting.localization.Localization(__file__, 781, 24), getitem___50067, result_add_50065)
        
        # Processing the call keyword arguments (line 781)
        kwargs_50069 = {}
        # Getting the type of 'ord' (line 781)
        ord_50060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 781, 20), 'ord', False)
        # Calling ord(args, kwargs) (line 781)
        ord_call_result_50070 = invoke(stypy.reporting.localization.Localization(__file__, 781, 20), ord_50060, *[subscript_call_result_50068], **kwargs_50069)
        
        # Assigning a type to the variable 'byte1' (line 781)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 781, 12), 'byte1', ord_call_result_50070)
        
        # Assigning a Call to a Subscript (line 782):
        
        # Assigning a Call to a Subscript (line 782):
        
        # Call to _fix2comp(...): (line 782)
        # Processing the call arguments (line 782)
        
        # Obtaining the type of the subscript
        # Getting the type of 'byte0' (line 782)
        byte0_50072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 782, 48), 'byte0', False)
        # Getting the type of 'widths' (line 782)
        widths_50073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 782, 41), 'widths', False)
        # Obtaining the member '__getitem__' of a type (line 782)
        getitem___50074 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 782, 41), widths_50073, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 782)
        subscript_call_result_50075 = invoke(stypy.reporting.localization.Localization(__file__, 782, 41), getitem___50074, byte0_50072)
        
        # Processing the call keyword arguments (line 782)
        kwargs_50076 = {}
        # Getting the type of '_fix2comp' (line 782)
        _fix2comp_50071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 782, 31), '_fix2comp', False)
        # Calling _fix2comp(args, kwargs) (line 782)
        _fix2comp_call_result_50077 = invoke(stypy.reporting.localization.Localization(__file__, 782, 31), _fix2comp_50071, *[subscript_call_result_50075], **kwargs_50076)
        
        # Getting the type of 'self' (line 782)
        self_50078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 782, 12), 'self')
        # Obtaining the member 'width' of a type (line 782)
        width_50079 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 782, 12), self_50078, 'width')
        # Getting the type of 'char' (line 782)
        char_50080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 782, 23), 'char')
        # Storing an element on a container (line 782)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 782, 12), width_50079, (char_50080, _fix2comp_call_result_50077))
        
        # Assigning a Call to a Subscript (line 783):
        
        # Assigning a Call to a Subscript (line 783):
        
        # Call to _fix2comp(...): (line 783)
        # Processing the call arguments (line 783)
        
        # Obtaining the type of the subscript
        # Getting the type of 'byte1' (line 783)
        byte1_50082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 783, 50), 'byte1', False)
        int_50083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 783, 59), 'int')
        # Applying the binary operator '>>' (line 783)
        result_rshift_50084 = python_operator(stypy.reporting.localization.Localization(__file__, 783, 50), '>>', byte1_50082, int_50083)
        
        # Getting the type of 'heights' (line 783)
        heights_50085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 783, 42), 'heights', False)
        # Obtaining the member '__getitem__' of a type (line 783)
        getitem___50086 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 783, 42), heights_50085, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 783)
        subscript_call_result_50087 = invoke(stypy.reporting.localization.Localization(__file__, 783, 42), getitem___50086, result_rshift_50084)
        
        # Processing the call keyword arguments (line 783)
        kwargs_50088 = {}
        # Getting the type of '_fix2comp' (line 783)
        _fix2comp_50081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 783, 32), '_fix2comp', False)
        # Calling _fix2comp(args, kwargs) (line 783)
        _fix2comp_call_result_50089 = invoke(stypy.reporting.localization.Localization(__file__, 783, 32), _fix2comp_50081, *[subscript_call_result_50087], **kwargs_50088)
        
        # Getting the type of 'self' (line 783)
        self_50090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 783, 12), 'self')
        # Obtaining the member 'height' of a type (line 783)
        height_50091 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 783, 12), self_50090, 'height')
        # Getting the type of 'char' (line 783)
        char_50092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 783, 24), 'char')
        # Storing an element on a container (line 783)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 783, 12), height_50091, (char_50092, _fix2comp_call_result_50089))
        
        # Assigning a Call to a Subscript (line 784):
        
        # Assigning a Call to a Subscript (line 784):
        
        # Call to _fix2comp(...): (line 784)
        # Processing the call arguments (line 784)
        
        # Obtaining the type of the subscript
        # Getting the type of 'byte1' (line 784)
        byte1_50094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 784, 48), 'byte1', False)
        int_50095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 784, 56), 'int')
        # Applying the binary operator '&' (line 784)
        result_and__50096 = python_operator(stypy.reporting.localization.Localization(__file__, 784, 48), '&', byte1_50094, int_50095)
        
        # Getting the type of 'depths' (line 784)
        depths_50097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 784, 41), 'depths', False)
        # Obtaining the member '__getitem__' of a type (line 784)
        getitem___50098 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 784, 41), depths_50097, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 784)
        subscript_call_result_50099 = invoke(stypy.reporting.localization.Localization(__file__, 784, 41), getitem___50098, result_and__50096)
        
        # Processing the call keyword arguments (line 784)
        kwargs_50100 = {}
        # Getting the type of '_fix2comp' (line 784)
        _fix2comp_50093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 784, 31), '_fix2comp', False)
        # Calling _fix2comp(args, kwargs) (line 784)
        _fix2comp_call_result_50101 = invoke(stypy.reporting.localization.Localization(__file__, 784, 31), _fix2comp_50093, *[subscript_call_result_50099], **kwargs_50100)
        
        # Getting the type of 'self' (line 784)
        self_50102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 784, 12), 'self')
        # Obtaining the member 'depth' of a type (line 784)
        depth_50103 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 784, 12), self_50102, 'depth')
        # Getting the type of 'char' (line 784)
        char_50104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 784, 23), 'char')
        # Storing an element on a container (line 784)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 784, 12), depth_50103, (char_50104, _fix2comp_call_result_50101))
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'Tfm' (line 734)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 734, 0), 'Tfm', Tfm)

# Assigning a Tuple to a Name (line 755):

# Obtaining an instance of the builtin type 'tuple' (line 755)
tuple_50105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 755, 17), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 755)
# Adding element type (line 755)
unicode_50106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 755, 17), 'unicode', u'checksum')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 755, 17), tuple_50105, unicode_50106)
# Adding element type (line 755)
unicode_50107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 755, 29), 'unicode', u'design_size')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 755, 17), tuple_50105, unicode_50107)
# Adding element type (line 755)
unicode_50108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 755, 44), 'unicode', u'width')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 755, 17), tuple_50105, unicode_50108)
# Adding element type (line 755)
unicode_50109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 755, 53), 'unicode', u'height')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 755, 17), tuple_50105, unicode_50109)
# Adding element type (line 755)
unicode_50110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 755, 63), 'unicode', u'depth')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 755, 17), tuple_50105, unicode_50110)

# Getting the type of 'Tfm'
Tfm_50111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Tfm')
# Setting the type of the member '__slots__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Tfm_50111, '__slots__', tuple_50105)

# Assigning a Call to a Name (line 787):

# Assigning a Call to a Name (line 787):

# Call to namedtuple(...): (line 787)
# Processing the call arguments (line 787)
unicode_50113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 787, 20), 'unicode', u'Font')
unicode_50114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 787, 28), 'unicode', u'texname psname effects encoding filename')
# Processing the call keyword arguments (line 787)
kwargs_50115 = {}
# Getting the type of 'namedtuple' (line 787)
namedtuple_50112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 787, 9), 'namedtuple', False)
# Calling namedtuple(args, kwargs) (line 787)
namedtuple_call_result_50116 = invoke(stypy.reporting.localization.Localization(__file__, 787, 9), namedtuple_50112, *[unicode_50113, unicode_50114], **kwargs_50115)

# Assigning a type to the variable 'PsFont' (line 787)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 787, 0), 'PsFont', namedtuple_call_result_50116)
# Declaration of the 'PsfontsMap' class

class PsfontsMap(object, ):
    unicode_50117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 835, (-1)), 'unicode', u'\n    A psfonts.map formatted file, mapping TeX fonts to PS fonts.\n\n    Usage::\n\n     >>> map = PsfontsMap(find_tex_file(\'pdftex.map\'))\n     >>> entry = map[b\'ptmbo8r\']\n     >>> entry.texname\n     b\'ptmbo8r\'\n     >>> entry.psname\n     b\'Times-Bold\'\n     >>> entry.encoding\n     \'/usr/local/texlive/2008/texmf-dist/fonts/enc/dvips/base/8r.enc\'\n     >>> entry.effects\n     {\'slant\': 0.16700000000000001}\n     >>> entry.filename\n\n    Parameters\n    ----------\n\n    filename : string or bytestring\n\n    Notes\n    -----\n\n    For historical reasons, TeX knows many Type-1 fonts by different\n    names than the outside world. (For one thing, the names have to\n    fit in eight characters.) Also, TeX\'s native fonts are not Type-1\n    but Metafont, which is nontrivial to convert to PostScript except\n    as a bitmap. While high-quality conversions to Type-1 format exist\n    and are shipped with modern TeX distributions, we need to know\n    which Type-1 fonts are the counterparts of which native fonts. For\n    these reasons a mapping is needed from internal font names to font\n    file names.\n\n    A texmf tree typically includes mapping files called e.g.\n    :file:`psfonts.map`, :file:`pdftex.map`, or :file:`dvipdfm.map`.\n    The file :file:`psfonts.map` is used by :program:`dvips`,\n    :file:`pdftex.map` by :program:`pdfTeX`, and :file:`dvipdfm.map`\n    by :program:`dvipdfm`. :file:`psfonts.map` might avoid embedding\n    the 35 PostScript fonts (i.e., have no filename for them, as in\n    the Times-Bold example above), while the pdf-related files perhaps\n    only avoid the "Base 14" pdf fonts. But the user may have\n    configured these files differently.\n    ')
    
    # Assigning a Tuple to a Name (line 836):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 838, 4, False)
        # Assigning a type to the variable 'self' (line 839)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 839, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PsfontsMap.__init__', ['filename'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['filename'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Dict to a Attribute (line 839):
        
        # Assigning a Dict to a Attribute (line 839):
        
        # Obtaining an instance of the builtin type 'dict' (line 839)
        dict_50118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 839, 21), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 839)
        
        # Getting the type of 'self' (line 839)
        self_50119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 839, 8), 'self')
        # Setting the type of the member '_font' of a type (line 839)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 839, 8), self_50119, '_font', dict_50118)
        
        # Assigning a Name to a Attribute (line 840):
        
        # Assigning a Name to a Attribute (line 840):
        # Getting the type of 'filename' (line 840)
        filename_50120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 840, 25), 'filename')
        # Getting the type of 'self' (line 840)
        self_50121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 840, 8), 'self')
        # Setting the type of the member '_filename' of a type (line 840)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 840, 8), self_50121, '_filename', filename_50120)
        
        
        # Evaluating a boolean operation
        # Getting the type of 'six' (line 841)
        six_50122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 841, 11), 'six')
        # Obtaining the member 'PY3' of a type (line 841)
        PY3_50123 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 841, 11), six_50122, 'PY3')
        
        # Call to isinstance(...): (line 841)
        # Processing the call arguments (line 841)
        # Getting the type of 'filename' (line 841)
        filename_50125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 841, 34), 'filename', False)
        # Getting the type of 'bytes' (line 841)
        bytes_50126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 841, 44), 'bytes', False)
        # Processing the call keyword arguments (line 841)
        kwargs_50127 = {}
        # Getting the type of 'isinstance' (line 841)
        isinstance_50124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 841, 23), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 841)
        isinstance_call_result_50128 = invoke(stypy.reporting.localization.Localization(__file__, 841, 23), isinstance_50124, *[filename_50125, bytes_50126], **kwargs_50127)
        
        # Applying the binary operator 'and' (line 841)
        result_and_keyword_50129 = python_operator(stypy.reporting.localization.Localization(__file__, 841, 11), 'and', PY3_50123, isinstance_call_result_50128)
        
        # Testing the type of an if condition (line 841)
        if_condition_50130 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 841, 8), result_and_keyword_50129)
        # Assigning a type to the variable 'if_condition_50130' (line 841)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 841, 8), 'if_condition_50130', if_condition_50130)
        # SSA begins for if statement (line 841)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BoolOp to a Name (line 842):
        
        # Assigning a BoolOp to a Name (line 842):
        
        # Evaluating a boolean operation
        
        # Call to getfilesystemencoding(...): (line 842)
        # Processing the call keyword arguments (line 842)
        kwargs_50133 = {}
        # Getting the type of 'sys' (line 842)
        sys_50131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 842, 23), 'sys', False)
        # Obtaining the member 'getfilesystemencoding' of a type (line 842)
        getfilesystemencoding_50132 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 842, 23), sys_50131, 'getfilesystemencoding')
        # Calling getfilesystemencoding(args, kwargs) (line 842)
        getfilesystemencoding_call_result_50134 = invoke(stypy.reporting.localization.Localization(__file__, 842, 23), getfilesystemencoding_50132, *[], **kwargs_50133)
        
        unicode_50135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 842, 54), 'unicode', u'utf-8')
        # Applying the binary operator 'or' (line 842)
        result_or_keyword_50136 = python_operator(stypy.reporting.localization.Localization(__file__, 842, 23), 'or', getfilesystemencoding_call_result_50134, unicode_50135)
        
        # Assigning a type to the variable 'encoding' (line 842)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 842, 12), 'encoding', result_or_keyword_50136)
        
        # Assigning a Call to a Attribute (line 843):
        
        # Assigning a Call to a Attribute (line 843):
        
        # Call to decode(...): (line 843)
        # Processing the call arguments (line 843)
        # Getting the type of 'encoding' (line 843)
        encoding_50139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 843, 45), 'encoding', False)
        # Processing the call keyword arguments (line 843)
        unicode_50140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 843, 62), 'unicode', u'replace')
        keyword_50141 = unicode_50140
        kwargs_50142 = {'errors': keyword_50141}
        # Getting the type of 'filename' (line 843)
        filename_50137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 843, 29), 'filename', False)
        # Obtaining the member 'decode' of a type (line 843)
        decode_50138 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 843, 29), filename_50137, 'decode')
        # Calling decode(args, kwargs) (line 843)
        decode_call_result_50143 = invoke(stypy.reporting.localization.Localization(__file__, 843, 29), decode_50138, *[encoding_50139], **kwargs_50142)
        
        # Getting the type of 'self' (line 843)
        self_50144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 843, 12), 'self')
        # Setting the type of the member '_filename' of a type (line 843)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 843, 12), self_50144, '_filename', decode_call_result_50143)
        # SSA join for if statement (line 841)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to open(...): (line 844)
        # Processing the call arguments (line 844)
        # Getting the type of 'filename' (line 844)
        filename_50146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 844, 18), 'filename', False)
        unicode_50147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 844, 28), 'unicode', u'rb')
        # Processing the call keyword arguments (line 844)
        kwargs_50148 = {}
        # Getting the type of 'open' (line 844)
        open_50145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 844, 13), 'open', False)
        # Calling open(args, kwargs) (line 844)
        open_call_result_50149 = invoke(stypy.reporting.localization.Localization(__file__, 844, 13), open_50145, *[filename_50146, unicode_50147], **kwargs_50148)
        
        with_50150 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 844, 13), open_call_result_50149, 'with parameter', '__enter__', '__exit__')

        if with_50150:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 844)
            enter___50151 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 844, 13), open_call_result_50149, '__enter__')
            with_enter_50152 = invoke(stypy.reporting.localization.Localization(__file__, 844, 13), enter___50151)
            # Assigning a type to the variable 'file' (line 844)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 844, 13), 'file', with_enter_50152)
            
            # Call to _parse(...): (line 845)
            # Processing the call arguments (line 845)
            # Getting the type of 'file' (line 845)
            file_50155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 845, 24), 'file', False)
            # Processing the call keyword arguments (line 845)
            kwargs_50156 = {}
            # Getting the type of 'self' (line 845)
            self_50153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 845, 12), 'self', False)
            # Obtaining the member '_parse' of a type (line 845)
            _parse_50154 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 845, 12), self_50153, '_parse')
            # Calling _parse(args, kwargs) (line 845)
            _parse_call_result_50157 = invoke(stypy.reporting.localization.Localization(__file__, 845, 12), _parse_50154, *[file_50155], **kwargs_50156)
            
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 844)
            exit___50158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 844, 13), open_call_result_50149, '__exit__')
            with_exit_50159 = invoke(stypy.reporting.localization.Localization(__file__, 844, 13), exit___50158, None, None, None)

        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def __getitem__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__getitem__'
        module_type_store = module_type_store.open_function_context('__getitem__', 847, 4, False)
        # Assigning a type to the variable 'self' (line 848)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 848, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PsfontsMap.__getitem__.__dict__.__setitem__('stypy_localization', localization)
        PsfontsMap.__getitem__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PsfontsMap.__getitem__.__dict__.__setitem__('stypy_type_store', module_type_store)
        PsfontsMap.__getitem__.__dict__.__setitem__('stypy_function_name', 'PsfontsMap.__getitem__')
        PsfontsMap.__getitem__.__dict__.__setitem__('stypy_param_names_list', ['texname'])
        PsfontsMap.__getitem__.__dict__.__setitem__('stypy_varargs_param_name', None)
        PsfontsMap.__getitem__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        PsfontsMap.__getitem__.__dict__.__setitem__('stypy_call_defaults', defaults)
        PsfontsMap.__getitem__.__dict__.__setitem__('stypy_call_varargs', varargs)
        PsfontsMap.__getitem__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PsfontsMap.__getitem__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PsfontsMap.__getitem__', ['texname'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__getitem__', localization, ['texname'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__getitem__(...)' code ##################

        # Evaluating assert statement condition
        
        # Call to isinstance(...): (line 848)
        # Processing the call arguments (line 848)
        # Getting the type of 'texname' (line 848)
        texname_50161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 848, 26), 'texname', False)
        # Getting the type of 'bytes' (line 848)
        bytes_50162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 848, 35), 'bytes', False)
        # Processing the call keyword arguments (line 848)
        kwargs_50163 = {}
        # Getting the type of 'isinstance' (line 848)
        isinstance_50160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 848, 15), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 848)
        isinstance_call_result_50164 = invoke(stypy.reporting.localization.Localization(__file__, 848, 15), isinstance_50160, *[texname_50161, bytes_50162], **kwargs_50163)
        
        
        
        # SSA begins for try-except statement (line 849)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Subscript to a Name (line 850):
        
        # Assigning a Subscript to a Name (line 850):
        
        # Obtaining the type of the subscript
        # Getting the type of 'texname' (line 850)
        texname_50165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 850, 32), 'texname')
        # Getting the type of 'self' (line 850)
        self_50166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 850, 21), 'self')
        # Obtaining the member '_font' of a type (line 850)
        _font_50167 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 850, 21), self_50166, '_font')
        # Obtaining the member '__getitem__' of a type (line 850)
        getitem___50168 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 850, 21), _font_50167, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 850)
        subscript_call_result_50169 = invoke(stypy.reporting.localization.Localization(__file__, 850, 21), getitem___50168, texname_50165)
        
        # Assigning a type to the variable 'result' (line 850)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 850, 12), 'result', subscript_call_result_50169)
        # SSA branch for the except part of a try statement (line 849)
        # SSA branch for the except 'KeyError' branch of a try statement (line 849)
        module_type_store.open_ssa_branch('except')
        
        # Assigning a Str to a Name (line 852):
        
        # Assigning a Str to a Name (line 852):
        unicode_50170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 852, 19), 'unicode', u'A PostScript file for the font whose TeX name is "{0}" could not be found in the file "{1}". The dviread module can only handle fonts that have an associated PostScript font file. This problem can often be solved by installing a suitable PostScript font package in your (TeX) package manager.')
        # Assigning a type to the variable 'fmt' (line 852)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 852, 12), 'fmt', unicode_50170)
        
        # Assigning a Call to a Name (line 859):
        
        # Assigning a Call to a Name (line 859):
        
        # Call to format(...): (line 859)
        # Processing the call arguments (line 859)
        
        # Call to decode(...): (line 859)
        # Processing the call arguments (line 859)
        unicode_50175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 859, 44), 'unicode', u'ascii')
        # Processing the call keyword arguments (line 859)
        kwargs_50176 = {}
        # Getting the type of 'texname' (line 859)
        texname_50173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 859, 29), 'texname', False)
        # Obtaining the member 'decode' of a type (line 859)
        decode_50174 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 859, 29), texname_50173, 'decode')
        # Calling decode(args, kwargs) (line 859)
        decode_call_result_50177 = invoke(stypy.reporting.localization.Localization(__file__, 859, 29), decode_50174, *[unicode_50175], **kwargs_50176)
        
        # Getting the type of 'self' (line 859)
        self_50178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 859, 54), 'self', False)
        # Obtaining the member '_filename' of a type (line 859)
        _filename_50179 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 859, 54), self_50178, '_filename')
        # Processing the call keyword arguments (line 859)
        kwargs_50180 = {}
        # Getting the type of 'fmt' (line 859)
        fmt_50171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 859, 18), 'fmt', False)
        # Obtaining the member 'format' of a type (line 859)
        format_50172 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 859, 18), fmt_50171, 'format')
        # Calling format(args, kwargs) (line 859)
        format_call_result_50181 = invoke(stypy.reporting.localization.Localization(__file__, 859, 18), format_50172, *[decode_call_result_50177, _filename_50179], **kwargs_50180)
        
        # Assigning a type to the variable 'msg' (line 859)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 859, 12), 'msg', format_call_result_50181)
        
        # Assigning a Call to a Name (line 860):
        
        # Assigning a Call to a Name (line 860):
        
        # Call to fill(...): (line 860)
        # Processing the call arguments (line 860)
        # Getting the type of 'msg' (line 860)
        msg_50184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 860, 32), 'msg', False)
        # Processing the call keyword arguments (line 860)
        # Getting the type of 'False' (line 860)
        False_50185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 860, 54), 'False', False)
        keyword_50186 = False_50185
        # Getting the type of 'False' (line 861)
        False_50187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 861, 49), 'False', False)
        keyword_50188 = False_50187
        kwargs_50189 = {'break_on_hyphens': keyword_50186, 'break_long_words': keyword_50188}
        # Getting the type of 'textwrap' (line 860)
        textwrap_50182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 860, 18), 'textwrap', False)
        # Obtaining the member 'fill' of a type (line 860)
        fill_50183 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 860, 18), textwrap_50182, 'fill')
        # Calling fill(args, kwargs) (line 860)
        fill_call_result_50190 = invoke(stypy.reporting.localization.Localization(__file__, 860, 18), fill_50183, *[msg_50184], **kwargs_50189)
        
        # Assigning a type to the variable 'msg' (line 860)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 860, 12), 'msg', fill_call_result_50190)
        
        # Call to report(...): (line 862)
        # Processing the call arguments (line 862)
        # Getting the type of 'msg' (line 862)
        msg_50194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 862, 38), 'msg', False)
        unicode_50195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 862, 43), 'unicode', u'helpful')
        # Processing the call keyword arguments (line 862)
        kwargs_50196 = {}
        # Getting the type of 'matplotlib' (line 862)
        matplotlib_50191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 862, 12), 'matplotlib', False)
        # Obtaining the member 'verbose' of a type (line 862)
        verbose_50192 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 862, 12), matplotlib_50191, 'verbose')
        # Obtaining the member 'report' of a type (line 862)
        report_50193 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 862, 12), verbose_50192, 'report')
        # Calling report(args, kwargs) (line 862)
        report_call_result_50197 = invoke(stypy.reporting.localization.Localization(__file__, 862, 12), report_50193, *[msg_50194, unicode_50195], **kwargs_50196)
        
        # SSA join for try-except statement (line 849)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Tuple to a Tuple (line 864):
        
        # Assigning a Attribute to a Name (line 864):
        # Getting the type of 'result' (line 864)
        result_50198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 864, 18), 'result')
        # Obtaining the member 'filename' of a type (line 864)
        filename_50199 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 864, 18), result_50198, 'filename')
        # Assigning a type to the variable 'tuple_assignment_47840' (line 864)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 864, 8), 'tuple_assignment_47840', filename_50199)
        
        # Assigning a Attribute to a Name (line 864):
        # Getting the type of 'result' (line 864)
        result_50200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 864, 35), 'result')
        # Obtaining the member 'encoding' of a type (line 864)
        encoding_50201 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 864, 35), result_50200, 'encoding')
        # Assigning a type to the variable 'tuple_assignment_47841' (line 864)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 864, 8), 'tuple_assignment_47841', encoding_50201)
        
        # Assigning a Name to a Name (line 864):
        # Getting the type of 'tuple_assignment_47840' (line 864)
        tuple_assignment_47840_50202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 864, 8), 'tuple_assignment_47840')
        # Assigning a type to the variable 'fn' (line 864)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 864, 8), 'fn', tuple_assignment_47840_50202)
        
        # Assigning a Name to a Name (line 864):
        # Getting the type of 'tuple_assignment_47841' (line 864)
        tuple_assignment_47841_50203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 864, 8), 'tuple_assignment_47841')
        # Assigning a type to the variable 'enc' (line 864)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 864, 12), 'enc', tuple_assignment_47841_50203)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'fn' (line 865)
        fn_50204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 865, 11), 'fn')
        # Getting the type of 'None' (line 865)
        None_50205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 865, 21), 'None')
        # Applying the binary operator 'isnot' (line 865)
        result_is_not_50206 = python_operator(stypy.reporting.localization.Localization(__file__, 865, 11), 'isnot', fn_50204, None_50205)
        
        
        
        # Call to startswith(...): (line 865)
        # Processing the call arguments (line 865)
        str_50209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 865, 48), 'str', '/')
        # Processing the call keyword arguments (line 865)
        kwargs_50210 = {}
        # Getting the type of 'fn' (line 865)
        fn_50207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 865, 34), 'fn', False)
        # Obtaining the member 'startswith' of a type (line 865)
        startswith_50208 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 865, 34), fn_50207, 'startswith')
        # Calling startswith(args, kwargs) (line 865)
        startswith_call_result_50211 = invoke(stypy.reporting.localization.Localization(__file__, 865, 34), startswith_50208, *[str_50209], **kwargs_50210)
        
        # Applying the 'not' unary operator (line 865)
        result_not__50212 = python_operator(stypy.reporting.localization.Localization(__file__, 865, 30), 'not', startswith_call_result_50211)
        
        # Applying the binary operator 'and' (line 865)
        result_and_keyword_50213 = python_operator(stypy.reporting.localization.Localization(__file__, 865, 11), 'and', result_is_not_50206, result_not__50212)
        
        # Testing the type of an if condition (line 865)
        if_condition_50214 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 865, 8), result_and_keyword_50213)
        # Assigning a type to the variable 'if_condition_50214' (line 865)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 865, 8), 'if_condition_50214', if_condition_50214)
        # SSA begins for if statement (line 865)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 866):
        
        # Assigning a Call to a Name (line 866):
        
        # Call to find_tex_file(...): (line 866)
        # Processing the call arguments (line 866)
        # Getting the type of 'fn' (line 866)
        fn_50216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 866, 31), 'fn', False)
        # Processing the call keyword arguments (line 866)
        kwargs_50217 = {}
        # Getting the type of 'find_tex_file' (line 866)
        find_tex_file_50215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 866, 17), 'find_tex_file', False)
        # Calling find_tex_file(args, kwargs) (line 866)
        find_tex_file_call_result_50218 = invoke(stypy.reporting.localization.Localization(__file__, 866, 17), find_tex_file_50215, *[fn_50216], **kwargs_50217)
        
        # Assigning a type to the variable 'fn' (line 866)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 866, 12), 'fn', find_tex_file_call_result_50218)
        # SSA join for if statement (line 865)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'enc' (line 867)
        enc_50219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 867, 11), 'enc')
        # Getting the type of 'None' (line 867)
        None_50220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 867, 22), 'None')
        # Applying the binary operator 'isnot' (line 867)
        result_is_not_50221 = python_operator(stypy.reporting.localization.Localization(__file__, 867, 11), 'isnot', enc_50219, None_50220)
        
        
        
        # Call to startswith(...): (line 867)
        # Processing the call arguments (line 867)
        str_50224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 867, 50), 'str', '/')
        # Processing the call keyword arguments (line 867)
        kwargs_50225 = {}
        # Getting the type of 'enc' (line 867)
        enc_50222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 867, 35), 'enc', False)
        # Obtaining the member 'startswith' of a type (line 867)
        startswith_50223 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 867, 35), enc_50222, 'startswith')
        # Calling startswith(args, kwargs) (line 867)
        startswith_call_result_50226 = invoke(stypy.reporting.localization.Localization(__file__, 867, 35), startswith_50223, *[str_50224], **kwargs_50225)
        
        # Applying the 'not' unary operator (line 867)
        result_not__50227 = python_operator(stypy.reporting.localization.Localization(__file__, 867, 31), 'not', startswith_call_result_50226)
        
        # Applying the binary operator 'and' (line 867)
        result_and_keyword_50228 = python_operator(stypy.reporting.localization.Localization(__file__, 867, 11), 'and', result_is_not_50221, result_not__50227)
        
        # Testing the type of an if condition (line 867)
        if_condition_50229 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 867, 8), result_and_keyword_50228)
        # Assigning a type to the variable 'if_condition_50229' (line 867)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 867, 8), 'if_condition_50229', if_condition_50229)
        # SSA begins for if statement (line 867)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 868):
        
        # Assigning a Call to a Name (line 868):
        
        # Call to find_tex_file(...): (line 868)
        # Processing the call arguments (line 868)
        # Getting the type of 'result' (line 868)
        result_50231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 868, 32), 'result', False)
        # Obtaining the member 'encoding' of a type (line 868)
        encoding_50232 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 868, 32), result_50231, 'encoding')
        # Processing the call keyword arguments (line 868)
        kwargs_50233 = {}
        # Getting the type of 'find_tex_file' (line 868)
        find_tex_file_50230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 868, 18), 'find_tex_file', False)
        # Calling find_tex_file(args, kwargs) (line 868)
        find_tex_file_call_result_50234 = invoke(stypy.reporting.localization.Localization(__file__, 868, 18), find_tex_file_50230, *[encoding_50232], **kwargs_50233)
        
        # Assigning a type to the variable 'enc' (line 868)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 868, 12), 'enc', find_tex_file_call_result_50234)
        # SSA join for if statement (line 867)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to _replace(...): (line 869)
        # Processing the call keyword arguments (line 869)
        # Getting the type of 'fn' (line 869)
        fn_50237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 869, 40), 'fn', False)
        keyword_50238 = fn_50237
        # Getting the type of 'enc' (line 869)
        enc_50239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 869, 53), 'enc', False)
        keyword_50240 = enc_50239
        kwargs_50241 = {'encoding': keyword_50240, 'filename': keyword_50238}
        # Getting the type of 'result' (line 869)
        result_50235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 869, 15), 'result', False)
        # Obtaining the member '_replace' of a type (line 869)
        _replace_50236 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 869, 15), result_50235, '_replace')
        # Calling _replace(args, kwargs) (line 869)
        _replace_call_result_50242 = invoke(stypy.reporting.localization.Localization(__file__, 869, 15), _replace_50236, *[], **kwargs_50241)
        
        # Assigning a type to the variable 'stypy_return_type' (line 869)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 869, 8), 'stypy_return_type', _replace_call_result_50242)
        
        # ################# End of '__getitem__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__getitem__' in the type store
        # Getting the type of 'stypy_return_type' (line 847)
        stypy_return_type_50243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 847, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_50243)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__getitem__'
        return stypy_return_type_50243


    @norecursion
    def _parse(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_parse'
        module_type_store = module_type_store.open_function_context('_parse', 871, 4, False)
        # Assigning a type to the variable 'self' (line 872)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 872, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PsfontsMap._parse.__dict__.__setitem__('stypy_localization', localization)
        PsfontsMap._parse.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PsfontsMap._parse.__dict__.__setitem__('stypy_type_store', module_type_store)
        PsfontsMap._parse.__dict__.__setitem__('stypy_function_name', 'PsfontsMap._parse')
        PsfontsMap._parse.__dict__.__setitem__('stypy_param_names_list', ['file'])
        PsfontsMap._parse.__dict__.__setitem__('stypy_varargs_param_name', None)
        PsfontsMap._parse.__dict__.__setitem__('stypy_kwargs_param_name', None)
        PsfontsMap._parse.__dict__.__setitem__('stypy_call_defaults', defaults)
        PsfontsMap._parse.__dict__.__setitem__('stypy_call_varargs', varargs)
        PsfontsMap._parse.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PsfontsMap._parse.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PsfontsMap._parse', ['file'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_parse', localization, ['file'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_parse(...)' code ##################

        unicode_50244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 884, (-1)), 'unicode', u'\n        Parse the font mapping file.\n\n        The format is, AFAIK: texname fontname [effects and filenames]\n        Effects are PostScript snippets like ".177 SlantFont",\n        filenames begin with one or two less-than signs. A filename\n        ending in enc is an encoding file, other filenames are font\n        files. This can be overridden with a left bracket: <[foobar\n        indicates an encoding file named foobar.\n\n        There is some difference between <foo.pfb and <<bar.pfb in\n        subsetting, but I have no example of << in my TeX installation.\n        ')
        
        # Assigning a Call to a Name (line 891):
        
        # Assigning a Call to a Name (line 891):
        
        # Call to compile(...): (line 891)
        # Processing the call arguments (line 891)
        str_50247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 891, 30), 'str', '%|\\s*$')
        # Processing the call keyword arguments (line 891)
        kwargs_50248 = {}
        # Getting the type of 're' (line 891)
        re_50245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 891, 19), 're', False)
        # Obtaining the member 'compile' of a type (line 891)
        compile_50246 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 891, 19), re_50245, 'compile')
        # Calling compile(args, kwargs) (line 891)
        compile_call_result_50249 = invoke(stypy.reporting.localization.Localization(__file__, 891, 19), compile_50246, *[str_50247], **kwargs_50248)
        
        # Assigning a type to the variable 'empty_re' (line 891)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 891, 8), 'empty_re', compile_call_result_50249)
        
        # Assigning a Call to a Name (line 892):
        
        # Assigning a Call to a Name (line 892):
        
        # Call to compile(...): (line 892)
        # Processing the call arguments (line 892)
        str_50252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 902, (-1)), 'str', '(?x) (?:\n                 "<\\[ (?P<enc1>  [^"]+    )" | # quoted encoding marked by [\n                 "<   (?P<enc2>  [^"]+.enc)" | # quoted encoding, ends in .enc\n                 "<<? (?P<file1> [^"]+    )" | # quoted font file name\n                 "    (?P<eff1>  [^"]+    )" | # quoted effects or font name\n                 <\\[  (?P<enc3>  \\S+      )  | # encoding marked by [\n                 <    (?P<enc4>  \\S+  .enc)  | # encoding, ends in .enc\n                 <<?  (?P<file2> \\S+      )  | # font file name\n                      (?P<eff2>  \\S+      )    # effects or font name\n            )')
        # Processing the call keyword arguments (line 892)
        kwargs_50253 = {}
        # Getting the type of 're' (line 892)
        re_50250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 892, 18), 're', False)
        # Obtaining the member 'compile' of a type (line 892)
        compile_50251 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 892, 18), re_50250, 'compile')
        # Calling compile(args, kwargs) (line 892)
        compile_call_result_50254 = invoke(stypy.reporting.localization.Localization(__file__, 892, 18), compile_50251, *[str_50252], **kwargs_50253)
        
        # Assigning a type to the variable 'word_re' (line 892)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 892, 8), 'word_re', compile_call_result_50254)
        
        # Assigning a Call to a Name (line 903):
        
        # Assigning a Call to a Name (line 903):
        
        # Call to compile(...): (line 903)
        # Processing the call arguments (line 903)
        str_50257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 905, (-1)), 'str', '(?x) (?P<slant> -?[0-9]*(?:\\.[0-9]+)) \\s* SlantFont\n                    | (?P<extend>-?[0-9]*(?:\\.[0-9]+)) \\s* ExtendFont')
        # Processing the call keyword arguments (line 903)
        kwargs_50258 = {}
        # Getting the type of 're' (line 903)
        re_50255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 903, 21), 're', False)
        # Obtaining the member 'compile' of a type (line 903)
        compile_50256 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 903, 21), re_50255, 'compile')
        # Calling compile(args, kwargs) (line 903)
        compile_call_result_50259 = invoke(stypy.reporting.localization.Localization(__file__, 903, 21), compile_50256, *[str_50257], **kwargs_50258)
        
        # Assigning a type to the variable 'effects_re' (line 903)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 903, 8), 'effects_re', compile_call_result_50259)
        
        # Assigning a GeneratorExp to a Name (line 907):
        
        # Assigning a GeneratorExp to a Name (line 907):
        # Calculating generator expression
        module_type_store = module_type_store.open_function_context('list comprehension expression', 907, 17, True)
        # Calculating comprehension expression
        # Getting the type of 'file' (line 908)
        file_50270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 908, 29), 'file')
        comprehension_50271 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 907, 17), file_50270)
        # Assigning a type to the variable 'line' (line 907)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 907, 17), 'line', comprehension_50271)
        
        
        # Call to match(...): (line 909)
        # Processing the call arguments (line 909)
        # Getting the type of 'line' (line 909)
        line_50266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 909, 39), 'line', False)
        # Processing the call keyword arguments (line 909)
        kwargs_50267 = {}
        # Getting the type of 'empty_re' (line 909)
        empty_re_50264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 909, 24), 'empty_re', False)
        # Obtaining the member 'match' of a type (line 909)
        match_50265 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 909, 24), empty_re_50264, 'match')
        # Calling match(args, kwargs) (line 909)
        match_call_result_50268 = invoke(stypy.reporting.localization.Localization(__file__, 909, 24), match_50265, *[line_50266], **kwargs_50267)
        
        # Applying the 'not' unary operator (line 909)
        result_not__50269 = python_operator(stypy.reporting.localization.Localization(__file__, 909, 20), 'not', match_call_result_50268)
        
        
        # Call to strip(...): (line 907)
        # Processing the call keyword arguments (line 907)
        kwargs_50262 = {}
        # Getting the type of 'line' (line 907)
        line_50260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 907, 17), 'line', False)
        # Obtaining the member 'strip' of a type (line 907)
        strip_50261 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 907, 17), line_50260, 'strip')
        # Calling strip(args, kwargs) (line 907)
        strip_call_result_50263 = invoke(stypy.reporting.localization.Localization(__file__, 907, 17), strip_50261, *[], **kwargs_50262)
        
        list_50272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 907, 17), 'list')
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 907, 17), list_50272, strip_call_result_50263)
        # Assigning a type to the variable 'lines' (line 907)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 907, 8), 'lines', list_50272)
        
        # Getting the type of 'lines' (line 910)
        lines_50273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 910, 20), 'lines')
        # Testing the type of a for loop iterable (line 910)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 910, 8), lines_50273)
        # Getting the type of the for loop variable (line 910)
        for_loop_var_50274 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 910, 8), lines_50273)
        # Assigning a type to the variable 'line' (line 910)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 910, 8), 'line', for_loop_var_50274)
        # SSA begins for a for statement (line 910)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Tuple to a Tuple (line 911):
        
        # Assigning a Str to a Name (line 911):
        str_50275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 911, 42), 'str', '')
        # Assigning a type to the variable 'tuple_assignment_47842' (line 911)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 911, 12), 'tuple_assignment_47842', str_50275)
        
        # Assigning a Name to a Name (line 911):
        # Getting the type of 'None' (line 911)
        None_50276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 911, 47), 'None')
        # Assigning a type to the variable 'tuple_assignment_47843' (line 911)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 911, 12), 'tuple_assignment_47843', None_50276)
        
        # Assigning a Name to a Name (line 911):
        # Getting the type of 'None' (line 911)
        None_50277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 911, 53), 'None')
        # Assigning a type to the variable 'tuple_assignment_47844' (line 911)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 911, 12), 'tuple_assignment_47844', None_50277)
        
        # Assigning a Name to a Name (line 911):
        # Getting the type of 'tuple_assignment_47842' (line 911)
        tuple_assignment_47842_50278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 911, 12), 'tuple_assignment_47842')
        # Assigning a type to the variable 'effects' (line 911)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 911, 12), 'effects', tuple_assignment_47842_50278)
        
        # Assigning a Name to a Name (line 911):
        # Getting the type of 'tuple_assignment_47843' (line 911)
        tuple_assignment_47843_50279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 911, 12), 'tuple_assignment_47843')
        # Assigning a type to the variable 'encoding' (line 911)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 911, 21), 'encoding', tuple_assignment_47843_50279)
        
        # Assigning a Name to a Name (line 911):
        # Getting the type of 'tuple_assignment_47844' (line 911)
        tuple_assignment_47844_50280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 911, 12), 'tuple_assignment_47844')
        # Assigning a type to the variable 'filename' (line 911)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 911, 31), 'filename', tuple_assignment_47844_50280)
        
        # Assigning a Call to a Name (line 912):
        
        # Assigning a Call to a Name (line 912):
        
        # Call to finditer(...): (line 912)
        # Processing the call arguments (line 912)
        # Getting the type of 'line' (line 912)
        line_50283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 912, 37), 'line', False)
        # Processing the call keyword arguments (line 912)
        kwargs_50284 = {}
        # Getting the type of 'word_re' (line 912)
        word_re_50281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 912, 20), 'word_re', False)
        # Obtaining the member 'finditer' of a type (line 912)
        finditer_50282 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 912, 20), word_re_50281, 'finditer')
        # Calling finditer(args, kwargs) (line 912)
        finditer_call_result_50285 = invoke(stypy.reporting.localization.Localization(__file__, 912, 20), finditer_50282, *[line_50283], **kwargs_50284)
        
        # Assigning a type to the variable 'words' (line 912)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 912, 12), 'words', finditer_call_result_50285)
        
        # Assigning a Call to a Name (line 918):
        
        # Assigning a Call to a Name (line 918):
        
        # Call to next(...): (line 918)
        # Processing the call arguments (line 918)
        # Getting the type of 'words' (line 918)
        words_50287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 918, 21), 'words', False)
        # Processing the call keyword arguments (line 918)
        kwargs_50288 = {}
        # Getting the type of 'next' (line 918)
        next_50286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 918, 16), 'next', False)
        # Calling next(args, kwargs) (line 918)
        next_call_result_50289 = invoke(stypy.reporting.localization.Localization(__file__, 918, 16), next_50286, *[words_50287], **kwargs_50288)
        
        # Assigning a type to the variable 'w' (line 918)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 918, 12), 'w', next_call_result_50289)
        
        # Assigning a BoolOp to a Name (line 919):
        
        # Assigning a BoolOp to a Name (line 919):
        
        # Evaluating a boolean operation
        
        # Call to group(...): (line 919)
        # Processing the call arguments (line 919)
        unicode_50292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 919, 30), 'unicode', u'eff2')
        # Processing the call keyword arguments (line 919)
        kwargs_50293 = {}
        # Getting the type of 'w' (line 919)
        w_50290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 919, 22), 'w', False)
        # Obtaining the member 'group' of a type (line 919)
        group_50291 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 919, 22), w_50290, 'group')
        # Calling group(args, kwargs) (line 919)
        group_call_result_50294 = invoke(stypy.reporting.localization.Localization(__file__, 919, 22), group_50291, *[unicode_50292], **kwargs_50293)
        
        
        # Call to group(...): (line 919)
        # Processing the call arguments (line 919)
        unicode_50297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 919, 49), 'unicode', u'eff1')
        # Processing the call keyword arguments (line 919)
        kwargs_50298 = {}
        # Getting the type of 'w' (line 919)
        w_50295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 919, 41), 'w', False)
        # Obtaining the member 'group' of a type (line 919)
        group_50296 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 919, 41), w_50295, 'group')
        # Calling group(args, kwargs) (line 919)
        group_call_result_50299 = invoke(stypy.reporting.localization.Localization(__file__, 919, 41), group_50296, *[unicode_50297], **kwargs_50298)
        
        # Applying the binary operator 'or' (line 919)
        result_or_keyword_50300 = python_operator(stypy.reporting.localization.Localization(__file__, 919, 22), 'or', group_call_result_50294, group_call_result_50299)
        
        # Assigning a type to the variable 'texname' (line 919)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 919, 12), 'texname', result_or_keyword_50300)
        
        # Assigning a Call to a Name (line 920):
        
        # Assigning a Call to a Name (line 920):
        
        # Call to next(...): (line 920)
        # Processing the call arguments (line 920)
        # Getting the type of 'words' (line 920)
        words_50302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 920, 21), 'words', False)
        # Processing the call keyword arguments (line 920)
        kwargs_50303 = {}
        # Getting the type of 'next' (line 920)
        next_50301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 920, 16), 'next', False)
        # Calling next(args, kwargs) (line 920)
        next_call_result_50304 = invoke(stypy.reporting.localization.Localization(__file__, 920, 16), next_50301, *[words_50302], **kwargs_50303)
        
        # Assigning a type to the variable 'w' (line 920)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 920, 12), 'w', next_call_result_50304)
        
        # Assigning a BoolOp to a Name (line 921):
        
        # Assigning a BoolOp to a Name (line 921):
        
        # Evaluating a boolean operation
        
        # Call to group(...): (line 921)
        # Processing the call arguments (line 921)
        unicode_50307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 921, 29), 'unicode', u'eff2')
        # Processing the call keyword arguments (line 921)
        kwargs_50308 = {}
        # Getting the type of 'w' (line 921)
        w_50305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 921, 21), 'w', False)
        # Obtaining the member 'group' of a type (line 921)
        group_50306 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 921, 21), w_50305, 'group')
        # Calling group(args, kwargs) (line 921)
        group_call_result_50309 = invoke(stypy.reporting.localization.Localization(__file__, 921, 21), group_50306, *[unicode_50307], **kwargs_50308)
        
        
        # Call to group(...): (line 921)
        # Processing the call arguments (line 921)
        unicode_50312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 921, 48), 'unicode', u'eff1')
        # Processing the call keyword arguments (line 921)
        kwargs_50313 = {}
        # Getting the type of 'w' (line 921)
        w_50310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 921, 40), 'w', False)
        # Obtaining the member 'group' of a type (line 921)
        group_50311 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 921, 40), w_50310, 'group')
        # Calling group(args, kwargs) (line 921)
        group_call_result_50314 = invoke(stypy.reporting.localization.Localization(__file__, 921, 40), group_50311, *[unicode_50312], **kwargs_50313)
        
        # Applying the binary operator 'or' (line 921)
        result_or_keyword_50315 = python_operator(stypy.reporting.localization.Localization(__file__, 921, 21), 'or', group_call_result_50309, group_call_result_50314)
        
        # Assigning a type to the variable 'psname' (line 921)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 921, 12), 'psname', result_or_keyword_50315)
        
        # Getting the type of 'words' (line 923)
        words_50316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 923, 21), 'words')
        # Testing the type of a for loop iterable (line 923)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 923, 12), words_50316)
        # Getting the type of the for loop variable (line 923)
        for_loop_var_50317 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 923, 12), words_50316)
        # Assigning a type to the variable 'w' (line 923)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 923, 12), 'w', for_loop_var_50317)
        # SSA begins for a for statement (line 923)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a BoolOp to a Name (line 925):
        
        # Assigning a BoolOp to a Name (line 925):
        
        # Evaluating a boolean operation
        
        # Call to group(...): (line 925)
        # Processing the call arguments (line 925)
        unicode_50320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 925, 30), 'unicode', u'eff1')
        # Processing the call keyword arguments (line 925)
        kwargs_50321 = {}
        # Getting the type of 'w' (line 925)
        w_50318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 925, 22), 'w', False)
        # Obtaining the member 'group' of a type (line 925)
        group_50319 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 925, 22), w_50318, 'group')
        # Calling group(args, kwargs) (line 925)
        group_call_result_50322 = invoke(stypy.reporting.localization.Localization(__file__, 925, 22), group_50319, *[unicode_50320], **kwargs_50321)
        
        
        # Call to group(...): (line 925)
        # Processing the call arguments (line 925)
        unicode_50325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 925, 49), 'unicode', u'eff2')
        # Processing the call keyword arguments (line 925)
        kwargs_50326 = {}
        # Getting the type of 'w' (line 925)
        w_50323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 925, 41), 'w', False)
        # Obtaining the member 'group' of a type (line 925)
        group_50324 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 925, 41), w_50323, 'group')
        # Calling group(args, kwargs) (line 925)
        group_call_result_50327 = invoke(stypy.reporting.localization.Localization(__file__, 925, 41), group_50324, *[unicode_50325], **kwargs_50326)
        
        # Applying the binary operator 'or' (line 925)
        result_or_keyword_50328 = python_operator(stypy.reporting.localization.Localization(__file__, 925, 22), 'or', group_call_result_50322, group_call_result_50327)
        
        # Assigning a type to the variable 'eff' (line 925)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 925, 16), 'eff', result_or_keyword_50328)
        
        # Getting the type of 'eff' (line 926)
        eff_50329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 926, 19), 'eff')
        # Testing the type of an if condition (line 926)
        if_condition_50330 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 926, 16), eff_50329)
        # Assigning a type to the variable 'if_condition_50330' (line 926)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 926, 16), 'if_condition_50330', if_condition_50330)
        # SSA begins for if statement (line 926)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 927):
        
        # Assigning a Name to a Name (line 927):
        # Getting the type of 'eff' (line 927)
        eff_50331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 927, 30), 'eff')
        # Assigning a type to the variable 'effects' (line 927)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 927, 20), 'effects', eff_50331)
        # SSA join for if statement (line 926)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BoolOp to a Name (line 931):
        
        # Assigning a BoolOp to a Name (line 931):
        
        # Evaluating a boolean operation
        
        # Call to group(...): (line 931)
        # Processing the call arguments (line 931)
        unicode_50334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 931, 31), 'unicode', u'enc4')
        # Processing the call keyword arguments (line 931)
        kwargs_50335 = {}
        # Getting the type of 'w' (line 931)
        w_50332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 931, 23), 'w', False)
        # Obtaining the member 'group' of a type (line 931)
        group_50333 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 931, 23), w_50332, 'group')
        # Calling group(args, kwargs) (line 931)
        group_call_result_50336 = invoke(stypy.reporting.localization.Localization(__file__, 931, 23), group_50333, *[unicode_50334], **kwargs_50335)
        
        
        # Call to group(...): (line 931)
        # Processing the call arguments (line 931)
        unicode_50339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 931, 50), 'unicode', u'enc3')
        # Processing the call keyword arguments (line 931)
        kwargs_50340 = {}
        # Getting the type of 'w' (line 931)
        w_50337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 931, 42), 'w', False)
        # Obtaining the member 'group' of a type (line 931)
        group_50338 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 931, 42), w_50337, 'group')
        # Calling group(args, kwargs) (line 931)
        group_call_result_50341 = invoke(stypy.reporting.localization.Localization(__file__, 931, 42), group_50338, *[unicode_50339], **kwargs_50340)
        
        # Applying the binary operator 'or' (line 931)
        result_or_keyword_50342 = python_operator(stypy.reporting.localization.Localization(__file__, 931, 23), 'or', group_call_result_50336, group_call_result_50341)
        
        # Call to group(...): (line 932)
        # Processing the call arguments (line 932)
        unicode_50345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 932, 31), 'unicode', u'enc2')
        # Processing the call keyword arguments (line 932)
        kwargs_50346 = {}
        # Getting the type of 'w' (line 932)
        w_50343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 932, 23), 'w', False)
        # Obtaining the member 'group' of a type (line 932)
        group_50344 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 932, 23), w_50343, 'group')
        # Calling group(args, kwargs) (line 932)
        group_call_result_50347 = invoke(stypy.reporting.localization.Localization(__file__, 932, 23), group_50344, *[unicode_50345], **kwargs_50346)
        
        # Applying the binary operator 'or' (line 931)
        result_or_keyword_50348 = python_operator(stypy.reporting.localization.Localization(__file__, 931, 23), 'or', result_or_keyword_50342, group_call_result_50347)
        
        # Call to group(...): (line 932)
        # Processing the call arguments (line 932)
        unicode_50351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 932, 50), 'unicode', u'enc1')
        # Processing the call keyword arguments (line 932)
        kwargs_50352 = {}
        # Getting the type of 'w' (line 932)
        w_50349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 932, 42), 'w', False)
        # Obtaining the member 'group' of a type (line 932)
        group_50350 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 932, 42), w_50349, 'group')
        # Calling group(args, kwargs) (line 932)
        group_call_result_50353 = invoke(stypy.reporting.localization.Localization(__file__, 932, 42), group_50350, *[unicode_50351], **kwargs_50352)
        
        # Applying the binary operator 'or' (line 931)
        result_or_keyword_50354 = python_operator(stypy.reporting.localization.Localization(__file__, 931, 23), 'or', result_or_keyword_50348, group_call_result_50353)
        
        # Assigning a type to the variable 'enc' (line 931)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 931, 16), 'enc', result_or_keyword_50354)
        
        # Getting the type of 'enc' (line 933)
        enc_50355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 933, 19), 'enc')
        # Testing the type of an if condition (line 933)
        if_condition_50356 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 933, 16), enc_50355)
        # Assigning a type to the variable 'if_condition_50356' (line 933)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 933, 16), 'if_condition_50356', if_condition_50356)
        # SSA begins for if statement (line 933)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Type idiom detected: calculating its left and rigth part (line 934)
        # Getting the type of 'encoding' (line 934)
        encoding_50357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 934, 20), 'encoding')
        # Getting the type of 'None' (line 934)
        None_50358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 934, 39), 'None')
        
        (may_be_50359, more_types_in_union_50360) = may_not_be_none(encoding_50357, None_50358)

        if may_be_50359:

            if more_types_in_union_50360:
                # Runtime conditional SSA (line 934)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to report(...): (line 935)
            # Processing the call arguments (line 935)
            unicode_50364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 936, 28), 'unicode', u'Multiple encodings for %s = %s')
            
            # Obtaining an instance of the builtin type 'tuple' (line 937)
            tuple_50365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 937, 31), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 937)
            # Adding element type (line 937)
            # Getting the type of 'texname' (line 937)
            texname_50366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 937, 31), 'texname', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 937, 31), tuple_50365, texname_50366)
            # Adding element type (line 937)
            # Getting the type of 'psname' (line 937)
            psname_50367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 937, 40), 'psname', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 937, 31), tuple_50365, psname_50367)
            
            # Applying the binary operator '%' (line 936)
            result_mod_50368 = python_operator(stypy.reporting.localization.Localization(__file__, 936, 28), '%', unicode_50364, tuple_50365)
            
            unicode_50369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 938, 28), 'unicode', u'debug')
            # Processing the call keyword arguments (line 935)
            kwargs_50370 = {}
            # Getting the type of 'matplotlib' (line 935)
            matplotlib_50361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 935, 24), 'matplotlib', False)
            # Obtaining the member 'verbose' of a type (line 935)
            verbose_50362 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 935, 24), matplotlib_50361, 'verbose')
            # Obtaining the member 'report' of a type (line 935)
            report_50363 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 935, 24), verbose_50362, 'report')
            # Calling report(args, kwargs) (line 935)
            report_call_result_50371 = invoke(stypy.reporting.localization.Localization(__file__, 935, 24), report_50363, *[result_mod_50368, unicode_50369], **kwargs_50370)
            

            if more_types_in_union_50360:
                # SSA join for if statement (line 934)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Name to a Name (line 939):
        
        # Assigning a Name to a Name (line 939):
        # Getting the type of 'enc' (line 939)
        enc_50372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 939, 31), 'enc')
        # Assigning a type to the variable 'encoding' (line 939)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 939, 20), 'encoding', enc_50372)
        # SSA join for if statement (line 933)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BoolOp to a Name (line 942):
        
        # Assigning a BoolOp to a Name (line 942):
        
        # Evaluating a boolean operation
        
        # Call to group(...): (line 942)
        # Processing the call arguments (line 942)
        unicode_50375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 942, 35), 'unicode', u'file2')
        # Processing the call keyword arguments (line 942)
        kwargs_50376 = {}
        # Getting the type of 'w' (line 942)
        w_50373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 942, 27), 'w', False)
        # Obtaining the member 'group' of a type (line 942)
        group_50374 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 942, 27), w_50373, 'group')
        # Calling group(args, kwargs) (line 942)
        group_call_result_50377 = invoke(stypy.reporting.localization.Localization(__file__, 942, 27), group_50374, *[unicode_50375], **kwargs_50376)
        
        
        # Call to group(...): (line 942)
        # Processing the call arguments (line 942)
        unicode_50380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 942, 55), 'unicode', u'file1')
        # Processing the call keyword arguments (line 942)
        kwargs_50381 = {}
        # Getting the type of 'w' (line 942)
        w_50378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 942, 47), 'w', False)
        # Obtaining the member 'group' of a type (line 942)
        group_50379 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 942, 47), w_50378, 'group')
        # Calling group(args, kwargs) (line 942)
        group_call_result_50382 = invoke(stypy.reporting.localization.Localization(__file__, 942, 47), group_50379, *[unicode_50380], **kwargs_50381)
        
        # Applying the binary operator 'or' (line 942)
        result_or_keyword_50383 = python_operator(stypy.reporting.localization.Localization(__file__, 942, 27), 'or', group_call_result_50377, group_call_result_50382)
        
        # Assigning a type to the variable 'filename' (line 942)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 942, 16), 'filename', result_or_keyword_50383)
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Dict to a Name (line 944):
        
        # Assigning a Dict to a Name (line 944):
        
        # Obtaining an instance of the builtin type 'dict' (line 944)
        dict_50384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 944, 27), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 944)
        
        # Assigning a type to the variable 'effects_dict' (line 944)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 944, 12), 'effects_dict', dict_50384)
        
        
        # Call to finditer(...): (line 945)
        # Processing the call arguments (line 945)
        # Getting the type of 'effects' (line 945)
        effects_50387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 945, 45), 'effects', False)
        # Processing the call keyword arguments (line 945)
        kwargs_50388 = {}
        # Getting the type of 'effects_re' (line 945)
        effects_re_50385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 945, 25), 'effects_re', False)
        # Obtaining the member 'finditer' of a type (line 945)
        finditer_50386 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 945, 25), effects_re_50385, 'finditer')
        # Calling finditer(args, kwargs) (line 945)
        finditer_call_result_50389 = invoke(stypy.reporting.localization.Localization(__file__, 945, 25), finditer_50386, *[effects_50387], **kwargs_50388)
        
        # Testing the type of a for loop iterable (line 945)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 945, 12), finditer_call_result_50389)
        # Getting the type of the for loop variable (line 945)
        for_loop_var_50390 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 945, 12), finditer_call_result_50389)
        # Assigning a type to the variable 'match' (line 945)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 945, 12), 'match', for_loop_var_50390)
        # SSA begins for a for statement (line 945)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 946):
        
        # Assigning a Call to a Name (line 946):
        
        # Call to group(...): (line 946)
        # Processing the call arguments (line 946)
        unicode_50393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 946, 36), 'unicode', u'slant')
        # Processing the call keyword arguments (line 946)
        kwargs_50394 = {}
        # Getting the type of 'match' (line 946)
        match_50391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 946, 24), 'match', False)
        # Obtaining the member 'group' of a type (line 946)
        group_50392 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 946, 24), match_50391, 'group')
        # Calling group(args, kwargs) (line 946)
        group_call_result_50395 = invoke(stypy.reporting.localization.Localization(__file__, 946, 24), group_50392, *[unicode_50393], **kwargs_50394)
        
        # Assigning a type to the variable 'slant' (line 946)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 946, 16), 'slant', group_call_result_50395)
        
        # Getting the type of 'slant' (line 947)
        slant_50396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 947, 19), 'slant')
        # Testing the type of an if condition (line 947)
        if_condition_50397 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 947, 16), slant_50396)
        # Assigning a type to the variable 'if_condition_50397' (line 947)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 947, 16), 'if_condition_50397', if_condition_50397)
        # SSA begins for if statement (line 947)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Subscript (line 948):
        
        # Assigning a Call to a Subscript (line 948):
        
        # Call to float(...): (line 948)
        # Processing the call arguments (line 948)
        # Getting the type of 'slant' (line 948)
        slant_50399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 948, 50), 'slant', False)
        # Processing the call keyword arguments (line 948)
        kwargs_50400 = {}
        # Getting the type of 'float' (line 948)
        float_50398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 948, 44), 'float', False)
        # Calling float(args, kwargs) (line 948)
        float_call_result_50401 = invoke(stypy.reporting.localization.Localization(__file__, 948, 44), float_50398, *[slant_50399], **kwargs_50400)
        
        # Getting the type of 'effects_dict' (line 948)
        effects_dict_50402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 948, 20), 'effects_dict')
        unicode_50403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 948, 33), 'unicode', u'slant')
        # Storing an element on a container (line 948)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 948, 20), effects_dict_50402, (unicode_50403, float_call_result_50401))
        # SSA branch for the else part of an if statement (line 947)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Subscript (line 950):
        
        # Assigning a Call to a Subscript (line 950):
        
        # Call to float(...): (line 950)
        # Processing the call arguments (line 950)
        
        # Call to group(...): (line 950)
        # Processing the call arguments (line 950)
        unicode_50407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 950, 63), 'unicode', u'extend')
        # Processing the call keyword arguments (line 950)
        kwargs_50408 = {}
        # Getting the type of 'match' (line 950)
        match_50405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 950, 51), 'match', False)
        # Obtaining the member 'group' of a type (line 950)
        group_50406 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 950, 51), match_50405, 'group')
        # Calling group(args, kwargs) (line 950)
        group_call_result_50409 = invoke(stypy.reporting.localization.Localization(__file__, 950, 51), group_50406, *[unicode_50407], **kwargs_50408)
        
        # Processing the call keyword arguments (line 950)
        kwargs_50410 = {}
        # Getting the type of 'float' (line 950)
        float_50404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 950, 45), 'float', False)
        # Calling float(args, kwargs) (line 950)
        float_call_result_50411 = invoke(stypy.reporting.localization.Localization(__file__, 950, 45), float_50404, *[group_call_result_50409], **kwargs_50410)
        
        # Getting the type of 'effects_dict' (line 950)
        effects_dict_50412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 950, 20), 'effects_dict')
        unicode_50413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 950, 33), 'unicode', u'extend')
        # Storing an element on a container (line 950)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 950, 20), effects_dict_50412, (unicode_50413, float_call_result_50411))
        # SSA join for if statement (line 947)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Subscript (line 952):
        
        # Assigning a Call to a Subscript (line 952):
        
        # Call to PsFont(...): (line 952)
        # Processing the call keyword arguments (line 952)
        # Getting the type of 'texname' (line 953)
        texname_50415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 953, 24), 'texname', False)
        keyword_50416 = texname_50415
        # Getting the type of 'psname' (line 953)
        psname_50417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 953, 40), 'psname', False)
        keyword_50418 = psname_50417
        # Getting the type of 'effects_dict' (line 953)
        effects_dict_50419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 953, 56), 'effects_dict', False)
        keyword_50420 = effects_dict_50419
        # Getting the type of 'encoding' (line 954)
        encoding_50421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 954, 25), 'encoding', False)
        keyword_50422 = encoding_50421
        # Getting the type of 'filename' (line 954)
        filename_50423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 954, 44), 'filename', False)
        keyword_50424 = filename_50423
        kwargs_50425 = {'psname': keyword_50418, 'texname': keyword_50416, 'filename': keyword_50424, 'effects': keyword_50420, 'encoding': keyword_50422}
        # Getting the type of 'PsFont' (line 952)
        PsFont_50414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 952, 34), 'PsFont', False)
        # Calling PsFont(args, kwargs) (line 952)
        PsFont_call_result_50426 = invoke(stypy.reporting.localization.Localization(__file__, 952, 34), PsFont_50414, *[], **kwargs_50425)
        
        # Getting the type of 'self' (line 952)
        self_50427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 952, 12), 'self')
        # Obtaining the member '_font' of a type (line 952)
        _font_50428 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 952, 12), self_50427, '_font')
        # Getting the type of 'texname' (line 952)
        texname_50429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 952, 23), 'texname')
        # Storing an element on a container (line 952)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 952, 12), _font_50428, (texname_50429, PsFont_call_result_50426))
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_parse(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_parse' in the type store
        # Getting the type of 'stypy_return_type' (line 871)
        stypy_return_type_50430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 871, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_50430)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_parse'
        return stypy_return_type_50430


# Assigning a type to the variable 'PsfontsMap' (line 790)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 790, 0), 'PsfontsMap', PsfontsMap)

# Assigning a Tuple to a Name (line 836):

# Obtaining an instance of the builtin type 'tuple' (line 836)
tuple_50431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 836, 17), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 836)
# Adding element type (line 836)
unicode_50432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 836, 17), 'unicode', u'_font')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 836, 17), tuple_50431, unicode_50432)
# Adding element type (line 836)
unicode_50433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 836, 26), 'unicode', u'_filename')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 836, 17), tuple_50431, unicode_50433)

# Getting the type of 'PsfontsMap'
PsfontsMap_50434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'PsfontsMap')
# Setting the type of the member '__slots__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), PsfontsMap_50434, '__slots__', tuple_50431)
# Declaration of the 'Encoding' class

class Encoding(object, ):
    unicode_50435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 976, (-1)), 'unicode', u'\n    Parses a \\*.enc file referenced from a psfonts.map style file.\n    The format this class understands is a very limited subset of\n    PostScript.\n\n    Usage (subject to change)::\n\n      for name in Encoding(filename):\n          whatever(name)\n\n    Parameters\n    ----------\n    filename : string or bytestring\n\n    Attributes\n    ----------\n    encoding : list\n        List of character names\n    ')
    
    # Assigning a Tuple to a Name (line 977):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 979, 4, False)
        # Assigning a type to the variable 'self' (line 980)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 980, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Encoding.__init__', ['filename'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['filename'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Call to open(...): (line 980)
        # Processing the call arguments (line 980)
        # Getting the type of 'filename' (line 980)
        filename_50437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 980, 18), 'filename', False)
        unicode_50438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 980, 28), 'unicode', u'rb')
        # Processing the call keyword arguments (line 980)
        kwargs_50439 = {}
        # Getting the type of 'open' (line 980)
        open_50436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 980, 13), 'open', False)
        # Calling open(args, kwargs) (line 980)
        open_call_result_50440 = invoke(stypy.reporting.localization.Localization(__file__, 980, 13), open_50436, *[filename_50437, unicode_50438], **kwargs_50439)
        
        with_50441 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 980, 13), open_call_result_50440, 'with parameter', '__enter__', '__exit__')

        if with_50441:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 980)
            enter___50442 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 980, 13), open_call_result_50440, '__enter__')
            with_enter_50443 = invoke(stypy.reporting.localization.Localization(__file__, 980, 13), enter___50442)
            # Assigning a type to the variable 'file' (line 980)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 980, 13), 'file', with_enter_50443)
            
            # Call to report(...): (line 981)
            # Processing the call arguments (line 981)
            unicode_50447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 981, 38), 'unicode', u'Parsing TeX encoding ')
            # Getting the type of 'filename' (line 981)
            filename_50448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 981, 64), 'filename', False)
            # Applying the binary operator '+' (line 981)
            result_add_50449 = python_operator(stypy.reporting.localization.Localization(__file__, 981, 38), '+', unicode_50447, filename_50448)
            
            unicode_50450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 982, 38), 'unicode', u'debug-annoying')
            # Processing the call keyword arguments (line 981)
            kwargs_50451 = {}
            # Getting the type of 'matplotlib' (line 981)
            matplotlib_50444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 981, 12), 'matplotlib', False)
            # Obtaining the member 'verbose' of a type (line 981)
            verbose_50445 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 981, 12), matplotlib_50444, 'verbose')
            # Obtaining the member 'report' of a type (line 981)
            report_50446 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 981, 12), verbose_50445, 'report')
            # Calling report(args, kwargs) (line 981)
            report_call_result_50452 = invoke(stypy.reporting.localization.Localization(__file__, 981, 12), report_50446, *[result_add_50449, unicode_50450], **kwargs_50451)
            
            
            # Assigning a Call to a Attribute (line 983):
            
            # Assigning a Call to a Attribute (line 983):
            
            # Call to _parse(...): (line 983)
            # Processing the call arguments (line 983)
            # Getting the type of 'file' (line 983)
            file_50455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 983, 40), 'file', False)
            # Processing the call keyword arguments (line 983)
            kwargs_50456 = {}
            # Getting the type of 'self' (line 983)
            self_50453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 983, 28), 'self', False)
            # Obtaining the member '_parse' of a type (line 983)
            _parse_50454 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 983, 28), self_50453, '_parse')
            # Calling _parse(args, kwargs) (line 983)
            _parse_call_result_50457 = invoke(stypy.reporting.localization.Localization(__file__, 983, 28), _parse_50454, *[file_50455], **kwargs_50456)
            
            # Getting the type of 'self' (line 983)
            self_50458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 983, 12), 'self')
            # Setting the type of the member 'encoding' of a type (line 983)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 983, 12), self_50458, 'encoding', _parse_call_result_50457)
            
            # Call to report(...): (line 984)
            # Processing the call arguments (line 984)
            unicode_50462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 984, 38), 'unicode', u'Result: ')
            
            # Call to repr(...): (line 984)
            # Processing the call arguments (line 984)
            # Getting the type of 'self' (line 984)
            self_50464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 984, 56), 'self', False)
            # Obtaining the member 'encoding' of a type (line 984)
            encoding_50465 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 984, 56), self_50464, 'encoding')
            # Processing the call keyword arguments (line 984)
            kwargs_50466 = {}
            # Getting the type of 'repr' (line 984)
            repr_50463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 984, 51), 'repr', False)
            # Calling repr(args, kwargs) (line 984)
            repr_call_result_50467 = invoke(stypy.reporting.localization.Localization(__file__, 984, 51), repr_50463, *[encoding_50465], **kwargs_50466)
            
            # Applying the binary operator '+' (line 984)
            result_add_50468 = python_operator(stypy.reporting.localization.Localization(__file__, 984, 38), '+', unicode_50462, repr_call_result_50467)
            
            unicode_50469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 985, 38), 'unicode', u'debug-annoying')
            # Processing the call keyword arguments (line 984)
            kwargs_50470 = {}
            # Getting the type of 'matplotlib' (line 984)
            matplotlib_50459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 984, 12), 'matplotlib', False)
            # Obtaining the member 'verbose' of a type (line 984)
            verbose_50460 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 984, 12), matplotlib_50459, 'verbose')
            # Obtaining the member 'report' of a type (line 984)
            report_50461 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 984, 12), verbose_50460, 'report')
            # Calling report(args, kwargs) (line 984)
            report_call_result_50471 = invoke(stypy.reporting.localization.Localization(__file__, 984, 12), report_50461, *[result_add_50468, unicode_50469], **kwargs_50470)
            
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 980)
            exit___50472 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 980, 13), open_call_result_50440, '__exit__')
            with_exit_50473 = invoke(stypy.reporting.localization.Localization(__file__, 980, 13), exit___50472, None, None, None)

        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def __iter__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__iter__'
        module_type_store = module_type_store.open_function_context('__iter__', 987, 4, False)
        # Assigning a type to the variable 'self' (line 988)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 988, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Encoding.__iter__.__dict__.__setitem__('stypy_localization', localization)
        Encoding.__iter__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Encoding.__iter__.__dict__.__setitem__('stypy_type_store', module_type_store)
        Encoding.__iter__.__dict__.__setitem__('stypy_function_name', 'Encoding.__iter__')
        Encoding.__iter__.__dict__.__setitem__('stypy_param_names_list', [])
        Encoding.__iter__.__dict__.__setitem__('stypy_varargs_param_name', None)
        Encoding.__iter__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Encoding.__iter__.__dict__.__setitem__('stypy_call_defaults', defaults)
        Encoding.__iter__.__dict__.__setitem__('stypy_call_varargs', varargs)
        Encoding.__iter__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Encoding.__iter__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Encoding.__iter__', [], None, None, defaults, varargs, kwargs)

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

        
        # Getting the type of 'self' (line 988)
        self_50474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 988, 20), 'self')
        # Obtaining the member 'encoding' of a type (line 988)
        encoding_50475 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 988, 20), self_50474, 'encoding')
        # Testing the type of a for loop iterable (line 988)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 988, 8), encoding_50475)
        # Getting the type of the for loop variable (line 988)
        for_loop_var_50476 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 988, 8), encoding_50475)
        # Assigning a type to the variable 'name' (line 988)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 988, 8), 'name', for_loop_var_50476)
        # SSA begins for a for statement (line 988)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        # Creating a generator
        # Getting the type of 'name' (line 989)
        name_50477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 989, 18), 'name')
        GeneratorType_50478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 989, 12), 'GeneratorType')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 989, 12), GeneratorType_50478, name_50477)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 989, 12), 'stypy_return_type', GeneratorType_50478)
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__iter__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__iter__' in the type store
        # Getting the type of 'stypy_return_type' (line 987)
        stypy_return_type_50479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 987, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_50479)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__iter__'
        return stypy_return_type_50479


    @norecursion
    def _parse(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_parse'
        module_type_store = module_type_store.open_function_context('_parse', 991, 4, False)
        # Assigning a type to the variable 'self' (line 992)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 992, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Encoding._parse.__dict__.__setitem__('stypy_localization', localization)
        Encoding._parse.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Encoding._parse.__dict__.__setitem__('stypy_type_store', module_type_store)
        Encoding._parse.__dict__.__setitem__('stypy_function_name', 'Encoding._parse')
        Encoding._parse.__dict__.__setitem__('stypy_param_names_list', ['file'])
        Encoding._parse.__dict__.__setitem__('stypy_varargs_param_name', None)
        Encoding._parse.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Encoding._parse.__dict__.__setitem__('stypy_call_defaults', defaults)
        Encoding._parse.__dict__.__setitem__('stypy_call_varargs', varargs)
        Encoding._parse.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Encoding._parse.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Encoding._parse', ['file'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_parse', localization, ['file'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_parse(...)' code ##################

        
        # Assigning a List to a Name (line 992):
        
        # Assigning a List to a Name (line 992):
        
        # Obtaining an instance of the builtin type 'list' (line 992)
        list_50480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 992, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 992)
        
        # Assigning a type to the variable 'result' (line 992)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 992, 8), 'result', list_50480)
        
        # Assigning a GeneratorExp to a Name (line 994):
        
        # Assigning a GeneratorExp to a Name (line 994):
        # Calculating generator expression
        module_type_store = module_type_store.open_function_context('list comprehension expression', 994, 17, True)
        # Calculating comprehension expression
        # Getting the type of 'file' (line 994)
        file_50493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 994, 60), 'file')
        comprehension_50494 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 994, 17), file_50493)
        # Assigning a type to the variable 'line' (line 994)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 994, 17), 'line', comprehension_50494)
        
        # Call to strip(...): (line 994)
        # Processing the call keyword arguments (line 994)
        kwargs_50491 = {}
        
        # Obtaining the type of the subscript
        int_50481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 994, 37), 'int')
        
        # Call to split(...): (line 994)
        # Processing the call arguments (line 994)
        str_50484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 994, 28), 'str', '%')
        int_50485 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 994, 34), 'int')
        # Processing the call keyword arguments (line 994)
        kwargs_50486 = {}
        # Getting the type of 'line' (line 994)
        line_50482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 994, 17), 'line', False)
        # Obtaining the member 'split' of a type (line 994)
        split_50483 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 994, 17), line_50482, 'split')
        # Calling split(args, kwargs) (line 994)
        split_call_result_50487 = invoke(stypy.reporting.localization.Localization(__file__, 994, 17), split_50483, *[str_50484, int_50485], **kwargs_50486)
        
        # Obtaining the member '__getitem__' of a type (line 994)
        getitem___50488 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 994, 17), split_call_result_50487, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 994)
        subscript_call_result_50489 = invoke(stypy.reporting.localization.Localization(__file__, 994, 17), getitem___50488, int_50481)
        
        # Obtaining the member 'strip' of a type (line 994)
        strip_50490 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 994, 17), subscript_call_result_50489, 'strip')
        # Calling strip(args, kwargs) (line 994)
        strip_call_result_50492 = invoke(stypy.reporting.localization.Localization(__file__, 994, 17), strip_50490, *[], **kwargs_50491)
        
        list_50495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 994, 17), 'list')
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 994, 17), list_50495, strip_call_result_50492)
        # Assigning a type to the variable 'lines' (line 994)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 994, 8), 'lines', list_50495)
        
        # Assigning a Call to a Name (line 995):
        
        # Assigning a Call to a Name (line 995):
        
        # Call to join(...): (line 995)
        # Processing the call arguments (line 995)
        # Getting the type of 'lines' (line 995)
        lines_50498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 995, 24), 'lines', False)
        # Processing the call keyword arguments (line 995)
        kwargs_50499 = {}
        str_50496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 995, 15), 'str', '')
        # Obtaining the member 'join' of a type (line 995)
        join_50497 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 995, 15), str_50496, 'join')
        # Calling join(args, kwargs) (line 995)
        join_call_result_50500 = invoke(stypy.reporting.localization.Localization(__file__, 995, 15), join_50497, *[lines_50498], **kwargs_50499)
        
        # Assigning a type to the variable 'data' (line 995)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 995, 8), 'data', join_call_result_50500)
        
        # Assigning a Call to a Name (line 996):
        
        # Assigning a Call to a Name (line 996):
        
        # Call to find(...): (line 996)
        # Processing the call arguments (line 996)
        str_50503 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 996, 30), 'str', '[')
        # Processing the call keyword arguments (line 996)
        kwargs_50504 = {}
        # Getting the type of 'data' (line 996)
        data_50501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 996, 20), 'data', False)
        # Obtaining the member 'find' of a type (line 996)
        find_50502 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 996, 20), data_50501, 'find')
        # Calling find(args, kwargs) (line 996)
        find_call_result_50505 = invoke(stypy.reporting.localization.Localization(__file__, 996, 20), find_50502, *[str_50503], **kwargs_50504)
        
        # Assigning a type to the variable 'beginning' (line 996)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 996, 8), 'beginning', find_call_result_50505)
        
        
        # Getting the type of 'beginning' (line 997)
        beginning_50506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 997, 11), 'beginning')
        int_50507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 997, 23), 'int')
        # Applying the binary operator '<' (line 997)
        result_lt_50508 = python_operator(stypy.reporting.localization.Localization(__file__, 997, 11), '<', beginning_50506, int_50507)
        
        # Testing the type of an if condition (line 997)
        if_condition_50509 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 997, 8), result_lt_50508)
        # Assigning a type to the variable 'if_condition_50509' (line 997)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 997, 8), 'if_condition_50509', if_condition_50509)
        # SSA begins for if statement (line 997)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 998)
        # Processing the call arguments (line 998)
        
        # Call to format(...): (line 998)
        # Processing the call arguments (line 998)
        # Getting the type of 'file' (line 999)
        file_50513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 999, 37), 'file', False)
        # Processing the call keyword arguments (line 998)
        kwargs_50514 = {}
        unicode_50511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 998, 29), 'unicode', u'Cannot locate beginning of encoding in {}')
        # Obtaining the member 'format' of a type (line 998)
        format_50512 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 998, 29), unicode_50511, 'format')
        # Calling format(args, kwargs) (line 998)
        format_call_result_50515 = invoke(stypy.reporting.localization.Localization(__file__, 998, 29), format_50512, *[file_50513], **kwargs_50514)
        
        # Processing the call keyword arguments (line 998)
        kwargs_50516 = {}
        # Getting the type of 'ValueError' (line 998)
        ValueError_50510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 998, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 998)
        ValueError_call_result_50517 = invoke(stypy.reporting.localization.Localization(__file__, 998, 18), ValueError_50510, *[format_call_result_50515], **kwargs_50516)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 998, 12), ValueError_call_result_50517, 'raise parameter', BaseException)
        # SSA join for if statement (line 997)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Subscript to a Name (line 1000):
        
        # Assigning a Subscript to a Name (line 1000):
        
        # Obtaining the type of the subscript
        # Getting the type of 'beginning' (line 1000)
        beginning_50518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1000, 20), 'beginning')
        slice_50519 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1000, 15), beginning_50518, None, None)
        # Getting the type of 'data' (line 1000)
        data_50520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1000, 15), 'data')
        # Obtaining the member '__getitem__' of a type (line 1000)
        getitem___50521 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1000, 15), data_50520, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1000)
        subscript_call_result_50522 = invoke(stypy.reporting.localization.Localization(__file__, 1000, 15), getitem___50521, slice_50519)
        
        # Assigning a type to the variable 'data' (line 1000)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1000, 8), 'data', subscript_call_result_50522)
        
        # Assigning a Call to a Name (line 1001):
        
        # Assigning a Call to a Name (line 1001):
        
        # Call to find(...): (line 1001)
        # Processing the call arguments (line 1001)
        str_50525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1001, 24), 'str', ']')
        # Processing the call keyword arguments (line 1001)
        kwargs_50526 = {}
        # Getting the type of 'data' (line 1001)
        data_50523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1001, 14), 'data', False)
        # Obtaining the member 'find' of a type (line 1001)
        find_50524 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1001, 14), data_50523, 'find')
        # Calling find(args, kwargs) (line 1001)
        find_call_result_50527 = invoke(stypy.reporting.localization.Localization(__file__, 1001, 14), find_50524, *[str_50525], **kwargs_50526)
        
        # Assigning a type to the variable 'end' (line 1001)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1001, 8), 'end', find_call_result_50527)
        
        
        # Getting the type of 'end' (line 1002)
        end_50528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1002, 11), 'end')
        int_50529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1002, 17), 'int')
        # Applying the binary operator '<' (line 1002)
        result_lt_50530 = python_operator(stypy.reporting.localization.Localization(__file__, 1002, 11), '<', end_50528, int_50529)
        
        # Testing the type of an if condition (line 1002)
        if_condition_50531 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1002, 8), result_lt_50530)
        # Assigning a type to the variable 'if_condition_50531' (line 1002)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1002, 8), 'if_condition_50531', if_condition_50531)
        # SSA begins for if statement (line 1002)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 1003)
        # Processing the call arguments (line 1003)
        
        # Call to format(...): (line 1003)
        # Processing the call arguments (line 1003)
        # Getting the type of 'file' (line 1004)
        file_50535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1004, 37), 'file', False)
        # Processing the call keyword arguments (line 1003)
        kwargs_50536 = {}
        unicode_50533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1003, 29), 'unicode', u'Cannot locate end of encoding in {}')
        # Obtaining the member 'format' of a type (line 1003)
        format_50534 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1003, 29), unicode_50533, 'format')
        # Calling format(args, kwargs) (line 1003)
        format_call_result_50537 = invoke(stypy.reporting.localization.Localization(__file__, 1003, 29), format_50534, *[file_50535], **kwargs_50536)
        
        # Processing the call keyword arguments (line 1003)
        kwargs_50538 = {}
        # Getting the type of 'ValueError' (line 1003)
        ValueError_50532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1003, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 1003)
        ValueError_call_result_50539 = invoke(stypy.reporting.localization.Localization(__file__, 1003, 18), ValueError_50532, *[format_call_result_50537], **kwargs_50538)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1003, 12), ValueError_call_result_50539, 'raise parameter', BaseException)
        # SSA join for if statement (line 1002)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Subscript to a Name (line 1005):
        
        # Assigning a Subscript to a Name (line 1005):
        
        # Obtaining the type of the subscript
        # Getting the type of 'end' (line 1005)
        end_50540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1005, 21), 'end')
        slice_50541 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1005, 15), None, end_50540, None)
        # Getting the type of 'data' (line 1005)
        data_50542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1005, 15), 'data')
        # Obtaining the member '__getitem__' of a type (line 1005)
        getitem___50543 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1005, 15), data_50542, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1005)
        subscript_call_result_50544 = invoke(stypy.reporting.localization.Localization(__file__, 1005, 15), getitem___50543, slice_50541)
        
        # Assigning a type to the variable 'data' (line 1005)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1005, 8), 'data', subscript_call_result_50544)
        
        # Call to findall(...): (line 1007)
        # Processing the call arguments (line 1007)
        str_50547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1007, 26), 'str', '/([^][{}<>\\s]+)')
        # Getting the type of 'data' (line 1007)
        data_50548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1007, 47), 'data', False)
        # Processing the call keyword arguments (line 1007)
        kwargs_50549 = {}
        # Getting the type of 're' (line 1007)
        re_50545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1007, 15), 're', False)
        # Obtaining the member 'findall' of a type (line 1007)
        findall_50546 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1007, 15), re_50545, 'findall')
        # Calling findall(args, kwargs) (line 1007)
        findall_call_result_50550 = invoke(stypy.reporting.localization.Localization(__file__, 1007, 15), findall_50546, *[str_50547, data_50548], **kwargs_50549)
        
        # Assigning a type to the variable 'stypy_return_type' (line 1007)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1007, 8), 'stypy_return_type', findall_call_result_50550)
        
        # ################# End of '_parse(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_parse' in the type store
        # Getting the type of 'stypy_return_type' (line 991)
        stypy_return_type_50551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 991, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_50551)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_parse'
        return stypy_return_type_50551


# Assigning a type to the variable 'Encoding' (line 957)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 957, 0), 'Encoding', Encoding)

# Assigning a Tuple to a Name (line 977):

# Obtaining an instance of the builtin type 'tuple' (line 977)
tuple_50552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 977, 17), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 977)
# Adding element type (line 977)
unicode_50553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 977, 17), 'unicode', u'encoding')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 977, 17), tuple_50552, unicode_50553)

# Getting the type of 'Encoding'
Encoding_50554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Encoding')
# Setting the type of the member '__slots__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Encoding_50554, '__slots__', tuple_50552)

@norecursion
def find_tex_file(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 1010)
    None_50555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1010, 35), 'None')
    defaults = [None_50555]
    # Create a new context for function 'find_tex_file'
    module_type_store = module_type_store.open_function_context('find_tex_file', 1010, 0, False)
    
    # Passed parameters checking function
    find_tex_file.stypy_localization = localization
    find_tex_file.stypy_type_of_self = None
    find_tex_file.stypy_type_store = module_type_store
    find_tex_file.stypy_function_name = 'find_tex_file'
    find_tex_file.stypy_param_names_list = ['filename', 'format']
    find_tex_file.stypy_varargs_param_name = None
    find_tex_file.stypy_kwargs_param_name = None
    find_tex_file.stypy_call_defaults = defaults
    find_tex_file.stypy_call_varargs = varargs
    find_tex_file.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'find_tex_file', ['filename', 'format'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'find_tex_file', localization, ['filename', 'format'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'find_tex_file(...)' code ##################

    unicode_50556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1031, (-1)), 'unicode', u"\n    Find a file in the texmf tree.\n\n    Calls :program:`kpsewhich` which is an interface to the kpathsea\n    library [1]_. Most existing TeX distributions on Unix-like systems use\n    kpathsea. It is also available as part of MikTeX, a popular\n    distribution on Windows.\n\n    Parameters\n    ----------\n    filename : string or bytestring\n    format : string or bytestring\n        Used as the value of the `--format` option to :program:`kpsewhich`.\n        Could be e.g. 'tfm' or 'vf' to limit the search to that type of files.\n\n    References\n    ----------\n\n    .. [1] `Kpathsea documentation <http://www.tug.org/kpathsea/>`_\n        The library that :program:`kpsewhich` is part of.\n    ")
    
    # Assigning a List to a Name (line 1033):
    
    # Assigning a List to a Name (line 1033):
    
    # Obtaining an instance of the builtin type 'list' (line 1033)
    list_50557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1033, 10), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1033)
    # Adding element type (line 1033)
    
    # Call to str(...): (line 1033)
    # Processing the call arguments (line 1033)
    unicode_50559 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1033, 15), 'unicode', u'kpsewhich')
    # Processing the call keyword arguments (line 1033)
    kwargs_50560 = {}
    # Getting the type of 'str' (line 1033)
    str_50558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1033, 11), 'str', False)
    # Calling str(args, kwargs) (line 1033)
    str_call_result_50561 = invoke(stypy.reporting.localization.Localization(__file__, 1033, 11), str_50558, *[unicode_50559], **kwargs_50560)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1033, 10), list_50557, str_call_result_50561)
    
    # Assigning a type to the variable 'cmd' (line 1033)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1033, 4), 'cmd', list_50557)
    
    # Type idiom detected: calculating its left and rigth part (line 1034)
    # Getting the type of 'format' (line 1034)
    format_50562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1034, 4), 'format')
    # Getting the type of 'None' (line 1034)
    None_50563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1034, 21), 'None')
    
    (may_be_50564, more_types_in_union_50565) = may_not_be_none(format_50562, None_50563)

    if may_be_50564:

        if more_types_in_union_50565:
            # Runtime conditional SSA (line 1034)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Getting the type of 'cmd' (line 1035)
        cmd_50566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1035, 8), 'cmd')
        
        # Obtaining an instance of the builtin type 'list' (line 1035)
        list_50567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1035, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 1035)
        # Adding element type (line 1035)
        unicode_50568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1035, 16), 'unicode', u'--format=')
        # Getting the type of 'format' (line 1035)
        format_50569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1035, 30), 'format')
        # Applying the binary operator '+' (line 1035)
        result_add_50570 = python_operator(stypy.reporting.localization.Localization(__file__, 1035, 16), '+', unicode_50568, format_50569)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1035, 15), list_50567, result_add_50570)
        
        # Applying the binary operator '+=' (line 1035)
        result_iadd_50571 = python_operator(stypy.reporting.localization.Localization(__file__, 1035, 8), '+=', cmd_50566, list_50567)
        # Assigning a type to the variable 'cmd' (line 1035)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1035, 8), 'cmd', result_iadd_50571)
        

        if more_types_in_union_50565:
            # SSA join for if statement (line 1034)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Getting the type of 'cmd' (line 1036)
    cmd_50572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1036, 4), 'cmd')
    
    # Obtaining an instance of the builtin type 'list' (line 1036)
    list_50573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1036, 11), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1036)
    # Adding element type (line 1036)
    # Getting the type of 'filename' (line 1036)
    filename_50574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1036, 12), 'filename')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1036, 11), list_50573, filename_50574)
    
    # Applying the binary operator '+=' (line 1036)
    result_iadd_50575 = python_operator(stypy.reporting.localization.Localization(__file__, 1036, 4), '+=', cmd_50572, list_50573)
    # Assigning a type to the variable 'cmd' (line 1036)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1036, 4), 'cmd', result_iadd_50575)
    
    
    # Call to report(...): (line 1038)
    # Processing the call arguments (line 1038)
    unicode_50579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1038, 30), 'unicode', u'find_tex_file(%s): %s')
    
    # Obtaining an instance of the builtin type 'tuple' (line 1039)
    tuple_50580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1039, 33), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1039)
    # Adding element type (line 1039)
    # Getting the type of 'filename' (line 1039)
    filename_50581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1039, 33), 'filename', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1039, 33), tuple_50580, filename_50581)
    # Adding element type (line 1039)
    # Getting the type of 'cmd' (line 1039)
    cmd_50582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1039, 43), 'cmd', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1039, 33), tuple_50580, cmd_50582)
    
    # Applying the binary operator '%' (line 1038)
    result_mod_50583 = python_operator(stypy.reporting.localization.Localization(__file__, 1038, 30), '%', unicode_50579, tuple_50580)
    
    unicode_50584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1039, 49), 'unicode', u'debug')
    # Processing the call keyword arguments (line 1038)
    kwargs_50585 = {}
    # Getting the type of 'matplotlib' (line 1038)
    matplotlib_50576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1038, 4), 'matplotlib', False)
    # Obtaining the member 'verbose' of a type (line 1038)
    verbose_50577 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1038, 4), matplotlib_50576, 'verbose')
    # Obtaining the member 'report' of a type (line 1038)
    report_50578 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1038, 4), verbose_50577, 'report')
    # Calling report(args, kwargs) (line 1038)
    report_call_result_50586 = invoke(stypy.reporting.localization.Localization(__file__, 1038, 4), report_50578, *[result_mod_50583, unicode_50584], **kwargs_50585)
    
    
    # Assigning a Call to a Name (line 1044):
    
    # Assigning a Call to a Name (line 1044):
    
    # Call to Popen(...): (line 1044)
    # Processing the call arguments (line 1044)
    # Getting the type of 'cmd' (line 1044)
    cmd_50589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1044, 28), 'cmd', False)
    # Processing the call keyword arguments (line 1044)
    # Getting the type of 'subprocess' (line 1044)
    subprocess_50590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1044, 40), 'subprocess', False)
    # Obtaining the member 'PIPE' of a type (line 1044)
    PIPE_50591 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1044, 40), subprocess_50590, 'PIPE')
    keyword_50592 = PIPE_50591
    # Getting the type of 'subprocess' (line 1045)
    subprocess_50593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1045, 35), 'subprocess', False)
    # Obtaining the member 'PIPE' of a type (line 1045)
    PIPE_50594 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1045, 35), subprocess_50593, 'PIPE')
    keyword_50595 = PIPE_50594
    kwargs_50596 = {'stderr': keyword_50595, 'stdout': keyword_50592}
    # Getting the type of 'subprocess' (line 1044)
    subprocess_50587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1044, 11), 'subprocess', False)
    # Obtaining the member 'Popen' of a type (line 1044)
    Popen_50588 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1044, 11), subprocess_50587, 'Popen')
    # Calling Popen(args, kwargs) (line 1044)
    Popen_call_result_50597 = invoke(stypy.reporting.localization.Localization(__file__, 1044, 11), Popen_50588, *[cmd_50589], **kwargs_50596)
    
    # Assigning a type to the variable 'pipe' (line 1044)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1044, 4), 'pipe', Popen_call_result_50597)
    
    # Assigning a Call to a Name (line 1046):
    
    # Assigning a Call to a Name (line 1046):
    
    # Call to rstrip(...): (line 1046)
    # Processing the call keyword arguments (line 1046)
    kwargs_50606 = {}
    
    # Obtaining the type of the subscript
    int_50598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1046, 32), 'int')
    
    # Call to communicate(...): (line 1046)
    # Processing the call keyword arguments (line 1046)
    kwargs_50601 = {}
    # Getting the type of 'pipe' (line 1046)
    pipe_50599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1046, 13), 'pipe', False)
    # Obtaining the member 'communicate' of a type (line 1046)
    communicate_50600 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1046, 13), pipe_50599, 'communicate')
    # Calling communicate(args, kwargs) (line 1046)
    communicate_call_result_50602 = invoke(stypy.reporting.localization.Localization(__file__, 1046, 13), communicate_50600, *[], **kwargs_50601)
    
    # Obtaining the member '__getitem__' of a type (line 1046)
    getitem___50603 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1046, 13), communicate_call_result_50602, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1046)
    subscript_call_result_50604 = invoke(stypy.reporting.localization.Localization(__file__, 1046, 13), getitem___50603, int_50598)
    
    # Obtaining the member 'rstrip' of a type (line 1046)
    rstrip_50605 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1046, 13), subscript_call_result_50604, 'rstrip')
    # Calling rstrip(args, kwargs) (line 1046)
    rstrip_call_result_50607 = invoke(stypy.reporting.localization.Localization(__file__, 1046, 13), rstrip_50605, *[], **kwargs_50606)
    
    # Assigning a type to the variable 'result' (line 1046)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1046, 4), 'result', rstrip_call_result_50607)
    
    # Call to report(...): (line 1047)
    # Processing the call arguments (line 1047)
    unicode_50611 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1047, 30), 'unicode', u'find_tex_file result: %s')
    # Getting the type of 'result' (line 1047)
    result_50612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1047, 59), 'result', False)
    # Applying the binary operator '%' (line 1047)
    result_mod_50613 = python_operator(stypy.reporting.localization.Localization(__file__, 1047, 30), '%', unicode_50611, result_50612)
    
    unicode_50614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1048, 30), 'unicode', u'debug')
    # Processing the call keyword arguments (line 1047)
    kwargs_50615 = {}
    # Getting the type of 'matplotlib' (line 1047)
    matplotlib_50608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1047, 4), 'matplotlib', False)
    # Obtaining the member 'verbose' of a type (line 1047)
    verbose_50609 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1047, 4), matplotlib_50608, 'verbose')
    # Obtaining the member 'report' of a type (line 1047)
    report_50610 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1047, 4), verbose_50609, 'report')
    # Calling report(args, kwargs) (line 1047)
    report_call_result_50616 = invoke(stypy.reporting.localization.Localization(__file__, 1047, 4), report_50610, *[result_mod_50613, unicode_50614], **kwargs_50615)
    
    
    # Call to decode(...): (line 1049)
    # Processing the call arguments (line 1049)
    unicode_50619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1049, 25), 'unicode', u'ascii')
    # Processing the call keyword arguments (line 1049)
    kwargs_50620 = {}
    # Getting the type of 'result' (line 1049)
    result_50617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1049, 11), 'result', False)
    # Obtaining the member 'decode' of a type (line 1049)
    decode_50618 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1049, 11), result_50617, 'decode')
    # Calling decode(args, kwargs) (line 1049)
    decode_call_result_50621 = invoke(stypy.reporting.localization.Localization(__file__, 1049, 11), decode_50618, *[unicode_50619], **kwargs_50620)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1049)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1049, 4), 'stypy_return_type', decode_call_result_50621)
    
    # ################# End of 'find_tex_file(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'find_tex_file' in the type store
    # Getting the type of 'stypy_return_type' (line 1010)
    stypy_return_type_50622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1010, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_50622)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'find_tex_file'
    return stypy_return_type_50622

# Assigning a type to the variable 'find_tex_file' (line 1010)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1010, 0), 'find_tex_file', find_tex_file)

# Assigning a Dict to a Name (line 1055):

# Assigning a Dict to a Name (line 1055):

# Obtaining an instance of the builtin type 'dict' (line 1055)
dict_50623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1055, 12), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 1055)

# Assigning a type to the variable '_tfmcache' (line 1055)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1055, 0), '_tfmcache', dict_50623)

# Assigning a Dict to a Name (line 1056):

# Assigning a Dict to a Name (line 1056):

# Obtaining an instance of the builtin type 'dict' (line 1056)
dict_50624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1056, 11), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 1056)

# Assigning a type to the variable '_vfcache' (line 1056)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1056, 0), '_vfcache', dict_50624)

@norecursion
def _fontfile(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_fontfile'
    module_type_store = module_type_store.open_function_context('_fontfile', 1059, 0, False)
    
    # Passed parameters checking function
    _fontfile.stypy_localization = localization
    _fontfile.stypy_type_of_self = None
    _fontfile.stypy_type_store = module_type_store
    _fontfile.stypy_function_name = '_fontfile'
    _fontfile.stypy_param_names_list = ['texname', 'class_', 'suffix', 'cache']
    _fontfile.stypy_varargs_param_name = None
    _fontfile.stypy_kwargs_param_name = None
    _fontfile.stypy_call_defaults = defaults
    _fontfile.stypy_call_varargs = varargs
    _fontfile.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_fontfile', ['texname', 'class_', 'suffix', 'cache'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_fontfile', localization, ['texname', 'class_', 'suffix', 'cache'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_fontfile(...)' code ##################

    
    
    # SSA begins for try-except statement (line 1060)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Obtaining the type of the subscript
    # Getting the type of 'texname' (line 1061)
    texname_50625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1061, 21), 'texname')
    # Getting the type of 'cache' (line 1061)
    cache_50626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1061, 15), 'cache')
    # Obtaining the member '__getitem__' of a type (line 1061)
    getitem___50627 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1061, 15), cache_50626, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1061)
    subscript_call_result_50628 = invoke(stypy.reporting.localization.Localization(__file__, 1061, 15), getitem___50627, texname_50625)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1061)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1061, 8), 'stypy_return_type', subscript_call_result_50628)
    # SSA branch for the except part of a try statement (line 1060)
    # SSA branch for the except 'KeyError' branch of a try statement (line 1060)
    module_type_store.open_ssa_branch('except')
    pass
    # SSA join for try-except statement (line 1060)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 1065):
    
    # Assigning a Call to a Name (line 1065):
    
    # Call to find_tex_file(...): (line 1065)
    # Processing the call arguments (line 1065)
    # Getting the type of 'texname' (line 1065)
    texname_50630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1065, 29), 'texname', False)
    # Getting the type of 'suffix' (line 1065)
    suffix_50631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1065, 39), 'suffix', False)
    # Applying the binary operator '+' (line 1065)
    result_add_50632 = python_operator(stypy.reporting.localization.Localization(__file__, 1065, 29), '+', texname_50630, suffix_50631)
    
    # Processing the call keyword arguments (line 1065)
    kwargs_50633 = {}
    # Getting the type of 'find_tex_file' (line 1065)
    find_tex_file_50629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1065, 15), 'find_tex_file', False)
    # Calling find_tex_file(args, kwargs) (line 1065)
    find_tex_file_call_result_50634 = invoke(stypy.reporting.localization.Localization(__file__, 1065, 15), find_tex_file_50629, *[result_add_50632], **kwargs_50633)
    
    # Assigning a type to the variable 'filename' (line 1065)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1065, 4), 'filename', find_tex_file_call_result_50634)
    
    # Getting the type of 'filename' (line 1066)
    filename_50635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1066, 7), 'filename')
    # Testing the type of an if condition (line 1066)
    if_condition_50636 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1066, 4), filename_50635)
    # Assigning a type to the variable 'if_condition_50636' (line 1066)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1066, 4), 'if_condition_50636', if_condition_50636)
    # SSA begins for if statement (line 1066)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 1067):
    
    # Assigning a Call to a Name (line 1067):
    
    # Call to class_(...): (line 1067)
    # Processing the call arguments (line 1067)
    # Getting the type of 'filename' (line 1067)
    filename_50638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1067, 24), 'filename', False)
    # Processing the call keyword arguments (line 1067)
    kwargs_50639 = {}
    # Getting the type of 'class_' (line 1067)
    class__50637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1067, 17), 'class_', False)
    # Calling class_(args, kwargs) (line 1067)
    class__call_result_50640 = invoke(stypy.reporting.localization.Localization(__file__, 1067, 17), class__50637, *[filename_50638], **kwargs_50639)
    
    # Assigning a type to the variable 'result' (line 1067)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1067, 8), 'result', class__call_result_50640)
    # SSA branch for the else part of an if statement (line 1066)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Name to a Name (line 1069):
    
    # Assigning a Name to a Name (line 1069):
    # Getting the type of 'None' (line 1069)
    None_50641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1069, 17), 'None')
    # Assigning a type to the variable 'result' (line 1069)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1069, 8), 'result', None_50641)
    # SSA join for if statement (line 1066)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Subscript (line 1071):
    
    # Assigning a Name to a Subscript (line 1071):
    # Getting the type of 'result' (line 1071)
    result_50642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1071, 21), 'result')
    # Getting the type of 'cache' (line 1071)
    cache_50643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1071, 4), 'cache')
    # Getting the type of 'texname' (line 1071)
    texname_50644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1071, 10), 'texname')
    # Storing an element on a container (line 1071)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1071, 4), cache_50643, (texname_50644, result_50642))
    # Getting the type of 'result' (line 1072)
    result_50645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1072, 11), 'result')
    # Assigning a type to the variable 'stypy_return_type' (line 1072)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1072, 4), 'stypy_return_type', result_50645)
    
    # ################# End of '_fontfile(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_fontfile' in the type store
    # Getting the type of 'stypy_return_type' (line 1059)
    stypy_return_type_50646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1059, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_50646)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_fontfile'
    return stypy_return_type_50646

# Assigning a type to the variable '_fontfile' (line 1059)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1059, 0), '_fontfile', _fontfile)

@norecursion
def _tfmfile(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_tfmfile'
    module_type_store = module_type_store.open_function_context('_tfmfile', 1075, 0, False)
    
    # Passed parameters checking function
    _tfmfile.stypy_localization = localization
    _tfmfile.stypy_type_of_self = None
    _tfmfile.stypy_type_store = module_type_store
    _tfmfile.stypy_function_name = '_tfmfile'
    _tfmfile.stypy_param_names_list = ['texname']
    _tfmfile.stypy_varargs_param_name = None
    _tfmfile.stypy_kwargs_param_name = None
    _tfmfile.stypy_call_defaults = defaults
    _tfmfile.stypy_call_varargs = varargs
    _tfmfile.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_tfmfile', ['texname'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_tfmfile', localization, ['texname'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_tfmfile(...)' code ##################

    
    # Call to _fontfile(...): (line 1076)
    # Processing the call arguments (line 1076)
    # Getting the type of 'texname' (line 1076)
    texname_50648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1076, 21), 'texname', False)
    # Getting the type of 'Tfm' (line 1076)
    Tfm_50649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1076, 30), 'Tfm', False)
    unicode_50650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1076, 35), 'unicode', u'.tfm')
    # Getting the type of '_tfmcache' (line 1076)
    _tfmcache_50651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1076, 43), '_tfmcache', False)
    # Processing the call keyword arguments (line 1076)
    kwargs_50652 = {}
    # Getting the type of '_fontfile' (line 1076)
    _fontfile_50647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1076, 11), '_fontfile', False)
    # Calling _fontfile(args, kwargs) (line 1076)
    _fontfile_call_result_50653 = invoke(stypy.reporting.localization.Localization(__file__, 1076, 11), _fontfile_50647, *[texname_50648, Tfm_50649, unicode_50650, _tfmcache_50651], **kwargs_50652)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1076)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1076, 4), 'stypy_return_type', _fontfile_call_result_50653)
    
    # ################# End of '_tfmfile(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_tfmfile' in the type store
    # Getting the type of 'stypy_return_type' (line 1075)
    stypy_return_type_50654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1075, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_50654)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_tfmfile'
    return stypy_return_type_50654

# Assigning a type to the variable '_tfmfile' (line 1075)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1075, 0), '_tfmfile', _tfmfile)

@norecursion
def _vffile(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_vffile'
    module_type_store = module_type_store.open_function_context('_vffile', 1079, 0, False)
    
    # Passed parameters checking function
    _vffile.stypy_localization = localization
    _vffile.stypy_type_of_self = None
    _vffile.stypy_type_store = module_type_store
    _vffile.stypy_function_name = '_vffile'
    _vffile.stypy_param_names_list = ['texname']
    _vffile.stypy_varargs_param_name = None
    _vffile.stypy_kwargs_param_name = None
    _vffile.stypy_call_defaults = defaults
    _vffile.stypy_call_varargs = varargs
    _vffile.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_vffile', ['texname'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_vffile', localization, ['texname'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_vffile(...)' code ##################

    
    # Call to _fontfile(...): (line 1080)
    # Processing the call arguments (line 1080)
    # Getting the type of 'texname' (line 1080)
    texname_50656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1080, 21), 'texname', False)
    # Getting the type of 'Vf' (line 1080)
    Vf_50657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1080, 30), 'Vf', False)
    unicode_50658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1080, 34), 'unicode', u'.vf')
    # Getting the type of '_vfcache' (line 1080)
    _vfcache_50659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1080, 41), '_vfcache', False)
    # Processing the call keyword arguments (line 1080)
    kwargs_50660 = {}
    # Getting the type of '_fontfile' (line 1080)
    _fontfile_50655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1080, 11), '_fontfile', False)
    # Calling _fontfile(args, kwargs) (line 1080)
    _fontfile_call_result_50661 = invoke(stypy.reporting.localization.Localization(__file__, 1080, 11), _fontfile_50655, *[texname_50656, Vf_50657, unicode_50658, _vfcache_50659], **kwargs_50660)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1080)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1080, 4), 'stypy_return_type', _fontfile_call_result_50661)
    
    # ################# End of '_vffile(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_vffile' in the type store
    # Getting the type of 'stypy_return_type' (line 1079)
    stypy_return_type_50662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1079, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_50662)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_vffile'
    return stypy_return_type_50662

# Assigning a type to the variable '_vffile' (line 1079)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1079, 0), '_vffile', _vffile)

if (__name__ == u'__main__'):
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1084, 4))
    
    # 'import sys' statement (line 1084)
    import sys

    import_module(stypy.reporting.localization.Localization(__file__, 1084, 4), 'sys', sys, module_type_store)
    
    
    # Call to set_level(...): (line 1085)
    # Processing the call arguments (line 1085)
    unicode_50666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1085, 33), 'unicode', u'debug-annoying')
    # Processing the call keyword arguments (line 1085)
    kwargs_50667 = {}
    # Getting the type of 'matplotlib' (line 1085)
    matplotlib_50663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1085, 4), 'matplotlib', False)
    # Obtaining the member 'verbose' of a type (line 1085)
    verbose_50664 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1085, 4), matplotlib_50663, 'verbose')
    # Obtaining the member 'set_level' of a type (line 1085)
    set_level_50665 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1085, 4), verbose_50664, 'set_level')
    # Calling set_level(args, kwargs) (line 1085)
    set_level_call_result_50668 = invoke(stypy.reporting.localization.Localization(__file__, 1085, 4), set_level_50665, *[unicode_50666], **kwargs_50667)
    
    
    # Assigning a Subscript to a Name (line 1086):
    
    # Assigning a Subscript to a Name (line 1086):
    
    # Obtaining the type of the subscript
    int_50669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1086, 21), 'int')
    # Getting the type of 'sys' (line 1086)
    sys_50670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1086, 12), 'sys')
    # Obtaining the member 'argv' of a type (line 1086)
    argv_50671 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1086, 12), sys_50670, 'argv')
    # Obtaining the member '__getitem__' of a type (line 1086)
    getitem___50672 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1086, 12), argv_50671, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1086)
    subscript_call_result_50673 = invoke(stypy.reporting.localization.Localization(__file__, 1086, 12), getitem___50672, int_50669)
    
    # Assigning a type to the variable 'fname' (line 1086)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1086, 4), 'fname', subscript_call_result_50673)
    
    
    # SSA begins for try-except statement (line 1087)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Name (line 1088):
    
    # Assigning a Call to a Name (line 1088):
    
    # Call to float(...): (line 1088)
    # Processing the call arguments (line 1088)
    
    # Obtaining the type of the subscript
    int_50675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1088, 29), 'int')
    # Getting the type of 'sys' (line 1088)
    sys_50676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1088, 20), 'sys', False)
    # Obtaining the member 'argv' of a type (line 1088)
    argv_50677 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1088, 20), sys_50676, 'argv')
    # Obtaining the member '__getitem__' of a type (line 1088)
    getitem___50678 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1088, 20), argv_50677, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1088)
    subscript_call_result_50679 = invoke(stypy.reporting.localization.Localization(__file__, 1088, 20), getitem___50678, int_50675)
    
    # Processing the call keyword arguments (line 1088)
    kwargs_50680 = {}
    # Getting the type of 'float' (line 1088)
    float_50674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1088, 14), 'float', False)
    # Calling float(args, kwargs) (line 1088)
    float_call_result_50681 = invoke(stypy.reporting.localization.Localization(__file__, 1088, 14), float_50674, *[subscript_call_result_50679], **kwargs_50680)
    
    # Assigning a type to the variable 'dpi' (line 1088)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1088, 8), 'dpi', float_call_result_50681)
    # SSA branch for the except part of a try statement (line 1087)
    # SSA branch for the except 'IndexError' branch of a try statement (line 1087)
    module_type_store.open_ssa_branch('except')
    
    # Assigning a Name to a Name (line 1090):
    
    # Assigning a Name to a Name (line 1090):
    # Getting the type of 'None' (line 1090)
    None_50682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1090, 14), 'None')
    # Assigning a type to the variable 'dpi' (line 1090)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1090, 8), 'dpi', None_50682)
    # SSA join for try-except statement (line 1087)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to Dvi(...): (line 1091)
    # Processing the call arguments (line 1091)
    # Getting the type of 'fname' (line 1091)
    fname_50684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1091, 13), 'fname', False)
    # Getting the type of 'dpi' (line 1091)
    dpi_50685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1091, 20), 'dpi', False)
    # Processing the call keyword arguments (line 1091)
    kwargs_50686 = {}
    # Getting the type of 'Dvi' (line 1091)
    Dvi_50683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1091, 9), 'Dvi', False)
    # Calling Dvi(args, kwargs) (line 1091)
    Dvi_call_result_50687 = invoke(stypy.reporting.localization.Localization(__file__, 1091, 9), Dvi_50683, *[fname_50684, dpi_50685], **kwargs_50686)
    
    with_50688 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 1091, 9), Dvi_call_result_50687, 'with parameter', '__enter__', '__exit__')

    if with_50688:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 1091)
        enter___50689 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1091, 9), Dvi_call_result_50687, '__enter__')
        with_enter_50690 = invoke(stypy.reporting.localization.Localization(__file__, 1091, 9), enter___50689)
        # Assigning a type to the variable 'dvi' (line 1091)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1091, 9), 'dvi', with_enter_50690)
        
        # Assigning a Call to a Name (line 1092):
        
        # Assigning a Call to a Name (line 1092):
        
        # Call to PsfontsMap(...): (line 1092)
        # Processing the call arguments (line 1092)
        
        # Call to find_tex_file(...): (line 1092)
        # Processing the call arguments (line 1092)
        unicode_50693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1092, 43), 'unicode', u'pdftex.map')
        # Processing the call keyword arguments (line 1092)
        kwargs_50694 = {}
        # Getting the type of 'find_tex_file' (line 1092)
        find_tex_file_50692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1092, 29), 'find_tex_file', False)
        # Calling find_tex_file(args, kwargs) (line 1092)
        find_tex_file_call_result_50695 = invoke(stypy.reporting.localization.Localization(__file__, 1092, 29), find_tex_file_50692, *[unicode_50693], **kwargs_50694)
        
        # Processing the call keyword arguments (line 1092)
        kwargs_50696 = {}
        # Getting the type of 'PsfontsMap' (line 1092)
        PsfontsMap_50691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1092, 18), 'PsfontsMap', False)
        # Calling PsfontsMap(args, kwargs) (line 1092)
        PsfontsMap_call_result_50697 = invoke(stypy.reporting.localization.Localization(__file__, 1092, 18), PsfontsMap_50691, *[find_tex_file_call_result_50695], **kwargs_50696)
        
        # Assigning a type to the variable 'fontmap' (line 1092)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1092, 8), 'fontmap', PsfontsMap_call_result_50697)
        
        # Getting the type of 'dvi' (line 1093)
        dvi_50698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1093, 20), 'dvi')
        # Testing the type of a for loop iterable (line 1093)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1093, 8), dvi_50698)
        # Getting the type of the for loop variable (line 1093)
        for_loop_var_50699 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1093, 8), dvi_50698)
        # Assigning a type to the variable 'page' (line 1093)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1093, 8), 'page', for_loop_var_50699)
        # SSA begins for a for statement (line 1093)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to print(...): (line 1094)
        # Processing the call arguments (line 1094)
        unicode_50701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1094, 18), 'unicode', u'=== new page ===')
        # Processing the call keyword arguments (line 1094)
        kwargs_50702 = {}
        # Getting the type of 'print' (line 1094)
        print_50700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1094, 12), 'print', False)
        # Calling print(args, kwargs) (line 1094)
        print_call_result_50703 = invoke(stypy.reporting.localization.Localization(__file__, 1094, 12), print_50700, *[unicode_50701], **kwargs_50702)
        
        
        # Assigning a Name to a Name (line 1095):
        
        # Assigning a Name to a Name (line 1095):
        # Getting the type of 'None' (line 1095)
        None_50704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1095, 20), 'None')
        # Assigning a type to the variable 'fPrev' (line 1095)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1095, 12), 'fPrev', None_50704)
        
        # Getting the type of 'page' (line 1096)
        page_50705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1096, 33), 'page')
        # Obtaining the member 'text' of a type (line 1096)
        text_50706 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1096, 33), page_50705, 'text')
        # Testing the type of a for loop iterable (line 1096)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1096, 12), text_50706)
        # Getting the type of the for loop variable (line 1096)
        for_loop_var_50707 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1096, 12), text_50706)
        # Assigning a type to the variable 'x' (line 1096)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1096, 12), 'x', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1096, 12), for_loop_var_50707))
        # Assigning a type to the variable 'y' (line 1096)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1096, 12), 'y', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1096, 12), for_loop_var_50707))
        # Assigning a type to the variable 'f' (line 1096)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1096, 12), 'f', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1096, 12), for_loop_var_50707))
        # Assigning a type to the variable 'c' (line 1096)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1096, 12), 'c', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1096, 12), for_loop_var_50707))
        # Assigning a type to the variable 'w' (line 1096)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1096, 12), 'w', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1096, 12), for_loop_var_50707))
        # SSA begins for a for statement (line 1096)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Getting the type of 'f' (line 1097)
        f_50708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1097, 19), 'f')
        # Getting the type of 'fPrev' (line 1097)
        fPrev_50709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1097, 24), 'fPrev')
        # Applying the binary operator '!=' (line 1097)
        result_ne_50710 = python_operator(stypy.reporting.localization.Localization(__file__, 1097, 19), '!=', f_50708, fPrev_50709)
        
        # Testing the type of an if condition (line 1097)
        if_condition_50711 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1097, 16), result_ne_50710)
        # Assigning a type to the variable 'if_condition_50711' (line 1097)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1097, 16), 'if_condition_50711', if_condition_50711)
        # SSA begins for if statement (line 1097)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to print(...): (line 1098)
        # Processing the call arguments (line 1098)
        unicode_50713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1098, 26), 'unicode', u'font')
        # Getting the type of 'f' (line 1098)
        f_50714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1098, 34), 'f', False)
        # Obtaining the member 'texname' of a type (line 1098)
        texname_50715 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1098, 34), f_50714, 'texname')
        unicode_50716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1098, 45), 'unicode', u'scaled')
        # Getting the type of 'f' (line 1098)
        f_50717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1098, 55), 'f', False)
        # Obtaining the member '_scale' of a type (line 1098)
        _scale_50718 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1098, 55), f_50717, '_scale')
        
        # Call to pow(...): (line 1098)
        # Processing the call arguments (line 1098)
        float_50720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1098, 68), 'float')
        int_50721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1098, 73), 'int')
        # Processing the call keyword arguments (line 1098)
        kwargs_50722 = {}
        # Getting the type of 'pow' (line 1098)
        pow_50719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1098, 64), 'pow', False)
        # Calling pow(args, kwargs) (line 1098)
        pow_call_result_50723 = invoke(stypy.reporting.localization.Localization(__file__, 1098, 64), pow_50719, *[float_50720, int_50721], **kwargs_50722)
        
        # Applying the binary operator 'div' (line 1098)
        result_div_50724 = python_operator(stypy.reporting.localization.Localization(__file__, 1098, 55), 'div', _scale_50718, pow_call_result_50723)
        
        # Processing the call keyword arguments (line 1098)
        kwargs_50725 = {}
        # Getting the type of 'print' (line 1098)
        print_50712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1098, 20), 'print', False)
        # Calling print(args, kwargs) (line 1098)
        print_call_result_50726 = invoke(stypy.reporting.localization.Localization(__file__, 1098, 20), print_50712, *[unicode_50713, texname_50715, unicode_50716, result_div_50724], **kwargs_50725)
        
        
        # Assigning a Name to a Name (line 1099):
        
        # Assigning a Name to a Name (line 1099):
        # Getting the type of 'f' (line 1099)
        f_50727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1099, 28), 'f')
        # Assigning a type to the variable 'fPrev' (line 1099)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1099, 20), 'fPrev', f_50727)
        # SSA join for if statement (line 1097)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to print(...): (line 1100)
        # Processing the call arguments (line 1100)
        # Getting the type of 'x' (line 1100)
        x_50729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1100, 22), 'x', False)
        # Getting the type of 'y' (line 1100)
        y_50730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1100, 25), 'y', False)
        # Getting the type of 'c' (line 1100)
        c_50731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1100, 28), 'c', False)
        
        # Evaluating a boolean operation
        
        # Evaluating a boolean operation
        
        int_50732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1100, 31), 'int')
        # Getting the type of 'c' (line 1100)
        c_50733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1100, 37), 'c', False)
        # Applying the binary operator '<=' (line 1100)
        result_le_50734 = python_operator(stypy.reporting.localization.Localization(__file__, 1100, 31), '<=', int_50732, c_50733)
        int_50735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1100, 41), 'int')
        # Applying the binary operator '<' (line 1100)
        result_lt_50736 = python_operator(stypy.reporting.localization.Localization(__file__, 1100, 31), '<', c_50733, int_50735)
        # Applying the binary operator '&' (line 1100)
        result_and__50737 = python_operator(stypy.reporting.localization.Localization(__file__, 1100, 31), '&', result_le_50734, result_lt_50736)
        
        
        # Call to chr(...): (line 1100)
        # Processing the call arguments (line 1100)
        # Getting the type of 'c' (line 1100)
        c_50739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1100, 53), 'c', False)
        # Processing the call keyword arguments (line 1100)
        kwargs_50740 = {}
        # Getting the type of 'chr' (line 1100)
        chr_50738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1100, 49), 'chr', False)
        # Calling chr(args, kwargs) (line 1100)
        chr_call_result_50741 = invoke(stypy.reporting.localization.Localization(__file__, 1100, 49), chr_50738, *[c_50739], **kwargs_50740)
        
        # Applying the binary operator 'and' (line 1100)
        result_and_keyword_50742 = python_operator(stypy.reporting.localization.Localization(__file__, 1100, 31), 'and', result_and__50737, chr_call_result_50741)
        
        unicode_50743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1100, 59), 'unicode', u'.')
        # Applying the binary operator 'or' (line 1100)
        result_or_keyword_50744 = python_operator(stypy.reporting.localization.Localization(__file__, 1100, 31), 'or', result_and_keyword_50742, unicode_50743)
        
        # Getting the type of 'w' (line 1100)
        w_50745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1100, 64), 'w', False)
        # Processing the call keyword arguments (line 1100)
        kwargs_50746 = {}
        # Getting the type of 'print' (line 1100)
        print_50728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1100, 16), 'print', False)
        # Calling print(args, kwargs) (line 1100)
        print_call_result_50747 = invoke(stypy.reporting.localization.Localization(__file__, 1100, 16), print_50728, *[x_50729, y_50730, c_50731, result_or_keyword_50744, w_50745], **kwargs_50746)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'page' (line 1101)
        page_50748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1101, 30), 'page')
        # Obtaining the member 'boxes' of a type (line 1101)
        boxes_50749 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1101, 30), page_50748, 'boxes')
        # Testing the type of a for loop iterable (line 1101)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1101, 12), boxes_50749)
        # Getting the type of the for loop variable (line 1101)
        for_loop_var_50750 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1101, 12), boxes_50749)
        # Assigning a type to the variable 'x' (line 1101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1101, 12), 'x', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1101, 12), for_loop_var_50750))
        # Assigning a type to the variable 'y' (line 1101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1101, 12), 'y', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1101, 12), for_loop_var_50750))
        # Assigning a type to the variable 'w' (line 1101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1101, 12), 'w', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1101, 12), for_loop_var_50750))
        # Assigning a type to the variable 'h' (line 1101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1101, 12), 'h', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1101, 12), for_loop_var_50750))
        # SSA begins for a for statement (line 1101)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to print(...): (line 1102)
        # Processing the call arguments (line 1102)
        # Getting the type of 'x' (line 1102)
        x_50752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1102, 22), 'x', False)
        # Getting the type of 'y' (line 1102)
        y_50753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1102, 25), 'y', False)
        unicode_50754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1102, 28), 'unicode', u'BOX')
        # Getting the type of 'w' (line 1102)
        w_50755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1102, 35), 'w', False)
        # Getting the type of 'h' (line 1102)
        h_50756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1102, 38), 'h', False)
        # Processing the call keyword arguments (line 1102)
        kwargs_50757 = {}
        # Getting the type of 'print' (line 1102)
        print_50751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1102, 16), 'print', False)
        # Calling print(args, kwargs) (line 1102)
        print_call_result_50758 = invoke(stypy.reporting.localization.Localization(__file__, 1102, 16), print_50751, *[x_50752, y_50753, unicode_50754, w_50755, h_50756], **kwargs_50757)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 1091)
        exit___50759 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1091, 9), Dvi_call_result_50687, '__exit__')
        with_exit_50760 = invoke(stypy.reporting.localization.Localization(__file__, 1091, 9), exit___50759, None, None, None)



# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
