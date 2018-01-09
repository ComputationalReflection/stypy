
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # -*- coding: utf-8 -*-
2: 
3: from __future__ import (absolute_import, division, print_function,
4:                         unicode_literals)
5: 
6: from collections import OrderedDict
7: 
8: import six
9: from six.moves import zip
10: 
11: import warnings
12: 
13: import numpy as np
14: 
15: from matplotlib.path import Path
16: from matplotlib import rcParams
17: import matplotlib.font_manager as font_manager
18: from matplotlib.ft2font import KERNING_DEFAULT, LOAD_NO_HINTING
19: from matplotlib.ft2font import LOAD_TARGET_LIGHT
20: from matplotlib.mathtext import MathTextParser
21: import matplotlib.dviread as dviread
22: from matplotlib.font_manager import FontProperties, get_font
23: from matplotlib.transforms import Affine2D
24: from six.moves.urllib.parse import quote as urllib_quote
25: 
26: 
27: class TextToPath(object):
28:     '''
29:     A class that convert a given text to a path using ttf fonts.
30:     '''
31: 
32:     FONT_SCALE = 100.
33:     DPI = 72
34: 
35:     def __init__(self):
36:         '''
37:         Initialization
38:         '''
39:         self.mathtext_parser = MathTextParser('path')
40:         self.tex_font_map = None
41: 
42:         from matplotlib.cbook import maxdict
43:         self._ps_fontd = maxdict(50)
44: 
45:         self._texmanager = None
46: 
47:         self._adobe_standard_encoding = None
48: 
49:     def _get_adobe_standard_encoding(self):
50:         enc_name = dviread.find_tex_file('8a.enc')
51:         enc = dviread.Encoding(enc_name)
52:         return {c: i for i, c in enumerate(enc.encoding)}
53: 
54:     def _get_font(self, prop):
55:         '''
56:         find a ttf font.
57:         '''
58:         fname = font_manager.findfont(prop)
59:         font = get_font(fname)
60:         font.set_size(self.FONT_SCALE, self.DPI)
61: 
62:         return font
63: 
64:     def _get_hinting_flag(self):
65:         return LOAD_NO_HINTING
66: 
67:     def _get_char_id(self, font, ccode):
68:         '''
69:         Return a unique id for the given font and character-code set.
70:         '''
71:         sfnt = font.get_sfnt()
72:         try:
73:             ps_name = sfnt[(1, 0, 0, 6)].decode('macroman')
74:         except KeyError:
75:             ps_name = sfnt[(3, 1, 0x0409, 6)].decode('utf-16be')
76:         char_id = urllib_quote('%s-%x' % (ps_name, ccode))
77:         return char_id
78: 
79:     def _get_char_id_ps(self, font, ccode):
80:         '''
81:         Return a unique id for the given font and character-code set (for tex).
82:         '''
83:         ps_name = font.get_ps_font_info()[2]
84:         char_id = urllib_quote('%s-%d' % (ps_name, ccode))
85:         return char_id
86: 
87:     def glyph_to_path(self, font, currx=0.):
88:         '''
89:         convert the ft2font glyph to vertices and codes.
90:         '''
91:         verts, codes = font.get_path()
92:         if currx != 0.0:
93:             verts[:, 0] += currx
94:         return verts, codes
95: 
96:     def get_text_width_height_descent(self, s, prop, ismath):
97:         if rcParams['text.usetex']:
98:             texmanager = self.get_texmanager()
99:             fontsize = prop.get_size_in_points()
100:             w, h, d = texmanager.get_text_width_height_descent(s, fontsize,
101:                                                                renderer=None)
102:             return w, h, d
103: 
104:         fontsize = prop.get_size_in_points()
105:         scale = float(fontsize) / self.FONT_SCALE
106: 
107:         if ismath:
108:             prop = prop.copy()
109:             prop.set_size(self.FONT_SCALE)
110: 
111:             width, height, descent, trash, used_characters = \
112:                 self.mathtext_parser.parse(s, 72, prop)
113:             return width * scale, height * scale, descent * scale
114: 
115:         font = self._get_font(prop)
116:         font.set_text(s, 0.0, flags=LOAD_NO_HINTING)
117:         w, h = font.get_width_height()
118:         w /= 64.0  # convert from subpixels
119:         h /= 64.0
120:         d = font.get_descent()
121:         d /= 64.0
122:         return w * scale, h * scale, d * scale
123: 
124:     def get_text_path(self, prop, s, ismath=False, usetex=False):
125:         '''
126:         convert text *s* to path (a tuple of vertices and codes for
127:         matplotlib.path.Path).
128: 
129:         *prop*
130:           font property
131: 
132:         *s*
133:           text to be converted
134: 
135:         *usetex*
136:           If True, use matplotlib usetex mode.
137: 
138:         *ismath*
139:           If True, use mathtext parser. Effective only if usetex == False.
140: 
141: 
142:         '''
143:         if not usetex:
144:             if not ismath:
145:                 font = self._get_font(prop)
146:                 glyph_info, glyph_map, rects = self.get_glyphs_with_font(
147:                                                     font, s)
148:             else:
149:                 glyph_info, glyph_map, rects = self.get_glyphs_mathtext(
150:                                                     prop, s)
151:         else:
152:             glyph_info, glyph_map, rects = self.get_glyphs_tex(prop, s)
153: 
154:         verts, codes = [], []
155: 
156:         for glyph_id, xposition, yposition, scale in glyph_info:
157:             verts1, codes1 = glyph_map[glyph_id]
158:             if len(verts1):
159:                 verts1 = np.array(verts1) * scale + [xposition, yposition]
160:                 verts.extend(verts1)
161:                 codes.extend(codes1)
162: 
163:         for verts1, codes1 in rects:
164:             verts.extend(verts1)
165:             codes.extend(codes1)
166: 
167:         return verts, codes
168: 
169:     def get_glyphs_with_font(self, font, s, glyph_map=None,
170:                              return_new_glyphs_only=False):
171:         '''
172:         convert the string *s* to vertices and codes using the
173:         provided ttf font.
174:         '''
175: 
176:         # Mostly copied from backend_svg.py.
177: 
178:         lastgind = None
179: 
180:         currx = 0
181:         xpositions = []
182:         glyph_ids = []
183: 
184:         if glyph_map is None:
185:             glyph_map = OrderedDict()
186: 
187:         if return_new_glyphs_only:
188:             glyph_map_new = OrderedDict()
189:         else:
190:             glyph_map_new = glyph_map
191: 
192:         # I'm not sure if I get kernings right. Needs to be verified. -JJL
193: 
194:         for c in s:
195:             ccode = ord(c)
196:             gind = font.get_char_index(ccode)
197:             if gind is None:
198:                 ccode = ord('?')
199:                 gind = 0
200: 
201:             if lastgind is not None:
202:                 kern = font.get_kerning(lastgind, gind, KERNING_DEFAULT)
203:             else:
204:                 kern = 0
205: 
206:             glyph = font.load_char(ccode, flags=LOAD_NO_HINTING)
207:             horiz_advance = (glyph.linearHoriAdvance / 65536.0)
208: 
209:             char_id = self._get_char_id(font, ccode)
210:             if char_id not in glyph_map:
211:                 glyph_map_new[char_id] = self.glyph_to_path(font)
212: 
213:             currx += (kern / 64.0)
214: 
215:             xpositions.append(currx)
216:             glyph_ids.append(char_id)
217: 
218:             currx += horiz_advance
219: 
220:             lastgind = gind
221: 
222:         ypositions = [0] * len(xpositions)
223:         sizes = [1.] * len(xpositions)
224: 
225:         rects = []
226: 
227:         return (list(zip(glyph_ids, xpositions, ypositions, sizes)),
228:                      glyph_map_new, rects)
229: 
230:     def get_glyphs_mathtext(self, prop, s, glyph_map=None,
231:                             return_new_glyphs_only=False):
232:         '''
233:         convert the string *s* to vertices and codes by parsing it with
234:         mathtext.
235:         '''
236: 
237:         prop = prop.copy()
238:         prop.set_size(self.FONT_SCALE)
239: 
240:         width, height, descent, glyphs, rects = self.mathtext_parser.parse(
241:             s, self.DPI, prop)
242: 
243:         if not glyph_map:
244:             glyph_map = OrderedDict()
245: 
246:         if return_new_glyphs_only:
247:             glyph_map_new = OrderedDict()
248:         else:
249:             glyph_map_new = glyph_map
250: 
251:         xpositions = []
252:         ypositions = []
253:         glyph_ids = []
254:         sizes = []
255: 
256:         currx, curry = 0, 0
257:         for font, fontsize, ccode, ox, oy in glyphs:
258:             char_id = self._get_char_id(font, ccode)
259:             if char_id not in glyph_map:
260:                 font.clear()
261:                 font.set_size(self.FONT_SCALE, self.DPI)
262:                 glyph = font.load_char(ccode, flags=LOAD_NO_HINTING)
263:                 glyph_map_new[char_id] = self.glyph_to_path(font)
264: 
265:             xpositions.append(ox)
266:             ypositions.append(oy)
267:             glyph_ids.append(char_id)
268:             size = fontsize / self.FONT_SCALE
269:             sizes.append(size)
270: 
271:         myrects = []
272:         for ox, oy, w, h in rects:
273:             vert1 = [(ox, oy), (ox, oy + h), (ox + w, oy + h),
274:                      (ox + w, oy), (ox, oy), (0, 0)]
275:             code1 = [Path.MOVETO,
276:                      Path.LINETO, Path.LINETO, Path.LINETO, Path.LINETO,
277:                      Path.CLOSEPOLY]
278:             myrects.append((vert1, code1))
279: 
280:         return (list(zip(glyph_ids, xpositions, ypositions, sizes)),
281:                 glyph_map_new, myrects)
282: 
283:     def get_texmanager(self):
284:         '''
285:         return the :class:`matplotlib.texmanager.TexManager` instance
286:         '''
287:         if self._texmanager is None:
288:             from matplotlib.texmanager import TexManager
289:             self._texmanager = TexManager()
290:         return self._texmanager
291: 
292:     def get_glyphs_tex(self, prop, s, glyph_map=None,
293:                        return_new_glyphs_only=False):
294:         '''
295:         convert the string *s* to vertices and codes using matplotlib's usetex
296:         mode.
297:         '''
298: 
299:         # codes are modstly borrowed from pdf backend.
300: 
301:         texmanager = self.get_texmanager()
302: 
303:         if self.tex_font_map is None:
304:             self.tex_font_map = dviread.PsfontsMap(
305:                                     dviread.find_tex_file('pdftex.map'))
306: 
307:         if self._adobe_standard_encoding is None:
308:             self._adobe_standard_encoding = self._get_adobe_standard_encoding()
309: 
310:         fontsize = prop.get_size_in_points()
311:         if hasattr(texmanager, "get_dvi"):
312:             dvifilelike = texmanager.get_dvi(s, self.FONT_SCALE)
313:             dvi = dviread.DviFromFileLike(dvifilelike, self.DPI)
314:         else:
315:             dvifile = texmanager.make_dvi(s, self.FONT_SCALE)
316:             dvi = dviread.Dvi(dvifile, self.DPI)
317:         with dvi:
318:             page = next(iter(dvi))
319: 
320:         if glyph_map is None:
321:             glyph_map = OrderedDict()
322: 
323:         if return_new_glyphs_only:
324:             glyph_map_new = OrderedDict()
325:         else:
326:             glyph_map_new = glyph_map
327: 
328:         glyph_ids, xpositions, ypositions, sizes = [], [], [], []
329: 
330:         # Gather font information and do some setup for combining
331:         # characters into strings.
332:         # oldfont, seq = None, []
333:         for x1, y1, dvifont, glyph, width in page.text:
334:             font_and_encoding = self._ps_fontd.get(dvifont.texname)
335:             font_bunch = self.tex_font_map[dvifont.texname]
336: 
337:             if font_and_encoding is None:
338:                 if font_bunch.filename is None:
339:                     raise ValueError(
340:                         ("No usable font file found for %s (%s). "
341:                          "The font may lack a Type-1 version.")
342:                         % (font_bunch.psname, dvifont.texname))
343: 
344:                 font = get_font(font_bunch.filename)
345: 
346:                 for charmap_name, charmap_code in [("ADOBE_CUSTOM",
347:                                                     1094992451),
348:                                                    ("ADOBE_STANDARD",
349:                                                     1094995778)]:
350:                     try:
351:                         font.select_charmap(charmap_code)
352:                     except (ValueError, RuntimeError):
353:                         pass
354:                     else:
355:                         break
356:                 else:
357:                     charmap_name = ""
358:                     warnings.warn("No supported encoding in font (%s)." %
359:                                   font_bunch.filename)
360: 
361:                 if charmap_name == "ADOBE_STANDARD" and font_bunch.encoding:
362:                     enc0 = dviread.Encoding(font_bunch.encoding)
363:                     enc = {i: self._adobe_standard_encoding.get(c, None)
364:                            for i, c in enumerate(enc0.encoding)}
365:                 else:
366:                     enc = {}
367:                 self._ps_fontd[dvifont.texname] = font, enc
368: 
369:             else:
370:                 font, enc = font_and_encoding
371: 
372:             ft2font_flag = LOAD_TARGET_LIGHT
373: 
374:             char_id = self._get_char_id_ps(font, glyph)
375: 
376:             if char_id not in glyph_map:
377:                 font.clear()
378:                 font.set_size(self.FONT_SCALE, self.DPI)
379:                 if enc:
380:                     charcode = enc.get(glyph, None)
381:                 else:
382:                     charcode = glyph
383: 
384:                 if charcode is not None:
385:                     glyph0 = font.load_char(charcode, flags=ft2font_flag)
386:                 else:
387:                     warnings.warn("The glyph (%d) of font (%s) cannot be "
388:                                   "converted with the encoding. Glyph may "
389:                                   "be wrong" % (glyph, font_bunch.filename))
390: 
391:                     glyph0 = font.load_char(glyph, flags=ft2font_flag)
392: 
393:                 glyph_map_new[char_id] = self.glyph_to_path(font)
394: 
395:             glyph_ids.append(char_id)
396:             xpositions.append(x1)
397:             ypositions.append(y1)
398:             sizes.append(dvifont.size / self.FONT_SCALE)
399: 
400:         myrects = []
401: 
402:         for ox, oy, h, w in page.boxes:
403:             vert1 = [(ox, oy), (ox + w, oy), (ox + w, oy + h),
404:                      (ox, oy + h), (ox, oy), (0, 0)]
405:             code1 = [Path.MOVETO,
406:                      Path.LINETO, Path.LINETO, Path.LINETO, Path.LINETO,
407:                      Path.CLOSEPOLY]
408:             myrects.append((vert1, code1))
409: 
410:         return (list(zip(glyph_ids, xpositions, ypositions, sizes)),
411:                 glyph_map_new, myrects)
412: 
413: 
414: text_to_path = TextToPath()
415: 
416: 
417: class TextPath(Path):
418:     '''
419:     Create a path from the text.
420:     '''
421: 
422:     def __init__(self, xy, s, size=None, prop=None,
423:                  _interpolation_steps=1, usetex=False,
424:                  *kl, **kwargs):
425:         '''
426:         Create a path from the text. No support for TeX yet. Note that
427:         it simply is a path, not an artist. You need to use the
428:         PathPatch (or other artists) to draw this path onto the
429:         canvas.
430: 
431:         xy : position of the text.
432:         s : text
433:         size : font size
434:         prop : font property
435:         '''
436: 
437:         if prop is None:
438:             prop = FontProperties()
439: 
440:         if size is None:
441:             size = prop.get_size_in_points()
442: 
443:         self._xy = xy
444:         self.set_size(size)
445: 
446:         self._cached_vertices = None
447: 
448:         self._vertices, self._codes = self.text_get_vertices_codes(
449:                                             prop, s,
450:                                             usetex=usetex)
451: 
452:         self._should_simplify = False
453:         self._simplify_threshold = rcParams['path.simplify_threshold']
454:         self._has_nonfinite = False
455:         self._interpolation_steps = _interpolation_steps
456: 
457:     def set_size(self, size):
458:         '''
459:         set the size of the text
460:         '''
461:         self._size = size
462:         self._invalid = True
463: 
464:     def get_size(self):
465:         '''
466:         get the size of the text
467:         '''
468:         return self._size
469: 
470:     def _get_vertices(self):
471:         '''
472:         Return the cached path after updating it if necessary.
473:         '''
474:         self._revalidate_path()
475:         return self._cached_vertices
476: 
477:     def _get_codes(self):
478:         '''
479:         Return the codes
480:         '''
481:         return self._codes
482: 
483:     vertices = property(_get_vertices)
484:     codes = property(_get_codes)
485: 
486:     def _revalidate_path(self):
487:         '''
488:         update the path if necessary.
489: 
490:         The path for the text is initially create with the font size
491:         of FONT_SCALE, and this path is rescaled to other size when
492:         necessary.
493: 
494:         '''
495:         if (self._invalid or
496:             (self._cached_vertices is None)):
497:             tr = Affine2D().scale(
498:                     self._size / text_to_path.FONT_SCALE,
499:                     self._size / text_to_path.FONT_SCALE).translate(*self._xy)
500:             self._cached_vertices = tr.transform(self._vertices)
501:             self._invalid = False
502: 
503:     def is_math_text(self, s):
504:         '''
505:         Returns True if the given string *s* contains any mathtext.
506:         '''
507:         # copied from Text.is_math_text -JJL
508: 
509:         # Did we find an even number of non-escaped dollar signs?
510:         # If so, treat is as math text.
511:         dollar_count = s.count(r'$') - s.count(r'\$')
512:         even_dollars = (dollar_count > 0 and dollar_count % 2 == 0)
513: 
514:         if rcParams['text.usetex']:
515:             return s, 'TeX'
516: 
517:         if even_dollars:
518:             return s, True
519:         else:
520:             return s.replace(r'\$', '$'), False
521: 
522:     def text_get_vertices_codes(self, prop, s, usetex):
523:         '''
524:         convert the string *s* to vertices and codes using the
525:         provided font property *prop*. Mostly copied from
526:         backend_svg.py.
527:         '''
528: 
529:         if usetex:
530:             verts, codes = text_to_path.get_text_path(prop, s, usetex=True)
531:         else:
532:             clean_line, ismath = self.is_math_text(s)
533:             verts, codes = text_to_path.get_text_path(prop, clean_line,
534:                                                       ismath=ismath)
535: 
536:         return verts, codes
537: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from collections import OrderedDict' statement (line 6)
try:
    from collections import OrderedDict

except:
    OrderedDict = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'collections', None, module_type_store, ['OrderedDict'], [OrderedDict])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'import six' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_144794 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'six')

if (type(import_144794) is not StypyTypeError):

    if (import_144794 != 'pyd_module'):
        __import__(import_144794)
        sys_modules_144795 = sys.modules[import_144794]
        import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'six', sys_modules_144795.module_type_store, module_type_store)
    else:
        import six

        import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'six', six, module_type_store)

else:
    # Assigning a type to the variable 'six' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'six', import_144794)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from six.moves import zip' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_144796 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'six.moves')

if (type(import_144796) is not StypyTypeError):

    if (import_144796 != 'pyd_module'):
        __import__(import_144796)
        sys_modules_144797 = sys.modules[import_144796]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'six.moves', sys_modules_144797.module_type_store, module_type_store, ['zip'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_144797, sys_modules_144797.module_type_store, module_type_store)
    else:
        from six.moves import zip

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'six.moves', None, module_type_store, ['zip'], [zip])

else:
    # Assigning a type to the variable 'six.moves' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'six.moves', import_144796)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'import warnings' statement (line 11)
import warnings

import_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'warnings', warnings, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'import numpy' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_144798 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'numpy')

if (type(import_144798) is not StypyTypeError):

    if (import_144798 != 'pyd_module'):
        __import__(import_144798)
        sys_modules_144799 = sys.modules[import_144798]
        import_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'np', sys_modules_144799.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'numpy', import_144798)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 0))

# 'from matplotlib.path import Path' statement (line 15)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_144800 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'matplotlib.path')

if (type(import_144800) is not StypyTypeError):

    if (import_144800 != 'pyd_module'):
        __import__(import_144800)
        sys_modules_144801 = sys.modules[import_144800]
        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'matplotlib.path', sys_modules_144801.module_type_store, module_type_store, ['Path'])
        nest_module(stypy.reporting.localization.Localization(__file__, 15, 0), __file__, sys_modules_144801, sys_modules_144801.module_type_store, module_type_store)
    else:
        from matplotlib.path import Path

        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'matplotlib.path', None, module_type_store, ['Path'], [Path])

else:
    # Assigning a type to the variable 'matplotlib.path' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'matplotlib.path', import_144800)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 16, 0))

# 'from matplotlib import rcParams' statement (line 16)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_144802 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'matplotlib')

if (type(import_144802) is not StypyTypeError):

    if (import_144802 != 'pyd_module'):
        __import__(import_144802)
        sys_modules_144803 = sys.modules[import_144802]
        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'matplotlib', sys_modules_144803.module_type_store, module_type_store, ['rcParams'])
        nest_module(stypy.reporting.localization.Localization(__file__, 16, 0), __file__, sys_modules_144803, sys_modules_144803.module_type_store, module_type_store)
    else:
        from matplotlib import rcParams

        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'matplotlib', None, module_type_store, ['rcParams'], [rcParams])

else:
    # Assigning a type to the variable 'matplotlib' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'matplotlib', import_144802)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 17, 0))

# 'import matplotlib.font_manager' statement (line 17)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_144804 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'matplotlib.font_manager')

if (type(import_144804) is not StypyTypeError):

    if (import_144804 != 'pyd_module'):
        __import__(import_144804)
        sys_modules_144805 = sys.modules[import_144804]
        import_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'font_manager', sys_modules_144805.module_type_store, module_type_store)
    else:
        import matplotlib.font_manager as font_manager

        import_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'font_manager', matplotlib.font_manager, module_type_store)

else:
    # Assigning a type to the variable 'matplotlib.font_manager' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'matplotlib.font_manager', import_144804)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 18, 0))

# 'from matplotlib.ft2font import KERNING_DEFAULT, LOAD_NO_HINTING' statement (line 18)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_144806 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'matplotlib.ft2font')

if (type(import_144806) is not StypyTypeError):

    if (import_144806 != 'pyd_module'):
        __import__(import_144806)
        sys_modules_144807 = sys.modules[import_144806]
        import_from_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'matplotlib.ft2font', sys_modules_144807.module_type_store, module_type_store, ['KERNING_DEFAULT', 'LOAD_NO_HINTING'])
        nest_module(stypy.reporting.localization.Localization(__file__, 18, 0), __file__, sys_modules_144807, sys_modules_144807.module_type_store, module_type_store)
    else:
        from matplotlib.ft2font import KERNING_DEFAULT, LOAD_NO_HINTING

        import_from_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'matplotlib.ft2font', None, module_type_store, ['KERNING_DEFAULT', 'LOAD_NO_HINTING'], [KERNING_DEFAULT, LOAD_NO_HINTING])

else:
    # Assigning a type to the variable 'matplotlib.ft2font' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'matplotlib.ft2font', import_144806)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 19, 0))

# 'from matplotlib.ft2font import LOAD_TARGET_LIGHT' statement (line 19)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_144808 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'matplotlib.ft2font')

if (type(import_144808) is not StypyTypeError):

    if (import_144808 != 'pyd_module'):
        __import__(import_144808)
        sys_modules_144809 = sys.modules[import_144808]
        import_from_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'matplotlib.ft2font', sys_modules_144809.module_type_store, module_type_store, ['LOAD_TARGET_LIGHT'])
        nest_module(stypy.reporting.localization.Localization(__file__, 19, 0), __file__, sys_modules_144809, sys_modules_144809.module_type_store, module_type_store)
    else:
        from matplotlib.ft2font import LOAD_TARGET_LIGHT

        import_from_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'matplotlib.ft2font', None, module_type_store, ['LOAD_TARGET_LIGHT'], [LOAD_TARGET_LIGHT])

else:
    # Assigning a type to the variable 'matplotlib.ft2font' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'matplotlib.ft2font', import_144808)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 20, 0))

# 'from matplotlib.mathtext import MathTextParser' statement (line 20)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_144810 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'matplotlib.mathtext')

if (type(import_144810) is not StypyTypeError):

    if (import_144810 != 'pyd_module'):
        __import__(import_144810)
        sys_modules_144811 = sys.modules[import_144810]
        import_from_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'matplotlib.mathtext', sys_modules_144811.module_type_store, module_type_store, ['MathTextParser'])
        nest_module(stypy.reporting.localization.Localization(__file__, 20, 0), __file__, sys_modules_144811, sys_modules_144811.module_type_store, module_type_store)
    else:
        from matplotlib.mathtext import MathTextParser

        import_from_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'matplotlib.mathtext', None, module_type_store, ['MathTextParser'], [MathTextParser])

else:
    # Assigning a type to the variable 'matplotlib.mathtext' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'matplotlib.mathtext', import_144810)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 21, 0))

# 'import matplotlib.dviread' statement (line 21)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_144812 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'matplotlib.dviread')

if (type(import_144812) is not StypyTypeError):

    if (import_144812 != 'pyd_module'):
        __import__(import_144812)
        sys_modules_144813 = sys.modules[import_144812]
        import_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'dviread', sys_modules_144813.module_type_store, module_type_store)
    else:
        import matplotlib.dviread as dviread

        import_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'dviread', matplotlib.dviread, module_type_store)

else:
    # Assigning a type to the variable 'matplotlib.dviread' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'matplotlib.dviread', import_144812)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 22, 0))

# 'from matplotlib.font_manager import FontProperties, get_font' statement (line 22)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_144814 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'matplotlib.font_manager')

if (type(import_144814) is not StypyTypeError):

    if (import_144814 != 'pyd_module'):
        __import__(import_144814)
        sys_modules_144815 = sys.modules[import_144814]
        import_from_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'matplotlib.font_manager', sys_modules_144815.module_type_store, module_type_store, ['FontProperties', 'get_font'])
        nest_module(stypy.reporting.localization.Localization(__file__, 22, 0), __file__, sys_modules_144815, sys_modules_144815.module_type_store, module_type_store)
    else:
        from matplotlib.font_manager import FontProperties, get_font

        import_from_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'matplotlib.font_manager', None, module_type_store, ['FontProperties', 'get_font'], [FontProperties, get_font])

else:
    # Assigning a type to the variable 'matplotlib.font_manager' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'matplotlib.font_manager', import_144814)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 23, 0))

# 'from matplotlib.transforms import Affine2D' statement (line 23)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_144816 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 23, 0), 'matplotlib.transforms')

if (type(import_144816) is not StypyTypeError):

    if (import_144816 != 'pyd_module'):
        __import__(import_144816)
        sys_modules_144817 = sys.modules[import_144816]
        import_from_module(stypy.reporting.localization.Localization(__file__, 23, 0), 'matplotlib.transforms', sys_modules_144817.module_type_store, module_type_store, ['Affine2D'])
        nest_module(stypy.reporting.localization.Localization(__file__, 23, 0), __file__, sys_modules_144817, sys_modules_144817.module_type_store, module_type_store)
    else:
        from matplotlib.transforms import Affine2D

        import_from_module(stypy.reporting.localization.Localization(__file__, 23, 0), 'matplotlib.transforms', None, module_type_store, ['Affine2D'], [Affine2D])

else:
    # Assigning a type to the variable 'matplotlib.transforms' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'matplotlib.transforms', import_144816)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 24, 0))

# 'from six.moves.urllib.parse import urllib_quote' statement (line 24)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_144818 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'six.moves.urllib.parse')

if (type(import_144818) is not StypyTypeError):

    if (import_144818 != 'pyd_module'):
        __import__(import_144818)
        sys_modules_144819 = sys.modules[import_144818]
        import_from_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'six.moves.urllib.parse', sys_modules_144819.module_type_store, module_type_store, ['quote'])
        nest_module(stypy.reporting.localization.Localization(__file__, 24, 0), __file__, sys_modules_144819, sys_modules_144819.module_type_store, module_type_store)
    else:
        from six.moves.urllib.parse import quote as urllib_quote

        import_from_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'six.moves.urllib.parse', None, module_type_store, ['quote'], [urllib_quote])

else:
    # Assigning a type to the variable 'six.moves.urllib.parse' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'six.moves.urllib.parse', import_144818)

# Adding an alias
module_type_store.add_alias('urllib_quote', 'quote')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

# Declaration of the 'TextToPath' class

class TextToPath(object, ):
    unicode_144820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, (-1)), 'unicode', u'\n    A class that convert a given text to a path using ttf fonts.\n    ')
    
    # Assigning a Num to a Name (line 32):
    
    # Assigning a Num to a Name (line 33):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 35, 4, False)
        # Assigning a type to the variable 'self' (line 36)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TextToPath.__init__', [], None, None, defaults, varargs, kwargs)

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

        unicode_144821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, (-1)), 'unicode', u'\n        Initialization\n        ')
        
        # Assigning a Call to a Attribute (line 39):
        
        # Assigning a Call to a Attribute (line 39):
        
        # Call to MathTextParser(...): (line 39)
        # Processing the call arguments (line 39)
        unicode_144823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 46), 'unicode', u'path')
        # Processing the call keyword arguments (line 39)
        kwargs_144824 = {}
        # Getting the type of 'MathTextParser' (line 39)
        MathTextParser_144822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 31), 'MathTextParser', False)
        # Calling MathTextParser(args, kwargs) (line 39)
        MathTextParser_call_result_144825 = invoke(stypy.reporting.localization.Localization(__file__, 39, 31), MathTextParser_144822, *[unicode_144823], **kwargs_144824)
        
        # Getting the type of 'self' (line 39)
        self_144826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'self')
        # Setting the type of the member 'mathtext_parser' of a type (line 39)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 8), self_144826, 'mathtext_parser', MathTextParser_call_result_144825)
        
        # Assigning a Name to a Attribute (line 40):
        
        # Assigning a Name to a Attribute (line 40):
        # Getting the type of 'None' (line 40)
        None_144827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 28), 'None')
        # Getting the type of 'self' (line 40)
        self_144828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'self')
        # Setting the type of the member 'tex_font_map' of a type (line 40)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 8), self_144828, 'tex_font_map', None_144827)
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 42, 8))
        
        # 'from matplotlib.cbook import maxdict' statement (line 42)
        update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
        import_144829 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 42, 8), 'matplotlib.cbook')

        if (type(import_144829) is not StypyTypeError):

            if (import_144829 != 'pyd_module'):
                __import__(import_144829)
                sys_modules_144830 = sys.modules[import_144829]
                import_from_module(stypy.reporting.localization.Localization(__file__, 42, 8), 'matplotlib.cbook', sys_modules_144830.module_type_store, module_type_store, ['maxdict'])
                nest_module(stypy.reporting.localization.Localization(__file__, 42, 8), __file__, sys_modules_144830, sys_modules_144830.module_type_store, module_type_store)
            else:
                from matplotlib.cbook import maxdict

                import_from_module(stypy.reporting.localization.Localization(__file__, 42, 8), 'matplotlib.cbook', None, module_type_store, ['maxdict'], [maxdict])

        else:
            # Assigning a type to the variable 'matplotlib.cbook' (line 42)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'matplotlib.cbook', import_144829)

        remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')
        
        
        # Assigning a Call to a Attribute (line 43):
        
        # Assigning a Call to a Attribute (line 43):
        
        # Call to maxdict(...): (line 43)
        # Processing the call arguments (line 43)
        int_144832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 33), 'int')
        # Processing the call keyword arguments (line 43)
        kwargs_144833 = {}
        # Getting the type of 'maxdict' (line 43)
        maxdict_144831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 25), 'maxdict', False)
        # Calling maxdict(args, kwargs) (line 43)
        maxdict_call_result_144834 = invoke(stypy.reporting.localization.Localization(__file__, 43, 25), maxdict_144831, *[int_144832], **kwargs_144833)
        
        # Getting the type of 'self' (line 43)
        self_144835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'self')
        # Setting the type of the member '_ps_fontd' of a type (line 43)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 8), self_144835, '_ps_fontd', maxdict_call_result_144834)
        
        # Assigning a Name to a Attribute (line 45):
        
        # Assigning a Name to a Attribute (line 45):
        # Getting the type of 'None' (line 45)
        None_144836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 27), 'None')
        # Getting the type of 'self' (line 45)
        self_144837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'self')
        # Setting the type of the member '_texmanager' of a type (line 45)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 8), self_144837, '_texmanager', None_144836)
        
        # Assigning a Name to a Attribute (line 47):
        
        # Assigning a Name to a Attribute (line 47):
        # Getting the type of 'None' (line 47)
        None_144838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 40), 'None')
        # Getting the type of 'self' (line 47)
        self_144839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'self')
        # Setting the type of the member '_adobe_standard_encoding' of a type (line 47)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 8), self_144839, '_adobe_standard_encoding', None_144838)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def _get_adobe_standard_encoding(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_get_adobe_standard_encoding'
        module_type_store = module_type_store.open_function_context('_get_adobe_standard_encoding', 49, 4, False)
        # Assigning a type to the variable 'self' (line 50)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TextToPath._get_adobe_standard_encoding.__dict__.__setitem__('stypy_localization', localization)
        TextToPath._get_adobe_standard_encoding.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TextToPath._get_adobe_standard_encoding.__dict__.__setitem__('stypy_type_store', module_type_store)
        TextToPath._get_adobe_standard_encoding.__dict__.__setitem__('stypy_function_name', 'TextToPath._get_adobe_standard_encoding')
        TextToPath._get_adobe_standard_encoding.__dict__.__setitem__('stypy_param_names_list', [])
        TextToPath._get_adobe_standard_encoding.__dict__.__setitem__('stypy_varargs_param_name', None)
        TextToPath._get_adobe_standard_encoding.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TextToPath._get_adobe_standard_encoding.__dict__.__setitem__('stypy_call_defaults', defaults)
        TextToPath._get_adobe_standard_encoding.__dict__.__setitem__('stypy_call_varargs', varargs)
        TextToPath._get_adobe_standard_encoding.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TextToPath._get_adobe_standard_encoding.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TextToPath._get_adobe_standard_encoding', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_get_adobe_standard_encoding', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_get_adobe_standard_encoding(...)' code ##################

        
        # Assigning a Call to a Name (line 50):
        
        # Assigning a Call to a Name (line 50):
        
        # Call to find_tex_file(...): (line 50)
        # Processing the call arguments (line 50)
        unicode_144842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 41), 'unicode', u'8a.enc')
        # Processing the call keyword arguments (line 50)
        kwargs_144843 = {}
        # Getting the type of 'dviread' (line 50)
        dviread_144840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 19), 'dviread', False)
        # Obtaining the member 'find_tex_file' of a type (line 50)
        find_tex_file_144841 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 19), dviread_144840, 'find_tex_file')
        # Calling find_tex_file(args, kwargs) (line 50)
        find_tex_file_call_result_144844 = invoke(stypy.reporting.localization.Localization(__file__, 50, 19), find_tex_file_144841, *[unicode_144842], **kwargs_144843)
        
        # Assigning a type to the variable 'enc_name' (line 50)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'enc_name', find_tex_file_call_result_144844)
        
        # Assigning a Call to a Name (line 51):
        
        # Assigning a Call to a Name (line 51):
        
        # Call to Encoding(...): (line 51)
        # Processing the call arguments (line 51)
        # Getting the type of 'enc_name' (line 51)
        enc_name_144847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 31), 'enc_name', False)
        # Processing the call keyword arguments (line 51)
        kwargs_144848 = {}
        # Getting the type of 'dviread' (line 51)
        dviread_144845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 14), 'dviread', False)
        # Obtaining the member 'Encoding' of a type (line 51)
        Encoding_144846 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 14), dviread_144845, 'Encoding')
        # Calling Encoding(args, kwargs) (line 51)
        Encoding_call_result_144849 = invoke(stypy.reporting.localization.Localization(__file__, 51, 14), Encoding_144846, *[enc_name_144847], **kwargs_144848)
        
        # Assigning a type to the variable 'enc' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'enc', Encoding_call_result_144849)
        # Calculating dict comprehension
        module_type_store = module_type_store.open_function_context('dict comprehension expression', 52, 16, True)
        # Calculating comprehension expression
        
        # Call to enumerate(...): (line 52)
        # Processing the call arguments (line 52)
        # Getting the type of 'enc' (line 52)
        enc_144853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 43), 'enc', False)
        # Obtaining the member 'encoding' of a type (line 52)
        encoding_144854 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 43), enc_144853, 'encoding')
        # Processing the call keyword arguments (line 52)
        kwargs_144855 = {}
        # Getting the type of 'enumerate' (line 52)
        enumerate_144852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 33), 'enumerate', False)
        # Calling enumerate(args, kwargs) (line 52)
        enumerate_call_result_144856 = invoke(stypy.reporting.localization.Localization(__file__, 52, 33), enumerate_144852, *[encoding_144854], **kwargs_144855)
        
        comprehension_144857 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 16), enumerate_call_result_144856)
        # Assigning a type to the variable 'i' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 16), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 16), comprehension_144857))
        # Assigning a type to the variable 'c' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 16), 'c', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 16), comprehension_144857))
        # Getting the type of 'c' (line 52)
        c_144850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 16), 'c')
        # Getting the type of 'i' (line 52)
        i_144851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 19), 'i')
        dict_144858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 16), 'dict')
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 16), dict_144858, (c_144850, i_144851))
        # Assigning a type to the variable 'stypy_return_type' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'stypy_return_type', dict_144858)
        
        # ################# End of '_get_adobe_standard_encoding(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_get_adobe_standard_encoding' in the type store
        # Getting the type of 'stypy_return_type' (line 49)
        stypy_return_type_144859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_144859)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_get_adobe_standard_encoding'
        return stypy_return_type_144859


    @norecursion
    def _get_font(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_get_font'
        module_type_store = module_type_store.open_function_context('_get_font', 54, 4, False)
        # Assigning a type to the variable 'self' (line 55)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TextToPath._get_font.__dict__.__setitem__('stypy_localization', localization)
        TextToPath._get_font.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TextToPath._get_font.__dict__.__setitem__('stypy_type_store', module_type_store)
        TextToPath._get_font.__dict__.__setitem__('stypy_function_name', 'TextToPath._get_font')
        TextToPath._get_font.__dict__.__setitem__('stypy_param_names_list', ['prop'])
        TextToPath._get_font.__dict__.__setitem__('stypy_varargs_param_name', None)
        TextToPath._get_font.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TextToPath._get_font.__dict__.__setitem__('stypy_call_defaults', defaults)
        TextToPath._get_font.__dict__.__setitem__('stypy_call_varargs', varargs)
        TextToPath._get_font.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TextToPath._get_font.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TextToPath._get_font', ['prop'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_get_font', localization, ['prop'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_get_font(...)' code ##################

        unicode_144860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, (-1)), 'unicode', u'\n        find a ttf font.\n        ')
        
        # Assigning a Call to a Name (line 58):
        
        # Assigning a Call to a Name (line 58):
        
        # Call to findfont(...): (line 58)
        # Processing the call arguments (line 58)
        # Getting the type of 'prop' (line 58)
        prop_144863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 38), 'prop', False)
        # Processing the call keyword arguments (line 58)
        kwargs_144864 = {}
        # Getting the type of 'font_manager' (line 58)
        font_manager_144861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 16), 'font_manager', False)
        # Obtaining the member 'findfont' of a type (line 58)
        findfont_144862 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 16), font_manager_144861, 'findfont')
        # Calling findfont(args, kwargs) (line 58)
        findfont_call_result_144865 = invoke(stypy.reporting.localization.Localization(__file__, 58, 16), findfont_144862, *[prop_144863], **kwargs_144864)
        
        # Assigning a type to the variable 'fname' (line 58)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'fname', findfont_call_result_144865)
        
        # Assigning a Call to a Name (line 59):
        
        # Assigning a Call to a Name (line 59):
        
        # Call to get_font(...): (line 59)
        # Processing the call arguments (line 59)
        # Getting the type of 'fname' (line 59)
        fname_144867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 24), 'fname', False)
        # Processing the call keyword arguments (line 59)
        kwargs_144868 = {}
        # Getting the type of 'get_font' (line 59)
        get_font_144866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 15), 'get_font', False)
        # Calling get_font(args, kwargs) (line 59)
        get_font_call_result_144869 = invoke(stypy.reporting.localization.Localization(__file__, 59, 15), get_font_144866, *[fname_144867], **kwargs_144868)
        
        # Assigning a type to the variable 'font' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'font', get_font_call_result_144869)
        
        # Call to set_size(...): (line 60)
        # Processing the call arguments (line 60)
        # Getting the type of 'self' (line 60)
        self_144872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 22), 'self', False)
        # Obtaining the member 'FONT_SCALE' of a type (line 60)
        FONT_SCALE_144873 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 22), self_144872, 'FONT_SCALE')
        # Getting the type of 'self' (line 60)
        self_144874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 39), 'self', False)
        # Obtaining the member 'DPI' of a type (line 60)
        DPI_144875 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 39), self_144874, 'DPI')
        # Processing the call keyword arguments (line 60)
        kwargs_144876 = {}
        # Getting the type of 'font' (line 60)
        font_144870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'font', False)
        # Obtaining the member 'set_size' of a type (line 60)
        set_size_144871 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 8), font_144870, 'set_size')
        # Calling set_size(args, kwargs) (line 60)
        set_size_call_result_144877 = invoke(stypy.reporting.localization.Localization(__file__, 60, 8), set_size_144871, *[FONT_SCALE_144873, DPI_144875], **kwargs_144876)
        
        # Getting the type of 'font' (line 62)
        font_144878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 15), 'font')
        # Assigning a type to the variable 'stypy_return_type' (line 62)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'stypy_return_type', font_144878)
        
        # ################# End of '_get_font(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_get_font' in the type store
        # Getting the type of 'stypy_return_type' (line 54)
        stypy_return_type_144879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_144879)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_get_font'
        return stypy_return_type_144879


    @norecursion
    def _get_hinting_flag(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_get_hinting_flag'
        module_type_store = module_type_store.open_function_context('_get_hinting_flag', 64, 4, False)
        # Assigning a type to the variable 'self' (line 65)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TextToPath._get_hinting_flag.__dict__.__setitem__('stypy_localization', localization)
        TextToPath._get_hinting_flag.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TextToPath._get_hinting_flag.__dict__.__setitem__('stypy_type_store', module_type_store)
        TextToPath._get_hinting_flag.__dict__.__setitem__('stypy_function_name', 'TextToPath._get_hinting_flag')
        TextToPath._get_hinting_flag.__dict__.__setitem__('stypy_param_names_list', [])
        TextToPath._get_hinting_flag.__dict__.__setitem__('stypy_varargs_param_name', None)
        TextToPath._get_hinting_flag.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TextToPath._get_hinting_flag.__dict__.__setitem__('stypy_call_defaults', defaults)
        TextToPath._get_hinting_flag.__dict__.__setitem__('stypy_call_varargs', varargs)
        TextToPath._get_hinting_flag.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TextToPath._get_hinting_flag.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TextToPath._get_hinting_flag', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_get_hinting_flag', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_get_hinting_flag(...)' code ##################

        # Getting the type of 'LOAD_NO_HINTING' (line 65)
        LOAD_NO_HINTING_144880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 15), 'LOAD_NO_HINTING')
        # Assigning a type to the variable 'stypy_return_type' (line 65)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'stypy_return_type', LOAD_NO_HINTING_144880)
        
        # ################# End of '_get_hinting_flag(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_get_hinting_flag' in the type store
        # Getting the type of 'stypy_return_type' (line 64)
        stypy_return_type_144881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_144881)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_get_hinting_flag'
        return stypy_return_type_144881


    @norecursion
    def _get_char_id(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_get_char_id'
        module_type_store = module_type_store.open_function_context('_get_char_id', 67, 4, False)
        # Assigning a type to the variable 'self' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TextToPath._get_char_id.__dict__.__setitem__('stypy_localization', localization)
        TextToPath._get_char_id.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TextToPath._get_char_id.__dict__.__setitem__('stypy_type_store', module_type_store)
        TextToPath._get_char_id.__dict__.__setitem__('stypy_function_name', 'TextToPath._get_char_id')
        TextToPath._get_char_id.__dict__.__setitem__('stypy_param_names_list', ['font', 'ccode'])
        TextToPath._get_char_id.__dict__.__setitem__('stypy_varargs_param_name', None)
        TextToPath._get_char_id.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TextToPath._get_char_id.__dict__.__setitem__('stypy_call_defaults', defaults)
        TextToPath._get_char_id.__dict__.__setitem__('stypy_call_varargs', varargs)
        TextToPath._get_char_id.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TextToPath._get_char_id.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TextToPath._get_char_id', ['font', 'ccode'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_get_char_id', localization, ['font', 'ccode'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_get_char_id(...)' code ##################

        unicode_144882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, (-1)), 'unicode', u'\n        Return a unique id for the given font and character-code set.\n        ')
        
        # Assigning a Call to a Name (line 71):
        
        # Assigning a Call to a Name (line 71):
        
        # Call to get_sfnt(...): (line 71)
        # Processing the call keyword arguments (line 71)
        kwargs_144885 = {}
        # Getting the type of 'font' (line 71)
        font_144883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 15), 'font', False)
        # Obtaining the member 'get_sfnt' of a type (line 71)
        get_sfnt_144884 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 15), font_144883, 'get_sfnt')
        # Calling get_sfnt(args, kwargs) (line 71)
        get_sfnt_call_result_144886 = invoke(stypy.reporting.localization.Localization(__file__, 71, 15), get_sfnt_144884, *[], **kwargs_144885)
        
        # Assigning a type to the variable 'sfnt' (line 71)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'sfnt', get_sfnt_call_result_144886)
        
        
        # SSA begins for try-except statement (line 72)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Name (line 73):
        
        # Assigning a Call to a Name (line 73):
        
        # Call to decode(...): (line 73)
        # Processing the call arguments (line 73)
        unicode_144896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 48), 'unicode', u'macroman')
        # Processing the call keyword arguments (line 73)
        kwargs_144897 = {}
        
        # Obtaining the type of the subscript
        
        # Obtaining an instance of the builtin type 'tuple' (line 73)
        tuple_144887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 73)
        # Adding element type (line 73)
        int_144888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 28), tuple_144887, int_144888)
        # Adding element type (line 73)
        int_144889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 28), tuple_144887, int_144889)
        # Adding element type (line 73)
        int_144890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 28), tuple_144887, int_144890)
        # Adding element type (line 73)
        int_144891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 28), tuple_144887, int_144891)
        
        # Getting the type of 'sfnt' (line 73)
        sfnt_144892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 22), 'sfnt', False)
        # Obtaining the member '__getitem__' of a type (line 73)
        getitem___144893 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 22), sfnt_144892, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 73)
        subscript_call_result_144894 = invoke(stypy.reporting.localization.Localization(__file__, 73, 22), getitem___144893, tuple_144887)
        
        # Obtaining the member 'decode' of a type (line 73)
        decode_144895 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 22), subscript_call_result_144894, 'decode')
        # Calling decode(args, kwargs) (line 73)
        decode_call_result_144898 = invoke(stypy.reporting.localization.Localization(__file__, 73, 22), decode_144895, *[unicode_144896], **kwargs_144897)
        
        # Assigning a type to the variable 'ps_name' (line 73)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 12), 'ps_name', decode_call_result_144898)
        # SSA branch for the except part of a try statement (line 72)
        # SSA branch for the except 'KeyError' branch of a try statement (line 72)
        module_type_store.open_ssa_branch('except')
        
        # Assigning a Call to a Name (line 75):
        
        # Assigning a Call to a Name (line 75):
        
        # Call to decode(...): (line 75)
        # Processing the call arguments (line 75)
        unicode_144908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 53), 'unicode', u'utf-16be')
        # Processing the call keyword arguments (line 75)
        kwargs_144909 = {}
        
        # Obtaining the type of the subscript
        
        # Obtaining an instance of the builtin type 'tuple' (line 75)
        tuple_144899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 75)
        # Adding element type (line 75)
        int_144900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 28), tuple_144899, int_144900)
        # Adding element type (line 75)
        int_144901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 28), tuple_144899, int_144901)
        # Adding element type (line 75)
        int_144902 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 28), tuple_144899, int_144902)
        # Adding element type (line 75)
        int_144903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 28), tuple_144899, int_144903)
        
        # Getting the type of 'sfnt' (line 75)
        sfnt_144904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 22), 'sfnt', False)
        # Obtaining the member '__getitem__' of a type (line 75)
        getitem___144905 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 22), sfnt_144904, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 75)
        subscript_call_result_144906 = invoke(stypy.reporting.localization.Localization(__file__, 75, 22), getitem___144905, tuple_144899)
        
        # Obtaining the member 'decode' of a type (line 75)
        decode_144907 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 22), subscript_call_result_144906, 'decode')
        # Calling decode(args, kwargs) (line 75)
        decode_call_result_144910 = invoke(stypy.reporting.localization.Localization(__file__, 75, 22), decode_144907, *[unicode_144908], **kwargs_144909)
        
        # Assigning a type to the variable 'ps_name' (line 75)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 12), 'ps_name', decode_call_result_144910)
        # SSA join for try-except statement (line 72)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 76):
        
        # Assigning a Call to a Name (line 76):
        
        # Call to urllib_quote(...): (line 76)
        # Processing the call arguments (line 76)
        unicode_144912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 31), 'unicode', u'%s-%x')
        
        # Obtaining an instance of the builtin type 'tuple' (line 76)
        tuple_144913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 42), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 76)
        # Adding element type (line 76)
        # Getting the type of 'ps_name' (line 76)
        ps_name_144914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 42), 'ps_name', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 42), tuple_144913, ps_name_144914)
        # Adding element type (line 76)
        # Getting the type of 'ccode' (line 76)
        ccode_144915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 51), 'ccode', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 42), tuple_144913, ccode_144915)
        
        # Applying the binary operator '%' (line 76)
        result_mod_144916 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 31), '%', unicode_144912, tuple_144913)
        
        # Processing the call keyword arguments (line 76)
        kwargs_144917 = {}
        # Getting the type of 'urllib_quote' (line 76)
        urllib_quote_144911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 18), 'urllib_quote', False)
        # Calling urllib_quote(args, kwargs) (line 76)
        urllib_quote_call_result_144918 = invoke(stypy.reporting.localization.Localization(__file__, 76, 18), urllib_quote_144911, *[result_mod_144916], **kwargs_144917)
        
        # Assigning a type to the variable 'char_id' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'char_id', urllib_quote_call_result_144918)
        # Getting the type of 'char_id' (line 77)
        char_id_144919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 15), 'char_id')
        # Assigning a type to the variable 'stypy_return_type' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'stypy_return_type', char_id_144919)
        
        # ################# End of '_get_char_id(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_get_char_id' in the type store
        # Getting the type of 'stypy_return_type' (line 67)
        stypy_return_type_144920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_144920)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_get_char_id'
        return stypy_return_type_144920


    @norecursion
    def _get_char_id_ps(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_get_char_id_ps'
        module_type_store = module_type_store.open_function_context('_get_char_id_ps', 79, 4, False)
        # Assigning a type to the variable 'self' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TextToPath._get_char_id_ps.__dict__.__setitem__('stypy_localization', localization)
        TextToPath._get_char_id_ps.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TextToPath._get_char_id_ps.__dict__.__setitem__('stypy_type_store', module_type_store)
        TextToPath._get_char_id_ps.__dict__.__setitem__('stypy_function_name', 'TextToPath._get_char_id_ps')
        TextToPath._get_char_id_ps.__dict__.__setitem__('stypy_param_names_list', ['font', 'ccode'])
        TextToPath._get_char_id_ps.__dict__.__setitem__('stypy_varargs_param_name', None)
        TextToPath._get_char_id_ps.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TextToPath._get_char_id_ps.__dict__.__setitem__('stypy_call_defaults', defaults)
        TextToPath._get_char_id_ps.__dict__.__setitem__('stypy_call_varargs', varargs)
        TextToPath._get_char_id_ps.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TextToPath._get_char_id_ps.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TextToPath._get_char_id_ps', ['font', 'ccode'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_get_char_id_ps', localization, ['font', 'ccode'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_get_char_id_ps(...)' code ##################

        unicode_144921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, (-1)), 'unicode', u'\n        Return a unique id for the given font and character-code set (for tex).\n        ')
        
        # Assigning a Subscript to a Name (line 83):
        
        # Assigning a Subscript to a Name (line 83):
        
        # Obtaining the type of the subscript
        int_144922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 42), 'int')
        
        # Call to get_ps_font_info(...): (line 83)
        # Processing the call keyword arguments (line 83)
        kwargs_144925 = {}
        # Getting the type of 'font' (line 83)
        font_144923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 18), 'font', False)
        # Obtaining the member 'get_ps_font_info' of a type (line 83)
        get_ps_font_info_144924 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 18), font_144923, 'get_ps_font_info')
        # Calling get_ps_font_info(args, kwargs) (line 83)
        get_ps_font_info_call_result_144926 = invoke(stypy.reporting.localization.Localization(__file__, 83, 18), get_ps_font_info_144924, *[], **kwargs_144925)
        
        # Obtaining the member '__getitem__' of a type (line 83)
        getitem___144927 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 18), get_ps_font_info_call_result_144926, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 83)
        subscript_call_result_144928 = invoke(stypy.reporting.localization.Localization(__file__, 83, 18), getitem___144927, int_144922)
        
        # Assigning a type to the variable 'ps_name' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'ps_name', subscript_call_result_144928)
        
        # Assigning a Call to a Name (line 84):
        
        # Assigning a Call to a Name (line 84):
        
        # Call to urllib_quote(...): (line 84)
        # Processing the call arguments (line 84)
        unicode_144930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 31), 'unicode', u'%s-%d')
        
        # Obtaining an instance of the builtin type 'tuple' (line 84)
        tuple_144931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 42), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 84)
        # Adding element type (line 84)
        # Getting the type of 'ps_name' (line 84)
        ps_name_144932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 42), 'ps_name', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 42), tuple_144931, ps_name_144932)
        # Adding element type (line 84)
        # Getting the type of 'ccode' (line 84)
        ccode_144933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 51), 'ccode', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 42), tuple_144931, ccode_144933)
        
        # Applying the binary operator '%' (line 84)
        result_mod_144934 = python_operator(stypy.reporting.localization.Localization(__file__, 84, 31), '%', unicode_144930, tuple_144931)
        
        # Processing the call keyword arguments (line 84)
        kwargs_144935 = {}
        # Getting the type of 'urllib_quote' (line 84)
        urllib_quote_144929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 18), 'urllib_quote', False)
        # Calling urllib_quote(args, kwargs) (line 84)
        urllib_quote_call_result_144936 = invoke(stypy.reporting.localization.Localization(__file__, 84, 18), urllib_quote_144929, *[result_mod_144934], **kwargs_144935)
        
        # Assigning a type to the variable 'char_id' (line 84)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'char_id', urllib_quote_call_result_144936)
        # Getting the type of 'char_id' (line 85)
        char_id_144937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 15), 'char_id')
        # Assigning a type to the variable 'stypy_return_type' (line 85)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'stypy_return_type', char_id_144937)
        
        # ################# End of '_get_char_id_ps(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_get_char_id_ps' in the type store
        # Getting the type of 'stypy_return_type' (line 79)
        stypy_return_type_144938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_144938)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_get_char_id_ps'
        return stypy_return_type_144938


    @norecursion
    def glyph_to_path(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        float_144939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 40), 'float')
        defaults = [float_144939]
        # Create a new context for function 'glyph_to_path'
        module_type_store = module_type_store.open_function_context('glyph_to_path', 87, 4, False)
        # Assigning a type to the variable 'self' (line 88)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TextToPath.glyph_to_path.__dict__.__setitem__('stypy_localization', localization)
        TextToPath.glyph_to_path.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TextToPath.glyph_to_path.__dict__.__setitem__('stypy_type_store', module_type_store)
        TextToPath.glyph_to_path.__dict__.__setitem__('stypy_function_name', 'TextToPath.glyph_to_path')
        TextToPath.glyph_to_path.__dict__.__setitem__('stypy_param_names_list', ['font', 'currx'])
        TextToPath.glyph_to_path.__dict__.__setitem__('stypy_varargs_param_name', None)
        TextToPath.glyph_to_path.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TextToPath.glyph_to_path.__dict__.__setitem__('stypy_call_defaults', defaults)
        TextToPath.glyph_to_path.__dict__.__setitem__('stypy_call_varargs', varargs)
        TextToPath.glyph_to_path.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TextToPath.glyph_to_path.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TextToPath.glyph_to_path', ['font', 'currx'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'glyph_to_path', localization, ['font', 'currx'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'glyph_to_path(...)' code ##################

        unicode_144940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, (-1)), 'unicode', u'\n        convert the ft2font glyph to vertices and codes.\n        ')
        
        # Assigning a Call to a Tuple (line 91):
        
        # Assigning a Call to a Name:
        
        # Call to get_path(...): (line 91)
        # Processing the call keyword arguments (line 91)
        kwargs_144943 = {}
        # Getting the type of 'font' (line 91)
        font_144941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 23), 'font', False)
        # Obtaining the member 'get_path' of a type (line 91)
        get_path_144942 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 23), font_144941, 'get_path')
        # Calling get_path(args, kwargs) (line 91)
        get_path_call_result_144944 = invoke(stypy.reporting.localization.Localization(__file__, 91, 23), get_path_144942, *[], **kwargs_144943)
        
        # Assigning a type to the variable 'call_assignment_144736' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'call_assignment_144736', get_path_call_result_144944)
        
        # Assigning a Call to a Name (line 91):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_144947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 8), 'int')
        # Processing the call keyword arguments
        kwargs_144948 = {}
        # Getting the type of 'call_assignment_144736' (line 91)
        call_assignment_144736_144945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'call_assignment_144736', False)
        # Obtaining the member '__getitem__' of a type (line 91)
        getitem___144946 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 8), call_assignment_144736_144945, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_144949 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___144946, *[int_144947], **kwargs_144948)
        
        # Assigning a type to the variable 'call_assignment_144737' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'call_assignment_144737', getitem___call_result_144949)
        
        # Assigning a Name to a Name (line 91):
        # Getting the type of 'call_assignment_144737' (line 91)
        call_assignment_144737_144950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'call_assignment_144737')
        # Assigning a type to the variable 'verts' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'verts', call_assignment_144737_144950)
        
        # Assigning a Call to a Name (line 91):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_144953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 8), 'int')
        # Processing the call keyword arguments
        kwargs_144954 = {}
        # Getting the type of 'call_assignment_144736' (line 91)
        call_assignment_144736_144951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'call_assignment_144736', False)
        # Obtaining the member '__getitem__' of a type (line 91)
        getitem___144952 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 8), call_assignment_144736_144951, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_144955 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___144952, *[int_144953], **kwargs_144954)
        
        # Assigning a type to the variable 'call_assignment_144738' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'call_assignment_144738', getitem___call_result_144955)
        
        # Assigning a Name to a Name (line 91):
        # Getting the type of 'call_assignment_144738' (line 91)
        call_assignment_144738_144956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'call_assignment_144738')
        # Assigning a type to the variable 'codes' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 15), 'codes', call_assignment_144738_144956)
        
        
        # Getting the type of 'currx' (line 92)
        currx_144957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 11), 'currx')
        float_144958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 20), 'float')
        # Applying the binary operator '!=' (line 92)
        result_ne_144959 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 11), '!=', currx_144957, float_144958)
        
        # Testing the type of an if condition (line 92)
        if_condition_144960 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 92, 8), result_ne_144959)
        # Assigning a type to the variable 'if_condition_144960' (line 92)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'if_condition_144960', if_condition_144960)
        # SSA begins for if statement (line 92)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'verts' (line 93)
        verts_144961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 12), 'verts')
        
        # Obtaining the type of the subscript
        slice_144962 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 93, 12), None, None, None)
        int_144963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 21), 'int')
        # Getting the type of 'verts' (line 93)
        verts_144964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 12), 'verts')
        # Obtaining the member '__getitem__' of a type (line 93)
        getitem___144965 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 12), verts_144964, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 93)
        subscript_call_result_144966 = invoke(stypy.reporting.localization.Localization(__file__, 93, 12), getitem___144965, (slice_144962, int_144963))
        
        # Getting the type of 'currx' (line 93)
        currx_144967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 27), 'currx')
        # Applying the binary operator '+=' (line 93)
        result_iadd_144968 = python_operator(stypy.reporting.localization.Localization(__file__, 93, 12), '+=', subscript_call_result_144966, currx_144967)
        # Getting the type of 'verts' (line 93)
        verts_144969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 12), 'verts')
        slice_144970 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 93, 12), None, None, None)
        int_144971 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 21), 'int')
        # Storing an element on a container (line 93)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 12), verts_144969, ((slice_144970, int_144971), result_iadd_144968))
        
        # SSA join for if statement (line 92)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 94)
        tuple_144972 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 94)
        # Adding element type (line 94)
        # Getting the type of 'verts' (line 94)
        verts_144973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 15), 'verts')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 94, 15), tuple_144972, verts_144973)
        # Adding element type (line 94)
        # Getting the type of 'codes' (line 94)
        codes_144974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 22), 'codes')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 94, 15), tuple_144972, codes_144974)
        
        # Assigning a type to the variable 'stypy_return_type' (line 94)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'stypy_return_type', tuple_144972)
        
        # ################# End of 'glyph_to_path(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'glyph_to_path' in the type store
        # Getting the type of 'stypy_return_type' (line 87)
        stypy_return_type_144975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_144975)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'glyph_to_path'
        return stypy_return_type_144975


    @norecursion
    def get_text_width_height_descent(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_text_width_height_descent'
        module_type_store = module_type_store.open_function_context('get_text_width_height_descent', 96, 4, False)
        # Assigning a type to the variable 'self' (line 97)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TextToPath.get_text_width_height_descent.__dict__.__setitem__('stypy_localization', localization)
        TextToPath.get_text_width_height_descent.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TextToPath.get_text_width_height_descent.__dict__.__setitem__('stypy_type_store', module_type_store)
        TextToPath.get_text_width_height_descent.__dict__.__setitem__('stypy_function_name', 'TextToPath.get_text_width_height_descent')
        TextToPath.get_text_width_height_descent.__dict__.__setitem__('stypy_param_names_list', ['s', 'prop', 'ismath'])
        TextToPath.get_text_width_height_descent.__dict__.__setitem__('stypy_varargs_param_name', None)
        TextToPath.get_text_width_height_descent.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TextToPath.get_text_width_height_descent.__dict__.__setitem__('stypy_call_defaults', defaults)
        TextToPath.get_text_width_height_descent.__dict__.__setitem__('stypy_call_varargs', varargs)
        TextToPath.get_text_width_height_descent.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TextToPath.get_text_width_height_descent.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TextToPath.get_text_width_height_descent', ['s', 'prop', 'ismath'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_text_width_height_descent', localization, ['s', 'prop', 'ismath'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_text_width_height_descent(...)' code ##################

        
        
        # Obtaining the type of the subscript
        unicode_144976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 20), 'unicode', u'text.usetex')
        # Getting the type of 'rcParams' (line 97)
        rcParams_144977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 11), 'rcParams')
        # Obtaining the member '__getitem__' of a type (line 97)
        getitem___144978 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 11), rcParams_144977, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 97)
        subscript_call_result_144979 = invoke(stypy.reporting.localization.Localization(__file__, 97, 11), getitem___144978, unicode_144976)
        
        # Testing the type of an if condition (line 97)
        if_condition_144980 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 97, 8), subscript_call_result_144979)
        # Assigning a type to the variable 'if_condition_144980' (line 97)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'if_condition_144980', if_condition_144980)
        # SSA begins for if statement (line 97)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 98):
        
        # Assigning a Call to a Name (line 98):
        
        # Call to get_texmanager(...): (line 98)
        # Processing the call keyword arguments (line 98)
        kwargs_144983 = {}
        # Getting the type of 'self' (line 98)
        self_144981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 25), 'self', False)
        # Obtaining the member 'get_texmanager' of a type (line 98)
        get_texmanager_144982 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 25), self_144981, 'get_texmanager')
        # Calling get_texmanager(args, kwargs) (line 98)
        get_texmanager_call_result_144984 = invoke(stypy.reporting.localization.Localization(__file__, 98, 25), get_texmanager_144982, *[], **kwargs_144983)
        
        # Assigning a type to the variable 'texmanager' (line 98)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 12), 'texmanager', get_texmanager_call_result_144984)
        
        # Assigning a Call to a Name (line 99):
        
        # Assigning a Call to a Name (line 99):
        
        # Call to get_size_in_points(...): (line 99)
        # Processing the call keyword arguments (line 99)
        kwargs_144987 = {}
        # Getting the type of 'prop' (line 99)
        prop_144985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 23), 'prop', False)
        # Obtaining the member 'get_size_in_points' of a type (line 99)
        get_size_in_points_144986 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 23), prop_144985, 'get_size_in_points')
        # Calling get_size_in_points(args, kwargs) (line 99)
        get_size_in_points_call_result_144988 = invoke(stypy.reporting.localization.Localization(__file__, 99, 23), get_size_in_points_144986, *[], **kwargs_144987)
        
        # Assigning a type to the variable 'fontsize' (line 99)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 12), 'fontsize', get_size_in_points_call_result_144988)
        
        # Assigning a Call to a Tuple (line 100):
        
        # Assigning a Call to a Name:
        
        # Call to get_text_width_height_descent(...): (line 100)
        # Processing the call arguments (line 100)
        # Getting the type of 's' (line 100)
        s_144991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 63), 's', False)
        # Getting the type of 'fontsize' (line 100)
        fontsize_144992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 66), 'fontsize', False)
        # Processing the call keyword arguments (line 100)
        # Getting the type of 'None' (line 101)
        None_144993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 72), 'None', False)
        keyword_144994 = None_144993
        kwargs_144995 = {'renderer': keyword_144994}
        # Getting the type of 'texmanager' (line 100)
        texmanager_144989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 22), 'texmanager', False)
        # Obtaining the member 'get_text_width_height_descent' of a type (line 100)
        get_text_width_height_descent_144990 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 22), texmanager_144989, 'get_text_width_height_descent')
        # Calling get_text_width_height_descent(args, kwargs) (line 100)
        get_text_width_height_descent_call_result_144996 = invoke(stypy.reporting.localization.Localization(__file__, 100, 22), get_text_width_height_descent_144990, *[s_144991, fontsize_144992], **kwargs_144995)
        
        # Assigning a type to the variable 'call_assignment_144739' (line 100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 12), 'call_assignment_144739', get_text_width_height_descent_call_result_144996)
        
        # Assigning a Call to a Name (line 100):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_144999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 12), 'int')
        # Processing the call keyword arguments
        kwargs_145000 = {}
        # Getting the type of 'call_assignment_144739' (line 100)
        call_assignment_144739_144997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 12), 'call_assignment_144739', False)
        # Obtaining the member '__getitem__' of a type (line 100)
        getitem___144998 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 12), call_assignment_144739_144997, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_145001 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___144998, *[int_144999], **kwargs_145000)
        
        # Assigning a type to the variable 'call_assignment_144740' (line 100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 12), 'call_assignment_144740', getitem___call_result_145001)
        
        # Assigning a Name to a Name (line 100):
        # Getting the type of 'call_assignment_144740' (line 100)
        call_assignment_144740_145002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 12), 'call_assignment_144740')
        # Assigning a type to the variable 'w' (line 100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 12), 'w', call_assignment_144740_145002)
        
        # Assigning a Call to a Name (line 100):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_145005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 12), 'int')
        # Processing the call keyword arguments
        kwargs_145006 = {}
        # Getting the type of 'call_assignment_144739' (line 100)
        call_assignment_144739_145003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 12), 'call_assignment_144739', False)
        # Obtaining the member '__getitem__' of a type (line 100)
        getitem___145004 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 12), call_assignment_144739_145003, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_145007 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___145004, *[int_145005], **kwargs_145006)
        
        # Assigning a type to the variable 'call_assignment_144741' (line 100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 12), 'call_assignment_144741', getitem___call_result_145007)
        
        # Assigning a Name to a Name (line 100):
        # Getting the type of 'call_assignment_144741' (line 100)
        call_assignment_144741_145008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 12), 'call_assignment_144741')
        # Assigning a type to the variable 'h' (line 100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 15), 'h', call_assignment_144741_145008)
        
        # Assigning a Call to a Name (line 100):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_145011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 12), 'int')
        # Processing the call keyword arguments
        kwargs_145012 = {}
        # Getting the type of 'call_assignment_144739' (line 100)
        call_assignment_144739_145009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 12), 'call_assignment_144739', False)
        # Obtaining the member '__getitem__' of a type (line 100)
        getitem___145010 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 12), call_assignment_144739_145009, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_145013 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___145010, *[int_145011], **kwargs_145012)
        
        # Assigning a type to the variable 'call_assignment_144742' (line 100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 12), 'call_assignment_144742', getitem___call_result_145013)
        
        # Assigning a Name to a Name (line 100):
        # Getting the type of 'call_assignment_144742' (line 100)
        call_assignment_144742_145014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 12), 'call_assignment_144742')
        # Assigning a type to the variable 'd' (line 100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 18), 'd', call_assignment_144742_145014)
        
        # Obtaining an instance of the builtin type 'tuple' (line 102)
        tuple_145015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 19), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 102)
        # Adding element type (line 102)
        # Getting the type of 'w' (line 102)
        w_145016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 19), 'w')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 19), tuple_145015, w_145016)
        # Adding element type (line 102)
        # Getting the type of 'h' (line 102)
        h_145017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 22), 'h')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 19), tuple_145015, h_145017)
        # Adding element type (line 102)
        # Getting the type of 'd' (line 102)
        d_145018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 25), 'd')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 19), tuple_145015, d_145018)
        
        # Assigning a type to the variable 'stypy_return_type' (line 102)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 12), 'stypy_return_type', tuple_145015)
        # SSA join for if statement (line 97)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 104):
        
        # Assigning a Call to a Name (line 104):
        
        # Call to get_size_in_points(...): (line 104)
        # Processing the call keyword arguments (line 104)
        kwargs_145021 = {}
        # Getting the type of 'prop' (line 104)
        prop_145019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 19), 'prop', False)
        # Obtaining the member 'get_size_in_points' of a type (line 104)
        get_size_in_points_145020 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 19), prop_145019, 'get_size_in_points')
        # Calling get_size_in_points(args, kwargs) (line 104)
        get_size_in_points_call_result_145022 = invoke(stypy.reporting.localization.Localization(__file__, 104, 19), get_size_in_points_145020, *[], **kwargs_145021)
        
        # Assigning a type to the variable 'fontsize' (line 104)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'fontsize', get_size_in_points_call_result_145022)
        
        # Assigning a BinOp to a Name (line 105):
        
        # Assigning a BinOp to a Name (line 105):
        
        # Call to float(...): (line 105)
        # Processing the call arguments (line 105)
        # Getting the type of 'fontsize' (line 105)
        fontsize_145024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 22), 'fontsize', False)
        # Processing the call keyword arguments (line 105)
        kwargs_145025 = {}
        # Getting the type of 'float' (line 105)
        float_145023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 16), 'float', False)
        # Calling float(args, kwargs) (line 105)
        float_call_result_145026 = invoke(stypy.reporting.localization.Localization(__file__, 105, 16), float_145023, *[fontsize_145024], **kwargs_145025)
        
        # Getting the type of 'self' (line 105)
        self_145027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 34), 'self')
        # Obtaining the member 'FONT_SCALE' of a type (line 105)
        FONT_SCALE_145028 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 34), self_145027, 'FONT_SCALE')
        # Applying the binary operator 'div' (line 105)
        result_div_145029 = python_operator(stypy.reporting.localization.Localization(__file__, 105, 16), 'div', float_call_result_145026, FONT_SCALE_145028)
        
        # Assigning a type to the variable 'scale' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'scale', result_div_145029)
        
        # Getting the type of 'ismath' (line 107)
        ismath_145030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 11), 'ismath')
        # Testing the type of an if condition (line 107)
        if_condition_145031 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 107, 8), ismath_145030)
        # Assigning a type to the variable 'if_condition_145031' (line 107)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 8), 'if_condition_145031', if_condition_145031)
        # SSA begins for if statement (line 107)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 108):
        
        # Assigning a Call to a Name (line 108):
        
        # Call to copy(...): (line 108)
        # Processing the call keyword arguments (line 108)
        kwargs_145034 = {}
        # Getting the type of 'prop' (line 108)
        prop_145032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 19), 'prop', False)
        # Obtaining the member 'copy' of a type (line 108)
        copy_145033 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 19), prop_145032, 'copy')
        # Calling copy(args, kwargs) (line 108)
        copy_call_result_145035 = invoke(stypy.reporting.localization.Localization(__file__, 108, 19), copy_145033, *[], **kwargs_145034)
        
        # Assigning a type to the variable 'prop' (line 108)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 12), 'prop', copy_call_result_145035)
        
        # Call to set_size(...): (line 109)
        # Processing the call arguments (line 109)
        # Getting the type of 'self' (line 109)
        self_145038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 26), 'self', False)
        # Obtaining the member 'FONT_SCALE' of a type (line 109)
        FONT_SCALE_145039 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 26), self_145038, 'FONT_SCALE')
        # Processing the call keyword arguments (line 109)
        kwargs_145040 = {}
        # Getting the type of 'prop' (line 109)
        prop_145036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 12), 'prop', False)
        # Obtaining the member 'set_size' of a type (line 109)
        set_size_145037 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 12), prop_145036, 'set_size')
        # Calling set_size(args, kwargs) (line 109)
        set_size_call_result_145041 = invoke(stypy.reporting.localization.Localization(__file__, 109, 12), set_size_145037, *[FONT_SCALE_145039], **kwargs_145040)
        
        
        # Assigning a Call to a Tuple (line 111):
        
        # Assigning a Call to a Name:
        
        # Call to parse(...): (line 112)
        # Processing the call arguments (line 112)
        # Getting the type of 's' (line 112)
        s_145045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 43), 's', False)
        int_145046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 46), 'int')
        # Getting the type of 'prop' (line 112)
        prop_145047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 50), 'prop', False)
        # Processing the call keyword arguments (line 112)
        kwargs_145048 = {}
        # Getting the type of 'self' (line 112)
        self_145042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 16), 'self', False)
        # Obtaining the member 'mathtext_parser' of a type (line 112)
        mathtext_parser_145043 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 16), self_145042, 'mathtext_parser')
        # Obtaining the member 'parse' of a type (line 112)
        parse_145044 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 16), mathtext_parser_145043, 'parse')
        # Calling parse(args, kwargs) (line 112)
        parse_call_result_145049 = invoke(stypy.reporting.localization.Localization(__file__, 112, 16), parse_145044, *[s_145045, int_145046, prop_145047], **kwargs_145048)
        
        # Assigning a type to the variable 'call_assignment_144743' (line 111)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'call_assignment_144743', parse_call_result_145049)
        
        # Assigning a Call to a Name (line 111):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_145052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 12), 'int')
        # Processing the call keyword arguments
        kwargs_145053 = {}
        # Getting the type of 'call_assignment_144743' (line 111)
        call_assignment_144743_145050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'call_assignment_144743', False)
        # Obtaining the member '__getitem__' of a type (line 111)
        getitem___145051 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 12), call_assignment_144743_145050, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_145054 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___145051, *[int_145052], **kwargs_145053)
        
        # Assigning a type to the variable 'call_assignment_144744' (line 111)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'call_assignment_144744', getitem___call_result_145054)
        
        # Assigning a Name to a Name (line 111):
        # Getting the type of 'call_assignment_144744' (line 111)
        call_assignment_144744_145055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'call_assignment_144744')
        # Assigning a type to the variable 'width' (line 111)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'width', call_assignment_144744_145055)
        
        # Assigning a Call to a Name (line 111):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_145058 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 12), 'int')
        # Processing the call keyword arguments
        kwargs_145059 = {}
        # Getting the type of 'call_assignment_144743' (line 111)
        call_assignment_144743_145056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'call_assignment_144743', False)
        # Obtaining the member '__getitem__' of a type (line 111)
        getitem___145057 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 12), call_assignment_144743_145056, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_145060 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___145057, *[int_145058], **kwargs_145059)
        
        # Assigning a type to the variable 'call_assignment_144745' (line 111)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'call_assignment_144745', getitem___call_result_145060)
        
        # Assigning a Name to a Name (line 111):
        # Getting the type of 'call_assignment_144745' (line 111)
        call_assignment_144745_145061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'call_assignment_144745')
        # Assigning a type to the variable 'height' (line 111)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 19), 'height', call_assignment_144745_145061)
        
        # Assigning a Call to a Name (line 111):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_145064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 12), 'int')
        # Processing the call keyword arguments
        kwargs_145065 = {}
        # Getting the type of 'call_assignment_144743' (line 111)
        call_assignment_144743_145062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'call_assignment_144743', False)
        # Obtaining the member '__getitem__' of a type (line 111)
        getitem___145063 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 12), call_assignment_144743_145062, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_145066 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___145063, *[int_145064], **kwargs_145065)
        
        # Assigning a type to the variable 'call_assignment_144746' (line 111)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'call_assignment_144746', getitem___call_result_145066)
        
        # Assigning a Name to a Name (line 111):
        # Getting the type of 'call_assignment_144746' (line 111)
        call_assignment_144746_145067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'call_assignment_144746')
        # Assigning a type to the variable 'descent' (line 111)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 27), 'descent', call_assignment_144746_145067)
        
        # Assigning a Call to a Name (line 111):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_145070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 12), 'int')
        # Processing the call keyword arguments
        kwargs_145071 = {}
        # Getting the type of 'call_assignment_144743' (line 111)
        call_assignment_144743_145068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'call_assignment_144743', False)
        # Obtaining the member '__getitem__' of a type (line 111)
        getitem___145069 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 12), call_assignment_144743_145068, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_145072 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___145069, *[int_145070], **kwargs_145071)
        
        # Assigning a type to the variable 'call_assignment_144747' (line 111)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'call_assignment_144747', getitem___call_result_145072)
        
        # Assigning a Name to a Name (line 111):
        # Getting the type of 'call_assignment_144747' (line 111)
        call_assignment_144747_145073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'call_assignment_144747')
        # Assigning a type to the variable 'trash' (line 111)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 36), 'trash', call_assignment_144747_145073)
        
        # Assigning a Call to a Name (line 111):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_145076 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 12), 'int')
        # Processing the call keyword arguments
        kwargs_145077 = {}
        # Getting the type of 'call_assignment_144743' (line 111)
        call_assignment_144743_145074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'call_assignment_144743', False)
        # Obtaining the member '__getitem__' of a type (line 111)
        getitem___145075 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 12), call_assignment_144743_145074, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_145078 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___145075, *[int_145076], **kwargs_145077)
        
        # Assigning a type to the variable 'call_assignment_144748' (line 111)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'call_assignment_144748', getitem___call_result_145078)
        
        # Assigning a Name to a Name (line 111):
        # Getting the type of 'call_assignment_144748' (line 111)
        call_assignment_144748_145079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'call_assignment_144748')
        # Assigning a type to the variable 'used_characters' (line 111)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 43), 'used_characters', call_assignment_144748_145079)
        
        # Obtaining an instance of the builtin type 'tuple' (line 113)
        tuple_145080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 19), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 113)
        # Adding element type (line 113)
        # Getting the type of 'width' (line 113)
        width_145081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 19), 'width')
        # Getting the type of 'scale' (line 113)
        scale_145082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 27), 'scale')
        # Applying the binary operator '*' (line 113)
        result_mul_145083 = python_operator(stypy.reporting.localization.Localization(__file__, 113, 19), '*', width_145081, scale_145082)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 19), tuple_145080, result_mul_145083)
        # Adding element type (line 113)
        # Getting the type of 'height' (line 113)
        height_145084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 34), 'height')
        # Getting the type of 'scale' (line 113)
        scale_145085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 43), 'scale')
        # Applying the binary operator '*' (line 113)
        result_mul_145086 = python_operator(stypy.reporting.localization.Localization(__file__, 113, 34), '*', height_145084, scale_145085)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 19), tuple_145080, result_mul_145086)
        # Adding element type (line 113)
        # Getting the type of 'descent' (line 113)
        descent_145087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 50), 'descent')
        # Getting the type of 'scale' (line 113)
        scale_145088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 60), 'scale')
        # Applying the binary operator '*' (line 113)
        result_mul_145089 = python_operator(stypy.reporting.localization.Localization(__file__, 113, 50), '*', descent_145087, scale_145088)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 19), tuple_145080, result_mul_145089)
        
        # Assigning a type to the variable 'stypy_return_type' (line 113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 12), 'stypy_return_type', tuple_145080)
        # SSA join for if statement (line 107)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 115):
        
        # Assigning a Call to a Name (line 115):
        
        # Call to _get_font(...): (line 115)
        # Processing the call arguments (line 115)
        # Getting the type of 'prop' (line 115)
        prop_145092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 30), 'prop', False)
        # Processing the call keyword arguments (line 115)
        kwargs_145093 = {}
        # Getting the type of 'self' (line 115)
        self_145090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 15), 'self', False)
        # Obtaining the member '_get_font' of a type (line 115)
        _get_font_145091 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 15), self_145090, '_get_font')
        # Calling _get_font(args, kwargs) (line 115)
        _get_font_call_result_145094 = invoke(stypy.reporting.localization.Localization(__file__, 115, 15), _get_font_145091, *[prop_145092], **kwargs_145093)
        
        # Assigning a type to the variable 'font' (line 115)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 8), 'font', _get_font_call_result_145094)
        
        # Call to set_text(...): (line 116)
        # Processing the call arguments (line 116)
        # Getting the type of 's' (line 116)
        s_145097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 22), 's', False)
        float_145098 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 25), 'float')
        # Processing the call keyword arguments (line 116)
        # Getting the type of 'LOAD_NO_HINTING' (line 116)
        LOAD_NO_HINTING_145099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 36), 'LOAD_NO_HINTING', False)
        keyword_145100 = LOAD_NO_HINTING_145099
        kwargs_145101 = {'flags': keyword_145100}
        # Getting the type of 'font' (line 116)
        font_145095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'font', False)
        # Obtaining the member 'set_text' of a type (line 116)
        set_text_145096 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 8), font_145095, 'set_text')
        # Calling set_text(args, kwargs) (line 116)
        set_text_call_result_145102 = invoke(stypy.reporting.localization.Localization(__file__, 116, 8), set_text_145096, *[s_145097, float_145098], **kwargs_145101)
        
        
        # Assigning a Call to a Tuple (line 117):
        
        # Assigning a Call to a Name:
        
        # Call to get_width_height(...): (line 117)
        # Processing the call keyword arguments (line 117)
        kwargs_145105 = {}
        # Getting the type of 'font' (line 117)
        font_145103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 15), 'font', False)
        # Obtaining the member 'get_width_height' of a type (line 117)
        get_width_height_145104 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 15), font_145103, 'get_width_height')
        # Calling get_width_height(args, kwargs) (line 117)
        get_width_height_call_result_145106 = invoke(stypy.reporting.localization.Localization(__file__, 117, 15), get_width_height_145104, *[], **kwargs_145105)
        
        # Assigning a type to the variable 'call_assignment_144749' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'call_assignment_144749', get_width_height_call_result_145106)
        
        # Assigning a Call to a Name (line 117):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_145109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 8), 'int')
        # Processing the call keyword arguments
        kwargs_145110 = {}
        # Getting the type of 'call_assignment_144749' (line 117)
        call_assignment_144749_145107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'call_assignment_144749', False)
        # Obtaining the member '__getitem__' of a type (line 117)
        getitem___145108 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 8), call_assignment_144749_145107, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_145111 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___145108, *[int_145109], **kwargs_145110)
        
        # Assigning a type to the variable 'call_assignment_144750' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'call_assignment_144750', getitem___call_result_145111)
        
        # Assigning a Name to a Name (line 117):
        # Getting the type of 'call_assignment_144750' (line 117)
        call_assignment_144750_145112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'call_assignment_144750')
        # Assigning a type to the variable 'w' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'w', call_assignment_144750_145112)
        
        # Assigning a Call to a Name (line 117):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_145115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 8), 'int')
        # Processing the call keyword arguments
        kwargs_145116 = {}
        # Getting the type of 'call_assignment_144749' (line 117)
        call_assignment_144749_145113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'call_assignment_144749', False)
        # Obtaining the member '__getitem__' of a type (line 117)
        getitem___145114 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 8), call_assignment_144749_145113, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_145117 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___145114, *[int_145115], **kwargs_145116)
        
        # Assigning a type to the variable 'call_assignment_144751' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'call_assignment_144751', getitem___call_result_145117)
        
        # Assigning a Name to a Name (line 117):
        # Getting the type of 'call_assignment_144751' (line 117)
        call_assignment_144751_145118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'call_assignment_144751')
        # Assigning a type to the variable 'h' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 11), 'h', call_assignment_144751_145118)
        
        # Getting the type of 'w' (line 118)
        w_145119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'w')
        float_145120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 13), 'float')
        # Applying the binary operator 'div=' (line 118)
        result_div_145121 = python_operator(stypy.reporting.localization.Localization(__file__, 118, 8), 'div=', w_145119, float_145120)
        # Assigning a type to the variable 'w' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'w', result_div_145121)
        
        
        # Getting the type of 'h' (line 119)
        h_145122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'h')
        float_145123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 13), 'float')
        # Applying the binary operator 'div=' (line 119)
        result_div_145124 = python_operator(stypy.reporting.localization.Localization(__file__, 119, 8), 'div=', h_145122, float_145123)
        # Assigning a type to the variable 'h' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'h', result_div_145124)
        
        
        # Assigning a Call to a Name (line 120):
        
        # Assigning a Call to a Name (line 120):
        
        # Call to get_descent(...): (line 120)
        # Processing the call keyword arguments (line 120)
        kwargs_145127 = {}
        # Getting the type of 'font' (line 120)
        font_145125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 12), 'font', False)
        # Obtaining the member 'get_descent' of a type (line 120)
        get_descent_145126 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 12), font_145125, 'get_descent')
        # Calling get_descent(args, kwargs) (line 120)
        get_descent_call_result_145128 = invoke(stypy.reporting.localization.Localization(__file__, 120, 12), get_descent_145126, *[], **kwargs_145127)
        
        # Assigning a type to the variable 'd' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'd', get_descent_call_result_145128)
        
        # Getting the type of 'd' (line 121)
        d_145129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'd')
        float_145130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 13), 'float')
        # Applying the binary operator 'div=' (line 121)
        result_div_145131 = python_operator(stypy.reporting.localization.Localization(__file__, 121, 8), 'div=', d_145129, float_145130)
        # Assigning a type to the variable 'd' (line 121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'd', result_div_145131)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 122)
        tuple_145132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 122)
        # Adding element type (line 122)
        # Getting the type of 'w' (line 122)
        w_145133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 15), 'w')
        # Getting the type of 'scale' (line 122)
        scale_145134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 19), 'scale')
        # Applying the binary operator '*' (line 122)
        result_mul_145135 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 15), '*', w_145133, scale_145134)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 15), tuple_145132, result_mul_145135)
        # Adding element type (line 122)
        # Getting the type of 'h' (line 122)
        h_145136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 26), 'h')
        # Getting the type of 'scale' (line 122)
        scale_145137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 30), 'scale')
        # Applying the binary operator '*' (line 122)
        result_mul_145138 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 26), '*', h_145136, scale_145137)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 15), tuple_145132, result_mul_145138)
        # Adding element type (line 122)
        # Getting the type of 'd' (line 122)
        d_145139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 37), 'd')
        # Getting the type of 'scale' (line 122)
        scale_145140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 41), 'scale')
        # Applying the binary operator '*' (line 122)
        result_mul_145141 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 37), '*', d_145139, scale_145140)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 15), tuple_145132, result_mul_145141)
        
        # Assigning a type to the variable 'stypy_return_type' (line 122)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 8), 'stypy_return_type', tuple_145132)
        
        # ################# End of 'get_text_width_height_descent(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_text_width_height_descent' in the type store
        # Getting the type of 'stypy_return_type' (line 96)
        stypy_return_type_145142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_145142)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_text_width_height_descent'
        return stypy_return_type_145142


    @norecursion
    def get_text_path(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'False' (line 124)
        False_145143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 44), 'False')
        # Getting the type of 'False' (line 124)
        False_145144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 58), 'False')
        defaults = [False_145143, False_145144]
        # Create a new context for function 'get_text_path'
        module_type_store = module_type_store.open_function_context('get_text_path', 124, 4, False)
        # Assigning a type to the variable 'self' (line 125)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TextToPath.get_text_path.__dict__.__setitem__('stypy_localization', localization)
        TextToPath.get_text_path.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TextToPath.get_text_path.__dict__.__setitem__('stypy_type_store', module_type_store)
        TextToPath.get_text_path.__dict__.__setitem__('stypy_function_name', 'TextToPath.get_text_path')
        TextToPath.get_text_path.__dict__.__setitem__('stypy_param_names_list', ['prop', 's', 'ismath', 'usetex'])
        TextToPath.get_text_path.__dict__.__setitem__('stypy_varargs_param_name', None)
        TextToPath.get_text_path.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TextToPath.get_text_path.__dict__.__setitem__('stypy_call_defaults', defaults)
        TextToPath.get_text_path.__dict__.__setitem__('stypy_call_varargs', varargs)
        TextToPath.get_text_path.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TextToPath.get_text_path.__dict__.__setitem__('stypy_declared_arg_number', 5)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TextToPath.get_text_path', ['prop', 's', 'ismath', 'usetex'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_text_path', localization, ['prop', 's', 'ismath', 'usetex'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_text_path(...)' code ##################

        unicode_145145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, (-1)), 'unicode', u'\n        convert text *s* to path (a tuple of vertices and codes for\n        matplotlib.path.Path).\n\n        *prop*\n          font property\n\n        *s*\n          text to be converted\n\n        *usetex*\n          If True, use matplotlib usetex mode.\n\n        *ismath*\n          If True, use mathtext parser. Effective only if usetex == False.\n\n\n        ')
        
        
        # Getting the type of 'usetex' (line 143)
        usetex_145146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 15), 'usetex')
        # Applying the 'not' unary operator (line 143)
        result_not__145147 = python_operator(stypy.reporting.localization.Localization(__file__, 143, 11), 'not', usetex_145146)
        
        # Testing the type of an if condition (line 143)
        if_condition_145148 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 143, 8), result_not__145147)
        # Assigning a type to the variable 'if_condition_145148' (line 143)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 8), 'if_condition_145148', if_condition_145148)
        # SSA begins for if statement (line 143)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Getting the type of 'ismath' (line 144)
        ismath_145149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 19), 'ismath')
        # Applying the 'not' unary operator (line 144)
        result_not__145150 = python_operator(stypy.reporting.localization.Localization(__file__, 144, 15), 'not', ismath_145149)
        
        # Testing the type of an if condition (line 144)
        if_condition_145151 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 144, 12), result_not__145150)
        # Assigning a type to the variable 'if_condition_145151' (line 144)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 12), 'if_condition_145151', if_condition_145151)
        # SSA begins for if statement (line 144)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 145):
        
        # Assigning a Call to a Name (line 145):
        
        # Call to _get_font(...): (line 145)
        # Processing the call arguments (line 145)
        # Getting the type of 'prop' (line 145)
        prop_145154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 38), 'prop', False)
        # Processing the call keyword arguments (line 145)
        kwargs_145155 = {}
        # Getting the type of 'self' (line 145)
        self_145152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 23), 'self', False)
        # Obtaining the member '_get_font' of a type (line 145)
        _get_font_145153 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 23), self_145152, '_get_font')
        # Calling _get_font(args, kwargs) (line 145)
        _get_font_call_result_145156 = invoke(stypy.reporting.localization.Localization(__file__, 145, 23), _get_font_145153, *[prop_145154], **kwargs_145155)
        
        # Assigning a type to the variable 'font' (line 145)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 16), 'font', _get_font_call_result_145156)
        
        # Assigning a Call to a Tuple (line 146):
        
        # Assigning a Call to a Name:
        
        # Call to get_glyphs_with_font(...): (line 146)
        # Processing the call arguments (line 146)
        # Getting the type of 'font' (line 147)
        font_145159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 52), 'font', False)
        # Getting the type of 's' (line 147)
        s_145160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 58), 's', False)
        # Processing the call keyword arguments (line 146)
        kwargs_145161 = {}
        # Getting the type of 'self' (line 146)
        self_145157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 47), 'self', False)
        # Obtaining the member 'get_glyphs_with_font' of a type (line 146)
        get_glyphs_with_font_145158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 47), self_145157, 'get_glyphs_with_font')
        # Calling get_glyphs_with_font(args, kwargs) (line 146)
        get_glyphs_with_font_call_result_145162 = invoke(stypy.reporting.localization.Localization(__file__, 146, 47), get_glyphs_with_font_145158, *[font_145159, s_145160], **kwargs_145161)
        
        # Assigning a type to the variable 'call_assignment_144752' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 16), 'call_assignment_144752', get_glyphs_with_font_call_result_145162)
        
        # Assigning a Call to a Name (line 146):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_145165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 16), 'int')
        # Processing the call keyword arguments
        kwargs_145166 = {}
        # Getting the type of 'call_assignment_144752' (line 146)
        call_assignment_144752_145163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 16), 'call_assignment_144752', False)
        # Obtaining the member '__getitem__' of a type (line 146)
        getitem___145164 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 16), call_assignment_144752_145163, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_145167 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___145164, *[int_145165], **kwargs_145166)
        
        # Assigning a type to the variable 'call_assignment_144753' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 16), 'call_assignment_144753', getitem___call_result_145167)
        
        # Assigning a Name to a Name (line 146):
        # Getting the type of 'call_assignment_144753' (line 146)
        call_assignment_144753_145168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 16), 'call_assignment_144753')
        # Assigning a type to the variable 'glyph_info' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 16), 'glyph_info', call_assignment_144753_145168)
        
        # Assigning a Call to a Name (line 146):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_145171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 16), 'int')
        # Processing the call keyword arguments
        kwargs_145172 = {}
        # Getting the type of 'call_assignment_144752' (line 146)
        call_assignment_144752_145169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 16), 'call_assignment_144752', False)
        # Obtaining the member '__getitem__' of a type (line 146)
        getitem___145170 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 16), call_assignment_144752_145169, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_145173 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___145170, *[int_145171], **kwargs_145172)
        
        # Assigning a type to the variable 'call_assignment_144754' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 16), 'call_assignment_144754', getitem___call_result_145173)
        
        # Assigning a Name to a Name (line 146):
        # Getting the type of 'call_assignment_144754' (line 146)
        call_assignment_144754_145174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 16), 'call_assignment_144754')
        # Assigning a type to the variable 'glyph_map' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 28), 'glyph_map', call_assignment_144754_145174)
        
        # Assigning a Call to a Name (line 146):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_145177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 16), 'int')
        # Processing the call keyword arguments
        kwargs_145178 = {}
        # Getting the type of 'call_assignment_144752' (line 146)
        call_assignment_144752_145175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 16), 'call_assignment_144752', False)
        # Obtaining the member '__getitem__' of a type (line 146)
        getitem___145176 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 16), call_assignment_144752_145175, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_145179 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___145176, *[int_145177], **kwargs_145178)
        
        # Assigning a type to the variable 'call_assignment_144755' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 16), 'call_assignment_144755', getitem___call_result_145179)
        
        # Assigning a Name to a Name (line 146):
        # Getting the type of 'call_assignment_144755' (line 146)
        call_assignment_144755_145180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 16), 'call_assignment_144755')
        # Assigning a type to the variable 'rects' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 39), 'rects', call_assignment_144755_145180)
        # SSA branch for the else part of an if statement (line 144)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Tuple (line 149):
        
        # Assigning a Call to a Name:
        
        # Call to get_glyphs_mathtext(...): (line 149)
        # Processing the call arguments (line 149)
        # Getting the type of 'prop' (line 150)
        prop_145183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 52), 'prop', False)
        # Getting the type of 's' (line 150)
        s_145184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 58), 's', False)
        # Processing the call keyword arguments (line 149)
        kwargs_145185 = {}
        # Getting the type of 'self' (line 149)
        self_145181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 47), 'self', False)
        # Obtaining the member 'get_glyphs_mathtext' of a type (line 149)
        get_glyphs_mathtext_145182 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 47), self_145181, 'get_glyphs_mathtext')
        # Calling get_glyphs_mathtext(args, kwargs) (line 149)
        get_glyphs_mathtext_call_result_145186 = invoke(stypy.reporting.localization.Localization(__file__, 149, 47), get_glyphs_mathtext_145182, *[prop_145183, s_145184], **kwargs_145185)
        
        # Assigning a type to the variable 'call_assignment_144756' (line 149)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 16), 'call_assignment_144756', get_glyphs_mathtext_call_result_145186)
        
        # Assigning a Call to a Name (line 149):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_145189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 16), 'int')
        # Processing the call keyword arguments
        kwargs_145190 = {}
        # Getting the type of 'call_assignment_144756' (line 149)
        call_assignment_144756_145187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 16), 'call_assignment_144756', False)
        # Obtaining the member '__getitem__' of a type (line 149)
        getitem___145188 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 16), call_assignment_144756_145187, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_145191 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___145188, *[int_145189], **kwargs_145190)
        
        # Assigning a type to the variable 'call_assignment_144757' (line 149)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 16), 'call_assignment_144757', getitem___call_result_145191)
        
        # Assigning a Name to a Name (line 149):
        # Getting the type of 'call_assignment_144757' (line 149)
        call_assignment_144757_145192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 16), 'call_assignment_144757')
        # Assigning a type to the variable 'glyph_info' (line 149)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 16), 'glyph_info', call_assignment_144757_145192)
        
        # Assigning a Call to a Name (line 149):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_145195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 16), 'int')
        # Processing the call keyword arguments
        kwargs_145196 = {}
        # Getting the type of 'call_assignment_144756' (line 149)
        call_assignment_144756_145193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 16), 'call_assignment_144756', False)
        # Obtaining the member '__getitem__' of a type (line 149)
        getitem___145194 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 16), call_assignment_144756_145193, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_145197 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___145194, *[int_145195], **kwargs_145196)
        
        # Assigning a type to the variable 'call_assignment_144758' (line 149)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 16), 'call_assignment_144758', getitem___call_result_145197)
        
        # Assigning a Name to a Name (line 149):
        # Getting the type of 'call_assignment_144758' (line 149)
        call_assignment_144758_145198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 16), 'call_assignment_144758')
        # Assigning a type to the variable 'glyph_map' (line 149)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 28), 'glyph_map', call_assignment_144758_145198)
        
        # Assigning a Call to a Name (line 149):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_145201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 16), 'int')
        # Processing the call keyword arguments
        kwargs_145202 = {}
        # Getting the type of 'call_assignment_144756' (line 149)
        call_assignment_144756_145199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 16), 'call_assignment_144756', False)
        # Obtaining the member '__getitem__' of a type (line 149)
        getitem___145200 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 16), call_assignment_144756_145199, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_145203 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___145200, *[int_145201], **kwargs_145202)
        
        # Assigning a type to the variable 'call_assignment_144759' (line 149)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 16), 'call_assignment_144759', getitem___call_result_145203)
        
        # Assigning a Name to a Name (line 149):
        # Getting the type of 'call_assignment_144759' (line 149)
        call_assignment_144759_145204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 16), 'call_assignment_144759')
        # Assigning a type to the variable 'rects' (line 149)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 39), 'rects', call_assignment_144759_145204)
        # SSA join for if statement (line 144)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 143)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Tuple (line 152):
        
        # Assigning a Call to a Name:
        
        # Call to get_glyphs_tex(...): (line 152)
        # Processing the call arguments (line 152)
        # Getting the type of 'prop' (line 152)
        prop_145207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 63), 'prop', False)
        # Getting the type of 's' (line 152)
        s_145208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 69), 's', False)
        # Processing the call keyword arguments (line 152)
        kwargs_145209 = {}
        # Getting the type of 'self' (line 152)
        self_145205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 43), 'self', False)
        # Obtaining the member 'get_glyphs_tex' of a type (line 152)
        get_glyphs_tex_145206 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 43), self_145205, 'get_glyphs_tex')
        # Calling get_glyphs_tex(args, kwargs) (line 152)
        get_glyphs_tex_call_result_145210 = invoke(stypy.reporting.localization.Localization(__file__, 152, 43), get_glyphs_tex_145206, *[prop_145207, s_145208], **kwargs_145209)
        
        # Assigning a type to the variable 'call_assignment_144760' (line 152)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 12), 'call_assignment_144760', get_glyphs_tex_call_result_145210)
        
        # Assigning a Call to a Name (line 152):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_145213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 12), 'int')
        # Processing the call keyword arguments
        kwargs_145214 = {}
        # Getting the type of 'call_assignment_144760' (line 152)
        call_assignment_144760_145211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 12), 'call_assignment_144760', False)
        # Obtaining the member '__getitem__' of a type (line 152)
        getitem___145212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 12), call_assignment_144760_145211, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_145215 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___145212, *[int_145213], **kwargs_145214)
        
        # Assigning a type to the variable 'call_assignment_144761' (line 152)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 12), 'call_assignment_144761', getitem___call_result_145215)
        
        # Assigning a Name to a Name (line 152):
        # Getting the type of 'call_assignment_144761' (line 152)
        call_assignment_144761_145216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 12), 'call_assignment_144761')
        # Assigning a type to the variable 'glyph_info' (line 152)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 12), 'glyph_info', call_assignment_144761_145216)
        
        # Assigning a Call to a Name (line 152):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_145219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 12), 'int')
        # Processing the call keyword arguments
        kwargs_145220 = {}
        # Getting the type of 'call_assignment_144760' (line 152)
        call_assignment_144760_145217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 12), 'call_assignment_144760', False)
        # Obtaining the member '__getitem__' of a type (line 152)
        getitem___145218 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 12), call_assignment_144760_145217, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_145221 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___145218, *[int_145219], **kwargs_145220)
        
        # Assigning a type to the variable 'call_assignment_144762' (line 152)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 12), 'call_assignment_144762', getitem___call_result_145221)
        
        # Assigning a Name to a Name (line 152):
        # Getting the type of 'call_assignment_144762' (line 152)
        call_assignment_144762_145222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 12), 'call_assignment_144762')
        # Assigning a type to the variable 'glyph_map' (line 152)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 24), 'glyph_map', call_assignment_144762_145222)
        
        # Assigning a Call to a Name (line 152):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_145225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 12), 'int')
        # Processing the call keyword arguments
        kwargs_145226 = {}
        # Getting the type of 'call_assignment_144760' (line 152)
        call_assignment_144760_145223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 12), 'call_assignment_144760', False)
        # Obtaining the member '__getitem__' of a type (line 152)
        getitem___145224 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 12), call_assignment_144760_145223, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_145227 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___145224, *[int_145225], **kwargs_145226)
        
        # Assigning a type to the variable 'call_assignment_144763' (line 152)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 12), 'call_assignment_144763', getitem___call_result_145227)
        
        # Assigning a Name to a Name (line 152):
        # Getting the type of 'call_assignment_144763' (line 152)
        call_assignment_144763_145228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 12), 'call_assignment_144763')
        # Assigning a type to the variable 'rects' (line 152)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 35), 'rects', call_assignment_144763_145228)
        # SSA join for if statement (line 143)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Tuple to a Tuple (line 154):
        
        # Assigning a List to a Name (line 154):
        
        # Obtaining an instance of the builtin type 'list' (line 154)
        list_145229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 154)
        
        # Assigning a type to the variable 'tuple_assignment_144764' (line 154)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'tuple_assignment_144764', list_145229)
        
        # Assigning a List to a Name (line 154):
        
        # Obtaining an instance of the builtin type 'list' (line 154)
        list_145230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 154)
        
        # Assigning a type to the variable 'tuple_assignment_144765' (line 154)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'tuple_assignment_144765', list_145230)
        
        # Assigning a Name to a Name (line 154):
        # Getting the type of 'tuple_assignment_144764' (line 154)
        tuple_assignment_144764_145231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'tuple_assignment_144764')
        # Assigning a type to the variable 'verts' (line 154)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'verts', tuple_assignment_144764_145231)
        
        # Assigning a Name to a Name (line 154):
        # Getting the type of 'tuple_assignment_144765' (line 154)
        tuple_assignment_144765_145232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'tuple_assignment_144765')
        # Assigning a type to the variable 'codes' (line 154)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 15), 'codes', tuple_assignment_144765_145232)
        
        # Getting the type of 'glyph_info' (line 156)
        glyph_info_145233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 53), 'glyph_info')
        # Testing the type of a for loop iterable (line 156)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 156, 8), glyph_info_145233)
        # Getting the type of the for loop variable (line 156)
        for_loop_var_145234 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 156, 8), glyph_info_145233)
        # Assigning a type to the variable 'glyph_id' (line 156)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 8), 'glyph_id', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 8), for_loop_var_145234))
        # Assigning a type to the variable 'xposition' (line 156)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 8), 'xposition', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 8), for_loop_var_145234))
        # Assigning a type to the variable 'yposition' (line 156)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 8), 'yposition', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 8), for_loop_var_145234))
        # Assigning a type to the variable 'scale' (line 156)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 8), 'scale', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 8), for_loop_var_145234))
        # SSA begins for a for statement (line 156)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Subscript to a Tuple (line 157):
        
        # Assigning a Subscript to a Name (line 157):
        
        # Obtaining the type of the subscript
        int_145235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 12), 'int')
        
        # Obtaining the type of the subscript
        # Getting the type of 'glyph_id' (line 157)
        glyph_id_145236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 39), 'glyph_id')
        # Getting the type of 'glyph_map' (line 157)
        glyph_map_145237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 29), 'glyph_map')
        # Obtaining the member '__getitem__' of a type (line 157)
        getitem___145238 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 29), glyph_map_145237, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 157)
        subscript_call_result_145239 = invoke(stypy.reporting.localization.Localization(__file__, 157, 29), getitem___145238, glyph_id_145236)
        
        # Obtaining the member '__getitem__' of a type (line 157)
        getitem___145240 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 12), subscript_call_result_145239, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 157)
        subscript_call_result_145241 = invoke(stypy.reporting.localization.Localization(__file__, 157, 12), getitem___145240, int_145235)
        
        # Assigning a type to the variable 'tuple_var_assignment_144766' (line 157)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 12), 'tuple_var_assignment_144766', subscript_call_result_145241)
        
        # Assigning a Subscript to a Name (line 157):
        
        # Obtaining the type of the subscript
        int_145242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 12), 'int')
        
        # Obtaining the type of the subscript
        # Getting the type of 'glyph_id' (line 157)
        glyph_id_145243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 39), 'glyph_id')
        # Getting the type of 'glyph_map' (line 157)
        glyph_map_145244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 29), 'glyph_map')
        # Obtaining the member '__getitem__' of a type (line 157)
        getitem___145245 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 29), glyph_map_145244, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 157)
        subscript_call_result_145246 = invoke(stypy.reporting.localization.Localization(__file__, 157, 29), getitem___145245, glyph_id_145243)
        
        # Obtaining the member '__getitem__' of a type (line 157)
        getitem___145247 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 12), subscript_call_result_145246, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 157)
        subscript_call_result_145248 = invoke(stypy.reporting.localization.Localization(__file__, 157, 12), getitem___145247, int_145242)
        
        # Assigning a type to the variable 'tuple_var_assignment_144767' (line 157)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 12), 'tuple_var_assignment_144767', subscript_call_result_145248)
        
        # Assigning a Name to a Name (line 157):
        # Getting the type of 'tuple_var_assignment_144766' (line 157)
        tuple_var_assignment_144766_145249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 12), 'tuple_var_assignment_144766')
        # Assigning a type to the variable 'verts1' (line 157)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 12), 'verts1', tuple_var_assignment_144766_145249)
        
        # Assigning a Name to a Name (line 157):
        # Getting the type of 'tuple_var_assignment_144767' (line 157)
        tuple_var_assignment_144767_145250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 12), 'tuple_var_assignment_144767')
        # Assigning a type to the variable 'codes1' (line 157)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 20), 'codes1', tuple_var_assignment_144767_145250)
        
        
        # Call to len(...): (line 158)
        # Processing the call arguments (line 158)
        # Getting the type of 'verts1' (line 158)
        verts1_145252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 19), 'verts1', False)
        # Processing the call keyword arguments (line 158)
        kwargs_145253 = {}
        # Getting the type of 'len' (line 158)
        len_145251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 15), 'len', False)
        # Calling len(args, kwargs) (line 158)
        len_call_result_145254 = invoke(stypy.reporting.localization.Localization(__file__, 158, 15), len_145251, *[verts1_145252], **kwargs_145253)
        
        # Testing the type of an if condition (line 158)
        if_condition_145255 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 158, 12), len_call_result_145254)
        # Assigning a type to the variable 'if_condition_145255' (line 158)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 12), 'if_condition_145255', if_condition_145255)
        # SSA begins for if statement (line 158)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 159):
        
        # Assigning a BinOp to a Name (line 159):
        
        # Call to array(...): (line 159)
        # Processing the call arguments (line 159)
        # Getting the type of 'verts1' (line 159)
        verts1_145258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 34), 'verts1', False)
        # Processing the call keyword arguments (line 159)
        kwargs_145259 = {}
        # Getting the type of 'np' (line 159)
        np_145256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 25), 'np', False)
        # Obtaining the member 'array' of a type (line 159)
        array_145257 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 25), np_145256, 'array')
        # Calling array(args, kwargs) (line 159)
        array_call_result_145260 = invoke(stypy.reporting.localization.Localization(__file__, 159, 25), array_145257, *[verts1_145258], **kwargs_145259)
        
        # Getting the type of 'scale' (line 159)
        scale_145261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 44), 'scale')
        # Applying the binary operator '*' (line 159)
        result_mul_145262 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 25), '*', array_call_result_145260, scale_145261)
        
        
        # Obtaining an instance of the builtin type 'list' (line 159)
        list_145263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 52), 'list')
        # Adding type elements to the builtin type 'list' instance (line 159)
        # Adding element type (line 159)
        # Getting the type of 'xposition' (line 159)
        xposition_145264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 53), 'xposition')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 159, 52), list_145263, xposition_145264)
        # Adding element type (line 159)
        # Getting the type of 'yposition' (line 159)
        yposition_145265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 64), 'yposition')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 159, 52), list_145263, yposition_145265)
        
        # Applying the binary operator '+' (line 159)
        result_add_145266 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 25), '+', result_mul_145262, list_145263)
        
        # Assigning a type to the variable 'verts1' (line 159)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 16), 'verts1', result_add_145266)
        
        # Call to extend(...): (line 160)
        # Processing the call arguments (line 160)
        # Getting the type of 'verts1' (line 160)
        verts1_145269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 29), 'verts1', False)
        # Processing the call keyword arguments (line 160)
        kwargs_145270 = {}
        # Getting the type of 'verts' (line 160)
        verts_145267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 16), 'verts', False)
        # Obtaining the member 'extend' of a type (line 160)
        extend_145268 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 16), verts_145267, 'extend')
        # Calling extend(args, kwargs) (line 160)
        extend_call_result_145271 = invoke(stypy.reporting.localization.Localization(__file__, 160, 16), extend_145268, *[verts1_145269], **kwargs_145270)
        
        
        # Call to extend(...): (line 161)
        # Processing the call arguments (line 161)
        # Getting the type of 'codes1' (line 161)
        codes1_145274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 29), 'codes1', False)
        # Processing the call keyword arguments (line 161)
        kwargs_145275 = {}
        # Getting the type of 'codes' (line 161)
        codes_145272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 16), 'codes', False)
        # Obtaining the member 'extend' of a type (line 161)
        extend_145273 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 16), codes_145272, 'extend')
        # Calling extend(args, kwargs) (line 161)
        extend_call_result_145276 = invoke(stypy.reporting.localization.Localization(__file__, 161, 16), extend_145273, *[codes1_145274], **kwargs_145275)
        
        # SSA join for if statement (line 158)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'rects' (line 163)
        rects_145277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 30), 'rects')
        # Testing the type of a for loop iterable (line 163)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 163, 8), rects_145277)
        # Getting the type of the for loop variable (line 163)
        for_loop_var_145278 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 163, 8), rects_145277)
        # Assigning a type to the variable 'verts1' (line 163)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 8), 'verts1', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 163, 8), for_loop_var_145278))
        # Assigning a type to the variable 'codes1' (line 163)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 8), 'codes1', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 163, 8), for_loop_var_145278))
        # SSA begins for a for statement (line 163)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to extend(...): (line 164)
        # Processing the call arguments (line 164)
        # Getting the type of 'verts1' (line 164)
        verts1_145281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 25), 'verts1', False)
        # Processing the call keyword arguments (line 164)
        kwargs_145282 = {}
        # Getting the type of 'verts' (line 164)
        verts_145279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 12), 'verts', False)
        # Obtaining the member 'extend' of a type (line 164)
        extend_145280 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 12), verts_145279, 'extend')
        # Calling extend(args, kwargs) (line 164)
        extend_call_result_145283 = invoke(stypy.reporting.localization.Localization(__file__, 164, 12), extend_145280, *[verts1_145281], **kwargs_145282)
        
        
        # Call to extend(...): (line 165)
        # Processing the call arguments (line 165)
        # Getting the type of 'codes1' (line 165)
        codes1_145286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 25), 'codes1', False)
        # Processing the call keyword arguments (line 165)
        kwargs_145287 = {}
        # Getting the type of 'codes' (line 165)
        codes_145284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 12), 'codes', False)
        # Obtaining the member 'extend' of a type (line 165)
        extend_145285 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 12), codes_145284, 'extend')
        # Calling extend(args, kwargs) (line 165)
        extend_call_result_145288 = invoke(stypy.reporting.localization.Localization(__file__, 165, 12), extend_145285, *[codes1_145286], **kwargs_145287)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 167)
        tuple_145289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 167)
        # Adding element type (line 167)
        # Getting the type of 'verts' (line 167)
        verts_145290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 15), 'verts')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 15), tuple_145289, verts_145290)
        # Adding element type (line 167)
        # Getting the type of 'codes' (line 167)
        codes_145291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 22), 'codes')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 15), tuple_145289, codes_145291)
        
        # Assigning a type to the variable 'stypy_return_type' (line 167)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 'stypy_return_type', tuple_145289)
        
        # ################# End of 'get_text_path(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_text_path' in the type store
        # Getting the type of 'stypy_return_type' (line 124)
        stypy_return_type_145292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_145292)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_text_path'
        return stypy_return_type_145292


    @norecursion
    def get_glyphs_with_font(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 169)
        None_145293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 54), 'None')
        # Getting the type of 'False' (line 170)
        False_145294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 52), 'False')
        defaults = [None_145293, False_145294]
        # Create a new context for function 'get_glyphs_with_font'
        module_type_store = module_type_store.open_function_context('get_glyphs_with_font', 169, 4, False)
        # Assigning a type to the variable 'self' (line 170)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TextToPath.get_glyphs_with_font.__dict__.__setitem__('stypy_localization', localization)
        TextToPath.get_glyphs_with_font.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TextToPath.get_glyphs_with_font.__dict__.__setitem__('stypy_type_store', module_type_store)
        TextToPath.get_glyphs_with_font.__dict__.__setitem__('stypy_function_name', 'TextToPath.get_glyphs_with_font')
        TextToPath.get_glyphs_with_font.__dict__.__setitem__('stypy_param_names_list', ['font', 's', 'glyph_map', 'return_new_glyphs_only'])
        TextToPath.get_glyphs_with_font.__dict__.__setitem__('stypy_varargs_param_name', None)
        TextToPath.get_glyphs_with_font.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TextToPath.get_glyphs_with_font.__dict__.__setitem__('stypy_call_defaults', defaults)
        TextToPath.get_glyphs_with_font.__dict__.__setitem__('stypy_call_varargs', varargs)
        TextToPath.get_glyphs_with_font.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TextToPath.get_glyphs_with_font.__dict__.__setitem__('stypy_declared_arg_number', 5)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TextToPath.get_glyphs_with_font', ['font', 's', 'glyph_map', 'return_new_glyphs_only'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_glyphs_with_font', localization, ['font', 's', 'glyph_map', 'return_new_glyphs_only'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_glyphs_with_font(...)' code ##################

        unicode_145295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, (-1)), 'unicode', u'\n        convert the string *s* to vertices and codes using the\n        provided ttf font.\n        ')
        
        # Assigning a Name to a Name (line 178):
        
        # Assigning a Name to a Name (line 178):
        # Getting the type of 'None' (line 178)
        None_145296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 19), 'None')
        # Assigning a type to the variable 'lastgind' (line 178)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 8), 'lastgind', None_145296)
        
        # Assigning a Num to a Name (line 180):
        
        # Assigning a Num to a Name (line 180):
        int_145297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 16), 'int')
        # Assigning a type to the variable 'currx' (line 180)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 8), 'currx', int_145297)
        
        # Assigning a List to a Name (line 181):
        
        # Assigning a List to a Name (line 181):
        
        # Obtaining an instance of the builtin type 'list' (line 181)
        list_145298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 181)
        
        # Assigning a type to the variable 'xpositions' (line 181)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 'xpositions', list_145298)
        
        # Assigning a List to a Name (line 182):
        
        # Assigning a List to a Name (line 182):
        
        # Obtaining an instance of the builtin type 'list' (line 182)
        list_145299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 182)
        
        # Assigning a type to the variable 'glyph_ids' (line 182)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), 'glyph_ids', list_145299)
        
        # Type idiom detected: calculating its left and rigth part (line 184)
        # Getting the type of 'glyph_map' (line 184)
        glyph_map_145300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 11), 'glyph_map')
        # Getting the type of 'None' (line 184)
        None_145301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 24), 'None')
        
        (may_be_145302, more_types_in_union_145303) = may_be_none(glyph_map_145300, None_145301)

        if may_be_145302:

            if more_types_in_union_145303:
                # Runtime conditional SSA (line 184)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 185):
            
            # Assigning a Call to a Name (line 185):
            
            # Call to OrderedDict(...): (line 185)
            # Processing the call keyword arguments (line 185)
            kwargs_145305 = {}
            # Getting the type of 'OrderedDict' (line 185)
            OrderedDict_145304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 24), 'OrderedDict', False)
            # Calling OrderedDict(args, kwargs) (line 185)
            OrderedDict_call_result_145306 = invoke(stypy.reporting.localization.Localization(__file__, 185, 24), OrderedDict_145304, *[], **kwargs_145305)
            
            # Assigning a type to the variable 'glyph_map' (line 185)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 12), 'glyph_map', OrderedDict_call_result_145306)

            if more_types_in_union_145303:
                # SSA join for if statement (line 184)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Getting the type of 'return_new_glyphs_only' (line 187)
        return_new_glyphs_only_145307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 11), 'return_new_glyphs_only')
        # Testing the type of an if condition (line 187)
        if_condition_145308 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 187, 8), return_new_glyphs_only_145307)
        # Assigning a type to the variable 'if_condition_145308' (line 187)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 8), 'if_condition_145308', if_condition_145308)
        # SSA begins for if statement (line 187)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 188):
        
        # Assigning a Call to a Name (line 188):
        
        # Call to OrderedDict(...): (line 188)
        # Processing the call keyword arguments (line 188)
        kwargs_145310 = {}
        # Getting the type of 'OrderedDict' (line 188)
        OrderedDict_145309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 28), 'OrderedDict', False)
        # Calling OrderedDict(args, kwargs) (line 188)
        OrderedDict_call_result_145311 = invoke(stypy.reporting.localization.Localization(__file__, 188, 28), OrderedDict_145309, *[], **kwargs_145310)
        
        # Assigning a type to the variable 'glyph_map_new' (line 188)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 12), 'glyph_map_new', OrderedDict_call_result_145311)
        # SSA branch for the else part of an if statement (line 187)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Name to a Name (line 190):
        
        # Assigning a Name to a Name (line 190):
        # Getting the type of 'glyph_map' (line 190)
        glyph_map_145312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 28), 'glyph_map')
        # Assigning a type to the variable 'glyph_map_new' (line 190)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 12), 'glyph_map_new', glyph_map_145312)
        # SSA join for if statement (line 187)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 's' (line 194)
        s_145313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 17), 's')
        # Testing the type of a for loop iterable (line 194)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 194, 8), s_145313)
        # Getting the type of the for loop variable (line 194)
        for_loop_var_145314 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 194, 8), s_145313)
        # Assigning a type to the variable 'c' (line 194)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 8), 'c', for_loop_var_145314)
        # SSA begins for a for statement (line 194)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 195):
        
        # Assigning a Call to a Name (line 195):
        
        # Call to ord(...): (line 195)
        # Processing the call arguments (line 195)
        # Getting the type of 'c' (line 195)
        c_145316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 24), 'c', False)
        # Processing the call keyword arguments (line 195)
        kwargs_145317 = {}
        # Getting the type of 'ord' (line 195)
        ord_145315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 20), 'ord', False)
        # Calling ord(args, kwargs) (line 195)
        ord_call_result_145318 = invoke(stypy.reporting.localization.Localization(__file__, 195, 20), ord_145315, *[c_145316], **kwargs_145317)
        
        # Assigning a type to the variable 'ccode' (line 195)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 12), 'ccode', ord_call_result_145318)
        
        # Assigning a Call to a Name (line 196):
        
        # Assigning a Call to a Name (line 196):
        
        # Call to get_char_index(...): (line 196)
        # Processing the call arguments (line 196)
        # Getting the type of 'ccode' (line 196)
        ccode_145321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 39), 'ccode', False)
        # Processing the call keyword arguments (line 196)
        kwargs_145322 = {}
        # Getting the type of 'font' (line 196)
        font_145319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 19), 'font', False)
        # Obtaining the member 'get_char_index' of a type (line 196)
        get_char_index_145320 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 19), font_145319, 'get_char_index')
        # Calling get_char_index(args, kwargs) (line 196)
        get_char_index_call_result_145323 = invoke(stypy.reporting.localization.Localization(__file__, 196, 19), get_char_index_145320, *[ccode_145321], **kwargs_145322)
        
        # Assigning a type to the variable 'gind' (line 196)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 12), 'gind', get_char_index_call_result_145323)
        
        # Type idiom detected: calculating its left and rigth part (line 197)
        # Getting the type of 'gind' (line 197)
        gind_145324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 15), 'gind')
        # Getting the type of 'None' (line 197)
        None_145325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 23), 'None')
        
        (may_be_145326, more_types_in_union_145327) = may_be_none(gind_145324, None_145325)

        if may_be_145326:

            if more_types_in_union_145327:
                # Runtime conditional SSA (line 197)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 198):
            
            # Assigning a Call to a Name (line 198):
            
            # Call to ord(...): (line 198)
            # Processing the call arguments (line 198)
            unicode_145329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 28), 'unicode', u'?')
            # Processing the call keyword arguments (line 198)
            kwargs_145330 = {}
            # Getting the type of 'ord' (line 198)
            ord_145328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 24), 'ord', False)
            # Calling ord(args, kwargs) (line 198)
            ord_call_result_145331 = invoke(stypy.reporting.localization.Localization(__file__, 198, 24), ord_145328, *[unicode_145329], **kwargs_145330)
            
            # Assigning a type to the variable 'ccode' (line 198)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 16), 'ccode', ord_call_result_145331)
            
            # Assigning a Num to a Name (line 199):
            
            # Assigning a Num to a Name (line 199):
            int_145332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 23), 'int')
            # Assigning a type to the variable 'gind' (line 199)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 16), 'gind', int_145332)

            if more_types_in_union_145327:
                # SSA join for if statement (line 197)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 201)
        # Getting the type of 'lastgind' (line 201)
        lastgind_145333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 12), 'lastgind')
        # Getting the type of 'None' (line 201)
        None_145334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 31), 'None')
        
        (may_be_145335, more_types_in_union_145336) = may_not_be_none(lastgind_145333, None_145334)

        if may_be_145335:

            if more_types_in_union_145336:
                # Runtime conditional SSA (line 201)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 202):
            
            # Assigning a Call to a Name (line 202):
            
            # Call to get_kerning(...): (line 202)
            # Processing the call arguments (line 202)
            # Getting the type of 'lastgind' (line 202)
            lastgind_145339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 40), 'lastgind', False)
            # Getting the type of 'gind' (line 202)
            gind_145340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 50), 'gind', False)
            # Getting the type of 'KERNING_DEFAULT' (line 202)
            KERNING_DEFAULT_145341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 56), 'KERNING_DEFAULT', False)
            # Processing the call keyword arguments (line 202)
            kwargs_145342 = {}
            # Getting the type of 'font' (line 202)
            font_145337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 23), 'font', False)
            # Obtaining the member 'get_kerning' of a type (line 202)
            get_kerning_145338 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 23), font_145337, 'get_kerning')
            # Calling get_kerning(args, kwargs) (line 202)
            get_kerning_call_result_145343 = invoke(stypy.reporting.localization.Localization(__file__, 202, 23), get_kerning_145338, *[lastgind_145339, gind_145340, KERNING_DEFAULT_145341], **kwargs_145342)
            
            # Assigning a type to the variable 'kern' (line 202)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 16), 'kern', get_kerning_call_result_145343)

            if more_types_in_union_145336:
                # Runtime conditional SSA for else branch (line 201)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_145335) or more_types_in_union_145336):
            
            # Assigning a Num to a Name (line 204):
            
            # Assigning a Num to a Name (line 204):
            int_145344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 23), 'int')
            # Assigning a type to the variable 'kern' (line 204)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 16), 'kern', int_145344)

            if (may_be_145335 and more_types_in_union_145336):
                # SSA join for if statement (line 201)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Name (line 206):
        
        # Assigning a Call to a Name (line 206):
        
        # Call to load_char(...): (line 206)
        # Processing the call arguments (line 206)
        # Getting the type of 'ccode' (line 206)
        ccode_145347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 35), 'ccode', False)
        # Processing the call keyword arguments (line 206)
        # Getting the type of 'LOAD_NO_HINTING' (line 206)
        LOAD_NO_HINTING_145348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 48), 'LOAD_NO_HINTING', False)
        keyword_145349 = LOAD_NO_HINTING_145348
        kwargs_145350 = {'flags': keyword_145349}
        # Getting the type of 'font' (line 206)
        font_145345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 20), 'font', False)
        # Obtaining the member 'load_char' of a type (line 206)
        load_char_145346 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 20), font_145345, 'load_char')
        # Calling load_char(args, kwargs) (line 206)
        load_char_call_result_145351 = invoke(stypy.reporting.localization.Localization(__file__, 206, 20), load_char_145346, *[ccode_145347], **kwargs_145350)
        
        # Assigning a type to the variable 'glyph' (line 206)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 12), 'glyph', load_char_call_result_145351)
        
        # Assigning a BinOp to a Name (line 207):
        
        # Assigning a BinOp to a Name (line 207):
        # Getting the type of 'glyph' (line 207)
        glyph_145352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 29), 'glyph')
        # Obtaining the member 'linearHoriAdvance' of a type (line 207)
        linearHoriAdvance_145353 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 29), glyph_145352, 'linearHoriAdvance')
        float_145354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 55), 'float')
        # Applying the binary operator 'div' (line 207)
        result_div_145355 = python_operator(stypy.reporting.localization.Localization(__file__, 207, 29), 'div', linearHoriAdvance_145353, float_145354)
        
        # Assigning a type to the variable 'horiz_advance' (line 207)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 12), 'horiz_advance', result_div_145355)
        
        # Assigning a Call to a Name (line 209):
        
        # Assigning a Call to a Name (line 209):
        
        # Call to _get_char_id(...): (line 209)
        # Processing the call arguments (line 209)
        # Getting the type of 'font' (line 209)
        font_145358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 40), 'font', False)
        # Getting the type of 'ccode' (line 209)
        ccode_145359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 46), 'ccode', False)
        # Processing the call keyword arguments (line 209)
        kwargs_145360 = {}
        # Getting the type of 'self' (line 209)
        self_145356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 22), 'self', False)
        # Obtaining the member '_get_char_id' of a type (line 209)
        _get_char_id_145357 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 22), self_145356, '_get_char_id')
        # Calling _get_char_id(args, kwargs) (line 209)
        _get_char_id_call_result_145361 = invoke(stypy.reporting.localization.Localization(__file__, 209, 22), _get_char_id_145357, *[font_145358, ccode_145359], **kwargs_145360)
        
        # Assigning a type to the variable 'char_id' (line 209)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 12), 'char_id', _get_char_id_call_result_145361)
        
        
        # Getting the type of 'char_id' (line 210)
        char_id_145362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 15), 'char_id')
        # Getting the type of 'glyph_map' (line 210)
        glyph_map_145363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 30), 'glyph_map')
        # Applying the binary operator 'notin' (line 210)
        result_contains_145364 = python_operator(stypy.reporting.localization.Localization(__file__, 210, 15), 'notin', char_id_145362, glyph_map_145363)
        
        # Testing the type of an if condition (line 210)
        if_condition_145365 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 210, 12), result_contains_145364)
        # Assigning a type to the variable 'if_condition_145365' (line 210)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 12), 'if_condition_145365', if_condition_145365)
        # SSA begins for if statement (line 210)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Subscript (line 211):
        
        # Assigning a Call to a Subscript (line 211):
        
        # Call to glyph_to_path(...): (line 211)
        # Processing the call arguments (line 211)
        # Getting the type of 'font' (line 211)
        font_145368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 60), 'font', False)
        # Processing the call keyword arguments (line 211)
        kwargs_145369 = {}
        # Getting the type of 'self' (line 211)
        self_145366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 41), 'self', False)
        # Obtaining the member 'glyph_to_path' of a type (line 211)
        glyph_to_path_145367 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 41), self_145366, 'glyph_to_path')
        # Calling glyph_to_path(args, kwargs) (line 211)
        glyph_to_path_call_result_145370 = invoke(stypy.reporting.localization.Localization(__file__, 211, 41), glyph_to_path_145367, *[font_145368], **kwargs_145369)
        
        # Getting the type of 'glyph_map_new' (line 211)
        glyph_map_new_145371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 16), 'glyph_map_new')
        # Getting the type of 'char_id' (line 211)
        char_id_145372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 30), 'char_id')
        # Storing an element on a container (line 211)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 211, 16), glyph_map_new_145371, (char_id_145372, glyph_to_path_call_result_145370))
        # SSA join for if statement (line 210)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'currx' (line 213)
        currx_145373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 12), 'currx')
        # Getting the type of 'kern' (line 213)
        kern_145374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 22), 'kern')
        float_145375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 29), 'float')
        # Applying the binary operator 'div' (line 213)
        result_div_145376 = python_operator(stypy.reporting.localization.Localization(__file__, 213, 22), 'div', kern_145374, float_145375)
        
        # Applying the binary operator '+=' (line 213)
        result_iadd_145377 = python_operator(stypy.reporting.localization.Localization(__file__, 213, 12), '+=', currx_145373, result_div_145376)
        # Assigning a type to the variable 'currx' (line 213)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 12), 'currx', result_iadd_145377)
        
        
        # Call to append(...): (line 215)
        # Processing the call arguments (line 215)
        # Getting the type of 'currx' (line 215)
        currx_145380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 30), 'currx', False)
        # Processing the call keyword arguments (line 215)
        kwargs_145381 = {}
        # Getting the type of 'xpositions' (line 215)
        xpositions_145378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 12), 'xpositions', False)
        # Obtaining the member 'append' of a type (line 215)
        append_145379 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 12), xpositions_145378, 'append')
        # Calling append(args, kwargs) (line 215)
        append_call_result_145382 = invoke(stypy.reporting.localization.Localization(__file__, 215, 12), append_145379, *[currx_145380], **kwargs_145381)
        
        
        # Call to append(...): (line 216)
        # Processing the call arguments (line 216)
        # Getting the type of 'char_id' (line 216)
        char_id_145385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 29), 'char_id', False)
        # Processing the call keyword arguments (line 216)
        kwargs_145386 = {}
        # Getting the type of 'glyph_ids' (line 216)
        glyph_ids_145383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 12), 'glyph_ids', False)
        # Obtaining the member 'append' of a type (line 216)
        append_145384 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 12), glyph_ids_145383, 'append')
        # Calling append(args, kwargs) (line 216)
        append_call_result_145387 = invoke(stypy.reporting.localization.Localization(__file__, 216, 12), append_145384, *[char_id_145385], **kwargs_145386)
        
        
        # Getting the type of 'currx' (line 218)
        currx_145388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 12), 'currx')
        # Getting the type of 'horiz_advance' (line 218)
        horiz_advance_145389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 21), 'horiz_advance')
        # Applying the binary operator '+=' (line 218)
        result_iadd_145390 = python_operator(stypy.reporting.localization.Localization(__file__, 218, 12), '+=', currx_145388, horiz_advance_145389)
        # Assigning a type to the variable 'currx' (line 218)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 12), 'currx', result_iadd_145390)
        
        
        # Assigning a Name to a Name (line 220):
        
        # Assigning a Name to a Name (line 220):
        # Getting the type of 'gind' (line 220)
        gind_145391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 23), 'gind')
        # Assigning a type to the variable 'lastgind' (line 220)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 12), 'lastgind', gind_145391)
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 222):
        
        # Assigning a BinOp to a Name (line 222):
        
        # Obtaining an instance of the builtin type 'list' (line 222)
        list_145392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 222)
        # Adding element type (line 222)
        int_145393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 222, 21), list_145392, int_145393)
        
        
        # Call to len(...): (line 222)
        # Processing the call arguments (line 222)
        # Getting the type of 'xpositions' (line 222)
        xpositions_145395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 31), 'xpositions', False)
        # Processing the call keyword arguments (line 222)
        kwargs_145396 = {}
        # Getting the type of 'len' (line 222)
        len_145394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 27), 'len', False)
        # Calling len(args, kwargs) (line 222)
        len_call_result_145397 = invoke(stypy.reporting.localization.Localization(__file__, 222, 27), len_145394, *[xpositions_145395], **kwargs_145396)
        
        # Applying the binary operator '*' (line 222)
        result_mul_145398 = python_operator(stypy.reporting.localization.Localization(__file__, 222, 21), '*', list_145392, len_call_result_145397)
        
        # Assigning a type to the variable 'ypositions' (line 222)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 8), 'ypositions', result_mul_145398)
        
        # Assigning a BinOp to a Name (line 223):
        
        # Assigning a BinOp to a Name (line 223):
        
        # Obtaining an instance of the builtin type 'list' (line 223)
        list_145399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 223)
        # Adding element type (line 223)
        float_145400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 17), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 223, 16), list_145399, float_145400)
        
        
        # Call to len(...): (line 223)
        # Processing the call arguments (line 223)
        # Getting the type of 'xpositions' (line 223)
        xpositions_145402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 27), 'xpositions', False)
        # Processing the call keyword arguments (line 223)
        kwargs_145403 = {}
        # Getting the type of 'len' (line 223)
        len_145401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 23), 'len', False)
        # Calling len(args, kwargs) (line 223)
        len_call_result_145404 = invoke(stypy.reporting.localization.Localization(__file__, 223, 23), len_145401, *[xpositions_145402], **kwargs_145403)
        
        # Applying the binary operator '*' (line 223)
        result_mul_145405 = python_operator(stypy.reporting.localization.Localization(__file__, 223, 16), '*', list_145399, len_call_result_145404)
        
        # Assigning a type to the variable 'sizes' (line 223)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'sizes', result_mul_145405)
        
        # Assigning a List to a Name (line 225):
        
        # Assigning a List to a Name (line 225):
        
        # Obtaining an instance of the builtin type 'list' (line 225)
        list_145406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 225)
        
        # Assigning a type to the variable 'rects' (line 225)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 8), 'rects', list_145406)
        
        # Obtaining an instance of the builtin type 'tuple' (line 227)
        tuple_145407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 16), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 227)
        # Adding element type (line 227)
        
        # Call to list(...): (line 227)
        # Processing the call arguments (line 227)
        
        # Call to zip(...): (line 227)
        # Processing the call arguments (line 227)
        # Getting the type of 'glyph_ids' (line 227)
        glyph_ids_145410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 25), 'glyph_ids', False)
        # Getting the type of 'xpositions' (line 227)
        xpositions_145411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 36), 'xpositions', False)
        # Getting the type of 'ypositions' (line 227)
        ypositions_145412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 48), 'ypositions', False)
        # Getting the type of 'sizes' (line 227)
        sizes_145413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 60), 'sizes', False)
        # Processing the call keyword arguments (line 227)
        kwargs_145414 = {}
        # Getting the type of 'zip' (line 227)
        zip_145409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 21), 'zip', False)
        # Calling zip(args, kwargs) (line 227)
        zip_call_result_145415 = invoke(stypy.reporting.localization.Localization(__file__, 227, 21), zip_145409, *[glyph_ids_145410, xpositions_145411, ypositions_145412, sizes_145413], **kwargs_145414)
        
        # Processing the call keyword arguments (line 227)
        kwargs_145416 = {}
        # Getting the type of 'list' (line 227)
        list_145408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 16), 'list', False)
        # Calling list(args, kwargs) (line 227)
        list_call_result_145417 = invoke(stypy.reporting.localization.Localization(__file__, 227, 16), list_145408, *[zip_call_result_145415], **kwargs_145416)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 227, 16), tuple_145407, list_call_result_145417)
        # Adding element type (line 227)
        # Getting the type of 'glyph_map_new' (line 228)
        glyph_map_new_145418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 21), 'glyph_map_new')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 227, 16), tuple_145407, glyph_map_new_145418)
        # Adding element type (line 227)
        # Getting the type of 'rects' (line 228)
        rects_145419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 36), 'rects')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 227, 16), tuple_145407, rects_145419)
        
        # Assigning a type to the variable 'stypy_return_type' (line 227)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'stypy_return_type', tuple_145407)
        
        # ################# End of 'get_glyphs_with_font(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_glyphs_with_font' in the type store
        # Getting the type of 'stypy_return_type' (line 169)
        stypy_return_type_145420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_145420)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_glyphs_with_font'
        return stypy_return_type_145420


    @norecursion
    def get_glyphs_mathtext(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 230)
        None_145421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 53), 'None')
        # Getting the type of 'False' (line 231)
        False_145422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 51), 'False')
        defaults = [None_145421, False_145422]
        # Create a new context for function 'get_glyphs_mathtext'
        module_type_store = module_type_store.open_function_context('get_glyphs_mathtext', 230, 4, False)
        # Assigning a type to the variable 'self' (line 231)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TextToPath.get_glyphs_mathtext.__dict__.__setitem__('stypy_localization', localization)
        TextToPath.get_glyphs_mathtext.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TextToPath.get_glyphs_mathtext.__dict__.__setitem__('stypy_type_store', module_type_store)
        TextToPath.get_glyphs_mathtext.__dict__.__setitem__('stypy_function_name', 'TextToPath.get_glyphs_mathtext')
        TextToPath.get_glyphs_mathtext.__dict__.__setitem__('stypy_param_names_list', ['prop', 's', 'glyph_map', 'return_new_glyphs_only'])
        TextToPath.get_glyphs_mathtext.__dict__.__setitem__('stypy_varargs_param_name', None)
        TextToPath.get_glyphs_mathtext.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TextToPath.get_glyphs_mathtext.__dict__.__setitem__('stypy_call_defaults', defaults)
        TextToPath.get_glyphs_mathtext.__dict__.__setitem__('stypy_call_varargs', varargs)
        TextToPath.get_glyphs_mathtext.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TextToPath.get_glyphs_mathtext.__dict__.__setitem__('stypy_declared_arg_number', 5)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TextToPath.get_glyphs_mathtext', ['prop', 's', 'glyph_map', 'return_new_glyphs_only'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_glyphs_mathtext', localization, ['prop', 's', 'glyph_map', 'return_new_glyphs_only'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_glyphs_mathtext(...)' code ##################

        unicode_145423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, (-1)), 'unicode', u'\n        convert the string *s* to vertices and codes by parsing it with\n        mathtext.\n        ')
        
        # Assigning a Call to a Name (line 237):
        
        # Assigning a Call to a Name (line 237):
        
        # Call to copy(...): (line 237)
        # Processing the call keyword arguments (line 237)
        kwargs_145426 = {}
        # Getting the type of 'prop' (line 237)
        prop_145424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 15), 'prop', False)
        # Obtaining the member 'copy' of a type (line 237)
        copy_145425 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 15), prop_145424, 'copy')
        # Calling copy(args, kwargs) (line 237)
        copy_call_result_145427 = invoke(stypy.reporting.localization.Localization(__file__, 237, 15), copy_145425, *[], **kwargs_145426)
        
        # Assigning a type to the variable 'prop' (line 237)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 8), 'prop', copy_call_result_145427)
        
        # Call to set_size(...): (line 238)
        # Processing the call arguments (line 238)
        # Getting the type of 'self' (line 238)
        self_145430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 22), 'self', False)
        # Obtaining the member 'FONT_SCALE' of a type (line 238)
        FONT_SCALE_145431 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 22), self_145430, 'FONT_SCALE')
        # Processing the call keyword arguments (line 238)
        kwargs_145432 = {}
        # Getting the type of 'prop' (line 238)
        prop_145428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 8), 'prop', False)
        # Obtaining the member 'set_size' of a type (line 238)
        set_size_145429 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 8), prop_145428, 'set_size')
        # Calling set_size(args, kwargs) (line 238)
        set_size_call_result_145433 = invoke(stypy.reporting.localization.Localization(__file__, 238, 8), set_size_145429, *[FONT_SCALE_145431], **kwargs_145432)
        
        
        # Assigning a Call to a Tuple (line 240):
        
        # Assigning a Call to a Name:
        
        # Call to parse(...): (line 240)
        # Processing the call arguments (line 240)
        # Getting the type of 's' (line 241)
        s_145437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 12), 's', False)
        # Getting the type of 'self' (line 241)
        self_145438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 15), 'self', False)
        # Obtaining the member 'DPI' of a type (line 241)
        DPI_145439 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 15), self_145438, 'DPI')
        # Getting the type of 'prop' (line 241)
        prop_145440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 25), 'prop', False)
        # Processing the call keyword arguments (line 240)
        kwargs_145441 = {}
        # Getting the type of 'self' (line 240)
        self_145434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 48), 'self', False)
        # Obtaining the member 'mathtext_parser' of a type (line 240)
        mathtext_parser_145435 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 48), self_145434, 'mathtext_parser')
        # Obtaining the member 'parse' of a type (line 240)
        parse_145436 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 48), mathtext_parser_145435, 'parse')
        # Calling parse(args, kwargs) (line 240)
        parse_call_result_145442 = invoke(stypy.reporting.localization.Localization(__file__, 240, 48), parse_145436, *[s_145437, DPI_145439, prop_145440], **kwargs_145441)
        
        # Assigning a type to the variable 'call_assignment_144768' (line 240)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'call_assignment_144768', parse_call_result_145442)
        
        # Assigning a Call to a Name (line 240):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_145445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 8), 'int')
        # Processing the call keyword arguments
        kwargs_145446 = {}
        # Getting the type of 'call_assignment_144768' (line 240)
        call_assignment_144768_145443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'call_assignment_144768', False)
        # Obtaining the member '__getitem__' of a type (line 240)
        getitem___145444 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 8), call_assignment_144768_145443, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_145447 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___145444, *[int_145445], **kwargs_145446)
        
        # Assigning a type to the variable 'call_assignment_144769' (line 240)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'call_assignment_144769', getitem___call_result_145447)
        
        # Assigning a Name to a Name (line 240):
        # Getting the type of 'call_assignment_144769' (line 240)
        call_assignment_144769_145448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'call_assignment_144769')
        # Assigning a type to the variable 'width' (line 240)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'width', call_assignment_144769_145448)
        
        # Assigning a Call to a Name (line 240):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_145451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 8), 'int')
        # Processing the call keyword arguments
        kwargs_145452 = {}
        # Getting the type of 'call_assignment_144768' (line 240)
        call_assignment_144768_145449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'call_assignment_144768', False)
        # Obtaining the member '__getitem__' of a type (line 240)
        getitem___145450 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 8), call_assignment_144768_145449, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_145453 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___145450, *[int_145451], **kwargs_145452)
        
        # Assigning a type to the variable 'call_assignment_144770' (line 240)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'call_assignment_144770', getitem___call_result_145453)
        
        # Assigning a Name to a Name (line 240):
        # Getting the type of 'call_assignment_144770' (line 240)
        call_assignment_144770_145454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'call_assignment_144770')
        # Assigning a type to the variable 'height' (line 240)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 15), 'height', call_assignment_144770_145454)
        
        # Assigning a Call to a Name (line 240):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_145457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 8), 'int')
        # Processing the call keyword arguments
        kwargs_145458 = {}
        # Getting the type of 'call_assignment_144768' (line 240)
        call_assignment_144768_145455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'call_assignment_144768', False)
        # Obtaining the member '__getitem__' of a type (line 240)
        getitem___145456 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 8), call_assignment_144768_145455, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_145459 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___145456, *[int_145457], **kwargs_145458)
        
        # Assigning a type to the variable 'call_assignment_144771' (line 240)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'call_assignment_144771', getitem___call_result_145459)
        
        # Assigning a Name to a Name (line 240):
        # Getting the type of 'call_assignment_144771' (line 240)
        call_assignment_144771_145460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'call_assignment_144771')
        # Assigning a type to the variable 'descent' (line 240)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 23), 'descent', call_assignment_144771_145460)
        
        # Assigning a Call to a Name (line 240):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_145463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 8), 'int')
        # Processing the call keyword arguments
        kwargs_145464 = {}
        # Getting the type of 'call_assignment_144768' (line 240)
        call_assignment_144768_145461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'call_assignment_144768', False)
        # Obtaining the member '__getitem__' of a type (line 240)
        getitem___145462 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 8), call_assignment_144768_145461, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_145465 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___145462, *[int_145463], **kwargs_145464)
        
        # Assigning a type to the variable 'call_assignment_144772' (line 240)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'call_assignment_144772', getitem___call_result_145465)
        
        # Assigning a Name to a Name (line 240):
        # Getting the type of 'call_assignment_144772' (line 240)
        call_assignment_144772_145466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'call_assignment_144772')
        # Assigning a type to the variable 'glyphs' (line 240)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 32), 'glyphs', call_assignment_144772_145466)
        
        # Assigning a Call to a Name (line 240):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_145469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 8), 'int')
        # Processing the call keyword arguments
        kwargs_145470 = {}
        # Getting the type of 'call_assignment_144768' (line 240)
        call_assignment_144768_145467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'call_assignment_144768', False)
        # Obtaining the member '__getitem__' of a type (line 240)
        getitem___145468 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 8), call_assignment_144768_145467, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_145471 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___145468, *[int_145469], **kwargs_145470)
        
        # Assigning a type to the variable 'call_assignment_144773' (line 240)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'call_assignment_144773', getitem___call_result_145471)
        
        # Assigning a Name to a Name (line 240):
        # Getting the type of 'call_assignment_144773' (line 240)
        call_assignment_144773_145472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'call_assignment_144773')
        # Assigning a type to the variable 'rects' (line 240)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 40), 'rects', call_assignment_144773_145472)
        
        
        # Getting the type of 'glyph_map' (line 243)
        glyph_map_145473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 15), 'glyph_map')
        # Applying the 'not' unary operator (line 243)
        result_not__145474 = python_operator(stypy.reporting.localization.Localization(__file__, 243, 11), 'not', glyph_map_145473)
        
        # Testing the type of an if condition (line 243)
        if_condition_145475 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 243, 8), result_not__145474)
        # Assigning a type to the variable 'if_condition_145475' (line 243)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 8), 'if_condition_145475', if_condition_145475)
        # SSA begins for if statement (line 243)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 244):
        
        # Assigning a Call to a Name (line 244):
        
        # Call to OrderedDict(...): (line 244)
        # Processing the call keyword arguments (line 244)
        kwargs_145477 = {}
        # Getting the type of 'OrderedDict' (line 244)
        OrderedDict_145476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 24), 'OrderedDict', False)
        # Calling OrderedDict(args, kwargs) (line 244)
        OrderedDict_call_result_145478 = invoke(stypy.reporting.localization.Localization(__file__, 244, 24), OrderedDict_145476, *[], **kwargs_145477)
        
        # Assigning a type to the variable 'glyph_map' (line 244)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 12), 'glyph_map', OrderedDict_call_result_145478)
        # SSA join for if statement (line 243)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'return_new_glyphs_only' (line 246)
        return_new_glyphs_only_145479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 11), 'return_new_glyphs_only')
        # Testing the type of an if condition (line 246)
        if_condition_145480 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 246, 8), return_new_glyphs_only_145479)
        # Assigning a type to the variable 'if_condition_145480' (line 246)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 8), 'if_condition_145480', if_condition_145480)
        # SSA begins for if statement (line 246)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 247):
        
        # Assigning a Call to a Name (line 247):
        
        # Call to OrderedDict(...): (line 247)
        # Processing the call keyword arguments (line 247)
        kwargs_145482 = {}
        # Getting the type of 'OrderedDict' (line 247)
        OrderedDict_145481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 28), 'OrderedDict', False)
        # Calling OrderedDict(args, kwargs) (line 247)
        OrderedDict_call_result_145483 = invoke(stypy.reporting.localization.Localization(__file__, 247, 28), OrderedDict_145481, *[], **kwargs_145482)
        
        # Assigning a type to the variable 'glyph_map_new' (line 247)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 12), 'glyph_map_new', OrderedDict_call_result_145483)
        # SSA branch for the else part of an if statement (line 246)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Name to a Name (line 249):
        
        # Assigning a Name to a Name (line 249):
        # Getting the type of 'glyph_map' (line 249)
        glyph_map_145484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 28), 'glyph_map')
        # Assigning a type to the variable 'glyph_map_new' (line 249)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 12), 'glyph_map_new', glyph_map_145484)
        # SSA join for if statement (line 246)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a List to a Name (line 251):
        
        # Assigning a List to a Name (line 251):
        
        # Obtaining an instance of the builtin type 'list' (line 251)
        list_145485 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 251)
        
        # Assigning a type to the variable 'xpositions' (line 251)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 8), 'xpositions', list_145485)
        
        # Assigning a List to a Name (line 252):
        
        # Assigning a List to a Name (line 252):
        
        # Obtaining an instance of the builtin type 'list' (line 252)
        list_145486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 252)
        
        # Assigning a type to the variable 'ypositions' (line 252)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 8), 'ypositions', list_145486)
        
        # Assigning a List to a Name (line 253):
        
        # Assigning a List to a Name (line 253):
        
        # Obtaining an instance of the builtin type 'list' (line 253)
        list_145487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 253)
        
        # Assigning a type to the variable 'glyph_ids' (line 253)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 8), 'glyph_ids', list_145487)
        
        # Assigning a List to a Name (line 254):
        
        # Assigning a List to a Name (line 254):
        
        # Obtaining an instance of the builtin type 'list' (line 254)
        list_145488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 254)
        
        # Assigning a type to the variable 'sizes' (line 254)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 8), 'sizes', list_145488)
        
        # Assigning a Tuple to a Tuple (line 256):
        
        # Assigning a Num to a Name (line 256):
        int_145489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 23), 'int')
        # Assigning a type to the variable 'tuple_assignment_144774' (line 256)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 8), 'tuple_assignment_144774', int_145489)
        
        # Assigning a Num to a Name (line 256):
        int_145490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 26), 'int')
        # Assigning a type to the variable 'tuple_assignment_144775' (line 256)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 8), 'tuple_assignment_144775', int_145490)
        
        # Assigning a Name to a Name (line 256):
        # Getting the type of 'tuple_assignment_144774' (line 256)
        tuple_assignment_144774_145491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 8), 'tuple_assignment_144774')
        # Assigning a type to the variable 'currx' (line 256)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 8), 'currx', tuple_assignment_144774_145491)
        
        # Assigning a Name to a Name (line 256):
        # Getting the type of 'tuple_assignment_144775' (line 256)
        tuple_assignment_144775_145492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 8), 'tuple_assignment_144775')
        # Assigning a type to the variable 'curry' (line 256)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 15), 'curry', tuple_assignment_144775_145492)
        
        # Getting the type of 'glyphs' (line 257)
        glyphs_145493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 45), 'glyphs')
        # Testing the type of a for loop iterable (line 257)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 257, 8), glyphs_145493)
        # Getting the type of the for loop variable (line 257)
        for_loop_var_145494 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 257, 8), glyphs_145493)
        # Assigning a type to the variable 'font' (line 257)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 8), 'font', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 257, 8), for_loop_var_145494))
        # Assigning a type to the variable 'fontsize' (line 257)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 8), 'fontsize', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 257, 8), for_loop_var_145494))
        # Assigning a type to the variable 'ccode' (line 257)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 8), 'ccode', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 257, 8), for_loop_var_145494))
        # Assigning a type to the variable 'ox' (line 257)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 8), 'ox', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 257, 8), for_loop_var_145494))
        # Assigning a type to the variable 'oy' (line 257)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 8), 'oy', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 257, 8), for_loop_var_145494))
        # SSA begins for a for statement (line 257)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 258):
        
        # Assigning a Call to a Name (line 258):
        
        # Call to _get_char_id(...): (line 258)
        # Processing the call arguments (line 258)
        # Getting the type of 'font' (line 258)
        font_145497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 40), 'font', False)
        # Getting the type of 'ccode' (line 258)
        ccode_145498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 46), 'ccode', False)
        # Processing the call keyword arguments (line 258)
        kwargs_145499 = {}
        # Getting the type of 'self' (line 258)
        self_145495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 22), 'self', False)
        # Obtaining the member '_get_char_id' of a type (line 258)
        _get_char_id_145496 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 22), self_145495, '_get_char_id')
        # Calling _get_char_id(args, kwargs) (line 258)
        _get_char_id_call_result_145500 = invoke(stypy.reporting.localization.Localization(__file__, 258, 22), _get_char_id_145496, *[font_145497, ccode_145498], **kwargs_145499)
        
        # Assigning a type to the variable 'char_id' (line 258)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 12), 'char_id', _get_char_id_call_result_145500)
        
        
        # Getting the type of 'char_id' (line 259)
        char_id_145501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 15), 'char_id')
        # Getting the type of 'glyph_map' (line 259)
        glyph_map_145502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 30), 'glyph_map')
        # Applying the binary operator 'notin' (line 259)
        result_contains_145503 = python_operator(stypy.reporting.localization.Localization(__file__, 259, 15), 'notin', char_id_145501, glyph_map_145502)
        
        # Testing the type of an if condition (line 259)
        if_condition_145504 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 259, 12), result_contains_145503)
        # Assigning a type to the variable 'if_condition_145504' (line 259)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 12), 'if_condition_145504', if_condition_145504)
        # SSA begins for if statement (line 259)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to clear(...): (line 260)
        # Processing the call keyword arguments (line 260)
        kwargs_145507 = {}
        # Getting the type of 'font' (line 260)
        font_145505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 16), 'font', False)
        # Obtaining the member 'clear' of a type (line 260)
        clear_145506 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 16), font_145505, 'clear')
        # Calling clear(args, kwargs) (line 260)
        clear_call_result_145508 = invoke(stypy.reporting.localization.Localization(__file__, 260, 16), clear_145506, *[], **kwargs_145507)
        
        
        # Call to set_size(...): (line 261)
        # Processing the call arguments (line 261)
        # Getting the type of 'self' (line 261)
        self_145511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 30), 'self', False)
        # Obtaining the member 'FONT_SCALE' of a type (line 261)
        FONT_SCALE_145512 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 30), self_145511, 'FONT_SCALE')
        # Getting the type of 'self' (line 261)
        self_145513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 47), 'self', False)
        # Obtaining the member 'DPI' of a type (line 261)
        DPI_145514 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 47), self_145513, 'DPI')
        # Processing the call keyword arguments (line 261)
        kwargs_145515 = {}
        # Getting the type of 'font' (line 261)
        font_145509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 16), 'font', False)
        # Obtaining the member 'set_size' of a type (line 261)
        set_size_145510 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 16), font_145509, 'set_size')
        # Calling set_size(args, kwargs) (line 261)
        set_size_call_result_145516 = invoke(stypy.reporting.localization.Localization(__file__, 261, 16), set_size_145510, *[FONT_SCALE_145512, DPI_145514], **kwargs_145515)
        
        
        # Assigning a Call to a Name (line 262):
        
        # Assigning a Call to a Name (line 262):
        
        # Call to load_char(...): (line 262)
        # Processing the call arguments (line 262)
        # Getting the type of 'ccode' (line 262)
        ccode_145519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 39), 'ccode', False)
        # Processing the call keyword arguments (line 262)
        # Getting the type of 'LOAD_NO_HINTING' (line 262)
        LOAD_NO_HINTING_145520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 52), 'LOAD_NO_HINTING', False)
        keyword_145521 = LOAD_NO_HINTING_145520
        kwargs_145522 = {'flags': keyword_145521}
        # Getting the type of 'font' (line 262)
        font_145517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 24), 'font', False)
        # Obtaining the member 'load_char' of a type (line 262)
        load_char_145518 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 24), font_145517, 'load_char')
        # Calling load_char(args, kwargs) (line 262)
        load_char_call_result_145523 = invoke(stypy.reporting.localization.Localization(__file__, 262, 24), load_char_145518, *[ccode_145519], **kwargs_145522)
        
        # Assigning a type to the variable 'glyph' (line 262)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 16), 'glyph', load_char_call_result_145523)
        
        # Assigning a Call to a Subscript (line 263):
        
        # Assigning a Call to a Subscript (line 263):
        
        # Call to glyph_to_path(...): (line 263)
        # Processing the call arguments (line 263)
        # Getting the type of 'font' (line 263)
        font_145526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 60), 'font', False)
        # Processing the call keyword arguments (line 263)
        kwargs_145527 = {}
        # Getting the type of 'self' (line 263)
        self_145524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 41), 'self', False)
        # Obtaining the member 'glyph_to_path' of a type (line 263)
        glyph_to_path_145525 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 41), self_145524, 'glyph_to_path')
        # Calling glyph_to_path(args, kwargs) (line 263)
        glyph_to_path_call_result_145528 = invoke(stypy.reporting.localization.Localization(__file__, 263, 41), glyph_to_path_145525, *[font_145526], **kwargs_145527)
        
        # Getting the type of 'glyph_map_new' (line 263)
        glyph_map_new_145529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 16), 'glyph_map_new')
        # Getting the type of 'char_id' (line 263)
        char_id_145530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 30), 'char_id')
        # Storing an element on a container (line 263)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 263, 16), glyph_map_new_145529, (char_id_145530, glyph_to_path_call_result_145528))
        # SSA join for if statement (line 259)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to append(...): (line 265)
        # Processing the call arguments (line 265)
        # Getting the type of 'ox' (line 265)
        ox_145533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 30), 'ox', False)
        # Processing the call keyword arguments (line 265)
        kwargs_145534 = {}
        # Getting the type of 'xpositions' (line 265)
        xpositions_145531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 12), 'xpositions', False)
        # Obtaining the member 'append' of a type (line 265)
        append_145532 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 12), xpositions_145531, 'append')
        # Calling append(args, kwargs) (line 265)
        append_call_result_145535 = invoke(stypy.reporting.localization.Localization(__file__, 265, 12), append_145532, *[ox_145533], **kwargs_145534)
        
        
        # Call to append(...): (line 266)
        # Processing the call arguments (line 266)
        # Getting the type of 'oy' (line 266)
        oy_145538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 30), 'oy', False)
        # Processing the call keyword arguments (line 266)
        kwargs_145539 = {}
        # Getting the type of 'ypositions' (line 266)
        ypositions_145536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 12), 'ypositions', False)
        # Obtaining the member 'append' of a type (line 266)
        append_145537 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 12), ypositions_145536, 'append')
        # Calling append(args, kwargs) (line 266)
        append_call_result_145540 = invoke(stypy.reporting.localization.Localization(__file__, 266, 12), append_145537, *[oy_145538], **kwargs_145539)
        
        
        # Call to append(...): (line 267)
        # Processing the call arguments (line 267)
        # Getting the type of 'char_id' (line 267)
        char_id_145543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 29), 'char_id', False)
        # Processing the call keyword arguments (line 267)
        kwargs_145544 = {}
        # Getting the type of 'glyph_ids' (line 267)
        glyph_ids_145541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 12), 'glyph_ids', False)
        # Obtaining the member 'append' of a type (line 267)
        append_145542 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 267, 12), glyph_ids_145541, 'append')
        # Calling append(args, kwargs) (line 267)
        append_call_result_145545 = invoke(stypy.reporting.localization.Localization(__file__, 267, 12), append_145542, *[char_id_145543], **kwargs_145544)
        
        
        # Assigning a BinOp to a Name (line 268):
        
        # Assigning a BinOp to a Name (line 268):
        # Getting the type of 'fontsize' (line 268)
        fontsize_145546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 19), 'fontsize')
        # Getting the type of 'self' (line 268)
        self_145547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 30), 'self')
        # Obtaining the member 'FONT_SCALE' of a type (line 268)
        FONT_SCALE_145548 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 30), self_145547, 'FONT_SCALE')
        # Applying the binary operator 'div' (line 268)
        result_div_145549 = python_operator(stypy.reporting.localization.Localization(__file__, 268, 19), 'div', fontsize_145546, FONT_SCALE_145548)
        
        # Assigning a type to the variable 'size' (line 268)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 12), 'size', result_div_145549)
        
        # Call to append(...): (line 269)
        # Processing the call arguments (line 269)
        # Getting the type of 'size' (line 269)
        size_145552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 25), 'size', False)
        # Processing the call keyword arguments (line 269)
        kwargs_145553 = {}
        # Getting the type of 'sizes' (line 269)
        sizes_145550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 12), 'sizes', False)
        # Obtaining the member 'append' of a type (line 269)
        append_145551 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 12), sizes_145550, 'append')
        # Calling append(args, kwargs) (line 269)
        append_call_result_145554 = invoke(stypy.reporting.localization.Localization(__file__, 269, 12), append_145551, *[size_145552], **kwargs_145553)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a List to a Name (line 271):
        
        # Assigning a List to a Name (line 271):
        
        # Obtaining an instance of the builtin type 'list' (line 271)
        list_145555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 271)
        
        # Assigning a type to the variable 'myrects' (line 271)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 8), 'myrects', list_145555)
        
        # Getting the type of 'rects' (line 272)
        rects_145556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 28), 'rects')
        # Testing the type of a for loop iterable (line 272)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 272, 8), rects_145556)
        # Getting the type of the for loop variable (line 272)
        for_loop_var_145557 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 272, 8), rects_145556)
        # Assigning a type to the variable 'ox' (line 272)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 8), 'ox', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 272, 8), for_loop_var_145557))
        # Assigning a type to the variable 'oy' (line 272)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 8), 'oy', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 272, 8), for_loop_var_145557))
        # Assigning a type to the variable 'w' (line 272)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 8), 'w', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 272, 8), for_loop_var_145557))
        # Assigning a type to the variable 'h' (line 272)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 8), 'h', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 272, 8), for_loop_var_145557))
        # SSA begins for a for statement (line 272)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a List to a Name (line 273):
        
        # Assigning a List to a Name (line 273):
        
        # Obtaining an instance of the builtin type 'list' (line 273)
        list_145558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 273)
        # Adding element type (line 273)
        
        # Obtaining an instance of the builtin type 'tuple' (line 273)
        tuple_145559 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 22), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 273)
        # Adding element type (line 273)
        # Getting the type of 'ox' (line 273)
        ox_145560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 22), 'ox')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 273, 22), tuple_145559, ox_145560)
        # Adding element type (line 273)
        # Getting the type of 'oy' (line 273)
        oy_145561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 26), 'oy')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 273, 22), tuple_145559, oy_145561)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 273, 20), list_145558, tuple_145559)
        # Adding element type (line 273)
        
        # Obtaining an instance of the builtin type 'tuple' (line 273)
        tuple_145562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 32), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 273)
        # Adding element type (line 273)
        # Getting the type of 'ox' (line 273)
        ox_145563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 32), 'ox')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 273, 32), tuple_145562, ox_145563)
        # Adding element type (line 273)
        # Getting the type of 'oy' (line 273)
        oy_145564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 36), 'oy')
        # Getting the type of 'h' (line 273)
        h_145565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 41), 'h')
        # Applying the binary operator '+' (line 273)
        result_add_145566 = python_operator(stypy.reporting.localization.Localization(__file__, 273, 36), '+', oy_145564, h_145565)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 273, 32), tuple_145562, result_add_145566)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 273, 20), list_145558, tuple_145562)
        # Adding element type (line 273)
        
        # Obtaining an instance of the builtin type 'tuple' (line 273)
        tuple_145567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 46), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 273)
        # Adding element type (line 273)
        # Getting the type of 'ox' (line 273)
        ox_145568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 46), 'ox')
        # Getting the type of 'w' (line 273)
        w_145569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 51), 'w')
        # Applying the binary operator '+' (line 273)
        result_add_145570 = python_operator(stypy.reporting.localization.Localization(__file__, 273, 46), '+', ox_145568, w_145569)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 273, 46), tuple_145567, result_add_145570)
        # Adding element type (line 273)
        # Getting the type of 'oy' (line 273)
        oy_145571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 54), 'oy')
        # Getting the type of 'h' (line 273)
        h_145572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 59), 'h')
        # Applying the binary operator '+' (line 273)
        result_add_145573 = python_operator(stypy.reporting.localization.Localization(__file__, 273, 54), '+', oy_145571, h_145572)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 273, 46), tuple_145567, result_add_145573)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 273, 20), list_145558, tuple_145567)
        # Adding element type (line 273)
        
        # Obtaining an instance of the builtin type 'tuple' (line 274)
        tuple_145574 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 22), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 274)
        # Adding element type (line 274)
        # Getting the type of 'ox' (line 274)
        ox_145575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 22), 'ox')
        # Getting the type of 'w' (line 274)
        w_145576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 27), 'w')
        # Applying the binary operator '+' (line 274)
        result_add_145577 = python_operator(stypy.reporting.localization.Localization(__file__, 274, 22), '+', ox_145575, w_145576)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 274, 22), tuple_145574, result_add_145577)
        # Adding element type (line 274)
        # Getting the type of 'oy' (line 274)
        oy_145578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 30), 'oy')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 274, 22), tuple_145574, oy_145578)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 273, 20), list_145558, tuple_145574)
        # Adding element type (line 273)
        
        # Obtaining an instance of the builtin type 'tuple' (line 274)
        tuple_145579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 274)
        # Adding element type (line 274)
        # Getting the type of 'ox' (line 274)
        ox_145580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 36), 'ox')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 274, 36), tuple_145579, ox_145580)
        # Adding element type (line 274)
        # Getting the type of 'oy' (line 274)
        oy_145581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 40), 'oy')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 274, 36), tuple_145579, oy_145581)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 273, 20), list_145558, tuple_145579)
        # Adding element type (line 273)
        
        # Obtaining an instance of the builtin type 'tuple' (line 274)
        tuple_145582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 46), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 274)
        # Adding element type (line 274)
        int_145583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 46), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 274, 46), tuple_145582, int_145583)
        # Adding element type (line 274)
        int_145584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 49), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 274, 46), tuple_145582, int_145584)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 273, 20), list_145558, tuple_145582)
        
        # Assigning a type to the variable 'vert1' (line 273)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 12), 'vert1', list_145558)
        
        # Assigning a List to a Name (line 275):
        
        # Assigning a List to a Name (line 275):
        
        # Obtaining an instance of the builtin type 'list' (line 275)
        list_145585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 275)
        # Adding element type (line 275)
        # Getting the type of 'Path' (line 275)
        Path_145586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 21), 'Path')
        # Obtaining the member 'MOVETO' of a type (line 275)
        MOVETO_145587 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 21), Path_145586, 'MOVETO')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 20), list_145585, MOVETO_145587)
        # Adding element type (line 275)
        # Getting the type of 'Path' (line 276)
        Path_145588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 21), 'Path')
        # Obtaining the member 'LINETO' of a type (line 276)
        LINETO_145589 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 21), Path_145588, 'LINETO')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 20), list_145585, LINETO_145589)
        # Adding element type (line 275)
        # Getting the type of 'Path' (line 276)
        Path_145590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 34), 'Path')
        # Obtaining the member 'LINETO' of a type (line 276)
        LINETO_145591 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 34), Path_145590, 'LINETO')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 20), list_145585, LINETO_145591)
        # Adding element type (line 275)
        # Getting the type of 'Path' (line 276)
        Path_145592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 47), 'Path')
        # Obtaining the member 'LINETO' of a type (line 276)
        LINETO_145593 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 47), Path_145592, 'LINETO')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 20), list_145585, LINETO_145593)
        # Adding element type (line 275)
        # Getting the type of 'Path' (line 276)
        Path_145594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 60), 'Path')
        # Obtaining the member 'LINETO' of a type (line 276)
        LINETO_145595 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 60), Path_145594, 'LINETO')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 20), list_145585, LINETO_145595)
        # Adding element type (line 275)
        # Getting the type of 'Path' (line 277)
        Path_145596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 21), 'Path')
        # Obtaining the member 'CLOSEPOLY' of a type (line 277)
        CLOSEPOLY_145597 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 21), Path_145596, 'CLOSEPOLY')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 20), list_145585, CLOSEPOLY_145597)
        
        # Assigning a type to the variable 'code1' (line 275)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 12), 'code1', list_145585)
        
        # Call to append(...): (line 278)
        # Processing the call arguments (line 278)
        
        # Obtaining an instance of the builtin type 'tuple' (line 278)
        tuple_145600 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 278)
        # Adding element type (line 278)
        # Getting the type of 'vert1' (line 278)
        vert1_145601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 28), 'vert1', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 278, 28), tuple_145600, vert1_145601)
        # Adding element type (line 278)
        # Getting the type of 'code1' (line 278)
        code1_145602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 35), 'code1', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 278, 28), tuple_145600, code1_145602)
        
        # Processing the call keyword arguments (line 278)
        kwargs_145603 = {}
        # Getting the type of 'myrects' (line 278)
        myrects_145598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 12), 'myrects', False)
        # Obtaining the member 'append' of a type (line 278)
        append_145599 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 12), myrects_145598, 'append')
        # Calling append(args, kwargs) (line 278)
        append_call_result_145604 = invoke(stypy.reporting.localization.Localization(__file__, 278, 12), append_145599, *[tuple_145600], **kwargs_145603)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 280)
        tuple_145605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 16), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 280)
        # Adding element type (line 280)
        
        # Call to list(...): (line 280)
        # Processing the call arguments (line 280)
        
        # Call to zip(...): (line 280)
        # Processing the call arguments (line 280)
        # Getting the type of 'glyph_ids' (line 280)
        glyph_ids_145608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 25), 'glyph_ids', False)
        # Getting the type of 'xpositions' (line 280)
        xpositions_145609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 36), 'xpositions', False)
        # Getting the type of 'ypositions' (line 280)
        ypositions_145610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 48), 'ypositions', False)
        # Getting the type of 'sizes' (line 280)
        sizes_145611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 60), 'sizes', False)
        # Processing the call keyword arguments (line 280)
        kwargs_145612 = {}
        # Getting the type of 'zip' (line 280)
        zip_145607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 21), 'zip', False)
        # Calling zip(args, kwargs) (line 280)
        zip_call_result_145613 = invoke(stypy.reporting.localization.Localization(__file__, 280, 21), zip_145607, *[glyph_ids_145608, xpositions_145609, ypositions_145610, sizes_145611], **kwargs_145612)
        
        # Processing the call keyword arguments (line 280)
        kwargs_145614 = {}
        # Getting the type of 'list' (line 280)
        list_145606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 16), 'list', False)
        # Calling list(args, kwargs) (line 280)
        list_call_result_145615 = invoke(stypy.reporting.localization.Localization(__file__, 280, 16), list_145606, *[zip_call_result_145613], **kwargs_145614)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 280, 16), tuple_145605, list_call_result_145615)
        # Adding element type (line 280)
        # Getting the type of 'glyph_map_new' (line 281)
        glyph_map_new_145616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 16), 'glyph_map_new')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 280, 16), tuple_145605, glyph_map_new_145616)
        # Adding element type (line 280)
        # Getting the type of 'myrects' (line 281)
        myrects_145617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 31), 'myrects')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 280, 16), tuple_145605, myrects_145617)
        
        # Assigning a type to the variable 'stypy_return_type' (line 280)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 8), 'stypy_return_type', tuple_145605)
        
        # ################# End of 'get_glyphs_mathtext(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_glyphs_mathtext' in the type store
        # Getting the type of 'stypy_return_type' (line 230)
        stypy_return_type_145618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_145618)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_glyphs_mathtext'
        return stypy_return_type_145618


    @norecursion
    def get_texmanager(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_texmanager'
        module_type_store = module_type_store.open_function_context('get_texmanager', 283, 4, False)
        # Assigning a type to the variable 'self' (line 284)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TextToPath.get_texmanager.__dict__.__setitem__('stypy_localization', localization)
        TextToPath.get_texmanager.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TextToPath.get_texmanager.__dict__.__setitem__('stypy_type_store', module_type_store)
        TextToPath.get_texmanager.__dict__.__setitem__('stypy_function_name', 'TextToPath.get_texmanager')
        TextToPath.get_texmanager.__dict__.__setitem__('stypy_param_names_list', [])
        TextToPath.get_texmanager.__dict__.__setitem__('stypy_varargs_param_name', None)
        TextToPath.get_texmanager.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TextToPath.get_texmanager.__dict__.__setitem__('stypy_call_defaults', defaults)
        TextToPath.get_texmanager.__dict__.__setitem__('stypy_call_varargs', varargs)
        TextToPath.get_texmanager.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TextToPath.get_texmanager.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TextToPath.get_texmanager', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_texmanager', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_texmanager(...)' code ##################

        unicode_145619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, (-1)), 'unicode', u'\n        return the :class:`matplotlib.texmanager.TexManager` instance\n        ')
        
        # Type idiom detected: calculating its left and rigth part (line 287)
        # Getting the type of 'self' (line 287)
        self_145620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 11), 'self')
        # Obtaining the member '_texmanager' of a type (line 287)
        _texmanager_145621 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 11), self_145620, '_texmanager')
        # Getting the type of 'None' (line 287)
        None_145622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 31), 'None')
        
        (may_be_145623, more_types_in_union_145624) = may_be_none(_texmanager_145621, None_145622)

        if may_be_145623:

            if more_types_in_union_145624:
                # Runtime conditional SSA (line 287)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 288, 12))
            
            # 'from matplotlib.texmanager import TexManager' statement (line 288)
            update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
            import_145625 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 288, 12), 'matplotlib.texmanager')

            if (type(import_145625) is not StypyTypeError):

                if (import_145625 != 'pyd_module'):
                    __import__(import_145625)
                    sys_modules_145626 = sys.modules[import_145625]
                    import_from_module(stypy.reporting.localization.Localization(__file__, 288, 12), 'matplotlib.texmanager', sys_modules_145626.module_type_store, module_type_store, ['TexManager'])
                    nest_module(stypy.reporting.localization.Localization(__file__, 288, 12), __file__, sys_modules_145626, sys_modules_145626.module_type_store, module_type_store)
                else:
                    from matplotlib.texmanager import TexManager

                    import_from_module(stypy.reporting.localization.Localization(__file__, 288, 12), 'matplotlib.texmanager', None, module_type_store, ['TexManager'], [TexManager])

            else:
                # Assigning a type to the variable 'matplotlib.texmanager' (line 288)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 12), 'matplotlib.texmanager', import_145625)

            remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')
            
            
            # Assigning a Call to a Attribute (line 289):
            
            # Assigning a Call to a Attribute (line 289):
            
            # Call to TexManager(...): (line 289)
            # Processing the call keyword arguments (line 289)
            kwargs_145628 = {}
            # Getting the type of 'TexManager' (line 289)
            TexManager_145627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 31), 'TexManager', False)
            # Calling TexManager(args, kwargs) (line 289)
            TexManager_call_result_145629 = invoke(stypy.reporting.localization.Localization(__file__, 289, 31), TexManager_145627, *[], **kwargs_145628)
            
            # Getting the type of 'self' (line 289)
            self_145630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 12), 'self')
            # Setting the type of the member '_texmanager' of a type (line 289)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 12), self_145630, '_texmanager', TexManager_call_result_145629)

            if more_types_in_union_145624:
                # SSA join for if statement (line 287)
                module_type_store = module_type_store.join_ssa_context()


        
        # Getting the type of 'self' (line 290)
        self_145631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 15), 'self')
        # Obtaining the member '_texmanager' of a type (line 290)
        _texmanager_145632 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 15), self_145631, '_texmanager')
        # Assigning a type to the variable 'stypy_return_type' (line 290)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 8), 'stypy_return_type', _texmanager_145632)
        
        # ################# End of 'get_texmanager(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_texmanager' in the type store
        # Getting the type of 'stypy_return_type' (line 283)
        stypy_return_type_145633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_145633)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_texmanager'
        return stypy_return_type_145633


    @norecursion
    def get_glyphs_tex(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 292)
        None_145634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 48), 'None')
        # Getting the type of 'False' (line 293)
        False_145635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 46), 'False')
        defaults = [None_145634, False_145635]
        # Create a new context for function 'get_glyphs_tex'
        module_type_store = module_type_store.open_function_context('get_glyphs_tex', 292, 4, False)
        # Assigning a type to the variable 'self' (line 293)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TextToPath.get_glyphs_tex.__dict__.__setitem__('stypy_localization', localization)
        TextToPath.get_glyphs_tex.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TextToPath.get_glyphs_tex.__dict__.__setitem__('stypy_type_store', module_type_store)
        TextToPath.get_glyphs_tex.__dict__.__setitem__('stypy_function_name', 'TextToPath.get_glyphs_tex')
        TextToPath.get_glyphs_tex.__dict__.__setitem__('stypy_param_names_list', ['prop', 's', 'glyph_map', 'return_new_glyphs_only'])
        TextToPath.get_glyphs_tex.__dict__.__setitem__('stypy_varargs_param_name', None)
        TextToPath.get_glyphs_tex.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TextToPath.get_glyphs_tex.__dict__.__setitem__('stypy_call_defaults', defaults)
        TextToPath.get_glyphs_tex.__dict__.__setitem__('stypy_call_varargs', varargs)
        TextToPath.get_glyphs_tex.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TextToPath.get_glyphs_tex.__dict__.__setitem__('stypy_declared_arg_number', 5)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TextToPath.get_glyphs_tex', ['prop', 's', 'glyph_map', 'return_new_glyphs_only'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_glyphs_tex', localization, ['prop', 's', 'glyph_map', 'return_new_glyphs_only'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_glyphs_tex(...)' code ##################

        unicode_145636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, (-1)), 'unicode', u"\n        convert the string *s* to vertices and codes using matplotlib's usetex\n        mode.\n        ")
        
        # Assigning a Call to a Name (line 301):
        
        # Assigning a Call to a Name (line 301):
        
        # Call to get_texmanager(...): (line 301)
        # Processing the call keyword arguments (line 301)
        kwargs_145639 = {}
        # Getting the type of 'self' (line 301)
        self_145637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 21), 'self', False)
        # Obtaining the member 'get_texmanager' of a type (line 301)
        get_texmanager_145638 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 21), self_145637, 'get_texmanager')
        # Calling get_texmanager(args, kwargs) (line 301)
        get_texmanager_call_result_145640 = invoke(stypy.reporting.localization.Localization(__file__, 301, 21), get_texmanager_145638, *[], **kwargs_145639)
        
        # Assigning a type to the variable 'texmanager' (line 301)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 8), 'texmanager', get_texmanager_call_result_145640)
        
        # Type idiom detected: calculating its left and rigth part (line 303)
        # Getting the type of 'self' (line 303)
        self_145641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 11), 'self')
        # Obtaining the member 'tex_font_map' of a type (line 303)
        tex_font_map_145642 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 11), self_145641, 'tex_font_map')
        # Getting the type of 'None' (line 303)
        None_145643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 32), 'None')
        
        (may_be_145644, more_types_in_union_145645) = may_be_none(tex_font_map_145642, None_145643)

        if may_be_145644:

            if more_types_in_union_145645:
                # Runtime conditional SSA (line 303)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Attribute (line 304):
            
            # Assigning a Call to a Attribute (line 304):
            
            # Call to PsfontsMap(...): (line 304)
            # Processing the call arguments (line 304)
            
            # Call to find_tex_file(...): (line 305)
            # Processing the call arguments (line 305)
            unicode_145650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 58), 'unicode', u'pdftex.map')
            # Processing the call keyword arguments (line 305)
            kwargs_145651 = {}
            # Getting the type of 'dviread' (line 305)
            dviread_145648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 36), 'dviread', False)
            # Obtaining the member 'find_tex_file' of a type (line 305)
            find_tex_file_145649 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 305, 36), dviread_145648, 'find_tex_file')
            # Calling find_tex_file(args, kwargs) (line 305)
            find_tex_file_call_result_145652 = invoke(stypy.reporting.localization.Localization(__file__, 305, 36), find_tex_file_145649, *[unicode_145650], **kwargs_145651)
            
            # Processing the call keyword arguments (line 304)
            kwargs_145653 = {}
            # Getting the type of 'dviread' (line 304)
            dviread_145646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 32), 'dviread', False)
            # Obtaining the member 'PsfontsMap' of a type (line 304)
            PsfontsMap_145647 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 32), dviread_145646, 'PsfontsMap')
            # Calling PsfontsMap(args, kwargs) (line 304)
            PsfontsMap_call_result_145654 = invoke(stypy.reporting.localization.Localization(__file__, 304, 32), PsfontsMap_145647, *[find_tex_file_call_result_145652], **kwargs_145653)
            
            # Getting the type of 'self' (line 304)
            self_145655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 12), 'self')
            # Setting the type of the member 'tex_font_map' of a type (line 304)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 12), self_145655, 'tex_font_map', PsfontsMap_call_result_145654)

            if more_types_in_union_145645:
                # SSA join for if statement (line 303)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 307)
        # Getting the type of 'self' (line 307)
        self_145656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 11), 'self')
        # Obtaining the member '_adobe_standard_encoding' of a type (line 307)
        _adobe_standard_encoding_145657 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 11), self_145656, '_adobe_standard_encoding')
        # Getting the type of 'None' (line 307)
        None_145658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 44), 'None')
        
        (may_be_145659, more_types_in_union_145660) = may_be_none(_adobe_standard_encoding_145657, None_145658)

        if may_be_145659:

            if more_types_in_union_145660:
                # Runtime conditional SSA (line 307)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Attribute (line 308):
            
            # Assigning a Call to a Attribute (line 308):
            
            # Call to _get_adobe_standard_encoding(...): (line 308)
            # Processing the call keyword arguments (line 308)
            kwargs_145663 = {}
            # Getting the type of 'self' (line 308)
            self_145661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 44), 'self', False)
            # Obtaining the member '_get_adobe_standard_encoding' of a type (line 308)
            _get_adobe_standard_encoding_145662 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 44), self_145661, '_get_adobe_standard_encoding')
            # Calling _get_adobe_standard_encoding(args, kwargs) (line 308)
            _get_adobe_standard_encoding_call_result_145664 = invoke(stypy.reporting.localization.Localization(__file__, 308, 44), _get_adobe_standard_encoding_145662, *[], **kwargs_145663)
            
            # Getting the type of 'self' (line 308)
            self_145665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 12), 'self')
            # Setting the type of the member '_adobe_standard_encoding' of a type (line 308)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 12), self_145665, '_adobe_standard_encoding', _get_adobe_standard_encoding_call_result_145664)

            if more_types_in_union_145660:
                # SSA join for if statement (line 307)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Name (line 310):
        
        # Assigning a Call to a Name (line 310):
        
        # Call to get_size_in_points(...): (line 310)
        # Processing the call keyword arguments (line 310)
        kwargs_145668 = {}
        # Getting the type of 'prop' (line 310)
        prop_145666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 19), 'prop', False)
        # Obtaining the member 'get_size_in_points' of a type (line 310)
        get_size_in_points_145667 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 19), prop_145666, 'get_size_in_points')
        # Calling get_size_in_points(args, kwargs) (line 310)
        get_size_in_points_call_result_145669 = invoke(stypy.reporting.localization.Localization(__file__, 310, 19), get_size_in_points_145667, *[], **kwargs_145668)
        
        # Assigning a type to the variable 'fontsize' (line 310)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 8), 'fontsize', get_size_in_points_call_result_145669)
        
        # Type idiom detected: calculating its left and rigth part (line 311)
        unicode_145670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 31), 'unicode', u'get_dvi')
        # Getting the type of 'texmanager' (line 311)
        texmanager_145671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 19), 'texmanager')
        
        (may_be_145672, more_types_in_union_145673) = may_provide_member(unicode_145670, texmanager_145671)

        if may_be_145672:

            if more_types_in_union_145673:
                # Runtime conditional SSA (line 311)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'texmanager' (line 311)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 8), 'texmanager', remove_not_member_provider_from_union(texmanager_145671, u'get_dvi'))
            
            # Assigning a Call to a Name (line 312):
            
            # Assigning a Call to a Name (line 312):
            
            # Call to get_dvi(...): (line 312)
            # Processing the call arguments (line 312)
            # Getting the type of 's' (line 312)
            s_145676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 45), 's', False)
            # Getting the type of 'self' (line 312)
            self_145677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 48), 'self', False)
            # Obtaining the member 'FONT_SCALE' of a type (line 312)
            FONT_SCALE_145678 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 48), self_145677, 'FONT_SCALE')
            # Processing the call keyword arguments (line 312)
            kwargs_145679 = {}
            # Getting the type of 'texmanager' (line 312)
            texmanager_145674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 26), 'texmanager', False)
            # Obtaining the member 'get_dvi' of a type (line 312)
            get_dvi_145675 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 26), texmanager_145674, 'get_dvi')
            # Calling get_dvi(args, kwargs) (line 312)
            get_dvi_call_result_145680 = invoke(stypy.reporting.localization.Localization(__file__, 312, 26), get_dvi_145675, *[s_145676, FONT_SCALE_145678], **kwargs_145679)
            
            # Assigning a type to the variable 'dvifilelike' (line 312)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 12), 'dvifilelike', get_dvi_call_result_145680)
            
            # Assigning a Call to a Name (line 313):
            
            # Assigning a Call to a Name (line 313):
            
            # Call to DviFromFileLike(...): (line 313)
            # Processing the call arguments (line 313)
            # Getting the type of 'dvifilelike' (line 313)
            dvifilelike_145683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 42), 'dvifilelike', False)
            # Getting the type of 'self' (line 313)
            self_145684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 55), 'self', False)
            # Obtaining the member 'DPI' of a type (line 313)
            DPI_145685 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 55), self_145684, 'DPI')
            # Processing the call keyword arguments (line 313)
            kwargs_145686 = {}
            # Getting the type of 'dviread' (line 313)
            dviread_145681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 18), 'dviread', False)
            # Obtaining the member 'DviFromFileLike' of a type (line 313)
            DviFromFileLike_145682 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 18), dviread_145681, 'DviFromFileLike')
            # Calling DviFromFileLike(args, kwargs) (line 313)
            DviFromFileLike_call_result_145687 = invoke(stypy.reporting.localization.Localization(__file__, 313, 18), DviFromFileLike_145682, *[dvifilelike_145683, DPI_145685], **kwargs_145686)
            
            # Assigning a type to the variable 'dvi' (line 313)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 313, 12), 'dvi', DviFromFileLike_call_result_145687)

            if more_types_in_union_145673:
                # Runtime conditional SSA for else branch (line 311)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_145672) or more_types_in_union_145673):
            # Assigning a type to the variable 'texmanager' (line 311)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 8), 'texmanager', remove_member_provider_from_union(texmanager_145671, u'get_dvi'))
            
            # Assigning a Call to a Name (line 315):
            
            # Assigning a Call to a Name (line 315):
            
            # Call to make_dvi(...): (line 315)
            # Processing the call arguments (line 315)
            # Getting the type of 's' (line 315)
            s_145690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 42), 's', False)
            # Getting the type of 'self' (line 315)
            self_145691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 45), 'self', False)
            # Obtaining the member 'FONT_SCALE' of a type (line 315)
            FONT_SCALE_145692 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 315, 45), self_145691, 'FONT_SCALE')
            # Processing the call keyword arguments (line 315)
            kwargs_145693 = {}
            # Getting the type of 'texmanager' (line 315)
            texmanager_145688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 22), 'texmanager', False)
            # Obtaining the member 'make_dvi' of a type (line 315)
            make_dvi_145689 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 315, 22), texmanager_145688, 'make_dvi')
            # Calling make_dvi(args, kwargs) (line 315)
            make_dvi_call_result_145694 = invoke(stypy.reporting.localization.Localization(__file__, 315, 22), make_dvi_145689, *[s_145690, FONT_SCALE_145692], **kwargs_145693)
            
            # Assigning a type to the variable 'dvifile' (line 315)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 12), 'dvifile', make_dvi_call_result_145694)
            
            # Assigning a Call to a Name (line 316):
            
            # Assigning a Call to a Name (line 316):
            
            # Call to Dvi(...): (line 316)
            # Processing the call arguments (line 316)
            # Getting the type of 'dvifile' (line 316)
            dvifile_145697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 30), 'dvifile', False)
            # Getting the type of 'self' (line 316)
            self_145698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 39), 'self', False)
            # Obtaining the member 'DPI' of a type (line 316)
            DPI_145699 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 39), self_145698, 'DPI')
            # Processing the call keyword arguments (line 316)
            kwargs_145700 = {}
            # Getting the type of 'dviread' (line 316)
            dviread_145695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 18), 'dviread', False)
            # Obtaining the member 'Dvi' of a type (line 316)
            Dvi_145696 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 18), dviread_145695, 'Dvi')
            # Calling Dvi(args, kwargs) (line 316)
            Dvi_call_result_145701 = invoke(stypy.reporting.localization.Localization(__file__, 316, 18), Dvi_145696, *[dvifile_145697, DPI_145699], **kwargs_145700)
            
            # Assigning a type to the variable 'dvi' (line 316)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 316, 12), 'dvi', Dvi_call_result_145701)

            if (may_be_145672 and more_types_in_union_145673):
                # SSA join for if statement (line 311)
                module_type_store = module_type_store.join_ssa_context()


        
        # Getting the type of 'dvi' (line 317)
        dvi_145702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 13), 'dvi')
        with_145703 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 317, 13), dvi_145702, 'with parameter', '__enter__', '__exit__')

        if with_145703:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 317)
            enter___145704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 13), dvi_145702, '__enter__')
            with_enter_145705 = invoke(stypy.reporting.localization.Localization(__file__, 317, 13), enter___145704)
            
            # Assigning a Call to a Name (line 318):
            
            # Assigning a Call to a Name (line 318):
            
            # Call to next(...): (line 318)
            # Processing the call arguments (line 318)
            
            # Call to iter(...): (line 318)
            # Processing the call arguments (line 318)
            # Getting the type of 'dvi' (line 318)
            dvi_145708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 29), 'dvi', False)
            # Processing the call keyword arguments (line 318)
            kwargs_145709 = {}
            # Getting the type of 'iter' (line 318)
            iter_145707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 24), 'iter', False)
            # Calling iter(args, kwargs) (line 318)
            iter_call_result_145710 = invoke(stypy.reporting.localization.Localization(__file__, 318, 24), iter_145707, *[dvi_145708], **kwargs_145709)
            
            # Processing the call keyword arguments (line 318)
            kwargs_145711 = {}
            # Getting the type of 'next' (line 318)
            next_145706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 19), 'next', False)
            # Calling next(args, kwargs) (line 318)
            next_call_result_145712 = invoke(stypy.reporting.localization.Localization(__file__, 318, 19), next_145706, *[iter_call_result_145710], **kwargs_145711)
            
            # Assigning a type to the variable 'page' (line 318)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 12), 'page', next_call_result_145712)
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 317)
            exit___145713 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 13), dvi_145702, '__exit__')
            with_exit_145714 = invoke(stypy.reporting.localization.Localization(__file__, 317, 13), exit___145713, None, None, None)

        
        # Type idiom detected: calculating its left and rigth part (line 320)
        # Getting the type of 'glyph_map' (line 320)
        glyph_map_145715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 11), 'glyph_map')
        # Getting the type of 'None' (line 320)
        None_145716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 24), 'None')
        
        (may_be_145717, more_types_in_union_145718) = may_be_none(glyph_map_145715, None_145716)

        if may_be_145717:

            if more_types_in_union_145718:
                # Runtime conditional SSA (line 320)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 321):
            
            # Assigning a Call to a Name (line 321):
            
            # Call to OrderedDict(...): (line 321)
            # Processing the call keyword arguments (line 321)
            kwargs_145720 = {}
            # Getting the type of 'OrderedDict' (line 321)
            OrderedDict_145719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 24), 'OrderedDict', False)
            # Calling OrderedDict(args, kwargs) (line 321)
            OrderedDict_call_result_145721 = invoke(stypy.reporting.localization.Localization(__file__, 321, 24), OrderedDict_145719, *[], **kwargs_145720)
            
            # Assigning a type to the variable 'glyph_map' (line 321)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 321, 12), 'glyph_map', OrderedDict_call_result_145721)

            if more_types_in_union_145718:
                # SSA join for if statement (line 320)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Getting the type of 'return_new_glyphs_only' (line 323)
        return_new_glyphs_only_145722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 11), 'return_new_glyphs_only')
        # Testing the type of an if condition (line 323)
        if_condition_145723 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 323, 8), return_new_glyphs_only_145722)
        # Assigning a type to the variable 'if_condition_145723' (line 323)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 8), 'if_condition_145723', if_condition_145723)
        # SSA begins for if statement (line 323)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 324):
        
        # Assigning a Call to a Name (line 324):
        
        # Call to OrderedDict(...): (line 324)
        # Processing the call keyword arguments (line 324)
        kwargs_145725 = {}
        # Getting the type of 'OrderedDict' (line 324)
        OrderedDict_145724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 28), 'OrderedDict', False)
        # Calling OrderedDict(args, kwargs) (line 324)
        OrderedDict_call_result_145726 = invoke(stypy.reporting.localization.Localization(__file__, 324, 28), OrderedDict_145724, *[], **kwargs_145725)
        
        # Assigning a type to the variable 'glyph_map_new' (line 324)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 324, 12), 'glyph_map_new', OrderedDict_call_result_145726)
        # SSA branch for the else part of an if statement (line 323)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Name to a Name (line 326):
        
        # Assigning a Name to a Name (line 326):
        # Getting the type of 'glyph_map' (line 326)
        glyph_map_145727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 28), 'glyph_map')
        # Assigning a type to the variable 'glyph_map_new' (line 326)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 12), 'glyph_map_new', glyph_map_145727)
        # SSA join for if statement (line 323)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Tuple to a Tuple (line 328):
        
        # Assigning a List to a Name (line 328):
        
        # Obtaining an instance of the builtin type 'list' (line 328)
        list_145728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 51), 'list')
        # Adding type elements to the builtin type 'list' instance (line 328)
        
        # Assigning a type to the variable 'tuple_assignment_144776' (line 328)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 328, 8), 'tuple_assignment_144776', list_145728)
        
        # Assigning a List to a Name (line 328):
        
        # Obtaining an instance of the builtin type 'list' (line 328)
        list_145729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 55), 'list')
        # Adding type elements to the builtin type 'list' instance (line 328)
        
        # Assigning a type to the variable 'tuple_assignment_144777' (line 328)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 328, 8), 'tuple_assignment_144777', list_145729)
        
        # Assigning a List to a Name (line 328):
        
        # Obtaining an instance of the builtin type 'list' (line 328)
        list_145730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 59), 'list')
        # Adding type elements to the builtin type 'list' instance (line 328)
        
        # Assigning a type to the variable 'tuple_assignment_144778' (line 328)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 328, 8), 'tuple_assignment_144778', list_145730)
        
        # Assigning a List to a Name (line 328):
        
        # Obtaining an instance of the builtin type 'list' (line 328)
        list_145731 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 63), 'list')
        # Adding type elements to the builtin type 'list' instance (line 328)
        
        # Assigning a type to the variable 'tuple_assignment_144779' (line 328)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 328, 8), 'tuple_assignment_144779', list_145731)
        
        # Assigning a Name to a Name (line 328):
        # Getting the type of 'tuple_assignment_144776' (line 328)
        tuple_assignment_144776_145732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 8), 'tuple_assignment_144776')
        # Assigning a type to the variable 'glyph_ids' (line 328)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 328, 8), 'glyph_ids', tuple_assignment_144776_145732)
        
        # Assigning a Name to a Name (line 328):
        # Getting the type of 'tuple_assignment_144777' (line 328)
        tuple_assignment_144777_145733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 8), 'tuple_assignment_144777')
        # Assigning a type to the variable 'xpositions' (line 328)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 328, 19), 'xpositions', tuple_assignment_144777_145733)
        
        # Assigning a Name to a Name (line 328):
        # Getting the type of 'tuple_assignment_144778' (line 328)
        tuple_assignment_144778_145734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 8), 'tuple_assignment_144778')
        # Assigning a type to the variable 'ypositions' (line 328)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 328, 31), 'ypositions', tuple_assignment_144778_145734)
        
        # Assigning a Name to a Name (line 328):
        # Getting the type of 'tuple_assignment_144779' (line 328)
        tuple_assignment_144779_145735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 8), 'tuple_assignment_144779')
        # Assigning a type to the variable 'sizes' (line 328)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 328, 43), 'sizes', tuple_assignment_144779_145735)
        
        # Getting the type of 'page' (line 333)
        page_145736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 45), 'page')
        # Obtaining the member 'text' of a type (line 333)
        text_145737 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 45), page_145736, 'text')
        # Testing the type of a for loop iterable (line 333)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 333, 8), text_145737)
        # Getting the type of the for loop variable (line 333)
        for_loop_var_145738 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 333, 8), text_145737)
        # Assigning a type to the variable 'x1' (line 333)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 333, 8), 'x1', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 333, 8), for_loop_var_145738))
        # Assigning a type to the variable 'y1' (line 333)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 333, 8), 'y1', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 333, 8), for_loop_var_145738))
        # Assigning a type to the variable 'dvifont' (line 333)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 333, 8), 'dvifont', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 333, 8), for_loop_var_145738))
        # Assigning a type to the variable 'glyph' (line 333)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 333, 8), 'glyph', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 333, 8), for_loop_var_145738))
        # Assigning a type to the variable 'width' (line 333)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 333, 8), 'width', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 333, 8), for_loop_var_145738))
        # SSA begins for a for statement (line 333)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 334):
        
        # Assigning a Call to a Name (line 334):
        
        # Call to get(...): (line 334)
        # Processing the call arguments (line 334)
        # Getting the type of 'dvifont' (line 334)
        dvifont_145742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 51), 'dvifont', False)
        # Obtaining the member 'texname' of a type (line 334)
        texname_145743 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 51), dvifont_145742, 'texname')
        # Processing the call keyword arguments (line 334)
        kwargs_145744 = {}
        # Getting the type of 'self' (line 334)
        self_145739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 32), 'self', False)
        # Obtaining the member '_ps_fontd' of a type (line 334)
        _ps_fontd_145740 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 32), self_145739, '_ps_fontd')
        # Obtaining the member 'get' of a type (line 334)
        get_145741 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 32), _ps_fontd_145740, 'get')
        # Calling get(args, kwargs) (line 334)
        get_call_result_145745 = invoke(stypy.reporting.localization.Localization(__file__, 334, 32), get_145741, *[texname_145743], **kwargs_145744)
        
        # Assigning a type to the variable 'font_and_encoding' (line 334)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 12), 'font_and_encoding', get_call_result_145745)
        
        # Assigning a Subscript to a Name (line 335):
        
        # Assigning a Subscript to a Name (line 335):
        
        # Obtaining the type of the subscript
        # Getting the type of 'dvifont' (line 335)
        dvifont_145746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 43), 'dvifont')
        # Obtaining the member 'texname' of a type (line 335)
        texname_145747 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 43), dvifont_145746, 'texname')
        # Getting the type of 'self' (line 335)
        self_145748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 25), 'self')
        # Obtaining the member 'tex_font_map' of a type (line 335)
        tex_font_map_145749 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 25), self_145748, 'tex_font_map')
        # Obtaining the member '__getitem__' of a type (line 335)
        getitem___145750 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 25), tex_font_map_145749, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 335)
        subscript_call_result_145751 = invoke(stypy.reporting.localization.Localization(__file__, 335, 25), getitem___145750, texname_145747)
        
        # Assigning a type to the variable 'font_bunch' (line 335)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 12), 'font_bunch', subscript_call_result_145751)
        
        # Type idiom detected: calculating its left and rigth part (line 337)
        # Getting the type of 'font_and_encoding' (line 337)
        font_and_encoding_145752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 15), 'font_and_encoding')
        # Getting the type of 'None' (line 337)
        None_145753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 36), 'None')
        
        (may_be_145754, more_types_in_union_145755) = may_be_none(font_and_encoding_145752, None_145753)

        if may_be_145754:

            if more_types_in_union_145755:
                # Runtime conditional SSA (line 337)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Type idiom detected: calculating its left and rigth part (line 338)
            # Getting the type of 'font_bunch' (line 338)
            font_bunch_145756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 19), 'font_bunch')
            # Obtaining the member 'filename' of a type (line 338)
            filename_145757 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 19), font_bunch_145756, 'filename')
            # Getting the type of 'None' (line 338)
            None_145758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 42), 'None')
            
            (may_be_145759, more_types_in_union_145760) = may_be_none(filename_145757, None_145758)

            if may_be_145759:

                if more_types_in_union_145760:
                    # Runtime conditional SSA (line 338)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                
                # Call to ValueError(...): (line 339)
                # Processing the call arguments (line 339)
                unicode_145762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 25), 'unicode', u'No usable font file found for %s (%s). The font may lack a Type-1 version.')
                
                # Obtaining an instance of the builtin type 'tuple' (line 342)
                tuple_145763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 27), 'tuple')
                # Adding type elements to the builtin type 'tuple' instance (line 342)
                # Adding element type (line 342)
                # Getting the type of 'font_bunch' (line 342)
                font_bunch_145764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 27), 'font_bunch', False)
                # Obtaining the member 'psname' of a type (line 342)
                psname_145765 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 27), font_bunch_145764, 'psname')
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 342, 27), tuple_145763, psname_145765)
                # Adding element type (line 342)
                # Getting the type of 'dvifont' (line 342)
                dvifont_145766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 46), 'dvifont', False)
                # Obtaining the member 'texname' of a type (line 342)
                texname_145767 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 46), dvifont_145766, 'texname')
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 342, 27), tuple_145763, texname_145767)
                
                # Applying the binary operator '%' (line 340)
                result_mod_145768 = python_operator(stypy.reporting.localization.Localization(__file__, 340, 24), '%', unicode_145762, tuple_145763)
                
                # Processing the call keyword arguments (line 339)
                kwargs_145769 = {}
                # Getting the type of 'ValueError' (line 339)
                ValueError_145761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 26), 'ValueError', False)
                # Calling ValueError(args, kwargs) (line 339)
                ValueError_call_result_145770 = invoke(stypy.reporting.localization.Localization(__file__, 339, 26), ValueError_145761, *[result_mod_145768], **kwargs_145769)
                
                ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 339, 20), ValueError_call_result_145770, 'raise parameter', BaseException)

                if more_types_in_union_145760:
                    # SSA join for if statement (line 338)
                    module_type_store = module_type_store.join_ssa_context()


            
            
            # Assigning a Call to a Name (line 344):
            
            # Assigning a Call to a Name (line 344):
            
            # Call to get_font(...): (line 344)
            # Processing the call arguments (line 344)
            # Getting the type of 'font_bunch' (line 344)
            font_bunch_145772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 32), 'font_bunch', False)
            # Obtaining the member 'filename' of a type (line 344)
            filename_145773 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 344, 32), font_bunch_145772, 'filename')
            # Processing the call keyword arguments (line 344)
            kwargs_145774 = {}
            # Getting the type of 'get_font' (line 344)
            get_font_145771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 23), 'get_font', False)
            # Calling get_font(args, kwargs) (line 344)
            get_font_call_result_145775 = invoke(stypy.reporting.localization.Localization(__file__, 344, 23), get_font_145771, *[filename_145773], **kwargs_145774)
            
            # Assigning a type to the variable 'font' (line 344)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 344, 16), 'font', get_font_call_result_145775)
            
            
            # Obtaining an instance of the builtin type 'list' (line 346)
            list_145776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 346, 50), 'list')
            # Adding type elements to the builtin type 'list' instance (line 346)
            # Adding element type (line 346)
            
            # Obtaining an instance of the builtin type 'tuple' (line 346)
            tuple_145777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 346, 52), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 346)
            # Adding element type (line 346)
            unicode_145778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 346, 52), 'unicode', u'ADOBE_CUSTOM')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 346, 52), tuple_145777, unicode_145778)
            # Adding element type (line 346)
            int_145779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 347, 52), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 346, 52), tuple_145777, int_145779)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 346, 50), list_145776, tuple_145777)
            # Adding element type (line 346)
            
            # Obtaining an instance of the builtin type 'tuple' (line 348)
            tuple_145780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 52), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 348)
            # Adding element type (line 348)
            unicode_145781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 52), 'unicode', u'ADOBE_STANDARD')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 348, 52), tuple_145780, unicode_145781)
            # Adding element type (line 348)
            int_145782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 52), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 348, 52), tuple_145780, int_145782)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 346, 50), list_145776, tuple_145780)
            
            # Testing the type of a for loop iterable (line 346)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 346, 16), list_145776)
            # Getting the type of the for loop variable (line 346)
            for_loop_var_145783 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 346, 16), list_145776)
            # Assigning a type to the variable 'charmap_name' (line 346)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 346, 16), 'charmap_name', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 346, 16), for_loop_var_145783))
            # Assigning a type to the variable 'charmap_code' (line 346)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 346, 16), 'charmap_code', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 346, 16), for_loop_var_145783))
            # SSA begins for a for statement (line 346)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            
            # SSA begins for try-except statement (line 350)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
            
            # Call to select_charmap(...): (line 351)
            # Processing the call arguments (line 351)
            # Getting the type of 'charmap_code' (line 351)
            charmap_code_145786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 44), 'charmap_code', False)
            # Processing the call keyword arguments (line 351)
            kwargs_145787 = {}
            # Getting the type of 'font' (line 351)
            font_145784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 24), 'font', False)
            # Obtaining the member 'select_charmap' of a type (line 351)
            select_charmap_145785 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 351, 24), font_145784, 'select_charmap')
            # Calling select_charmap(args, kwargs) (line 351)
            select_charmap_call_result_145788 = invoke(stypy.reporting.localization.Localization(__file__, 351, 24), select_charmap_145785, *[charmap_code_145786], **kwargs_145787)
            
            # SSA branch for the except part of a try statement (line 350)
            # SSA branch for the except 'Tuple' branch of a try statement (line 350)
            module_type_store.open_ssa_branch('except')
            pass
            # SSA branch for the else branch of a try statement (line 350)
            module_type_store.open_ssa_branch('except else')
            # SSA join for try-except statement (line 350)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA branch for the else part of a for statement (line 346)
            module_type_store.open_ssa_branch('for loop else')
            
            # Assigning a Str to a Name (line 357):
            
            # Assigning a Str to a Name (line 357):
            unicode_145789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 357, 35), 'unicode', u'')
            # Assigning a type to the variable 'charmap_name' (line 357)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 357, 20), 'charmap_name', unicode_145789)
            
            # Call to warn(...): (line 358)
            # Processing the call arguments (line 358)
            unicode_145792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 358, 34), 'unicode', u'No supported encoding in font (%s).')
            # Getting the type of 'font_bunch' (line 359)
            font_bunch_145793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 34), 'font_bunch', False)
            # Obtaining the member 'filename' of a type (line 359)
            filename_145794 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 34), font_bunch_145793, 'filename')
            # Applying the binary operator '%' (line 358)
            result_mod_145795 = python_operator(stypy.reporting.localization.Localization(__file__, 358, 34), '%', unicode_145792, filename_145794)
            
            # Processing the call keyword arguments (line 358)
            kwargs_145796 = {}
            # Getting the type of 'warnings' (line 358)
            warnings_145790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 20), 'warnings', False)
            # Obtaining the member 'warn' of a type (line 358)
            warn_145791 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 20), warnings_145790, 'warn')
            # Calling warn(args, kwargs) (line 358)
            warn_call_result_145797 = invoke(stypy.reporting.localization.Localization(__file__, 358, 20), warn_145791, *[result_mod_145795], **kwargs_145796)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()
            
            
            
            # Evaluating a boolean operation
            
            # Getting the type of 'charmap_name' (line 361)
            charmap_name_145798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 19), 'charmap_name')
            unicode_145799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 35), 'unicode', u'ADOBE_STANDARD')
            # Applying the binary operator '==' (line 361)
            result_eq_145800 = python_operator(stypy.reporting.localization.Localization(__file__, 361, 19), '==', charmap_name_145798, unicode_145799)
            
            # Getting the type of 'font_bunch' (line 361)
            font_bunch_145801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 56), 'font_bunch')
            # Obtaining the member 'encoding' of a type (line 361)
            encoding_145802 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 361, 56), font_bunch_145801, 'encoding')
            # Applying the binary operator 'and' (line 361)
            result_and_keyword_145803 = python_operator(stypy.reporting.localization.Localization(__file__, 361, 19), 'and', result_eq_145800, encoding_145802)
            
            # Testing the type of an if condition (line 361)
            if_condition_145804 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 361, 16), result_and_keyword_145803)
            # Assigning a type to the variable 'if_condition_145804' (line 361)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 361, 16), 'if_condition_145804', if_condition_145804)
            # SSA begins for if statement (line 361)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 362):
            
            # Assigning a Call to a Name (line 362):
            
            # Call to Encoding(...): (line 362)
            # Processing the call arguments (line 362)
            # Getting the type of 'font_bunch' (line 362)
            font_bunch_145807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 44), 'font_bunch', False)
            # Obtaining the member 'encoding' of a type (line 362)
            encoding_145808 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 44), font_bunch_145807, 'encoding')
            # Processing the call keyword arguments (line 362)
            kwargs_145809 = {}
            # Getting the type of 'dviread' (line 362)
            dviread_145805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 27), 'dviread', False)
            # Obtaining the member 'Encoding' of a type (line 362)
            Encoding_145806 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 27), dviread_145805, 'Encoding')
            # Calling Encoding(args, kwargs) (line 362)
            Encoding_call_result_145810 = invoke(stypy.reporting.localization.Localization(__file__, 362, 27), Encoding_145806, *[encoding_145808], **kwargs_145809)
            
            # Assigning a type to the variable 'enc0' (line 362)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 20), 'enc0', Encoding_call_result_145810)
            
            # Assigning a DictComp to a Name (line 363):
            
            # Assigning a DictComp to a Name (line 363):
            # Calculating dict comprehension
            module_type_store = module_type_store.open_function_context('dict comprehension expression', 363, 27, True)
            # Calculating comprehension expression
            
            # Call to enumerate(...): (line 364)
            # Processing the call arguments (line 364)
            # Getting the type of 'enc0' (line 364)
            enc0_145820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 49), 'enc0', False)
            # Obtaining the member 'encoding' of a type (line 364)
            encoding_145821 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 364, 49), enc0_145820, 'encoding')
            # Processing the call keyword arguments (line 364)
            kwargs_145822 = {}
            # Getting the type of 'enumerate' (line 364)
            enumerate_145819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 39), 'enumerate', False)
            # Calling enumerate(args, kwargs) (line 364)
            enumerate_call_result_145823 = invoke(stypy.reporting.localization.Localization(__file__, 364, 39), enumerate_145819, *[encoding_145821], **kwargs_145822)
            
            comprehension_145824 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 363, 27), enumerate_call_result_145823)
            # Assigning a type to the variable 'i' (line 363)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 27), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 363, 27), comprehension_145824))
            # Assigning a type to the variable 'c' (line 363)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 27), 'c', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 363, 27), comprehension_145824))
            # Getting the type of 'i' (line 363)
            i_145811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 27), 'i')
            
            # Call to get(...): (line 363)
            # Processing the call arguments (line 363)
            # Getting the type of 'c' (line 363)
            c_145815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 64), 'c', False)
            # Getting the type of 'None' (line 363)
            None_145816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 67), 'None', False)
            # Processing the call keyword arguments (line 363)
            kwargs_145817 = {}
            # Getting the type of 'self' (line 363)
            self_145812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 30), 'self', False)
            # Obtaining the member '_adobe_standard_encoding' of a type (line 363)
            _adobe_standard_encoding_145813 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 363, 30), self_145812, '_adobe_standard_encoding')
            # Obtaining the member 'get' of a type (line 363)
            get_145814 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 363, 30), _adobe_standard_encoding_145813, 'get')
            # Calling get(args, kwargs) (line 363)
            get_call_result_145818 = invoke(stypy.reporting.localization.Localization(__file__, 363, 30), get_145814, *[c_145815, None_145816], **kwargs_145817)
            
            dict_145825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 363, 27), 'dict')
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 363, 27), dict_145825, (i_145811, get_call_result_145818))
            # Assigning a type to the variable 'enc' (line 363)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 20), 'enc', dict_145825)
            # SSA branch for the else part of an if statement (line 361)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Dict to a Name (line 366):
            
            # Assigning a Dict to a Name (line 366):
            
            # Obtaining an instance of the builtin type 'dict' (line 366)
            dict_145826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 26), 'dict')
            # Adding type elements to the builtin type 'dict' instance (line 366)
            
            # Assigning a type to the variable 'enc' (line 366)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 366, 20), 'enc', dict_145826)
            # SSA join for if statement (line 361)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Assigning a Tuple to a Subscript (line 367):
            
            # Assigning a Tuple to a Subscript (line 367):
            
            # Obtaining an instance of the builtin type 'tuple' (line 367)
            tuple_145827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 367, 50), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 367)
            # Adding element type (line 367)
            # Getting the type of 'font' (line 367)
            font_145828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 50), 'font')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 367, 50), tuple_145827, font_145828)
            # Adding element type (line 367)
            # Getting the type of 'enc' (line 367)
            enc_145829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 56), 'enc')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 367, 50), tuple_145827, enc_145829)
            
            # Getting the type of 'self' (line 367)
            self_145830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 16), 'self')
            # Obtaining the member '_ps_fontd' of a type (line 367)
            _ps_fontd_145831 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 367, 16), self_145830, '_ps_fontd')
            # Getting the type of 'dvifont' (line 367)
            dvifont_145832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 31), 'dvifont')
            # Obtaining the member 'texname' of a type (line 367)
            texname_145833 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 367, 31), dvifont_145832, 'texname')
            # Storing an element on a container (line 367)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 367, 16), _ps_fontd_145831, (texname_145833, tuple_145827))

            if more_types_in_union_145755:
                # Runtime conditional SSA for else branch (line 337)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_145754) or more_types_in_union_145755):
            
            # Assigning a Name to a Tuple (line 370):
            
            # Assigning a Subscript to a Name (line 370):
            
            # Obtaining the type of the subscript
            int_145834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 16), 'int')
            # Getting the type of 'font_and_encoding' (line 370)
            font_and_encoding_145835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 28), 'font_and_encoding')
            # Obtaining the member '__getitem__' of a type (line 370)
            getitem___145836 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 16), font_and_encoding_145835, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 370)
            subscript_call_result_145837 = invoke(stypy.reporting.localization.Localization(__file__, 370, 16), getitem___145836, int_145834)
            
            # Assigning a type to the variable 'tuple_var_assignment_144780' (line 370)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 16), 'tuple_var_assignment_144780', subscript_call_result_145837)
            
            # Assigning a Subscript to a Name (line 370):
            
            # Obtaining the type of the subscript
            int_145838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 16), 'int')
            # Getting the type of 'font_and_encoding' (line 370)
            font_and_encoding_145839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 28), 'font_and_encoding')
            # Obtaining the member '__getitem__' of a type (line 370)
            getitem___145840 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 16), font_and_encoding_145839, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 370)
            subscript_call_result_145841 = invoke(stypy.reporting.localization.Localization(__file__, 370, 16), getitem___145840, int_145838)
            
            # Assigning a type to the variable 'tuple_var_assignment_144781' (line 370)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 16), 'tuple_var_assignment_144781', subscript_call_result_145841)
            
            # Assigning a Name to a Name (line 370):
            # Getting the type of 'tuple_var_assignment_144780' (line 370)
            tuple_var_assignment_144780_145842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 16), 'tuple_var_assignment_144780')
            # Assigning a type to the variable 'font' (line 370)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 16), 'font', tuple_var_assignment_144780_145842)
            
            # Assigning a Name to a Name (line 370):
            # Getting the type of 'tuple_var_assignment_144781' (line 370)
            tuple_var_assignment_144781_145843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 16), 'tuple_var_assignment_144781')
            # Assigning a type to the variable 'enc' (line 370)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 22), 'enc', tuple_var_assignment_144781_145843)

            if (may_be_145754 and more_types_in_union_145755):
                # SSA join for if statement (line 337)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Name to a Name (line 372):
        
        # Assigning a Name to a Name (line 372):
        # Getting the type of 'LOAD_TARGET_LIGHT' (line 372)
        LOAD_TARGET_LIGHT_145844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 27), 'LOAD_TARGET_LIGHT')
        # Assigning a type to the variable 'ft2font_flag' (line 372)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 372, 12), 'ft2font_flag', LOAD_TARGET_LIGHT_145844)
        
        # Assigning a Call to a Name (line 374):
        
        # Assigning a Call to a Name (line 374):
        
        # Call to _get_char_id_ps(...): (line 374)
        # Processing the call arguments (line 374)
        # Getting the type of 'font' (line 374)
        font_145847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 43), 'font', False)
        # Getting the type of 'glyph' (line 374)
        glyph_145848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 49), 'glyph', False)
        # Processing the call keyword arguments (line 374)
        kwargs_145849 = {}
        # Getting the type of 'self' (line 374)
        self_145845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 22), 'self', False)
        # Obtaining the member '_get_char_id_ps' of a type (line 374)
        _get_char_id_ps_145846 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 374, 22), self_145845, '_get_char_id_ps')
        # Calling _get_char_id_ps(args, kwargs) (line 374)
        _get_char_id_ps_call_result_145850 = invoke(stypy.reporting.localization.Localization(__file__, 374, 22), _get_char_id_ps_145846, *[font_145847, glyph_145848], **kwargs_145849)
        
        # Assigning a type to the variable 'char_id' (line 374)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 374, 12), 'char_id', _get_char_id_ps_call_result_145850)
        
        
        # Getting the type of 'char_id' (line 376)
        char_id_145851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 15), 'char_id')
        # Getting the type of 'glyph_map' (line 376)
        glyph_map_145852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 30), 'glyph_map')
        # Applying the binary operator 'notin' (line 376)
        result_contains_145853 = python_operator(stypy.reporting.localization.Localization(__file__, 376, 15), 'notin', char_id_145851, glyph_map_145852)
        
        # Testing the type of an if condition (line 376)
        if_condition_145854 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 376, 12), result_contains_145853)
        # Assigning a type to the variable 'if_condition_145854' (line 376)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 376, 12), 'if_condition_145854', if_condition_145854)
        # SSA begins for if statement (line 376)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to clear(...): (line 377)
        # Processing the call keyword arguments (line 377)
        kwargs_145857 = {}
        # Getting the type of 'font' (line 377)
        font_145855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 16), 'font', False)
        # Obtaining the member 'clear' of a type (line 377)
        clear_145856 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 377, 16), font_145855, 'clear')
        # Calling clear(args, kwargs) (line 377)
        clear_call_result_145858 = invoke(stypy.reporting.localization.Localization(__file__, 377, 16), clear_145856, *[], **kwargs_145857)
        
        
        # Call to set_size(...): (line 378)
        # Processing the call arguments (line 378)
        # Getting the type of 'self' (line 378)
        self_145861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 30), 'self', False)
        # Obtaining the member 'FONT_SCALE' of a type (line 378)
        FONT_SCALE_145862 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 378, 30), self_145861, 'FONT_SCALE')
        # Getting the type of 'self' (line 378)
        self_145863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 47), 'self', False)
        # Obtaining the member 'DPI' of a type (line 378)
        DPI_145864 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 378, 47), self_145863, 'DPI')
        # Processing the call keyword arguments (line 378)
        kwargs_145865 = {}
        # Getting the type of 'font' (line 378)
        font_145859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 16), 'font', False)
        # Obtaining the member 'set_size' of a type (line 378)
        set_size_145860 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 378, 16), font_145859, 'set_size')
        # Calling set_size(args, kwargs) (line 378)
        set_size_call_result_145866 = invoke(stypy.reporting.localization.Localization(__file__, 378, 16), set_size_145860, *[FONT_SCALE_145862, DPI_145864], **kwargs_145865)
        
        
        # Getting the type of 'enc' (line 379)
        enc_145867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 19), 'enc')
        # Testing the type of an if condition (line 379)
        if_condition_145868 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 379, 16), enc_145867)
        # Assigning a type to the variable 'if_condition_145868' (line 379)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 379, 16), 'if_condition_145868', if_condition_145868)
        # SSA begins for if statement (line 379)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 380):
        
        # Assigning a Call to a Name (line 380):
        
        # Call to get(...): (line 380)
        # Processing the call arguments (line 380)
        # Getting the type of 'glyph' (line 380)
        glyph_145871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 39), 'glyph', False)
        # Getting the type of 'None' (line 380)
        None_145872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 46), 'None', False)
        # Processing the call keyword arguments (line 380)
        kwargs_145873 = {}
        # Getting the type of 'enc' (line 380)
        enc_145869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 31), 'enc', False)
        # Obtaining the member 'get' of a type (line 380)
        get_145870 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 380, 31), enc_145869, 'get')
        # Calling get(args, kwargs) (line 380)
        get_call_result_145874 = invoke(stypy.reporting.localization.Localization(__file__, 380, 31), get_145870, *[glyph_145871, None_145872], **kwargs_145873)
        
        # Assigning a type to the variable 'charcode' (line 380)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 380, 20), 'charcode', get_call_result_145874)
        # SSA branch for the else part of an if statement (line 379)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Name to a Name (line 382):
        
        # Assigning a Name to a Name (line 382):
        # Getting the type of 'glyph' (line 382)
        glyph_145875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 31), 'glyph')
        # Assigning a type to the variable 'charcode' (line 382)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 382, 20), 'charcode', glyph_145875)
        # SSA join for if statement (line 379)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Type idiom detected: calculating its left and rigth part (line 384)
        # Getting the type of 'charcode' (line 384)
        charcode_145876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 16), 'charcode')
        # Getting the type of 'None' (line 384)
        None_145877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 35), 'None')
        
        (may_be_145878, more_types_in_union_145879) = may_not_be_none(charcode_145876, None_145877)

        if may_be_145878:

            if more_types_in_union_145879:
                # Runtime conditional SSA (line 384)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 385):
            
            # Assigning a Call to a Name (line 385):
            
            # Call to load_char(...): (line 385)
            # Processing the call arguments (line 385)
            # Getting the type of 'charcode' (line 385)
            charcode_145882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 44), 'charcode', False)
            # Processing the call keyword arguments (line 385)
            # Getting the type of 'ft2font_flag' (line 385)
            ft2font_flag_145883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 60), 'ft2font_flag', False)
            keyword_145884 = ft2font_flag_145883
            kwargs_145885 = {'flags': keyword_145884}
            # Getting the type of 'font' (line 385)
            font_145880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 29), 'font', False)
            # Obtaining the member 'load_char' of a type (line 385)
            load_char_145881 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 385, 29), font_145880, 'load_char')
            # Calling load_char(args, kwargs) (line 385)
            load_char_call_result_145886 = invoke(stypy.reporting.localization.Localization(__file__, 385, 29), load_char_145881, *[charcode_145882], **kwargs_145885)
            
            # Assigning a type to the variable 'glyph0' (line 385)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 385, 20), 'glyph0', load_char_call_result_145886)

            if more_types_in_union_145879:
                # Runtime conditional SSA for else branch (line 384)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_145878) or more_types_in_union_145879):
            
            # Call to warn(...): (line 387)
            # Processing the call arguments (line 387)
            unicode_145889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 387, 34), 'unicode', u'The glyph (%d) of font (%s) cannot be converted with the encoding. Glyph may be wrong')
            
            # Obtaining an instance of the builtin type 'tuple' (line 389)
            tuple_145890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 48), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 389)
            # Adding element type (line 389)
            # Getting the type of 'glyph' (line 389)
            glyph_145891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 48), 'glyph', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 389, 48), tuple_145890, glyph_145891)
            # Adding element type (line 389)
            # Getting the type of 'font_bunch' (line 389)
            font_bunch_145892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 55), 'font_bunch', False)
            # Obtaining the member 'filename' of a type (line 389)
            filename_145893 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 55), font_bunch_145892, 'filename')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 389, 48), tuple_145890, filename_145893)
            
            # Applying the binary operator '%' (line 387)
            result_mod_145894 = python_operator(stypy.reporting.localization.Localization(__file__, 387, 34), '%', unicode_145889, tuple_145890)
            
            # Processing the call keyword arguments (line 387)
            kwargs_145895 = {}
            # Getting the type of 'warnings' (line 387)
            warnings_145887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 20), 'warnings', False)
            # Obtaining the member 'warn' of a type (line 387)
            warn_145888 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 387, 20), warnings_145887, 'warn')
            # Calling warn(args, kwargs) (line 387)
            warn_call_result_145896 = invoke(stypy.reporting.localization.Localization(__file__, 387, 20), warn_145888, *[result_mod_145894], **kwargs_145895)
            
            
            # Assigning a Call to a Name (line 391):
            
            # Assigning a Call to a Name (line 391):
            
            # Call to load_char(...): (line 391)
            # Processing the call arguments (line 391)
            # Getting the type of 'glyph' (line 391)
            glyph_145899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 44), 'glyph', False)
            # Processing the call keyword arguments (line 391)
            # Getting the type of 'ft2font_flag' (line 391)
            ft2font_flag_145900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 57), 'ft2font_flag', False)
            keyword_145901 = ft2font_flag_145900
            kwargs_145902 = {'flags': keyword_145901}
            # Getting the type of 'font' (line 391)
            font_145897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 29), 'font', False)
            # Obtaining the member 'load_char' of a type (line 391)
            load_char_145898 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 391, 29), font_145897, 'load_char')
            # Calling load_char(args, kwargs) (line 391)
            load_char_call_result_145903 = invoke(stypy.reporting.localization.Localization(__file__, 391, 29), load_char_145898, *[glyph_145899], **kwargs_145902)
            
            # Assigning a type to the variable 'glyph0' (line 391)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 391, 20), 'glyph0', load_char_call_result_145903)

            if (may_be_145878 and more_types_in_union_145879):
                # SSA join for if statement (line 384)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Subscript (line 393):
        
        # Assigning a Call to a Subscript (line 393):
        
        # Call to glyph_to_path(...): (line 393)
        # Processing the call arguments (line 393)
        # Getting the type of 'font' (line 393)
        font_145906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 60), 'font', False)
        # Processing the call keyword arguments (line 393)
        kwargs_145907 = {}
        # Getting the type of 'self' (line 393)
        self_145904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 41), 'self', False)
        # Obtaining the member 'glyph_to_path' of a type (line 393)
        glyph_to_path_145905 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 393, 41), self_145904, 'glyph_to_path')
        # Calling glyph_to_path(args, kwargs) (line 393)
        glyph_to_path_call_result_145908 = invoke(stypy.reporting.localization.Localization(__file__, 393, 41), glyph_to_path_145905, *[font_145906], **kwargs_145907)
        
        # Getting the type of 'glyph_map_new' (line 393)
        glyph_map_new_145909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 16), 'glyph_map_new')
        # Getting the type of 'char_id' (line 393)
        char_id_145910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 30), 'char_id')
        # Storing an element on a container (line 393)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 393, 16), glyph_map_new_145909, (char_id_145910, glyph_to_path_call_result_145908))
        # SSA join for if statement (line 376)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to append(...): (line 395)
        # Processing the call arguments (line 395)
        # Getting the type of 'char_id' (line 395)
        char_id_145913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 29), 'char_id', False)
        # Processing the call keyword arguments (line 395)
        kwargs_145914 = {}
        # Getting the type of 'glyph_ids' (line 395)
        glyph_ids_145911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 12), 'glyph_ids', False)
        # Obtaining the member 'append' of a type (line 395)
        append_145912 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 395, 12), glyph_ids_145911, 'append')
        # Calling append(args, kwargs) (line 395)
        append_call_result_145915 = invoke(stypy.reporting.localization.Localization(__file__, 395, 12), append_145912, *[char_id_145913], **kwargs_145914)
        
        
        # Call to append(...): (line 396)
        # Processing the call arguments (line 396)
        # Getting the type of 'x1' (line 396)
        x1_145918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 30), 'x1', False)
        # Processing the call keyword arguments (line 396)
        kwargs_145919 = {}
        # Getting the type of 'xpositions' (line 396)
        xpositions_145916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 12), 'xpositions', False)
        # Obtaining the member 'append' of a type (line 396)
        append_145917 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 396, 12), xpositions_145916, 'append')
        # Calling append(args, kwargs) (line 396)
        append_call_result_145920 = invoke(stypy.reporting.localization.Localization(__file__, 396, 12), append_145917, *[x1_145918], **kwargs_145919)
        
        
        # Call to append(...): (line 397)
        # Processing the call arguments (line 397)
        # Getting the type of 'y1' (line 397)
        y1_145923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 30), 'y1', False)
        # Processing the call keyword arguments (line 397)
        kwargs_145924 = {}
        # Getting the type of 'ypositions' (line 397)
        ypositions_145921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 12), 'ypositions', False)
        # Obtaining the member 'append' of a type (line 397)
        append_145922 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 397, 12), ypositions_145921, 'append')
        # Calling append(args, kwargs) (line 397)
        append_call_result_145925 = invoke(stypy.reporting.localization.Localization(__file__, 397, 12), append_145922, *[y1_145923], **kwargs_145924)
        
        
        # Call to append(...): (line 398)
        # Processing the call arguments (line 398)
        # Getting the type of 'dvifont' (line 398)
        dvifont_145928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 25), 'dvifont', False)
        # Obtaining the member 'size' of a type (line 398)
        size_145929 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 25), dvifont_145928, 'size')
        # Getting the type of 'self' (line 398)
        self_145930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 40), 'self', False)
        # Obtaining the member 'FONT_SCALE' of a type (line 398)
        FONT_SCALE_145931 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 40), self_145930, 'FONT_SCALE')
        # Applying the binary operator 'div' (line 398)
        result_div_145932 = python_operator(stypy.reporting.localization.Localization(__file__, 398, 25), 'div', size_145929, FONT_SCALE_145931)
        
        # Processing the call keyword arguments (line 398)
        kwargs_145933 = {}
        # Getting the type of 'sizes' (line 398)
        sizes_145926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 12), 'sizes', False)
        # Obtaining the member 'append' of a type (line 398)
        append_145927 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 12), sizes_145926, 'append')
        # Calling append(args, kwargs) (line 398)
        append_call_result_145934 = invoke(stypy.reporting.localization.Localization(__file__, 398, 12), append_145927, *[result_div_145932], **kwargs_145933)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a List to a Name (line 400):
        
        # Assigning a List to a Name (line 400):
        
        # Obtaining an instance of the builtin type 'list' (line 400)
        list_145935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 400)
        
        # Assigning a type to the variable 'myrects' (line 400)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 400, 8), 'myrects', list_145935)
        
        # Getting the type of 'page' (line 402)
        page_145936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 28), 'page')
        # Obtaining the member 'boxes' of a type (line 402)
        boxes_145937 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 402, 28), page_145936, 'boxes')
        # Testing the type of a for loop iterable (line 402)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 402, 8), boxes_145937)
        # Getting the type of the for loop variable (line 402)
        for_loop_var_145938 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 402, 8), boxes_145937)
        # Assigning a type to the variable 'ox' (line 402)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 402, 8), 'ox', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 402, 8), for_loop_var_145938))
        # Assigning a type to the variable 'oy' (line 402)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 402, 8), 'oy', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 402, 8), for_loop_var_145938))
        # Assigning a type to the variable 'h' (line 402)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 402, 8), 'h', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 402, 8), for_loop_var_145938))
        # Assigning a type to the variable 'w' (line 402)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 402, 8), 'w', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 402, 8), for_loop_var_145938))
        # SSA begins for a for statement (line 402)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a List to a Name (line 403):
        
        # Assigning a List to a Name (line 403):
        
        # Obtaining an instance of the builtin type 'list' (line 403)
        list_145939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 403, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 403)
        # Adding element type (line 403)
        
        # Obtaining an instance of the builtin type 'tuple' (line 403)
        tuple_145940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 403, 22), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 403)
        # Adding element type (line 403)
        # Getting the type of 'ox' (line 403)
        ox_145941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 22), 'ox')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 403, 22), tuple_145940, ox_145941)
        # Adding element type (line 403)
        # Getting the type of 'oy' (line 403)
        oy_145942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 26), 'oy')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 403, 22), tuple_145940, oy_145942)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 403, 20), list_145939, tuple_145940)
        # Adding element type (line 403)
        
        # Obtaining an instance of the builtin type 'tuple' (line 403)
        tuple_145943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 403, 32), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 403)
        # Adding element type (line 403)
        # Getting the type of 'ox' (line 403)
        ox_145944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 32), 'ox')
        # Getting the type of 'w' (line 403)
        w_145945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 37), 'w')
        # Applying the binary operator '+' (line 403)
        result_add_145946 = python_operator(stypy.reporting.localization.Localization(__file__, 403, 32), '+', ox_145944, w_145945)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 403, 32), tuple_145943, result_add_145946)
        # Adding element type (line 403)
        # Getting the type of 'oy' (line 403)
        oy_145947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 40), 'oy')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 403, 32), tuple_145943, oy_145947)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 403, 20), list_145939, tuple_145943)
        # Adding element type (line 403)
        
        # Obtaining an instance of the builtin type 'tuple' (line 403)
        tuple_145948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 403, 46), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 403)
        # Adding element type (line 403)
        # Getting the type of 'ox' (line 403)
        ox_145949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 46), 'ox')
        # Getting the type of 'w' (line 403)
        w_145950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 51), 'w')
        # Applying the binary operator '+' (line 403)
        result_add_145951 = python_operator(stypy.reporting.localization.Localization(__file__, 403, 46), '+', ox_145949, w_145950)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 403, 46), tuple_145948, result_add_145951)
        # Adding element type (line 403)
        # Getting the type of 'oy' (line 403)
        oy_145952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 54), 'oy')
        # Getting the type of 'h' (line 403)
        h_145953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 59), 'h')
        # Applying the binary operator '+' (line 403)
        result_add_145954 = python_operator(stypy.reporting.localization.Localization(__file__, 403, 54), '+', oy_145952, h_145953)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 403, 46), tuple_145948, result_add_145954)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 403, 20), list_145939, tuple_145948)
        # Adding element type (line 403)
        
        # Obtaining an instance of the builtin type 'tuple' (line 404)
        tuple_145955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, 22), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 404)
        # Adding element type (line 404)
        # Getting the type of 'ox' (line 404)
        ox_145956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 22), 'ox')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 404, 22), tuple_145955, ox_145956)
        # Adding element type (line 404)
        # Getting the type of 'oy' (line 404)
        oy_145957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 26), 'oy')
        # Getting the type of 'h' (line 404)
        h_145958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 31), 'h')
        # Applying the binary operator '+' (line 404)
        result_add_145959 = python_operator(stypy.reporting.localization.Localization(__file__, 404, 26), '+', oy_145957, h_145958)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 404, 22), tuple_145955, result_add_145959)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 403, 20), list_145939, tuple_145955)
        # Adding element type (line 403)
        
        # Obtaining an instance of the builtin type 'tuple' (line 404)
        tuple_145960 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 404)
        # Adding element type (line 404)
        # Getting the type of 'ox' (line 404)
        ox_145961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 36), 'ox')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 404, 36), tuple_145960, ox_145961)
        # Adding element type (line 404)
        # Getting the type of 'oy' (line 404)
        oy_145962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 40), 'oy')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 404, 36), tuple_145960, oy_145962)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 403, 20), list_145939, tuple_145960)
        # Adding element type (line 403)
        
        # Obtaining an instance of the builtin type 'tuple' (line 404)
        tuple_145963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, 46), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 404)
        # Adding element type (line 404)
        int_145964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, 46), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 404, 46), tuple_145963, int_145964)
        # Adding element type (line 404)
        int_145965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, 49), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 404, 46), tuple_145963, int_145965)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 403, 20), list_145939, tuple_145963)
        
        # Assigning a type to the variable 'vert1' (line 403)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 403, 12), 'vert1', list_145939)
        
        # Assigning a List to a Name (line 405):
        
        # Assigning a List to a Name (line 405):
        
        # Obtaining an instance of the builtin type 'list' (line 405)
        list_145966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 405)
        # Adding element type (line 405)
        # Getting the type of 'Path' (line 405)
        Path_145967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 21), 'Path')
        # Obtaining the member 'MOVETO' of a type (line 405)
        MOVETO_145968 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 405, 21), Path_145967, 'MOVETO')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 405, 20), list_145966, MOVETO_145968)
        # Adding element type (line 405)
        # Getting the type of 'Path' (line 406)
        Path_145969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 21), 'Path')
        # Obtaining the member 'LINETO' of a type (line 406)
        LINETO_145970 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 21), Path_145969, 'LINETO')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 405, 20), list_145966, LINETO_145970)
        # Adding element type (line 405)
        # Getting the type of 'Path' (line 406)
        Path_145971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 34), 'Path')
        # Obtaining the member 'LINETO' of a type (line 406)
        LINETO_145972 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 34), Path_145971, 'LINETO')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 405, 20), list_145966, LINETO_145972)
        # Adding element type (line 405)
        # Getting the type of 'Path' (line 406)
        Path_145973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 47), 'Path')
        # Obtaining the member 'LINETO' of a type (line 406)
        LINETO_145974 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 47), Path_145973, 'LINETO')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 405, 20), list_145966, LINETO_145974)
        # Adding element type (line 405)
        # Getting the type of 'Path' (line 406)
        Path_145975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 60), 'Path')
        # Obtaining the member 'LINETO' of a type (line 406)
        LINETO_145976 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 60), Path_145975, 'LINETO')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 405, 20), list_145966, LINETO_145976)
        # Adding element type (line 405)
        # Getting the type of 'Path' (line 407)
        Path_145977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 21), 'Path')
        # Obtaining the member 'CLOSEPOLY' of a type (line 407)
        CLOSEPOLY_145978 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 407, 21), Path_145977, 'CLOSEPOLY')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 405, 20), list_145966, CLOSEPOLY_145978)
        
        # Assigning a type to the variable 'code1' (line 405)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 405, 12), 'code1', list_145966)
        
        # Call to append(...): (line 408)
        # Processing the call arguments (line 408)
        
        # Obtaining an instance of the builtin type 'tuple' (line 408)
        tuple_145981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 408, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 408)
        # Adding element type (line 408)
        # Getting the type of 'vert1' (line 408)
        vert1_145982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 28), 'vert1', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 408, 28), tuple_145981, vert1_145982)
        # Adding element type (line 408)
        # Getting the type of 'code1' (line 408)
        code1_145983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 35), 'code1', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 408, 28), tuple_145981, code1_145983)
        
        # Processing the call keyword arguments (line 408)
        kwargs_145984 = {}
        # Getting the type of 'myrects' (line 408)
        myrects_145979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 12), 'myrects', False)
        # Obtaining the member 'append' of a type (line 408)
        append_145980 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 408, 12), myrects_145979, 'append')
        # Calling append(args, kwargs) (line 408)
        append_call_result_145985 = invoke(stypy.reporting.localization.Localization(__file__, 408, 12), append_145980, *[tuple_145981], **kwargs_145984)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 410)
        tuple_145986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 410, 16), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 410)
        # Adding element type (line 410)
        
        # Call to list(...): (line 410)
        # Processing the call arguments (line 410)
        
        # Call to zip(...): (line 410)
        # Processing the call arguments (line 410)
        # Getting the type of 'glyph_ids' (line 410)
        glyph_ids_145989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 25), 'glyph_ids', False)
        # Getting the type of 'xpositions' (line 410)
        xpositions_145990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 36), 'xpositions', False)
        # Getting the type of 'ypositions' (line 410)
        ypositions_145991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 48), 'ypositions', False)
        # Getting the type of 'sizes' (line 410)
        sizes_145992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 60), 'sizes', False)
        # Processing the call keyword arguments (line 410)
        kwargs_145993 = {}
        # Getting the type of 'zip' (line 410)
        zip_145988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 21), 'zip', False)
        # Calling zip(args, kwargs) (line 410)
        zip_call_result_145994 = invoke(stypy.reporting.localization.Localization(__file__, 410, 21), zip_145988, *[glyph_ids_145989, xpositions_145990, ypositions_145991, sizes_145992], **kwargs_145993)
        
        # Processing the call keyword arguments (line 410)
        kwargs_145995 = {}
        # Getting the type of 'list' (line 410)
        list_145987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 16), 'list', False)
        # Calling list(args, kwargs) (line 410)
        list_call_result_145996 = invoke(stypy.reporting.localization.Localization(__file__, 410, 16), list_145987, *[zip_call_result_145994], **kwargs_145995)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 410, 16), tuple_145986, list_call_result_145996)
        # Adding element type (line 410)
        # Getting the type of 'glyph_map_new' (line 411)
        glyph_map_new_145997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 16), 'glyph_map_new')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 410, 16), tuple_145986, glyph_map_new_145997)
        # Adding element type (line 410)
        # Getting the type of 'myrects' (line 411)
        myrects_145998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 31), 'myrects')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 410, 16), tuple_145986, myrects_145998)
        
        # Assigning a type to the variable 'stypy_return_type' (line 410)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 410, 8), 'stypy_return_type', tuple_145986)
        
        # ################# End of 'get_glyphs_tex(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_glyphs_tex' in the type store
        # Getting the type of 'stypy_return_type' (line 292)
        stypy_return_type_145999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_145999)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_glyphs_tex'
        return stypy_return_type_145999


# Assigning a type to the variable 'TextToPath' (line 27)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 0), 'TextToPath', TextToPath)

# Assigning a Num to a Name (line 32):
float_146000 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 17), 'float')
# Getting the type of 'TextToPath'
TextToPath_146001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TextToPath')
# Setting the type of the member 'FONT_SCALE' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TextToPath_146001, 'FONT_SCALE', float_146000)

# Assigning a Num to a Name (line 33):
int_146002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 10), 'int')
# Getting the type of 'TextToPath'
TextToPath_146003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TextToPath')
# Setting the type of the member 'DPI' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TextToPath_146003, 'DPI', int_146002)

# Assigning a Call to a Name (line 414):

# Assigning a Call to a Name (line 414):

# Call to TextToPath(...): (line 414)
# Processing the call keyword arguments (line 414)
kwargs_146005 = {}
# Getting the type of 'TextToPath' (line 414)
TextToPath_146004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 15), 'TextToPath', False)
# Calling TextToPath(args, kwargs) (line 414)
TextToPath_call_result_146006 = invoke(stypy.reporting.localization.Localization(__file__, 414, 15), TextToPath_146004, *[], **kwargs_146005)

# Assigning a type to the variable 'text_to_path' (line 414)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 414, 0), 'text_to_path', TextToPath_call_result_146006)
# Declaration of the 'TextPath' class
# Getting the type of 'Path' (line 417)
Path_146007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 15), 'Path')

class TextPath(Path_146007, ):
    unicode_146008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 420, (-1)), 'unicode', u'\n    Create a path from the text.\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 422)
        None_146009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 35), 'None')
        # Getting the type of 'None' (line 422)
        None_146010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 46), 'None')
        int_146011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, 38), 'int')
        # Getting the type of 'False' (line 423)
        False_146012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 48), 'False')
        defaults = [None_146009, None_146010, int_146011, False_146012]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 422, 4, False)
        # Assigning a type to the variable 'self' (line 423)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 423, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TextPath.__init__', ['xy', 's', 'size', 'prop', '_interpolation_steps', 'usetex'], 'kl', 'kwargs', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['xy', 's', 'size', 'prop', '_interpolation_steps', 'usetex'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        unicode_146013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 435, (-1)), 'unicode', u'\n        Create a path from the text. No support for TeX yet. Note that\n        it simply is a path, not an artist. You need to use the\n        PathPatch (or other artists) to draw this path onto the\n        canvas.\n\n        xy : position of the text.\n        s : text\n        size : font size\n        prop : font property\n        ')
        
        # Type idiom detected: calculating its left and rigth part (line 437)
        # Getting the type of 'prop' (line 437)
        prop_146014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 11), 'prop')
        # Getting the type of 'None' (line 437)
        None_146015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 19), 'None')
        
        (may_be_146016, more_types_in_union_146017) = may_be_none(prop_146014, None_146015)

        if may_be_146016:

            if more_types_in_union_146017:
                # Runtime conditional SSA (line 437)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 438):
            
            # Assigning a Call to a Name (line 438):
            
            # Call to FontProperties(...): (line 438)
            # Processing the call keyword arguments (line 438)
            kwargs_146019 = {}
            # Getting the type of 'FontProperties' (line 438)
            FontProperties_146018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 19), 'FontProperties', False)
            # Calling FontProperties(args, kwargs) (line 438)
            FontProperties_call_result_146020 = invoke(stypy.reporting.localization.Localization(__file__, 438, 19), FontProperties_146018, *[], **kwargs_146019)
            
            # Assigning a type to the variable 'prop' (line 438)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 438, 12), 'prop', FontProperties_call_result_146020)

            if more_types_in_union_146017:
                # SSA join for if statement (line 437)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 440)
        # Getting the type of 'size' (line 440)
        size_146021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 11), 'size')
        # Getting the type of 'None' (line 440)
        None_146022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 19), 'None')
        
        (may_be_146023, more_types_in_union_146024) = may_be_none(size_146021, None_146022)

        if may_be_146023:

            if more_types_in_union_146024:
                # Runtime conditional SSA (line 440)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 441):
            
            # Assigning a Call to a Name (line 441):
            
            # Call to get_size_in_points(...): (line 441)
            # Processing the call keyword arguments (line 441)
            kwargs_146027 = {}
            # Getting the type of 'prop' (line 441)
            prop_146025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 19), 'prop', False)
            # Obtaining the member 'get_size_in_points' of a type (line 441)
            get_size_in_points_146026 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 441, 19), prop_146025, 'get_size_in_points')
            # Calling get_size_in_points(args, kwargs) (line 441)
            get_size_in_points_call_result_146028 = invoke(stypy.reporting.localization.Localization(__file__, 441, 19), get_size_in_points_146026, *[], **kwargs_146027)
            
            # Assigning a type to the variable 'size' (line 441)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 441, 12), 'size', get_size_in_points_call_result_146028)

            if more_types_in_union_146024:
                # SSA join for if statement (line 440)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Name to a Attribute (line 443):
        
        # Assigning a Name to a Attribute (line 443):
        # Getting the type of 'xy' (line 443)
        xy_146029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 19), 'xy')
        # Getting the type of 'self' (line 443)
        self_146030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 8), 'self')
        # Setting the type of the member '_xy' of a type (line 443)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 443, 8), self_146030, '_xy', xy_146029)
        
        # Call to set_size(...): (line 444)
        # Processing the call arguments (line 444)
        # Getting the type of 'size' (line 444)
        size_146033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 22), 'size', False)
        # Processing the call keyword arguments (line 444)
        kwargs_146034 = {}
        # Getting the type of 'self' (line 444)
        self_146031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 8), 'self', False)
        # Obtaining the member 'set_size' of a type (line 444)
        set_size_146032 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 444, 8), self_146031, 'set_size')
        # Calling set_size(args, kwargs) (line 444)
        set_size_call_result_146035 = invoke(stypy.reporting.localization.Localization(__file__, 444, 8), set_size_146032, *[size_146033], **kwargs_146034)
        
        
        # Assigning a Name to a Attribute (line 446):
        
        # Assigning a Name to a Attribute (line 446):
        # Getting the type of 'None' (line 446)
        None_146036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 32), 'None')
        # Getting the type of 'self' (line 446)
        self_146037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 8), 'self')
        # Setting the type of the member '_cached_vertices' of a type (line 446)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 446, 8), self_146037, '_cached_vertices', None_146036)
        
        # Assigning a Call to a Tuple (line 448):
        
        # Assigning a Call to a Name:
        
        # Call to text_get_vertices_codes(...): (line 448)
        # Processing the call arguments (line 448)
        # Getting the type of 'prop' (line 449)
        prop_146040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 44), 'prop', False)
        # Getting the type of 's' (line 449)
        s_146041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 50), 's', False)
        # Processing the call keyword arguments (line 448)
        # Getting the type of 'usetex' (line 450)
        usetex_146042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 51), 'usetex', False)
        keyword_146043 = usetex_146042
        kwargs_146044 = {'usetex': keyword_146043}
        # Getting the type of 'self' (line 448)
        self_146038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 38), 'self', False)
        # Obtaining the member 'text_get_vertices_codes' of a type (line 448)
        text_get_vertices_codes_146039 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 448, 38), self_146038, 'text_get_vertices_codes')
        # Calling text_get_vertices_codes(args, kwargs) (line 448)
        text_get_vertices_codes_call_result_146045 = invoke(stypy.reporting.localization.Localization(__file__, 448, 38), text_get_vertices_codes_146039, *[prop_146040, s_146041], **kwargs_146044)
        
        # Assigning a type to the variable 'call_assignment_144782' (line 448)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 448, 8), 'call_assignment_144782', text_get_vertices_codes_call_result_146045)
        
        # Assigning a Call to a Name (line 448):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_146048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 448, 8), 'int')
        # Processing the call keyword arguments
        kwargs_146049 = {}
        # Getting the type of 'call_assignment_144782' (line 448)
        call_assignment_144782_146046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 8), 'call_assignment_144782', False)
        # Obtaining the member '__getitem__' of a type (line 448)
        getitem___146047 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 448, 8), call_assignment_144782_146046, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_146050 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___146047, *[int_146048], **kwargs_146049)
        
        # Assigning a type to the variable 'call_assignment_144783' (line 448)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 448, 8), 'call_assignment_144783', getitem___call_result_146050)
        
        # Assigning a Name to a Attribute (line 448):
        # Getting the type of 'call_assignment_144783' (line 448)
        call_assignment_144783_146051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 8), 'call_assignment_144783')
        # Getting the type of 'self' (line 448)
        self_146052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 8), 'self')
        # Setting the type of the member '_vertices' of a type (line 448)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 448, 8), self_146052, '_vertices', call_assignment_144783_146051)
        
        # Assigning a Call to a Name (line 448):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_146055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 448, 8), 'int')
        # Processing the call keyword arguments
        kwargs_146056 = {}
        # Getting the type of 'call_assignment_144782' (line 448)
        call_assignment_144782_146053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 8), 'call_assignment_144782', False)
        # Obtaining the member '__getitem__' of a type (line 448)
        getitem___146054 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 448, 8), call_assignment_144782_146053, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_146057 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___146054, *[int_146055], **kwargs_146056)
        
        # Assigning a type to the variable 'call_assignment_144784' (line 448)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 448, 8), 'call_assignment_144784', getitem___call_result_146057)
        
        # Assigning a Name to a Attribute (line 448):
        # Getting the type of 'call_assignment_144784' (line 448)
        call_assignment_144784_146058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 8), 'call_assignment_144784')
        # Getting the type of 'self' (line 448)
        self_146059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 24), 'self')
        # Setting the type of the member '_codes' of a type (line 448)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 448, 24), self_146059, '_codes', call_assignment_144784_146058)
        
        # Assigning a Name to a Attribute (line 452):
        
        # Assigning a Name to a Attribute (line 452):
        # Getting the type of 'False' (line 452)
        False_146060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 32), 'False')
        # Getting the type of 'self' (line 452)
        self_146061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 8), 'self')
        # Setting the type of the member '_should_simplify' of a type (line 452)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 452, 8), self_146061, '_should_simplify', False_146060)
        
        # Assigning a Subscript to a Attribute (line 453):
        
        # Assigning a Subscript to a Attribute (line 453):
        
        # Obtaining the type of the subscript
        unicode_146062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 453, 44), 'unicode', u'path.simplify_threshold')
        # Getting the type of 'rcParams' (line 453)
        rcParams_146063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 35), 'rcParams')
        # Obtaining the member '__getitem__' of a type (line 453)
        getitem___146064 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 453, 35), rcParams_146063, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 453)
        subscript_call_result_146065 = invoke(stypy.reporting.localization.Localization(__file__, 453, 35), getitem___146064, unicode_146062)
        
        # Getting the type of 'self' (line 453)
        self_146066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 8), 'self')
        # Setting the type of the member '_simplify_threshold' of a type (line 453)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 453, 8), self_146066, '_simplify_threshold', subscript_call_result_146065)
        
        # Assigning a Name to a Attribute (line 454):
        
        # Assigning a Name to a Attribute (line 454):
        # Getting the type of 'False' (line 454)
        False_146067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 30), 'False')
        # Getting the type of 'self' (line 454)
        self_146068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 8), 'self')
        # Setting the type of the member '_has_nonfinite' of a type (line 454)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 454, 8), self_146068, '_has_nonfinite', False_146067)
        
        # Assigning a Name to a Attribute (line 455):
        
        # Assigning a Name to a Attribute (line 455):
        # Getting the type of '_interpolation_steps' (line 455)
        _interpolation_steps_146069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 36), '_interpolation_steps')
        # Getting the type of 'self' (line 455)
        self_146070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 8), 'self')
        # Setting the type of the member '_interpolation_steps' of a type (line 455)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 455, 8), self_146070, '_interpolation_steps', _interpolation_steps_146069)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def set_size(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_size'
        module_type_store = module_type_store.open_function_context('set_size', 457, 4, False)
        # Assigning a type to the variable 'self' (line 458)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 458, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TextPath.set_size.__dict__.__setitem__('stypy_localization', localization)
        TextPath.set_size.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TextPath.set_size.__dict__.__setitem__('stypy_type_store', module_type_store)
        TextPath.set_size.__dict__.__setitem__('stypy_function_name', 'TextPath.set_size')
        TextPath.set_size.__dict__.__setitem__('stypy_param_names_list', ['size'])
        TextPath.set_size.__dict__.__setitem__('stypy_varargs_param_name', None)
        TextPath.set_size.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TextPath.set_size.__dict__.__setitem__('stypy_call_defaults', defaults)
        TextPath.set_size.__dict__.__setitem__('stypy_call_varargs', varargs)
        TextPath.set_size.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TextPath.set_size.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TextPath.set_size', ['size'], None, None, defaults, varargs, kwargs)

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

        unicode_146071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 460, (-1)), 'unicode', u'\n        set the size of the text\n        ')
        
        # Assigning a Name to a Attribute (line 461):
        
        # Assigning a Name to a Attribute (line 461):
        # Getting the type of 'size' (line 461)
        size_146072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 21), 'size')
        # Getting the type of 'self' (line 461)
        self_146073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 8), 'self')
        # Setting the type of the member '_size' of a type (line 461)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 461, 8), self_146073, '_size', size_146072)
        
        # Assigning a Name to a Attribute (line 462):
        
        # Assigning a Name to a Attribute (line 462):
        # Getting the type of 'True' (line 462)
        True_146074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 24), 'True')
        # Getting the type of 'self' (line 462)
        self_146075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 8), 'self')
        # Setting the type of the member '_invalid' of a type (line 462)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 462, 8), self_146075, '_invalid', True_146074)
        
        # ################# End of 'set_size(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_size' in the type store
        # Getting the type of 'stypy_return_type' (line 457)
        stypy_return_type_146076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_146076)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_size'
        return stypy_return_type_146076


    @norecursion
    def get_size(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_size'
        module_type_store = module_type_store.open_function_context('get_size', 464, 4, False)
        # Assigning a type to the variable 'self' (line 465)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 465, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TextPath.get_size.__dict__.__setitem__('stypy_localization', localization)
        TextPath.get_size.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TextPath.get_size.__dict__.__setitem__('stypy_type_store', module_type_store)
        TextPath.get_size.__dict__.__setitem__('stypy_function_name', 'TextPath.get_size')
        TextPath.get_size.__dict__.__setitem__('stypy_param_names_list', [])
        TextPath.get_size.__dict__.__setitem__('stypy_varargs_param_name', None)
        TextPath.get_size.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TextPath.get_size.__dict__.__setitem__('stypy_call_defaults', defaults)
        TextPath.get_size.__dict__.__setitem__('stypy_call_varargs', varargs)
        TextPath.get_size.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TextPath.get_size.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TextPath.get_size', [], None, None, defaults, varargs, kwargs)

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

        unicode_146077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 467, (-1)), 'unicode', u'\n        get the size of the text\n        ')
        # Getting the type of 'self' (line 468)
        self_146078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 15), 'self')
        # Obtaining the member '_size' of a type (line 468)
        _size_146079 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 468, 15), self_146078, '_size')
        # Assigning a type to the variable 'stypy_return_type' (line 468)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 468, 8), 'stypy_return_type', _size_146079)
        
        # ################# End of 'get_size(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_size' in the type store
        # Getting the type of 'stypy_return_type' (line 464)
        stypy_return_type_146080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_146080)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_size'
        return stypy_return_type_146080


    @norecursion
    def _get_vertices(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_get_vertices'
        module_type_store = module_type_store.open_function_context('_get_vertices', 470, 4, False)
        # Assigning a type to the variable 'self' (line 471)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 471, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TextPath._get_vertices.__dict__.__setitem__('stypy_localization', localization)
        TextPath._get_vertices.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TextPath._get_vertices.__dict__.__setitem__('stypy_type_store', module_type_store)
        TextPath._get_vertices.__dict__.__setitem__('stypy_function_name', 'TextPath._get_vertices')
        TextPath._get_vertices.__dict__.__setitem__('stypy_param_names_list', [])
        TextPath._get_vertices.__dict__.__setitem__('stypy_varargs_param_name', None)
        TextPath._get_vertices.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TextPath._get_vertices.__dict__.__setitem__('stypy_call_defaults', defaults)
        TextPath._get_vertices.__dict__.__setitem__('stypy_call_varargs', varargs)
        TextPath._get_vertices.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TextPath._get_vertices.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TextPath._get_vertices', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_get_vertices', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_get_vertices(...)' code ##################

        unicode_146081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 473, (-1)), 'unicode', u'\n        Return the cached path after updating it if necessary.\n        ')
        
        # Call to _revalidate_path(...): (line 474)
        # Processing the call keyword arguments (line 474)
        kwargs_146084 = {}
        # Getting the type of 'self' (line 474)
        self_146082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 8), 'self', False)
        # Obtaining the member '_revalidate_path' of a type (line 474)
        _revalidate_path_146083 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 474, 8), self_146082, '_revalidate_path')
        # Calling _revalidate_path(args, kwargs) (line 474)
        _revalidate_path_call_result_146085 = invoke(stypy.reporting.localization.Localization(__file__, 474, 8), _revalidate_path_146083, *[], **kwargs_146084)
        
        # Getting the type of 'self' (line 475)
        self_146086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 15), 'self')
        # Obtaining the member '_cached_vertices' of a type (line 475)
        _cached_vertices_146087 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 475, 15), self_146086, '_cached_vertices')
        # Assigning a type to the variable 'stypy_return_type' (line 475)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 475, 8), 'stypy_return_type', _cached_vertices_146087)
        
        # ################# End of '_get_vertices(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_get_vertices' in the type store
        # Getting the type of 'stypy_return_type' (line 470)
        stypy_return_type_146088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_146088)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_get_vertices'
        return stypy_return_type_146088


    @norecursion
    def _get_codes(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_get_codes'
        module_type_store = module_type_store.open_function_context('_get_codes', 477, 4, False)
        # Assigning a type to the variable 'self' (line 478)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 478, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TextPath._get_codes.__dict__.__setitem__('stypy_localization', localization)
        TextPath._get_codes.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TextPath._get_codes.__dict__.__setitem__('stypy_type_store', module_type_store)
        TextPath._get_codes.__dict__.__setitem__('stypy_function_name', 'TextPath._get_codes')
        TextPath._get_codes.__dict__.__setitem__('stypy_param_names_list', [])
        TextPath._get_codes.__dict__.__setitem__('stypy_varargs_param_name', None)
        TextPath._get_codes.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TextPath._get_codes.__dict__.__setitem__('stypy_call_defaults', defaults)
        TextPath._get_codes.__dict__.__setitem__('stypy_call_varargs', varargs)
        TextPath._get_codes.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TextPath._get_codes.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TextPath._get_codes', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_get_codes', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_get_codes(...)' code ##################

        unicode_146089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 480, (-1)), 'unicode', u'\n        Return the codes\n        ')
        # Getting the type of 'self' (line 481)
        self_146090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 15), 'self')
        # Obtaining the member '_codes' of a type (line 481)
        _codes_146091 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 481, 15), self_146090, '_codes')
        # Assigning a type to the variable 'stypy_return_type' (line 481)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 481, 8), 'stypy_return_type', _codes_146091)
        
        # ################# End of '_get_codes(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_get_codes' in the type store
        # Getting the type of 'stypy_return_type' (line 477)
        stypy_return_type_146092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_146092)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_get_codes'
        return stypy_return_type_146092

    
    # Assigning a Call to a Name (line 483):
    
    # Assigning a Call to a Name (line 484):

    @norecursion
    def _revalidate_path(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_revalidate_path'
        module_type_store = module_type_store.open_function_context('_revalidate_path', 486, 4, False)
        # Assigning a type to the variable 'self' (line 487)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 487, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TextPath._revalidate_path.__dict__.__setitem__('stypy_localization', localization)
        TextPath._revalidate_path.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TextPath._revalidate_path.__dict__.__setitem__('stypy_type_store', module_type_store)
        TextPath._revalidate_path.__dict__.__setitem__('stypy_function_name', 'TextPath._revalidate_path')
        TextPath._revalidate_path.__dict__.__setitem__('stypy_param_names_list', [])
        TextPath._revalidate_path.__dict__.__setitem__('stypy_varargs_param_name', None)
        TextPath._revalidate_path.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TextPath._revalidate_path.__dict__.__setitem__('stypy_call_defaults', defaults)
        TextPath._revalidate_path.__dict__.__setitem__('stypy_call_varargs', varargs)
        TextPath._revalidate_path.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TextPath._revalidate_path.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TextPath._revalidate_path', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_revalidate_path', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_revalidate_path(...)' code ##################

        unicode_146093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 494, (-1)), 'unicode', u'\n        update the path if necessary.\n\n        The path for the text is initially create with the font size\n        of FONT_SCALE, and this path is rescaled to other size when\n        necessary.\n\n        ')
        
        
        # Evaluating a boolean operation
        # Getting the type of 'self' (line 495)
        self_146094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 495, 12), 'self')
        # Obtaining the member '_invalid' of a type (line 495)
        _invalid_146095 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 495, 12), self_146094, '_invalid')
        
        # Getting the type of 'self' (line 496)
        self_146096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 13), 'self')
        # Obtaining the member '_cached_vertices' of a type (line 496)
        _cached_vertices_146097 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 496, 13), self_146096, '_cached_vertices')
        # Getting the type of 'None' (line 496)
        None_146098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 38), 'None')
        # Applying the binary operator 'is' (line 496)
        result_is__146099 = python_operator(stypy.reporting.localization.Localization(__file__, 496, 13), 'is', _cached_vertices_146097, None_146098)
        
        # Applying the binary operator 'or' (line 495)
        result_or_keyword_146100 = python_operator(stypy.reporting.localization.Localization(__file__, 495, 12), 'or', _invalid_146095, result_is__146099)
        
        # Testing the type of an if condition (line 495)
        if_condition_146101 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 495, 8), result_or_keyword_146100)
        # Assigning a type to the variable 'if_condition_146101' (line 495)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 495, 8), 'if_condition_146101', if_condition_146101)
        # SSA begins for if statement (line 495)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 497):
        
        # Assigning a Call to a Name (line 497):
        
        # Call to translate(...): (line 497)
        # Getting the type of 'self' (line 499)
        self_146119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 69), 'self', False)
        # Obtaining the member '_xy' of a type (line 499)
        _xy_146120 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 499, 69), self_146119, '_xy')
        # Processing the call keyword arguments (line 497)
        kwargs_146121 = {}
        
        # Call to scale(...): (line 497)
        # Processing the call arguments (line 497)
        # Getting the type of 'self' (line 498)
        self_146106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 20), 'self', False)
        # Obtaining the member '_size' of a type (line 498)
        _size_146107 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 498, 20), self_146106, '_size')
        # Getting the type of 'text_to_path' (line 498)
        text_to_path_146108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 33), 'text_to_path', False)
        # Obtaining the member 'FONT_SCALE' of a type (line 498)
        FONT_SCALE_146109 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 498, 33), text_to_path_146108, 'FONT_SCALE')
        # Applying the binary operator 'div' (line 498)
        result_div_146110 = python_operator(stypy.reporting.localization.Localization(__file__, 498, 20), 'div', _size_146107, FONT_SCALE_146109)
        
        # Getting the type of 'self' (line 499)
        self_146111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 20), 'self', False)
        # Obtaining the member '_size' of a type (line 499)
        _size_146112 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 499, 20), self_146111, '_size')
        # Getting the type of 'text_to_path' (line 499)
        text_to_path_146113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 33), 'text_to_path', False)
        # Obtaining the member 'FONT_SCALE' of a type (line 499)
        FONT_SCALE_146114 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 499, 33), text_to_path_146113, 'FONT_SCALE')
        # Applying the binary operator 'div' (line 499)
        result_div_146115 = python_operator(stypy.reporting.localization.Localization(__file__, 499, 20), 'div', _size_146112, FONT_SCALE_146114)
        
        # Processing the call keyword arguments (line 497)
        kwargs_146116 = {}
        
        # Call to Affine2D(...): (line 497)
        # Processing the call keyword arguments (line 497)
        kwargs_146103 = {}
        # Getting the type of 'Affine2D' (line 497)
        Affine2D_146102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 17), 'Affine2D', False)
        # Calling Affine2D(args, kwargs) (line 497)
        Affine2D_call_result_146104 = invoke(stypy.reporting.localization.Localization(__file__, 497, 17), Affine2D_146102, *[], **kwargs_146103)
        
        # Obtaining the member 'scale' of a type (line 497)
        scale_146105 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 497, 17), Affine2D_call_result_146104, 'scale')
        # Calling scale(args, kwargs) (line 497)
        scale_call_result_146117 = invoke(stypy.reporting.localization.Localization(__file__, 497, 17), scale_146105, *[result_div_146110, result_div_146115], **kwargs_146116)
        
        # Obtaining the member 'translate' of a type (line 497)
        translate_146118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 497, 17), scale_call_result_146117, 'translate')
        # Calling translate(args, kwargs) (line 497)
        translate_call_result_146122 = invoke(stypy.reporting.localization.Localization(__file__, 497, 17), translate_146118, *[_xy_146120], **kwargs_146121)
        
        # Assigning a type to the variable 'tr' (line 497)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 497, 12), 'tr', translate_call_result_146122)
        
        # Assigning a Call to a Attribute (line 500):
        
        # Assigning a Call to a Attribute (line 500):
        
        # Call to transform(...): (line 500)
        # Processing the call arguments (line 500)
        # Getting the type of 'self' (line 500)
        self_146125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 49), 'self', False)
        # Obtaining the member '_vertices' of a type (line 500)
        _vertices_146126 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 500, 49), self_146125, '_vertices')
        # Processing the call keyword arguments (line 500)
        kwargs_146127 = {}
        # Getting the type of 'tr' (line 500)
        tr_146123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 36), 'tr', False)
        # Obtaining the member 'transform' of a type (line 500)
        transform_146124 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 500, 36), tr_146123, 'transform')
        # Calling transform(args, kwargs) (line 500)
        transform_call_result_146128 = invoke(stypy.reporting.localization.Localization(__file__, 500, 36), transform_146124, *[_vertices_146126], **kwargs_146127)
        
        # Getting the type of 'self' (line 500)
        self_146129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 12), 'self')
        # Setting the type of the member '_cached_vertices' of a type (line 500)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 500, 12), self_146129, '_cached_vertices', transform_call_result_146128)
        
        # Assigning a Name to a Attribute (line 501):
        
        # Assigning a Name to a Attribute (line 501):
        # Getting the type of 'False' (line 501)
        False_146130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 28), 'False')
        # Getting the type of 'self' (line 501)
        self_146131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 12), 'self')
        # Setting the type of the member '_invalid' of a type (line 501)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 501, 12), self_146131, '_invalid', False_146130)
        # SSA join for if statement (line 495)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_revalidate_path(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_revalidate_path' in the type store
        # Getting the type of 'stypy_return_type' (line 486)
        stypy_return_type_146132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_146132)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_revalidate_path'
        return stypy_return_type_146132


    @norecursion
    def is_math_text(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'is_math_text'
        module_type_store = module_type_store.open_function_context('is_math_text', 503, 4, False)
        # Assigning a type to the variable 'self' (line 504)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 504, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TextPath.is_math_text.__dict__.__setitem__('stypy_localization', localization)
        TextPath.is_math_text.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TextPath.is_math_text.__dict__.__setitem__('stypy_type_store', module_type_store)
        TextPath.is_math_text.__dict__.__setitem__('stypy_function_name', 'TextPath.is_math_text')
        TextPath.is_math_text.__dict__.__setitem__('stypy_param_names_list', ['s'])
        TextPath.is_math_text.__dict__.__setitem__('stypy_varargs_param_name', None)
        TextPath.is_math_text.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TextPath.is_math_text.__dict__.__setitem__('stypy_call_defaults', defaults)
        TextPath.is_math_text.__dict__.__setitem__('stypy_call_varargs', varargs)
        TextPath.is_math_text.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TextPath.is_math_text.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TextPath.is_math_text', ['s'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'is_math_text', localization, ['s'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'is_math_text(...)' code ##################

        unicode_146133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 506, (-1)), 'unicode', u'\n        Returns True if the given string *s* contains any mathtext.\n        ')
        
        # Assigning a BinOp to a Name (line 511):
        
        # Assigning a BinOp to a Name (line 511):
        
        # Call to count(...): (line 511)
        # Processing the call arguments (line 511)
        unicode_146136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 511, 31), 'unicode', u'$')
        # Processing the call keyword arguments (line 511)
        kwargs_146137 = {}
        # Getting the type of 's' (line 511)
        s_146134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 23), 's', False)
        # Obtaining the member 'count' of a type (line 511)
        count_146135 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 511, 23), s_146134, 'count')
        # Calling count(args, kwargs) (line 511)
        count_call_result_146138 = invoke(stypy.reporting.localization.Localization(__file__, 511, 23), count_146135, *[unicode_146136], **kwargs_146137)
        
        
        # Call to count(...): (line 511)
        # Processing the call arguments (line 511)
        unicode_146141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 511, 47), 'unicode', u'\\$')
        # Processing the call keyword arguments (line 511)
        kwargs_146142 = {}
        # Getting the type of 's' (line 511)
        s_146139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 39), 's', False)
        # Obtaining the member 'count' of a type (line 511)
        count_146140 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 511, 39), s_146139, 'count')
        # Calling count(args, kwargs) (line 511)
        count_call_result_146143 = invoke(stypy.reporting.localization.Localization(__file__, 511, 39), count_146140, *[unicode_146141], **kwargs_146142)
        
        # Applying the binary operator '-' (line 511)
        result_sub_146144 = python_operator(stypy.reporting.localization.Localization(__file__, 511, 23), '-', count_call_result_146138, count_call_result_146143)
        
        # Assigning a type to the variable 'dollar_count' (line 511)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 511, 8), 'dollar_count', result_sub_146144)
        
        # Assigning a BoolOp to a Name (line 512):
        
        # Assigning a BoolOp to a Name (line 512):
        
        # Evaluating a boolean operation
        
        # Getting the type of 'dollar_count' (line 512)
        dollar_count_146145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 24), 'dollar_count')
        int_146146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 512, 39), 'int')
        # Applying the binary operator '>' (line 512)
        result_gt_146147 = python_operator(stypy.reporting.localization.Localization(__file__, 512, 24), '>', dollar_count_146145, int_146146)
        
        
        # Getting the type of 'dollar_count' (line 512)
        dollar_count_146148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 45), 'dollar_count')
        int_146149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 512, 60), 'int')
        # Applying the binary operator '%' (line 512)
        result_mod_146150 = python_operator(stypy.reporting.localization.Localization(__file__, 512, 45), '%', dollar_count_146148, int_146149)
        
        int_146151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 512, 65), 'int')
        # Applying the binary operator '==' (line 512)
        result_eq_146152 = python_operator(stypy.reporting.localization.Localization(__file__, 512, 45), '==', result_mod_146150, int_146151)
        
        # Applying the binary operator 'and' (line 512)
        result_and_keyword_146153 = python_operator(stypy.reporting.localization.Localization(__file__, 512, 24), 'and', result_gt_146147, result_eq_146152)
        
        # Assigning a type to the variable 'even_dollars' (line 512)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 512, 8), 'even_dollars', result_and_keyword_146153)
        
        
        # Obtaining the type of the subscript
        unicode_146154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 514, 20), 'unicode', u'text.usetex')
        # Getting the type of 'rcParams' (line 514)
        rcParams_146155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 11), 'rcParams')
        # Obtaining the member '__getitem__' of a type (line 514)
        getitem___146156 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 514, 11), rcParams_146155, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 514)
        subscript_call_result_146157 = invoke(stypy.reporting.localization.Localization(__file__, 514, 11), getitem___146156, unicode_146154)
        
        # Testing the type of an if condition (line 514)
        if_condition_146158 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 514, 8), subscript_call_result_146157)
        # Assigning a type to the variable 'if_condition_146158' (line 514)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 514, 8), 'if_condition_146158', if_condition_146158)
        # SSA begins for if statement (line 514)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Obtaining an instance of the builtin type 'tuple' (line 515)
        tuple_146159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 515, 19), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 515)
        # Adding element type (line 515)
        # Getting the type of 's' (line 515)
        s_146160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 19), 's')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 515, 19), tuple_146159, s_146160)
        # Adding element type (line 515)
        unicode_146161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 515, 22), 'unicode', u'TeX')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 515, 19), tuple_146159, unicode_146161)
        
        # Assigning a type to the variable 'stypy_return_type' (line 515)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 515, 12), 'stypy_return_type', tuple_146159)
        # SSA join for if statement (line 514)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'even_dollars' (line 517)
        even_dollars_146162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 11), 'even_dollars')
        # Testing the type of an if condition (line 517)
        if_condition_146163 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 517, 8), even_dollars_146162)
        # Assigning a type to the variable 'if_condition_146163' (line 517)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 517, 8), 'if_condition_146163', if_condition_146163)
        # SSA begins for if statement (line 517)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Obtaining an instance of the builtin type 'tuple' (line 518)
        tuple_146164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 518, 19), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 518)
        # Adding element type (line 518)
        # Getting the type of 's' (line 518)
        s_146165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 19), 's')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 518, 19), tuple_146164, s_146165)
        # Adding element type (line 518)
        # Getting the type of 'True' (line 518)
        True_146166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 22), 'True')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 518, 19), tuple_146164, True_146166)
        
        # Assigning a type to the variable 'stypy_return_type' (line 518)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 518, 12), 'stypy_return_type', tuple_146164)
        # SSA branch for the else part of an if statement (line 517)
        module_type_store.open_ssa_branch('else')
        
        # Obtaining an instance of the builtin type 'tuple' (line 520)
        tuple_146167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 520, 19), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 520)
        # Adding element type (line 520)
        
        # Call to replace(...): (line 520)
        # Processing the call arguments (line 520)
        unicode_146170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 520, 29), 'unicode', u'\\$')
        unicode_146171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 520, 36), 'unicode', u'$')
        # Processing the call keyword arguments (line 520)
        kwargs_146172 = {}
        # Getting the type of 's' (line 520)
        s_146168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 19), 's', False)
        # Obtaining the member 'replace' of a type (line 520)
        replace_146169 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 520, 19), s_146168, 'replace')
        # Calling replace(args, kwargs) (line 520)
        replace_call_result_146173 = invoke(stypy.reporting.localization.Localization(__file__, 520, 19), replace_146169, *[unicode_146170, unicode_146171], **kwargs_146172)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 520, 19), tuple_146167, replace_call_result_146173)
        # Adding element type (line 520)
        # Getting the type of 'False' (line 520)
        False_146174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 42), 'False')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 520, 19), tuple_146167, False_146174)
        
        # Assigning a type to the variable 'stypy_return_type' (line 520)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 520, 12), 'stypy_return_type', tuple_146167)
        # SSA join for if statement (line 517)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'is_math_text(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'is_math_text' in the type store
        # Getting the type of 'stypy_return_type' (line 503)
        stypy_return_type_146175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_146175)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'is_math_text'
        return stypy_return_type_146175


    @norecursion
    def text_get_vertices_codes(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'text_get_vertices_codes'
        module_type_store = module_type_store.open_function_context('text_get_vertices_codes', 522, 4, False)
        # Assigning a type to the variable 'self' (line 523)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 523, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TextPath.text_get_vertices_codes.__dict__.__setitem__('stypy_localization', localization)
        TextPath.text_get_vertices_codes.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TextPath.text_get_vertices_codes.__dict__.__setitem__('stypy_type_store', module_type_store)
        TextPath.text_get_vertices_codes.__dict__.__setitem__('stypy_function_name', 'TextPath.text_get_vertices_codes')
        TextPath.text_get_vertices_codes.__dict__.__setitem__('stypy_param_names_list', ['prop', 's', 'usetex'])
        TextPath.text_get_vertices_codes.__dict__.__setitem__('stypy_varargs_param_name', None)
        TextPath.text_get_vertices_codes.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TextPath.text_get_vertices_codes.__dict__.__setitem__('stypy_call_defaults', defaults)
        TextPath.text_get_vertices_codes.__dict__.__setitem__('stypy_call_varargs', varargs)
        TextPath.text_get_vertices_codes.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TextPath.text_get_vertices_codes.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TextPath.text_get_vertices_codes', ['prop', 's', 'usetex'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'text_get_vertices_codes', localization, ['prop', 's', 'usetex'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'text_get_vertices_codes(...)' code ##################

        unicode_146176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 527, (-1)), 'unicode', u'\n        convert the string *s* to vertices and codes using the\n        provided font property *prop*. Mostly copied from\n        backend_svg.py.\n        ')
        
        # Getting the type of 'usetex' (line 529)
        usetex_146177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 11), 'usetex')
        # Testing the type of an if condition (line 529)
        if_condition_146178 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 529, 8), usetex_146177)
        # Assigning a type to the variable 'if_condition_146178' (line 529)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 529, 8), 'if_condition_146178', if_condition_146178)
        # SSA begins for if statement (line 529)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Tuple (line 530):
        
        # Assigning a Call to a Name:
        
        # Call to get_text_path(...): (line 530)
        # Processing the call arguments (line 530)
        # Getting the type of 'prop' (line 530)
        prop_146181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 54), 'prop', False)
        # Getting the type of 's' (line 530)
        s_146182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 60), 's', False)
        # Processing the call keyword arguments (line 530)
        # Getting the type of 'True' (line 530)
        True_146183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 70), 'True', False)
        keyword_146184 = True_146183
        kwargs_146185 = {'usetex': keyword_146184}
        # Getting the type of 'text_to_path' (line 530)
        text_to_path_146179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 27), 'text_to_path', False)
        # Obtaining the member 'get_text_path' of a type (line 530)
        get_text_path_146180 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 530, 27), text_to_path_146179, 'get_text_path')
        # Calling get_text_path(args, kwargs) (line 530)
        get_text_path_call_result_146186 = invoke(stypy.reporting.localization.Localization(__file__, 530, 27), get_text_path_146180, *[prop_146181, s_146182], **kwargs_146185)
        
        # Assigning a type to the variable 'call_assignment_144785' (line 530)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 530, 12), 'call_assignment_144785', get_text_path_call_result_146186)
        
        # Assigning a Call to a Name (line 530):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_146189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 530, 12), 'int')
        # Processing the call keyword arguments
        kwargs_146190 = {}
        # Getting the type of 'call_assignment_144785' (line 530)
        call_assignment_144785_146187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 12), 'call_assignment_144785', False)
        # Obtaining the member '__getitem__' of a type (line 530)
        getitem___146188 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 530, 12), call_assignment_144785_146187, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_146191 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___146188, *[int_146189], **kwargs_146190)
        
        # Assigning a type to the variable 'call_assignment_144786' (line 530)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 530, 12), 'call_assignment_144786', getitem___call_result_146191)
        
        # Assigning a Name to a Name (line 530):
        # Getting the type of 'call_assignment_144786' (line 530)
        call_assignment_144786_146192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 12), 'call_assignment_144786')
        # Assigning a type to the variable 'verts' (line 530)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 530, 12), 'verts', call_assignment_144786_146192)
        
        # Assigning a Call to a Name (line 530):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_146195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 530, 12), 'int')
        # Processing the call keyword arguments
        kwargs_146196 = {}
        # Getting the type of 'call_assignment_144785' (line 530)
        call_assignment_144785_146193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 12), 'call_assignment_144785', False)
        # Obtaining the member '__getitem__' of a type (line 530)
        getitem___146194 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 530, 12), call_assignment_144785_146193, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_146197 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___146194, *[int_146195], **kwargs_146196)
        
        # Assigning a type to the variable 'call_assignment_144787' (line 530)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 530, 12), 'call_assignment_144787', getitem___call_result_146197)
        
        # Assigning a Name to a Name (line 530):
        # Getting the type of 'call_assignment_144787' (line 530)
        call_assignment_144787_146198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 12), 'call_assignment_144787')
        # Assigning a type to the variable 'codes' (line 530)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 530, 19), 'codes', call_assignment_144787_146198)
        # SSA branch for the else part of an if statement (line 529)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Tuple (line 532):
        
        # Assigning a Call to a Name:
        
        # Call to is_math_text(...): (line 532)
        # Processing the call arguments (line 532)
        # Getting the type of 's' (line 532)
        s_146201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 51), 's', False)
        # Processing the call keyword arguments (line 532)
        kwargs_146202 = {}
        # Getting the type of 'self' (line 532)
        self_146199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 33), 'self', False)
        # Obtaining the member 'is_math_text' of a type (line 532)
        is_math_text_146200 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 532, 33), self_146199, 'is_math_text')
        # Calling is_math_text(args, kwargs) (line 532)
        is_math_text_call_result_146203 = invoke(stypy.reporting.localization.Localization(__file__, 532, 33), is_math_text_146200, *[s_146201], **kwargs_146202)
        
        # Assigning a type to the variable 'call_assignment_144788' (line 532)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 532, 12), 'call_assignment_144788', is_math_text_call_result_146203)
        
        # Assigning a Call to a Name (line 532):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_146206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 532, 12), 'int')
        # Processing the call keyword arguments
        kwargs_146207 = {}
        # Getting the type of 'call_assignment_144788' (line 532)
        call_assignment_144788_146204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 12), 'call_assignment_144788', False)
        # Obtaining the member '__getitem__' of a type (line 532)
        getitem___146205 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 532, 12), call_assignment_144788_146204, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_146208 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___146205, *[int_146206], **kwargs_146207)
        
        # Assigning a type to the variable 'call_assignment_144789' (line 532)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 532, 12), 'call_assignment_144789', getitem___call_result_146208)
        
        # Assigning a Name to a Name (line 532):
        # Getting the type of 'call_assignment_144789' (line 532)
        call_assignment_144789_146209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 12), 'call_assignment_144789')
        # Assigning a type to the variable 'clean_line' (line 532)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 532, 12), 'clean_line', call_assignment_144789_146209)
        
        # Assigning a Call to a Name (line 532):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_146212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 532, 12), 'int')
        # Processing the call keyword arguments
        kwargs_146213 = {}
        # Getting the type of 'call_assignment_144788' (line 532)
        call_assignment_144788_146210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 12), 'call_assignment_144788', False)
        # Obtaining the member '__getitem__' of a type (line 532)
        getitem___146211 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 532, 12), call_assignment_144788_146210, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_146214 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___146211, *[int_146212], **kwargs_146213)
        
        # Assigning a type to the variable 'call_assignment_144790' (line 532)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 532, 12), 'call_assignment_144790', getitem___call_result_146214)
        
        # Assigning a Name to a Name (line 532):
        # Getting the type of 'call_assignment_144790' (line 532)
        call_assignment_144790_146215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 12), 'call_assignment_144790')
        # Assigning a type to the variable 'ismath' (line 532)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 532, 24), 'ismath', call_assignment_144790_146215)
        
        # Assigning a Call to a Tuple (line 533):
        
        # Assigning a Call to a Name:
        
        # Call to get_text_path(...): (line 533)
        # Processing the call arguments (line 533)
        # Getting the type of 'prop' (line 533)
        prop_146218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 54), 'prop', False)
        # Getting the type of 'clean_line' (line 533)
        clean_line_146219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 60), 'clean_line', False)
        # Processing the call keyword arguments (line 533)
        # Getting the type of 'ismath' (line 534)
        ismath_146220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 61), 'ismath', False)
        keyword_146221 = ismath_146220
        kwargs_146222 = {'ismath': keyword_146221}
        # Getting the type of 'text_to_path' (line 533)
        text_to_path_146216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 27), 'text_to_path', False)
        # Obtaining the member 'get_text_path' of a type (line 533)
        get_text_path_146217 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 533, 27), text_to_path_146216, 'get_text_path')
        # Calling get_text_path(args, kwargs) (line 533)
        get_text_path_call_result_146223 = invoke(stypy.reporting.localization.Localization(__file__, 533, 27), get_text_path_146217, *[prop_146218, clean_line_146219], **kwargs_146222)
        
        # Assigning a type to the variable 'call_assignment_144791' (line 533)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 533, 12), 'call_assignment_144791', get_text_path_call_result_146223)
        
        # Assigning a Call to a Name (line 533):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_146226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 533, 12), 'int')
        # Processing the call keyword arguments
        kwargs_146227 = {}
        # Getting the type of 'call_assignment_144791' (line 533)
        call_assignment_144791_146224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 12), 'call_assignment_144791', False)
        # Obtaining the member '__getitem__' of a type (line 533)
        getitem___146225 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 533, 12), call_assignment_144791_146224, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_146228 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___146225, *[int_146226], **kwargs_146227)
        
        # Assigning a type to the variable 'call_assignment_144792' (line 533)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 533, 12), 'call_assignment_144792', getitem___call_result_146228)
        
        # Assigning a Name to a Name (line 533):
        # Getting the type of 'call_assignment_144792' (line 533)
        call_assignment_144792_146229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 12), 'call_assignment_144792')
        # Assigning a type to the variable 'verts' (line 533)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 533, 12), 'verts', call_assignment_144792_146229)
        
        # Assigning a Call to a Name (line 533):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_146232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 533, 12), 'int')
        # Processing the call keyword arguments
        kwargs_146233 = {}
        # Getting the type of 'call_assignment_144791' (line 533)
        call_assignment_144791_146230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 12), 'call_assignment_144791', False)
        # Obtaining the member '__getitem__' of a type (line 533)
        getitem___146231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 533, 12), call_assignment_144791_146230, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_146234 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___146231, *[int_146232], **kwargs_146233)
        
        # Assigning a type to the variable 'call_assignment_144793' (line 533)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 533, 12), 'call_assignment_144793', getitem___call_result_146234)
        
        # Assigning a Name to a Name (line 533):
        # Getting the type of 'call_assignment_144793' (line 533)
        call_assignment_144793_146235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 12), 'call_assignment_144793')
        # Assigning a type to the variable 'codes' (line 533)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 533, 19), 'codes', call_assignment_144793_146235)
        # SSA join for if statement (line 529)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 536)
        tuple_146236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 536, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 536)
        # Adding element type (line 536)
        # Getting the type of 'verts' (line 536)
        verts_146237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 15), 'verts')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 536, 15), tuple_146236, verts_146237)
        # Adding element type (line 536)
        # Getting the type of 'codes' (line 536)
        codes_146238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 22), 'codes')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 536, 15), tuple_146236, codes_146238)
        
        # Assigning a type to the variable 'stypy_return_type' (line 536)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 536, 8), 'stypy_return_type', tuple_146236)
        
        # ################# End of 'text_get_vertices_codes(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'text_get_vertices_codes' in the type store
        # Getting the type of 'stypy_return_type' (line 522)
        stypy_return_type_146239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_146239)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'text_get_vertices_codes'
        return stypy_return_type_146239


# Assigning a type to the variable 'TextPath' (line 417)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 417, 0), 'TextPath', TextPath)

# Assigning a Call to a Name (line 483):

# Call to property(...): (line 483)
# Processing the call arguments (line 483)
# Getting the type of 'TextPath'
TextPath_146241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TextPath', False)
# Obtaining the member '_get_vertices' of a type
_get_vertices_146242 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TextPath_146241, '_get_vertices')
# Processing the call keyword arguments (line 483)
kwargs_146243 = {}
# Getting the type of 'property' (line 483)
property_146240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 15), 'property', False)
# Calling property(args, kwargs) (line 483)
property_call_result_146244 = invoke(stypy.reporting.localization.Localization(__file__, 483, 15), property_146240, *[_get_vertices_146242], **kwargs_146243)

# Getting the type of 'TextPath'
TextPath_146245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TextPath')
# Setting the type of the member 'vertices' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TextPath_146245, 'vertices', property_call_result_146244)

# Assigning a Call to a Name (line 484):

# Call to property(...): (line 484)
# Processing the call arguments (line 484)
# Getting the type of 'TextPath'
TextPath_146247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TextPath', False)
# Obtaining the member '_get_codes' of a type
_get_codes_146248 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TextPath_146247, '_get_codes')
# Processing the call keyword arguments (line 484)
kwargs_146249 = {}
# Getting the type of 'property' (line 484)
property_146246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 12), 'property', False)
# Calling property(args, kwargs) (line 484)
property_call_result_146250 = invoke(stypy.reporting.localization.Localization(__file__, 484, 12), property_146246, *[_get_codes_146248], **kwargs_146249)

# Getting the type of 'TextPath'
TextPath_146251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TextPath')
# Setting the type of the member 'codes' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TextPath_146251, 'codes', property_call_result_146250)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
