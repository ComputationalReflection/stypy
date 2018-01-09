
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: A Cairo backend for matplotlib
3: Author: Steve Chaplin
4: 
5: Cairo is a vector graphics library with cross-device output support.
6: Features of Cairo:
7:  * anti-aliasing
8:  * alpha channel
9:  * saves image files as PNG, PostScript, PDF
10: 
11: http://cairographics.org
12: Requires (in order, all available from Cairo website):
13:     cairo, pycairo
14: 
15: Naming Conventions
16:   * classes MixedUpperCase
17:   * varables lowerUpper
18:   * functions underscore_separated
19: '''
20: 
21: from __future__ import (absolute_import, division, print_function,
22:                         unicode_literals)
23: 
24: import six
25: 
26: import gzip
27: import os
28: import sys
29: import warnings
30: 
31: import numpy as np
32: 
33: try:
34:     import cairocffi as cairo
35: except ImportError:
36:     try:
37:         import cairo
38:     except ImportError:
39:         raise ImportError("Cairo backend requires that cairocffi or pycairo "
40:                           "is installed.")
41:     else:
42:         HAS_CAIRO_CFFI = False
43: else:
44:     HAS_CAIRO_CFFI = True
45: 
46: _version_required = (1, 2, 0)
47: if cairo.version_info < _version_required:
48:     raise ImportError("Pycairo %d.%d.%d is installed\n"
49:                       "Pycairo %d.%d.%d or later is required"
50:                       % (cairo.version_info + _version_required))
51: backend_version = cairo.version
52: del _version_required
53: 
54: from matplotlib.backend_bases import (
55:     _Backend, FigureCanvasBase, FigureManagerBase, GraphicsContextBase,
56:     RendererBase)
57: from matplotlib.figure import Figure
58: from matplotlib.mathtext import MathTextParser
59: from matplotlib.path import Path
60: from matplotlib.transforms import Bbox, Affine2D
61: from matplotlib.font_manager import ttfFontProperty
62: 
63: 
64: class ArrayWrapper:
65:     '''Thin wrapper around numpy ndarray to expose the interface
66:        expected by cairocffi. Basically replicates the
67:        array.array interface.
68:     '''
69:     def __init__(self, myarray):
70:         self.__array = myarray
71:         self.__data = myarray.ctypes.data
72:         self.__size = len(myarray.flatten())
73:         self.itemsize = myarray.itemsize
74: 
75:     def buffer_info(self):
76:         return (self.__data, self.__size)
77: 
78: 
79: class RendererCairo(RendererBase):
80:     fontweights = {
81:         100          : cairo.FONT_WEIGHT_NORMAL,
82:         200          : cairo.FONT_WEIGHT_NORMAL,
83:         300          : cairo.FONT_WEIGHT_NORMAL,
84:         400          : cairo.FONT_WEIGHT_NORMAL,
85:         500          : cairo.FONT_WEIGHT_NORMAL,
86:         600          : cairo.FONT_WEIGHT_BOLD,
87:         700          : cairo.FONT_WEIGHT_BOLD,
88:         800          : cairo.FONT_WEIGHT_BOLD,
89:         900          : cairo.FONT_WEIGHT_BOLD,
90:         'ultralight' : cairo.FONT_WEIGHT_NORMAL,
91:         'light'      : cairo.FONT_WEIGHT_NORMAL,
92:         'normal'     : cairo.FONT_WEIGHT_NORMAL,
93:         'medium'     : cairo.FONT_WEIGHT_NORMAL,
94:         'regular'    : cairo.FONT_WEIGHT_NORMAL,
95:         'semibold'   : cairo.FONT_WEIGHT_BOLD,
96:         'bold'       : cairo.FONT_WEIGHT_BOLD,
97:         'heavy'      : cairo.FONT_WEIGHT_BOLD,
98:         'ultrabold'  : cairo.FONT_WEIGHT_BOLD,
99:         'black'      : cairo.FONT_WEIGHT_BOLD,
100:                    }
101:     fontangles = {
102:         'italic'  : cairo.FONT_SLANT_ITALIC,
103:         'normal'  : cairo.FONT_SLANT_NORMAL,
104:         'oblique' : cairo.FONT_SLANT_OBLIQUE,
105:         }
106: 
107: 
108:     def __init__(self, dpi):
109:         self.dpi = dpi
110:         self.gc = GraphicsContextCairo(renderer=self)
111:         self.text_ctx = cairo.Context(
112:            cairo.ImageSurface(cairo.FORMAT_ARGB32, 1, 1))
113:         self.mathtext_parser = MathTextParser('Cairo')
114:         RendererBase.__init__(self)
115: 
116:     def set_ctx_from_surface(self, surface):
117:         self.gc.ctx = cairo.Context(surface)
118: 
119:     def set_width_height(self, width, height):
120:         self.width  = width
121:         self.height = height
122:         self.matrix_flipy = cairo.Matrix(yy=-1, y0=self.height)
123:         # use matrix_flipy for ALL rendering?
124:         # - problem with text? - will need to switch matrix_flipy off, or do a
125:         # font transform?
126: 
127:     def _fill_and_stroke(self, ctx, fill_c, alpha, alpha_overrides):
128:         if fill_c is not None:
129:             ctx.save()
130:             if len(fill_c) == 3 or alpha_overrides:
131:                 ctx.set_source_rgba(fill_c[0], fill_c[1], fill_c[2], alpha)
132:             else:
133:                 ctx.set_source_rgba(fill_c[0], fill_c[1], fill_c[2], fill_c[3])
134:             ctx.fill_preserve()
135:             ctx.restore()
136:         ctx.stroke()
137: 
138:     @staticmethod
139:     def convert_path(ctx, path, transform, clip=None):
140:         for points, code in path.iter_segments(transform, clip=clip):
141:             if code == Path.MOVETO:
142:                 ctx.move_to(*points)
143:             elif code == Path.CLOSEPOLY:
144:                 ctx.close_path()
145:             elif code == Path.LINETO:
146:                 ctx.line_to(*points)
147:             elif code == Path.CURVE3:
148:                 ctx.curve_to(points[0], points[1],
149:                              points[0], points[1],
150:                              points[2], points[3])
151:             elif code == Path.CURVE4:
152:                 ctx.curve_to(*points)
153: 
154:     def draw_path(self, gc, path, transform, rgbFace=None):
155:         ctx = gc.ctx
156: 
157:         # We'll clip the path to the actual rendering extents
158:         # if the path isn't filled.
159:         if rgbFace is None and gc.get_hatch() is None:
160:             clip = ctx.clip_extents()
161:         else:
162:             clip = None
163: 
164:         transform = (transform
165:                      + Affine2D().scale(1.0, -1.0).translate(0, self.height))
166: 
167:         ctx.new_path()
168:         self.convert_path(ctx, path, transform, clip)
169: 
170:         self._fill_and_stroke(
171:             ctx, rgbFace, gc.get_alpha(), gc.get_forced_alpha())
172: 
173:     def draw_markers(
174:             self, gc, marker_path, marker_trans, path, transform, rgbFace=None):
175:         ctx = gc.ctx
176: 
177:         ctx.new_path()
178:         # Create the path for the marker; it needs to be flipped here already!
179:         self.convert_path(
180:             ctx, marker_path, marker_trans + Affine2D().scale(1.0, -1.0))
181:         marker_path = ctx.copy_path_flat()
182: 
183:         # Figure out whether the path has a fill
184:         x1, y1, x2, y2 = ctx.fill_extents()
185:         if x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0:
186:             filled = False
187:             # No fill, just unset this (so we don't try to fill it later on)
188:             rgbFace = None
189:         else:
190:             filled = True
191: 
192:         transform = (transform
193:                      + Affine2D().scale(1.0, -1.0).translate(0, self.height))
194: 
195:         ctx.new_path()
196:         for i, (vertices, codes) in enumerate(
197:                 path.iter_segments(transform, simplify=False)):
198:             if len(vertices):
199:                 x, y = vertices[-2:]
200:                 ctx.save()
201: 
202:                 # Translate and apply path
203:                 ctx.translate(x, y)
204:                 ctx.append_path(marker_path)
205: 
206:                 ctx.restore()
207: 
208:                 # Slower code path if there is a fill; we need to draw
209:                 # the fill and stroke for each marker at the same time.
210:                 # Also flush out the drawing every once in a while to
211:                 # prevent the paths from getting way too long.
212:                 if filled or i % 1000 == 0:
213:                     self._fill_and_stroke(
214:                         ctx, rgbFace, gc.get_alpha(), gc.get_forced_alpha())
215: 
216:         # Fast path, if there is no fill, draw everything in one step
217:         if not filled:
218:             self._fill_and_stroke(
219:                 ctx, rgbFace, gc.get_alpha(), gc.get_forced_alpha())
220: 
221:     def draw_image(self, gc, x, y, im):
222:         # bbox - not currently used
223:         if sys.byteorder == 'little':
224:             im = im[:, :, (2, 1, 0, 3)]
225:         else:
226:             im = im[:, :, (3, 0, 1, 2)]
227:         if HAS_CAIRO_CFFI:
228:             # cairocffi tries to use the buffer_info from array.array
229:             # that we replicate in ArrayWrapper and alternatively falls back
230:             # on ctypes to get a pointer to the numpy array. This works
231:             # correctly on a numpy array in python3 but not 2.7. We replicate
232:             # the array.array functionality here to get cross version support.
233:             imbuffer = ArrayWrapper(im.flatten())
234:         else:
235:             # py2cairo uses PyObject_AsWriteBuffer
236:             # to get a pointer to the numpy array this works correctly
237:             # on a regular numpy array but not on a memory view.
238:             # At the time of writing the latest release version of
239:             # py3cairo still does not support create_for_data
240:             imbuffer = im.flatten()
241:         surface = cairo.ImageSurface.create_for_data(
242:             imbuffer, cairo.FORMAT_ARGB32,
243:             im.shape[1], im.shape[0], im.shape[1]*4)
244:         ctx = gc.ctx
245:         y = self.height - y - im.shape[0]
246: 
247:         ctx.save()
248:         ctx.set_source_surface(surface, float(x), float(y))
249:         if gc.get_alpha() != 1.0:
250:             ctx.paint_with_alpha(gc.get_alpha())
251:         else:
252:             ctx.paint()
253:         ctx.restore()
254: 
255:     def draw_text(self, gc, x, y, s, prop, angle, ismath=False, mtext=None):
256:         # Note: x,y are device/display coords, not user-coords, unlike other
257:         # draw_* methods
258:         if ismath:
259:             self._draw_mathtext(gc, x, y, s, prop, angle)
260: 
261:         else:
262:             ctx = gc.ctx
263:             ctx.new_path()
264:             ctx.move_to(x, y)
265:             ctx.select_font_face(prop.get_name(),
266:                                  self.fontangles[prop.get_style()],
267:                                  self.fontweights[prop.get_weight()])
268: 
269:             size = prop.get_size_in_points() * self.dpi / 72.0
270: 
271:             ctx.save()
272:             if angle:
273:                 ctx.rotate(np.deg2rad(-angle))
274:             ctx.set_font_size(size)
275: 
276:             if HAS_CAIRO_CFFI:
277:                 if not isinstance(s, six.text_type):
278:                     s = six.text_type(s)
279:             else:
280:                 if not six.PY3 and isinstance(s, six.text_type):
281:                     s = s.encode("utf-8")
282: 
283:             ctx.show_text(s)
284:             ctx.restore()
285: 
286:     def _draw_mathtext(self, gc, x, y, s, prop, angle):
287:         ctx = gc.ctx
288:         width, height, descent, glyphs, rects = self.mathtext_parser.parse(
289:             s, self.dpi, prop)
290: 
291:         ctx.save()
292:         ctx.translate(x, y)
293:         if angle:
294:             ctx.rotate(np.deg2rad(-angle))
295: 
296:         for font, fontsize, s, ox, oy in glyphs:
297:             ctx.new_path()
298:             ctx.move_to(ox, oy)
299: 
300:             fontProp = ttfFontProperty(font)
301:             ctx.save()
302:             ctx.select_font_face(fontProp.name,
303:                                  self.fontangles[fontProp.style],
304:                                  self.fontweights[fontProp.weight])
305: 
306:             size = fontsize * self.dpi / 72.0
307:             ctx.set_font_size(size)
308:             if not six.PY3 and isinstance(s, six.text_type):
309:                 s = s.encode("utf-8")
310:             ctx.show_text(s)
311:             ctx.restore()
312: 
313:         for ox, oy, w, h in rects:
314:             ctx.new_path()
315:             ctx.rectangle(ox, oy, w, h)
316:             ctx.set_source_rgb(0, 0, 0)
317:             ctx.fill_preserve()
318: 
319:         ctx.restore()
320: 
321:     def flipy(self):
322:         return True
323:         #return False # tried - all draw objects ok except text (and images?)
324:         # which comes out mirrored!
325: 
326:     def get_canvas_width_height(self):
327:         return self.width, self.height
328: 
329:     def get_text_width_height_descent(self, s, prop, ismath):
330:         if ismath:
331:             width, height, descent, fonts, used_characters = self.mathtext_parser.parse(
332:                s, self.dpi, prop)
333:             return width, height, descent
334: 
335:         ctx = self.text_ctx
336:         ctx.save()
337:         ctx.select_font_face(prop.get_name(),
338:                              self.fontangles[prop.get_style()],
339:                              self.fontweights[prop.get_weight()])
340: 
341:         # Cairo (says it) uses 1/96 inch user space units, ref: cairo_gstate.c
342:         # but if /96.0 is used the font is too small
343:         size = prop.get_size_in_points() * self.dpi / 72
344: 
345:         # problem - scale remembers last setting and font can become
346:         # enormous causing program to crash
347:         # save/restore prevents the problem
348:         ctx.set_font_size(size)
349: 
350:         y_bearing, w, h = ctx.text_extents(s)[1:4]
351:         ctx.restore()
352: 
353:         return w, h, h + y_bearing
354: 
355:     def new_gc(self):
356:         self.gc.ctx.save()
357:         self.gc._alpha = 1.0
358:         self.gc._forced_alpha = False # if True, _alpha overrides A from RGBA
359:         return self.gc
360: 
361:     def points_to_pixels(self, points):
362:         return points / 72 * self.dpi
363: 
364: 
365: class GraphicsContextCairo(GraphicsContextBase):
366:     _joind = {
367:         'bevel' : cairo.LINE_JOIN_BEVEL,
368:         'miter' : cairo.LINE_JOIN_MITER,
369:         'round' : cairo.LINE_JOIN_ROUND,
370:         }
371: 
372:     _capd = {
373:         'butt'       : cairo.LINE_CAP_BUTT,
374:         'projecting' : cairo.LINE_CAP_SQUARE,
375:         'round'      : cairo.LINE_CAP_ROUND,
376:         }
377: 
378:     def __init__(self, renderer):
379:         GraphicsContextBase.__init__(self)
380:         self.renderer = renderer
381: 
382:     def restore(self):
383:         self.ctx.restore()
384: 
385:     def set_alpha(self, alpha):
386:         GraphicsContextBase.set_alpha(self, alpha)
387:         _alpha = self.get_alpha()
388:         rgb = self._rgb
389:         if self.get_forced_alpha():
390:             self.ctx.set_source_rgba(rgb[0], rgb[1], rgb[2], _alpha)
391:         else:
392:             self.ctx.set_source_rgba(rgb[0], rgb[1], rgb[2], rgb[3])
393: 
394:     #def set_antialiased(self, b):
395:         # enable/disable anti-aliasing is not (yet) supported by Cairo
396: 
397:     def set_capstyle(self, cs):
398:         if cs in ('butt', 'round', 'projecting'):
399:             self._capstyle = cs
400:             self.ctx.set_line_cap(self._capd[cs])
401:         else:
402:             raise ValueError('Unrecognized cap style.  Found %s' % cs)
403: 
404:     def set_clip_rectangle(self, rectangle):
405:         if not rectangle:
406:             return
407:         x, y, w, h = rectangle.bounds
408:         # pixel-aligned clip-regions are faster
409:         x,y,w,h = np.round(x), np.round(y), np.round(w), np.round(h)
410:         ctx = self.ctx
411:         ctx.new_path()
412:         ctx.rectangle(x, self.renderer.height - h - y, w, h)
413:         ctx.clip()
414: 
415:     def set_clip_path(self, path):
416:         if not path:
417:             return
418:         tpath, affine = path.get_transformed_path_and_affine()
419:         ctx = self.ctx
420:         ctx.new_path()
421:         affine = (affine
422:                   + Affine2D().scale(1, -1).translate(0, self.renderer.height))
423:         RendererCairo.convert_path(ctx, tpath, affine)
424:         ctx.clip()
425: 
426:     def set_dashes(self, offset, dashes):
427:         self._dashes = offset, dashes
428:         if dashes == None:
429:             self.ctx.set_dash([], 0)  # switch dashes off
430:         else:
431:             self.ctx.set_dash(
432:                 list(self.renderer.points_to_pixels(np.asarray(dashes))),
433:                 offset)
434: 
435:     def set_foreground(self, fg, isRGBA=None):
436:         GraphicsContextBase.set_foreground(self, fg, isRGBA)
437:         if len(self._rgb) == 3:
438:             self.ctx.set_source_rgb(*self._rgb)
439:         else:
440:             self.ctx.set_source_rgba(*self._rgb)
441: 
442:     def get_rgb(self):
443:         return self.ctx.get_source().get_rgba()[:3]
444: 
445:     def set_joinstyle(self, js):
446:         if js in ('miter', 'round', 'bevel'):
447:             self._joinstyle = js
448:             self.ctx.set_line_join(self._joind[js])
449:         else:
450:             raise ValueError('Unrecognized join style.  Found %s' % js)
451: 
452:     def set_linewidth(self, w):
453:         self._linewidth = float(w)
454:         self.ctx.set_line_width(self.renderer.points_to_pixels(w))
455: 
456: 
457: class FigureCanvasCairo(FigureCanvasBase):
458:     def print_png(self, fobj, *args, **kwargs):
459:         width, height = self.get_width_height()
460: 
461:         renderer = RendererCairo(self.figure.dpi)
462:         renderer.set_width_height(width, height)
463:         surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
464:         renderer.set_ctx_from_surface(surface)
465: 
466:         self.figure.draw(renderer)
467:         surface.write_to_png(fobj)
468: 
469:     def print_pdf(self, fobj, *args, **kwargs):
470:         return self._save(fobj, 'pdf', *args, **kwargs)
471: 
472:     def print_ps(self, fobj, *args, **kwargs):
473:         return self._save(fobj, 'ps', *args, **kwargs)
474: 
475:     def print_svg(self, fobj, *args, **kwargs):
476:         return self._save(fobj, 'svg', *args, **kwargs)
477: 
478:     def print_svgz(self, fobj, *args, **kwargs):
479:         return self._save(fobj, 'svgz', *args, **kwargs)
480: 
481:     def _save(self, fo, fmt, **kwargs):
482:         # save PDF/PS/SVG
483:         orientation = kwargs.get('orientation', 'portrait')
484: 
485:         dpi = 72
486:         self.figure.dpi = dpi
487:         w_in, h_in = self.figure.get_size_inches()
488:         width_in_points, height_in_points = w_in * dpi, h_in * dpi
489: 
490:         if orientation == 'landscape':
491:             width_in_points, height_in_points = (
492:                 height_in_points, width_in_points)
493: 
494:         if fmt == 'ps':
495:             if not hasattr(cairo, 'PSSurface'):
496:                 raise RuntimeError('cairo has not been compiled with PS '
497:                                    'support enabled')
498:             surface = cairo.PSSurface(fo, width_in_points, height_in_points)
499:         elif fmt == 'pdf':
500:             if not hasattr(cairo, 'PDFSurface'):
501:                 raise RuntimeError('cairo has not been compiled with PDF '
502:                                    'support enabled')
503:             surface = cairo.PDFSurface(fo, width_in_points, height_in_points)
504:         elif fmt in ('svg', 'svgz'):
505:             if not hasattr(cairo, 'SVGSurface'):
506:                 raise RuntimeError('cairo has not been compiled with SVG '
507:                                    'support enabled')
508:             if fmt == 'svgz':
509:                 if isinstance(fo, six.string_types):
510:                     fo = gzip.GzipFile(fo, 'wb')
511:                 else:
512:                     fo = gzip.GzipFile(None, 'wb', fileobj=fo)
513:             surface = cairo.SVGSurface(fo, width_in_points, height_in_points)
514:         else:
515:             warnings.warn("unknown format: %s" % fmt)
516:             return
517: 
518:         # surface.set_dpi() can be used
519:         renderer = RendererCairo(self.figure.dpi)
520:         renderer.set_width_height(width_in_points, height_in_points)
521:         renderer.set_ctx_from_surface(surface)
522:         ctx = renderer.gc.ctx
523: 
524:         if orientation == 'landscape':
525:             ctx.rotate(np.pi/2)
526:             ctx.translate(0, -height_in_points)
527:             # cairo/src/cairo_ps_surface.c
528:             # '%%Orientation: Portrait' is always written to the file header
529:             # '%%Orientation: Landscape' would possibly cause problems
530:             # since some printers would rotate again ?
531:             # TODO:
532:             # add portrait/landscape checkbox to FileChooser
533: 
534:         self.figure.draw(renderer)
535: 
536:         ctx.show_page()
537:         surface.finish()
538:         if fmt == 'svgz':
539:             fo.close()
540: 
541: 
542: @_Backend.export
543: class _BackendCairo(_Backend):
544:     FigureCanvas = FigureCanvasCairo
545:     FigureManager = FigureManagerBase
546: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

unicode_219664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, (-1)), 'unicode', u'\nA Cairo backend for matplotlib\nAuthor: Steve Chaplin\n\nCairo is a vector graphics library with cross-device output support.\nFeatures of Cairo:\n * anti-aliasing\n * alpha channel\n * saves image files as PNG, PostScript, PDF\n\nhttp://cairographics.org\nRequires (in order, all available from Cairo website):\n    cairo, pycairo\n\nNaming Conventions\n  * classes MixedUpperCase\n  * varables lowerUpper\n  * functions underscore_separated\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 24, 0))

# 'import six' statement (line 24)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_219665 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'six')

if (type(import_219665) is not StypyTypeError):

    if (import_219665 != 'pyd_module'):
        __import__(import_219665)
        sys_modules_219666 = sys.modules[import_219665]
        import_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'six', sys_modules_219666.module_type_store, module_type_store)
    else:
        import six

        import_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'six', six, module_type_store)

else:
    # Assigning a type to the variable 'six' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'six', import_219665)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 26, 0))

# 'import gzip' statement (line 26)
import gzip

import_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'gzip', gzip, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 27, 0))

# 'import os' statement (line 27)
import os

import_module(stypy.reporting.localization.Localization(__file__, 27, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 28, 0))

# 'import sys' statement (line 28)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 28, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 29, 0))

# 'import warnings' statement (line 29)
import warnings

import_module(stypy.reporting.localization.Localization(__file__, 29, 0), 'warnings', warnings, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 31, 0))

# 'import numpy' statement (line 31)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_219667 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 31, 0), 'numpy')

if (type(import_219667) is not StypyTypeError):

    if (import_219667 != 'pyd_module'):
        __import__(import_219667)
        sys_modules_219668 = sys.modules[import_219667]
        import_module(stypy.reporting.localization.Localization(__file__, 31, 0), 'np', sys_modules_219668.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 31, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 0), 'numpy', import_219667)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')



# SSA begins for try-except statement (line 33)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 34, 4))

# 'import cairocffi' statement (line 34)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_219669 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 34, 4), 'cairocffi')

if (type(import_219669) is not StypyTypeError):

    if (import_219669 != 'pyd_module'):
        __import__(import_219669)
        sys_modules_219670 = sys.modules[import_219669]
        import_module(stypy.reporting.localization.Localization(__file__, 34, 4), 'cairo', sys_modules_219670.module_type_store, module_type_store)
    else:
        import cairocffi as cairo

        import_module(stypy.reporting.localization.Localization(__file__, 34, 4), 'cairo', cairocffi, module_type_store)

else:
    # Assigning a type to the variable 'cairocffi' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'cairocffi', import_219669)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

# SSA branch for the except part of a try statement (line 33)
# SSA branch for the except 'ImportError' branch of a try statement (line 33)
module_type_store.open_ssa_branch('except')


# SSA begins for try-except statement (line 36)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 37, 8))

# 'import cairo' statement (line 37)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_219671 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 37, 8), 'cairo')

if (type(import_219671) is not StypyTypeError):

    if (import_219671 != 'pyd_module'):
        __import__(import_219671)
        sys_modules_219672 = sys.modules[import_219671]
        import_module(stypy.reporting.localization.Localization(__file__, 37, 8), 'cairo', sys_modules_219672.module_type_store, module_type_store)
    else:
        import cairo

        import_module(stypy.reporting.localization.Localization(__file__, 37, 8), 'cairo', cairo, module_type_store)

else:
    # Assigning a type to the variable 'cairo' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'cairo', import_219671)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

# SSA branch for the except part of a try statement (line 36)
# SSA branch for the except 'ImportError' branch of a try statement (line 36)
module_type_store.open_ssa_branch('except')

# Call to ImportError(...): (line 39)
# Processing the call arguments (line 39)
unicode_219674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 26), 'unicode', u'Cairo backend requires that cairocffi or pycairo is installed.')
# Processing the call keyword arguments (line 39)
kwargs_219675 = {}
# Getting the type of 'ImportError' (line 39)
ImportError_219673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 14), 'ImportError', False)
# Calling ImportError(args, kwargs) (line 39)
ImportError_call_result_219676 = invoke(stypy.reporting.localization.Localization(__file__, 39, 14), ImportError_219673, *[unicode_219674], **kwargs_219675)

ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 39, 8), ImportError_call_result_219676, 'raise parameter', BaseException)
# SSA branch for the else branch of a try statement (line 36)
module_type_store.open_ssa_branch('except else')

# Assigning a Name to a Name (line 42):

# Assigning a Name to a Name (line 42):
# Getting the type of 'False' (line 42)
False_219677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 25), 'False')
# Assigning a type to the variable 'HAS_CAIRO_CFFI' (line 42)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'HAS_CAIRO_CFFI', False_219677)
# SSA join for try-except statement (line 36)
module_type_store = module_type_store.join_ssa_context()

# SSA branch for the else branch of a try statement (line 33)
module_type_store.open_ssa_branch('except else')

# Assigning a Name to a Name (line 44):

# Assigning a Name to a Name (line 44):
# Getting the type of 'True' (line 44)
True_219678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 21), 'True')
# Assigning a type to the variable 'HAS_CAIRO_CFFI' (line 44)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'HAS_CAIRO_CFFI', True_219678)
# SSA join for try-except statement (line 33)
module_type_store = module_type_store.join_ssa_context()


# Assigning a Tuple to a Name (line 46):

# Assigning a Tuple to a Name (line 46):

# Obtaining an instance of the builtin type 'tuple' (line 46)
tuple_219679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 21), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 46)
# Adding element type (line 46)
int_219680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 21), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 21), tuple_219679, int_219680)
# Adding element type (line 46)
int_219681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 24), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 21), tuple_219679, int_219681)
# Adding element type (line 46)
int_219682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 27), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 21), tuple_219679, int_219682)

# Assigning a type to the variable '_version_required' (line 46)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 0), '_version_required', tuple_219679)


# Getting the type of 'cairo' (line 47)
cairo_219683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 3), 'cairo')
# Obtaining the member 'version_info' of a type (line 47)
version_info_219684 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 3), cairo_219683, 'version_info')
# Getting the type of '_version_required' (line 47)
_version_required_219685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 24), '_version_required')
# Applying the binary operator '<' (line 47)
result_lt_219686 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 3), '<', version_info_219684, _version_required_219685)

# Testing the type of an if condition (line 47)
if_condition_219687 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 47, 0), result_lt_219686)
# Assigning a type to the variable 'if_condition_219687' (line 47)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 0), 'if_condition_219687', if_condition_219687)
# SSA begins for if statement (line 47)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Call to ImportError(...): (line 48)
# Processing the call arguments (line 48)
unicode_219689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 22), 'unicode', u'Pycairo %d.%d.%d is installed\nPycairo %d.%d.%d or later is required')
# Getting the type of 'cairo' (line 50)
cairo_219690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 25), 'cairo', False)
# Obtaining the member 'version_info' of a type (line 50)
version_info_219691 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 25), cairo_219690, 'version_info')
# Getting the type of '_version_required' (line 50)
_version_required_219692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 46), '_version_required', False)
# Applying the binary operator '+' (line 50)
result_add_219693 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 25), '+', version_info_219691, _version_required_219692)

# Applying the binary operator '%' (line 48)
result_mod_219694 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 22), '%', unicode_219689, result_add_219693)

# Processing the call keyword arguments (line 48)
kwargs_219695 = {}
# Getting the type of 'ImportError' (line 48)
ImportError_219688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 10), 'ImportError', False)
# Calling ImportError(args, kwargs) (line 48)
ImportError_call_result_219696 = invoke(stypy.reporting.localization.Localization(__file__, 48, 10), ImportError_219688, *[result_mod_219694], **kwargs_219695)

ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 48, 4), ImportError_call_result_219696, 'raise parameter', BaseException)
# SSA join for if statement (line 47)
module_type_store = module_type_store.join_ssa_context()


# Assigning a Attribute to a Name (line 51):

# Assigning a Attribute to a Name (line 51):
# Getting the type of 'cairo' (line 51)
cairo_219697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 18), 'cairo')
# Obtaining the member 'version' of a type (line 51)
version_219698 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 18), cairo_219697, 'version')
# Assigning a type to the variable 'backend_version' (line 51)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 0), 'backend_version', version_219698)
# Deleting a member
module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 52, 0), module_type_store, '_version_required')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 54, 0))

# 'from matplotlib.backend_bases import _Backend, FigureCanvasBase, FigureManagerBase, GraphicsContextBase, RendererBase' statement (line 54)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_219699 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 54, 0), 'matplotlib.backend_bases')

if (type(import_219699) is not StypyTypeError):

    if (import_219699 != 'pyd_module'):
        __import__(import_219699)
        sys_modules_219700 = sys.modules[import_219699]
        import_from_module(stypy.reporting.localization.Localization(__file__, 54, 0), 'matplotlib.backend_bases', sys_modules_219700.module_type_store, module_type_store, ['_Backend', 'FigureCanvasBase', 'FigureManagerBase', 'GraphicsContextBase', 'RendererBase'])
        nest_module(stypy.reporting.localization.Localization(__file__, 54, 0), __file__, sys_modules_219700, sys_modules_219700.module_type_store, module_type_store)
    else:
        from matplotlib.backend_bases import _Backend, FigureCanvasBase, FigureManagerBase, GraphicsContextBase, RendererBase

        import_from_module(stypy.reporting.localization.Localization(__file__, 54, 0), 'matplotlib.backend_bases', None, module_type_store, ['_Backend', 'FigureCanvasBase', 'FigureManagerBase', 'GraphicsContextBase', 'RendererBase'], [_Backend, FigureCanvasBase, FigureManagerBase, GraphicsContextBase, RendererBase])

else:
    # Assigning a type to the variable 'matplotlib.backend_bases' (line 54)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 0), 'matplotlib.backend_bases', import_219699)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 57, 0))

# 'from matplotlib.figure import Figure' statement (line 57)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_219701 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 57, 0), 'matplotlib.figure')

if (type(import_219701) is not StypyTypeError):

    if (import_219701 != 'pyd_module'):
        __import__(import_219701)
        sys_modules_219702 = sys.modules[import_219701]
        import_from_module(stypy.reporting.localization.Localization(__file__, 57, 0), 'matplotlib.figure', sys_modules_219702.module_type_store, module_type_store, ['Figure'])
        nest_module(stypy.reporting.localization.Localization(__file__, 57, 0), __file__, sys_modules_219702, sys_modules_219702.module_type_store, module_type_store)
    else:
        from matplotlib.figure import Figure

        import_from_module(stypy.reporting.localization.Localization(__file__, 57, 0), 'matplotlib.figure', None, module_type_store, ['Figure'], [Figure])

else:
    # Assigning a type to the variable 'matplotlib.figure' (line 57)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 0), 'matplotlib.figure', import_219701)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 58, 0))

# 'from matplotlib.mathtext import MathTextParser' statement (line 58)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_219703 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 58, 0), 'matplotlib.mathtext')

if (type(import_219703) is not StypyTypeError):

    if (import_219703 != 'pyd_module'):
        __import__(import_219703)
        sys_modules_219704 = sys.modules[import_219703]
        import_from_module(stypy.reporting.localization.Localization(__file__, 58, 0), 'matplotlib.mathtext', sys_modules_219704.module_type_store, module_type_store, ['MathTextParser'])
        nest_module(stypy.reporting.localization.Localization(__file__, 58, 0), __file__, sys_modules_219704, sys_modules_219704.module_type_store, module_type_store)
    else:
        from matplotlib.mathtext import MathTextParser

        import_from_module(stypy.reporting.localization.Localization(__file__, 58, 0), 'matplotlib.mathtext', None, module_type_store, ['MathTextParser'], [MathTextParser])

else:
    # Assigning a type to the variable 'matplotlib.mathtext' (line 58)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 0), 'matplotlib.mathtext', import_219703)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 59, 0))

# 'from matplotlib.path import Path' statement (line 59)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_219705 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 59, 0), 'matplotlib.path')

if (type(import_219705) is not StypyTypeError):

    if (import_219705 != 'pyd_module'):
        __import__(import_219705)
        sys_modules_219706 = sys.modules[import_219705]
        import_from_module(stypy.reporting.localization.Localization(__file__, 59, 0), 'matplotlib.path', sys_modules_219706.module_type_store, module_type_store, ['Path'])
        nest_module(stypy.reporting.localization.Localization(__file__, 59, 0), __file__, sys_modules_219706, sys_modules_219706.module_type_store, module_type_store)
    else:
        from matplotlib.path import Path

        import_from_module(stypy.reporting.localization.Localization(__file__, 59, 0), 'matplotlib.path', None, module_type_store, ['Path'], [Path])

else:
    # Assigning a type to the variable 'matplotlib.path' (line 59)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 0), 'matplotlib.path', import_219705)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 60, 0))

# 'from matplotlib.transforms import Bbox, Affine2D' statement (line 60)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_219707 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 60, 0), 'matplotlib.transforms')

if (type(import_219707) is not StypyTypeError):

    if (import_219707 != 'pyd_module'):
        __import__(import_219707)
        sys_modules_219708 = sys.modules[import_219707]
        import_from_module(stypy.reporting.localization.Localization(__file__, 60, 0), 'matplotlib.transforms', sys_modules_219708.module_type_store, module_type_store, ['Bbox', 'Affine2D'])
        nest_module(stypy.reporting.localization.Localization(__file__, 60, 0), __file__, sys_modules_219708, sys_modules_219708.module_type_store, module_type_store)
    else:
        from matplotlib.transforms import Bbox, Affine2D

        import_from_module(stypy.reporting.localization.Localization(__file__, 60, 0), 'matplotlib.transforms', None, module_type_store, ['Bbox', 'Affine2D'], [Bbox, Affine2D])

else:
    # Assigning a type to the variable 'matplotlib.transforms' (line 60)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 0), 'matplotlib.transforms', import_219707)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 61, 0))

# 'from matplotlib.font_manager import ttfFontProperty' statement (line 61)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_219709 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 61, 0), 'matplotlib.font_manager')

if (type(import_219709) is not StypyTypeError):

    if (import_219709 != 'pyd_module'):
        __import__(import_219709)
        sys_modules_219710 = sys.modules[import_219709]
        import_from_module(stypy.reporting.localization.Localization(__file__, 61, 0), 'matplotlib.font_manager', sys_modules_219710.module_type_store, module_type_store, ['ttfFontProperty'])
        nest_module(stypy.reporting.localization.Localization(__file__, 61, 0), __file__, sys_modules_219710, sys_modules_219710.module_type_store, module_type_store)
    else:
        from matplotlib.font_manager import ttfFontProperty

        import_from_module(stypy.reporting.localization.Localization(__file__, 61, 0), 'matplotlib.font_manager', None, module_type_store, ['ttfFontProperty'], [ttfFontProperty])

else:
    # Assigning a type to the variable 'matplotlib.font_manager' (line 61)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 0), 'matplotlib.font_manager', import_219709)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

# Declaration of the 'ArrayWrapper' class

class ArrayWrapper:
    unicode_219711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, (-1)), 'unicode', u'Thin wrapper around numpy ndarray to expose the interface\n       expected by cairocffi. Basically replicates the\n       array.array interface.\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 69, 4, False)
        # Assigning a type to the variable 'self' (line 70)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ArrayWrapper.__init__', ['myarray'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['myarray'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 70):
        
        # Assigning a Name to a Attribute (line 70):
        # Getting the type of 'myarray' (line 70)
        myarray_219712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 23), 'myarray')
        # Getting the type of 'self' (line 70)
        self_219713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'self')
        # Setting the type of the member '__array' of a type (line 70)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 8), self_219713, '__array', myarray_219712)
        
        # Assigning a Attribute to a Attribute (line 71):
        
        # Assigning a Attribute to a Attribute (line 71):
        # Getting the type of 'myarray' (line 71)
        myarray_219714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 22), 'myarray')
        # Obtaining the member 'ctypes' of a type (line 71)
        ctypes_219715 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 22), myarray_219714, 'ctypes')
        # Obtaining the member 'data' of a type (line 71)
        data_219716 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 22), ctypes_219715, 'data')
        # Getting the type of 'self' (line 71)
        self_219717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'self')
        # Setting the type of the member '__data' of a type (line 71)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 8), self_219717, '__data', data_219716)
        
        # Assigning a Call to a Attribute (line 72):
        
        # Assigning a Call to a Attribute (line 72):
        
        # Call to len(...): (line 72)
        # Processing the call arguments (line 72)
        
        # Call to flatten(...): (line 72)
        # Processing the call keyword arguments (line 72)
        kwargs_219721 = {}
        # Getting the type of 'myarray' (line 72)
        myarray_219719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 26), 'myarray', False)
        # Obtaining the member 'flatten' of a type (line 72)
        flatten_219720 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 26), myarray_219719, 'flatten')
        # Calling flatten(args, kwargs) (line 72)
        flatten_call_result_219722 = invoke(stypy.reporting.localization.Localization(__file__, 72, 26), flatten_219720, *[], **kwargs_219721)
        
        # Processing the call keyword arguments (line 72)
        kwargs_219723 = {}
        # Getting the type of 'len' (line 72)
        len_219718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 22), 'len', False)
        # Calling len(args, kwargs) (line 72)
        len_call_result_219724 = invoke(stypy.reporting.localization.Localization(__file__, 72, 22), len_219718, *[flatten_call_result_219722], **kwargs_219723)
        
        # Getting the type of 'self' (line 72)
        self_219725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'self')
        # Setting the type of the member '__size' of a type (line 72)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 8), self_219725, '__size', len_call_result_219724)
        
        # Assigning a Attribute to a Attribute (line 73):
        
        # Assigning a Attribute to a Attribute (line 73):
        # Getting the type of 'myarray' (line 73)
        myarray_219726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 24), 'myarray')
        # Obtaining the member 'itemsize' of a type (line 73)
        itemsize_219727 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 24), myarray_219726, 'itemsize')
        # Getting the type of 'self' (line 73)
        self_219728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'self')
        # Setting the type of the member 'itemsize' of a type (line 73)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 8), self_219728, 'itemsize', itemsize_219727)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def buffer_info(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'buffer_info'
        module_type_store = module_type_store.open_function_context('buffer_info', 75, 4, False)
        # Assigning a type to the variable 'self' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ArrayWrapper.buffer_info.__dict__.__setitem__('stypy_localization', localization)
        ArrayWrapper.buffer_info.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ArrayWrapper.buffer_info.__dict__.__setitem__('stypy_type_store', module_type_store)
        ArrayWrapper.buffer_info.__dict__.__setitem__('stypy_function_name', 'ArrayWrapper.buffer_info')
        ArrayWrapper.buffer_info.__dict__.__setitem__('stypy_param_names_list', [])
        ArrayWrapper.buffer_info.__dict__.__setitem__('stypy_varargs_param_name', None)
        ArrayWrapper.buffer_info.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ArrayWrapper.buffer_info.__dict__.__setitem__('stypy_call_defaults', defaults)
        ArrayWrapper.buffer_info.__dict__.__setitem__('stypy_call_varargs', varargs)
        ArrayWrapper.buffer_info.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ArrayWrapper.buffer_info.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ArrayWrapper.buffer_info', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'buffer_info', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'buffer_info(...)' code ##################

        
        # Obtaining an instance of the builtin type 'tuple' (line 76)
        tuple_219729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 16), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 76)
        # Adding element type (line 76)
        # Getting the type of 'self' (line 76)
        self_219730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 16), 'self')
        # Obtaining the member '__data' of a type (line 76)
        data_219731 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 16), self_219730, '__data')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 16), tuple_219729, data_219731)
        # Adding element type (line 76)
        # Getting the type of 'self' (line 76)
        self_219732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 29), 'self')
        # Obtaining the member '__size' of a type (line 76)
        size_219733 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 29), self_219732, '__size')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 16), tuple_219729, size_219733)
        
        # Assigning a type to the variable 'stypy_return_type' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'stypy_return_type', tuple_219729)
        
        # ################# End of 'buffer_info(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'buffer_info' in the type store
        # Getting the type of 'stypy_return_type' (line 75)
        stypy_return_type_219734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_219734)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'buffer_info'
        return stypy_return_type_219734


# Assigning a type to the variable 'ArrayWrapper' (line 64)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 0), 'ArrayWrapper', ArrayWrapper)
# Declaration of the 'RendererCairo' class
# Getting the type of 'RendererBase' (line 79)
RendererBase_219735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 20), 'RendererBase')

class RendererCairo(RendererBase_219735, ):
    
    # Assigning a Dict to a Name (line 80):
    
    # Assigning a Dict to a Name (line 101):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 108, 4, False)
        # Assigning a type to the variable 'self' (line 109)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RendererCairo.__init__', ['dpi'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['dpi'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 109):
        
        # Assigning a Name to a Attribute (line 109):
        # Getting the type of 'dpi' (line 109)
        dpi_219736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 19), 'dpi')
        # Getting the type of 'self' (line 109)
        self_219737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'self')
        # Setting the type of the member 'dpi' of a type (line 109)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 8), self_219737, 'dpi', dpi_219736)
        
        # Assigning a Call to a Attribute (line 110):
        
        # Assigning a Call to a Attribute (line 110):
        
        # Call to GraphicsContextCairo(...): (line 110)
        # Processing the call keyword arguments (line 110)
        # Getting the type of 'self' (line 110)
        self_219739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 48), 'self', False)
        keyword_219740 = self_219739
        kwargs_219741 = {'renderer': keyword_219740}
        # Getting the type of 'GraphicsContextCairo' (line 110)
        GraphicsContextCairo_219738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 18), 'GraphicsContextCairo', False)
        # Calling GraphicsContextCairo(args, kwargs) (line 110)
        GraphicsContextCairo_call_result_219742 = invoke(stypy.reporting.localization.Localization(__file__, 110, 18), GraphicsContextCairo_219738, *[], **kwargs_219741)
        
        # Getting the type of 'self' (line 110)
        self_219743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'self')
        # Setting the type of the member 'gc' of a type (line 110)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 8), self_219743, 'gc', GraphicsContextCairo_call_result_219742)
        
        # Assigning a Call to a Attribute (line 111):
        
        # Assigning a Call to a Attribute (line 111):
        
        # Call to Context(...): (line 111)
        # Processing the call arguments (line 111)
        
        # Call to ImageSurface(...): (line 112)
        # Processing the call arguments (line 112)
        # Getting the type of 'cairo' (line 112)
        cairo_219748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 30), 'cairo', False)
        # Obtaining the member 'FORMAT_ARGB32' of a type (line 112)
        FORMAT_ARGB32_219749 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 30), cairo_219748, 'FORMAT_ARGB32')
        int_219750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 51), 'int')
        int_219751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 54), 'int')
        # Processing the call keyword arguments (line 112)
        kwargs_219752 = {}
        # Getting the type of 'cairo' (line 112)
        cairo_219746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 11), 'cairo', False)
        # Obtaining the member 'ImageSurface' of a type (line 112)
        ImageSurface_219747 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 11), cairo_219746, 'ImageSurface')
        # Calling ImageSurface(args, kwargs) (line 112)
        ImageSurface_call_result_219753 = invoke(stypy.reporting.localization.Localization(__file__, 112, 11), ImageSurface_219747, *[FORMAT_ARGB32_219749, int_219750, int_219751], **kwargs_219752)
        
        # Processing the call keyword arguments (line 111)
        kwargs_219754 = {}
        # Getting the type of 'cairo' (line 111)
        cairo_219744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 24), 'cairo', False)
        # Obtaining the member 'Context' of a type (line 111)
        Context_219745 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 24), cairo_219744, 'Context')
        # Calling Context(args, kwargs) (line 111)
        Context_call_result_219755 = invoke(stypy.reporting.localization.Localization(__file__, 111, 24), Context_219745, *[ImageSurface_call_result_219753], **kwargs_219754)
        
        # Getting the type of 'self' (line 111)
        self_219756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), 'self')
        # Setting the type of the member 'text_ctx' of a type (line 111)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 8), self_219756, 'text_ctx', Context_call_result_219755)
        
        # Assigning a Call to a Attribute (line 113):
        
        # Assigning a Call to a Attribute (line 113):
        
        # Call to MathTextParser(...): (line 113)
        # Processing the call arguments (line 113)
        unicode_219758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 46), 'unicode', u'Cairo')
        # Processing the call keyword arguments (line 113)
        kwargs_219759 = {}
        # Getting the type of 'MathTextParser' (line 113)
        MathTextParser_219757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 31), 'MathTextParser', False)
        # Calling MathTextParser(args, kwargs) (line 113)
        MathTextParser_call_result_219760 = invoke(stypy.reporting.localization.Localization(__file__, 113, 31), MathTextParser_219757, *[unicode_219758], **kwargs_219759)
        
        # Getting the type of 'self' (line 113)
        self_219761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'self')
        # Setting the type of the member 'mathtext_parser' of a type (line 113)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 8), self_219761, 'mathtext_parser', MathTextParser_call_result_219760)
        
        # Call to __init__(...): (line 114)
        # Processing the call arguments (line 114)
        # Getting the type of 'self' (line 114)
        self_219764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 30), 'self', False)
        # Processing the call keyword arguments (line 114)
        kwargs_219765 = {}
        # Getting the type of 'RendererBase' (line 114)
        RendererBase_219762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 8), 'RendererBase', False)
        # Obtaining the member '__init__' of a type (line 114)
        init___219763 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 8), RendererBase_219762, '__init__')
        # Calling __init__(args, kwargs) (line 114)
        init___call_result_219766 = invoke(stypy.reporting.localization.Localization(__file__, 114, 8), init___219763, *[self_219764], **kwargs_219765)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def set_ctx_from_surface(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_ctx_from_surface'
        module_type_store = module_type_store.open_function_context('set_ctx_from_surface', 116, 4, False)
        # Assigning a type to the variable 'self' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        RendererCairo.set_ctx_from_surface.__dict__.__setitem__('stypy_localization', localization)
        RendererCairo.set_ctx_from_surface.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        RendererCairo.set_ctx_from_surface.__dict__.__setitem__('stypy_type_store', module_type_store)
        RendererCairo.set_ctx_from_surface.__dict__.__setitem__('stypy_function_name', 'RendererCairo.set_ctx_from_surface')
        RendererCairo.set_ctx_from_surface.__dict__.__setitem__('stypy_param_names_list', ['surface'])
        RendererCairo.set_ctx_from_surface.__dict__.__setitem__('stypy_varargs_param_name', None)
        RendererCairo.set_ctx_from_surface.__dict__.__setitem__('stypy_kwargs_param_name', None)
        RendererCairo.set_ctx_from_surface.__dict__.__setitem__('stypy_call_defaults', defaults)
        RendererCairo.set_ctx_from_surface.__dict__.__setitem__('stypy_call_varargs', varargs)
        RendererCairo.set_ctx_from_surface.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        RendererCairo.set_ctx_from_surface.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RendererCairo.set_ctx_from_surface', ['surface'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_ctx_from_surface', localization, ['surface'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_ctx_from_surface(...)' code ##################

        
        # Assigning a Call to a Attribute (line 117):
        
        # Assigning a Call to a Attribute (line 117):
        
        # Call to Context(...): (line 117)
        # Processing the call arguments (line 117)
        # Getting the type of 'surface' (line 117)
        surface_219769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 36), 'surface', False)
        # Processing the call keyword arguments (line 117)
        kwargs_219770 = {}
        # Getting the type of 'cairo' (line 117)
        cairo_219767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 22), 'cairo', False)
        # Obtaining the member 'Context' of a type (line 117)
        Context_219768 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 22), cairo_219767, 'Context')
        # Calling Context(args, kwargs) (line 117)
        Context_call_result_219771 = invoke(stypy.reporting.localization.Localization(__file__, 117, 22), Context_219768, *[surface_219769], **kwargs_219770)
        
        # Getting the type of 'self' (line 117)
        self_219772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'self')
        # Obtaining the member 'gc' of a type (line 117)
        gc_219773 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 8), self_219772, 'gc')
        # Setting the type of the member 'ctx' of a type (line 117)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 8), gc_219773, 'ctx', Context_call_result_219771)
        
        # ################# End of 'set_ctx_from_surface(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_ctx_from_surface' in the type store
        # Getting the type of 'stypy_return_type' (line 116)
        stypy_return_type_219774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_219774)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_ctx_from_surface'
        return stypy_return_type_219774


    @norecursion
    def set_width_height(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_width_height'
        module_type_store = module_type_store.open_function_context('set_width_height', 119, 4, False)
        # Assigning a type to the variable 'self' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        RendererCairo.set_width_height.__dict__.__setitem__('stypy_localization', localization)
        RendererCairo.set_width_height.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        RendererCairo.set_width_height.__dict__.__setitem__('stypy_type_store', module_type_store)
        RendererCairo.set_width_height.__dict__.__setitem__('stypy_function_name', 'RendererCairo.set_width_height')
        RendererCairo.set_width_height.__dict__.__setitem__('stypy_param_names_list', ['width', 'height'])
        RendererCairo.set_width_height.__dict__.__setitem__('stypy_varargs_param_name', None)
        RendererCairo.set_width_height.__dict__.__setitem__('stypy_kwargs_param_name', None)
        RendererCairo.set_width_height.__dict__.__setitem__('stypy_call_defaults', defaults)
        RendererCairo.set_width_height.__dict__.__setitem__('stypy_call_varargs', varargs)
        RendererCairo.set_width_height.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        RendererCairo.set_width_height.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RendererCairo.set_width_height', ['width', 'height'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_width_height', localization, ['width', 'height'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_width_height(...)' code ##################

        
        # Assigning a Name to a Attribute (line 120):
        
        # Assigning a Name to a Attribute (line 120):
        # Getting the type of 'width' (line 120)
        width_219775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 22), 'width')
        # Getting the type of 'self' (line 120)
        self_219776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'self')
        # Setting the type of the member 'width' of a type (line 120)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 8), self_219776, 'width', width_219775)
        
        # Assigning a Name to a Attribute (line 121):
        
        # Assigning a Name to a Attribute (line 121):
        # Getting the type of 'height' (line 121)
        height_219777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 22), 'height')
        # Getting the type of 'self' (line 121)
        self_219778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'self')
        # Setting the type of the member 'height' of a type (line 121)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 8), self_219778, 'height', height_219777)
        
        # Assigning a Call to a Attribute (line 122):
        
        # Assigning a Call to a Attribute (line 122):
        
        # Call to Matrix(...): (line 122)
        # Processing the call keyword arguments (line 122)
        int_219781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 44), 'int')
        keyword_219782 = int_219781
        # Getting the type of 'self' (line 122)
        self_219783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 51), 'self', False)
        # Obtaining the member 'height' of a type (line 122)
        height_219784 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 51), self_219783, 'height')
        keyword_219785 = height_219784
        kwargs_219786 = {'yy': keyword_219782, 'y0': keyword_219785}
        # Getting the type of 'cairo' (line 122)
        cairo_219779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 28), 'cairo', False)
        # Obtaining the member 'Matrix' of a type (line 122)
        Matrix_219780 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 28), cairo_219779, 'Matrix')
        # Calling Matrix(args, kwargs) (line 122)
        Matrix_call_result_219787 = invoke(stypy.reporting.localization.Localization(__file__, 122, 28), Matrix_219780, *[], **kwargs_219786)
        
        # Getting the type of 'self' (line 122)
        self_219788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 8), 'self')
        # Setting the type of the member 'matrix_flipy' of a type (line 122)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 8), self_219788, 'matrix_flipy', Matrix_call_result_219787)
        
        # ################# End of 'set_width_height(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_width_height' in the type store
        # Getting the type of 'stypy_return_type' (line 119)
        stypy_return_type_219789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_219789)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_width_height'
        return stypy_return_type_219789


    @norecursion
    def _fill_and_stroke(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_fill_and_stroke'
        module_type_store = module_type_store.open_function_context('_fill_and_stroke', 127, 4, False)
        # Assigning a type to the variable 'self' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        RendererCairo._fill_and_stroke.__dict__.__setitem__('stypy_localization', localization)
        RendererCairo._fill_and_stroke.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        RendererCairo._fill_and_stroke.__dict__.__setitem__('stypy_type_store', module_type_store)
        RendererCairo._fill_and_stroke.__dict__.__setitem__('stypy_function_name', 'RendererCairo._fill_and_stroke')
        RendererCairo._fill_and_stroke.__dict__.__setitem__('stypy_param_names_list', ['ctx', 'fill_c', 'alpha', 'alpha_overrides'])
        RendererCairo._fill_and_stroke.__dict__.__setitem__('stypy_varargs_param_name', None)
        RendererCairo._fill_and_stroke.__dict__.__setitem__('stypy_kwargs_param_name', None)
        RendererCairo._fill_and_stroke.__dict__.__setitem__('stypy_call_defaults', defaults)
        RendererCairo._fill_and_stroke.__dict__.__setitem__('stypy_call_varargs', varargs)
        RendererCairo._fill_and_stroke.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        RendererCairo._fill_and_stroke.__dict__.__setitem__('stypy_declared_arg_number', 5)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RendererCairo._fill_and_stroke', ['ctx', 'fill_c', 'alpha', 'alpha_overrides'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_fill_and_stroke', localization, ['ctx', 'fill_c', 'alpha', 'alpha_overrides'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_fill_and_stroke(...)' code ##################

        
        # Type idiom detected: calculating its left and rigth part (line 128)
        # Getting the type of 'fill_c' (line 128)
        fill_c_219790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'fill_c')
        # Getting the type of 'None' (line 128)
        None_219791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 25), 'None')
        
        (may_be_219792, more_types_in_union_219793) = may_not_be_none(fill_c_219790, None_219791)

        if may_be_219792:

            if more_types_in_union_219793:
                # Runtime conditional SSA (line 128)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to save(...): (line 129)
            # Processing the call keyword arguments (line 129)
            kwargs_219796 = {}
            # Getting the type of 'ctx' (line 129)
            ctx_219794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 12), 'ctx', False)
            # Obtaining the member 'save' of a type (line 129)
            save_219795 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 12), ctx_219794, 'save')
            # Calling save(args, kwargs) (line 129)
            save_call_result_219797 = invoke(stypy.reporting.localization.Localization(__file__, 129, 12), save_219795, *[], **kwargs_219796)
            
            
            
            # Evaluating a boolean operation
            
            
            # Call to len(...): (line 130)
            # Processing the call arguments (line 130)
            # Getting the type of 'fill_c' (line 130)
            fill_c_219799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 19), 'fill_c', False)
            # Processing the call keyword arguments (line 130)
            kwargs_219800 = {}
            # Getting the type of 'len' (line 130)
            len_219798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 15), 'len', False)
            # Calling len(args, kwargs) (line 130)
            len_call_result_219801 = invoke(stypy.reporting.localization.Localization(__file__, 130, 15), len_219798, *[fill_c_219799], **kwargs_219800)
            
            int_219802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 30), 'int')
            # Applying the binary operator '==' (line 130)
            result_eq_219803 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 15), '==', len_call_result_219801, int_219802)
            
            # Getting the type of 'alpha_overrides' (line 130)
            alpha_overrides_219804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 35), 'alpha_overrides')
            # Applying the binary operator 'or' (line 130)
            result_or_keyword_219805 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 15), 'or', result_eq_219803, alpha_overrides_219804)
            
            # Testing the type of an if condition (line 130)
            if_condition_219806 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 130, 12), result_or_keyword_219805)
            # Assigning a type to the variable 'if_condition_219806' (line 130)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 12), 'if_condition_219806', if_condition_219806)
            # SSA begins for if statement (line 130)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to set_source_rgba(...): (line 131)
            # Processing the call arguments (line 131)
            
            # Obtaining the type of the subscript
            int_219809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 43), 'int')
            # Getting the type of 'fill_c' (line 131)
            fill_c_219810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 36), 'fill_c', False)
            # Obtaining the member '__getitem__' of a type (line 131)
            getitem___219811 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 36), fill_c_219810, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 131)
            subscript_call_result_219812 = invoke(stypy.reporting.localization.Localization(__file__, 131, 36), getitem___219811, int_219809)
            
            
            # Obtaining the type of the subscript
            int_219813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 54), 'int')
            # Getting the type of 'fill_c' (line 131)
            fill_c_219814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 47), 'fill_c', False)
            # Obtaining the member '__getitem__' of a type (line 131)
            getitem___219815 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 47), fill_c_219814, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 131)
            subscript_call_result_219816 = invoke(stypy.reporting.localization.Localization(__file__, 131, 47), getitem___219815, int_219813)
            
            
            # Obtaining the type of the subscript
            int_219817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 65), 'int')
            # Getting the type of 'fill_c' (line 131)
            fill_c_219818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 58), 'fill_c', False)
            # Obtaining the member '__getitem__' of a type (line 131)
            getitem___219819 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 58), fill_c_219818, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 131)
            subscript_call_result_219820 = invoke(stypy.reporting.localization.Localization(__file__, 131, 58), getitem___219819, int_219817)
            
            # Getting the type of 'alpha' (line 131)
            alpha_219821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 69), 'alpha', False)
            # Processing the call keyword arguments (line 131)
            kwargs_219822 = {}
            # Getting the type of 'ctx' (line 131)
            ctx_219807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 16), 'ctx', False)
            # Obtaining the member 'set_source_rgba' of a type (line 131)
            set_source_rgba_219808 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 16), ctx_219807, 'set_source_rgba')
            # Calling set_source_rgba(args, kwargs) (line 131)
            set_source_rgba_call_result_219823 = invoke(stypy.reporting.localization.Localization(__file__, 131, 16), set_source_rgba_219808, *[subscript_call_result_219812, subscript_call_result_219816, subscript_call_result_219820, alpha_219821], **kwargs_219822)
            
            # SSA branch for the else part of an if statement (line 130)
            module_type_store.open_ssa_branch('else')
            
            # Call to set_source_rgba(...): (line 133)
            # Processing the call arguments (line 133)
            
            # Obtaining the type of the subscript
            int_219826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 43), 'int')
            # Getting the type of 'fill_c' (line 133)
            fill_c_219827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 36), 'fill_c', False)
            # Obtaining the member '__getitem__' of a type (line 133)
            getitem___219828 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 36), fill_c_219827, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 133)
            subscript_call_result_219829 = invoke(stypy.reporting.localization.Localization(__file__, 133, 36), getitem___219828, int_219826)
            
            
            # Obtaining the type of the subscript
            int_219830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 54), 'int')
            # Getting the type of 'fill_c' (line 133)
            fill_c_219831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 47), 'fill_c', False)
            # Obtaining the member '__getitem__' of a type (line 133)
            getitem___219832 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 47), fill_c_219831, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 133)
            subscript_call_result_219833 = invoke(stypy.reporting.localization.Localization(__file__, 133, 47), getitem___219832, int_219830)
            
            
            # Obtaining the type of the subscript
            int_219834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 65), 'int')
            # Getting the type of 'fill_c' (line 133)
            fill_c_219835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 58), 'fill_c', False)
            # Obtaining the member '__getitem__' of a type (line 133)
            getitem___219836 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 58), fill_c_219835, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 133)
            subscript_call_result_219837 = invoke(stypy.reporting.localization.Localization(__file__, 133, 58), getitem___219836, int_219834)
            
            
            # Obtaining the type of the subscript
            int_219838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 76), 'int')
            # Getting the type of 'fill_c' (line 133)
            fill_c_219839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 69), 'fill_c', False)
            # Obtaining the member '__getitem__' of a type (line 133)
            getitem___219840 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 69), fill_c_219839, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 133)
            subscript_call_result_219841 = invoke(stypy.reporting.localization.Localization(__file__, 133, 69), getitem___219840, int_219838)
            
            # Processing the call keyword arguments (line 133)
            kwargs_219842 = {}
            # Getting the type of 'ctx' (line 133)
            ctx_219824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 16), 'ctx', False)
            # Obtaining the member 'set_source_rgba' of a type (line 133)
            set_source_rgba_219825 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 16), ctx_219824, 'set_source_rgba')
            # Calling set_source_rgba(args, kwargs) (line 133)
            set_source_rgba_call_result_219843 = invoke(stypy.reporting.localization.Localization(__file__, 133, 16), set_source_rgba_219825, *[subscript_call_result_219829, subscript_call_result_219833, subscript_call_result_219837, subscript_call_result_219841], **kwargs_219842)
            
            # SSA join for if statement (line 130)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Call to fill_preserve(...): (line 134)
            # Processing the call keyword arguments (line 134)
            kwargs_219846 = {}
            # Getting the type of 'ctx' (line 134)
            ctx_219844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 12), 'ctx', False)
            # Obtaining the member 'fill_preserve' of a type (line 134)
            fill_preserve_219845 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 12), ctx_219844, 'fill_preserve')
            # Calling fill_preserve(args, kwargs) (line 134)
            fill_preserve_call_result_219847 = invoke(stypy.reporting.localization.Localization(__file__, 134, 12), fill_preserve_219845, *[], **kwargs_219846)
            
            
            # Call to restore(...): (line 135)
            # Processing the call keyword arguments (line 135)
            kwargs_219850 = {}
            # Getting the type of 'ctx' (line 135)
            ctx_219848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 12), 'ctx', False)
            # Obtaining the member 'restore' of a type (line 135)
            restore_219849 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 12), ctx_219848, 'restore')
            # Calling restore(args, kwargs) (line 135)
            restore_call_result_219851 = invoke(stypy.reporting.localization.Localization(__file__, 135, 12), restore_219849, *[], **kwargs_219850)
            

            if more_types_in_union_219793:
                # SSA join for if statement (line 128)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Call to stroke(...): (line 136)
        # Processing the call keyword arguments (line 136)
        kwargs_219854 = {}
        # Getting the type of 'ctx' (line 136)
        ctx_219852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 8), 'ctx', False)
        # Obtaining the member 'stroke' of a type (line 136)
        stroke_219853 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 8), ctx_219852, 'stroke')
        # Calling stroke(args, kwargs) (line 136)
        stroke_call_result_219855 = invoke(stypy.reporting.localization.Localization(__file__, 136, 8), stroke_219853, *[], **kwargs_219854)
        
        
        # ################# End of '_fill_and_stroke(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_fill_and_stroke' in the type store
        # Getting the type of 'stypy_return_type' (line 127)
        stypy_return_type_219856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_219856)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_fill_and_stroke'
        return stypy_return_type_219856


    @staticmethod
    @norecursion
    def convert_path(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 139)
        None_219857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 48), 'None')
        defaults = [None_219857]
        # Create a new context for function 'convert_path'
        module_type_store = module_type_store.open_function_context('convert_path', 138, 4, False)
        
        # Passed parameters checking function
        RendererCairo.convert_path.__dict__.__setitem__('stypy_localization', localization)
        RendererCairo.convert_path.__dict__.__setitem__('stypy_type_of_self', None)
        RendererCairo.convert_path.__dict__.__setitem__('stypy_type_store', module_type_store)
        RendererCairo.convert_path.__dict__.__setitem__('stypy_function_name', 'convert_path')
        RendererCairo.convert_path.__dict__.__setitem__('stypy_param_names_list', ['ctx', 'path', 'transform', 'clip'])
        RendererCairo.convert_path.__dict__.__setitem__('stypy_varargs_param_name', None)
        RendererCairo.convert_path.__dict__.__setitem__('stypy_kwargs_param_name', None)
        RendererCairo.convert_path.__dict__.__setitem__('stypy_call_defaults', defaults)
        RendererCairo.convert_path.__dict__.__setitem__('stypy_call_varargs', varargs)
        RendererCairo.convert_path.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        RendererCairo.convert_path.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, None, module_type_store, 'convert_path', ['ctx', 'path', 'transform', 'clip'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'convert_path', localization, ['path', 'transform', 'clip'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'convert_path(...)' code ##################

        
        
        # Call to iter_segments(...): (line 140)
        # Processing the call arguments (line 140)
        # Getting the type of 'transform' (line 140)
        transform_219860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 47), 'transform', False)
        # Processing the call keyword arguments (line 140)
        # Getting the type of 'clip' (line 140)
        clip_219861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 63), 'clip', False)
        keyword_219862 = clip_219861
        kwargs_219863 = {'clip': keyword_219862}
        # Getting the type of 'path' (line 140)
        path_219858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 28), 'path', False)
        # Obtaining the member 'iter_segments' of a type (line 140)
        iter_segments_219859 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 28), path_219858, 'iter_segments')
        # Calling iter_segments(args, kwargs) (line 140)
        iter_segments_call_result_219864 = invoke(stypy.reporting.localization.Localization(__file__, 140, 28), iter_segments_219859, *[transform_219860], **kwargs_219863)
        
        # Testing the type of a for loop iterable (line 140)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 140, 8), iter_segments_call_result_219864)
        # Getting the type of the for loop variable (line 140)
        for_loop_var_219865 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 140, 8), iter_segments_call_result_219864)
        # Assigning a type to the variable 'points' (line 140)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 8), 'points', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 8), for_loop_var_219865))
        # Assigning a type to the variable 'code' (line 140)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 8), 'code', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 8), for_loop_var_219865))
        # SSA begins for a for statement (line 140)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Getting the type of 'code' (line 141)
        code_219866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 15), 'code')
        # Getting the type of 'Path' (line 141)
        Path_219867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 23), 'Path')
        # Obtaining the member 'MOVETO' of a type (line 141)
        MOVETO_219868 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 23), Path_219867, 'MOVETO')
        # Applying the binary operator '==' (line 141)
        result_eq_219869 = python_operator(stypy.reporting.localization.Localization(__file__, 141, 15), '==', code_219866, MOVETO_219868)
        
        # Testing the type of an if condition (line 141)
        if_condition_219870 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 141, 12), result_eq_219869)
        # Assigning a type to the variable 'if_condition_219870' (line 141)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 12), 'if_condition_219870', if_condition_219870)
        # SSA begins for if statement (line 141)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to move_to(...): (line 142)
        # Getting the type of 'points' (line 142)
        points_219873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 29), 'points', False)
        # Processing the call keyword arguments (line 142)
        kwargs_219874 = {}
        # Getting the type of 'ctx' (line 142)
        ctx_219871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 16), 'ctx', False)
        # Obtaining the member 'move_to' of a type (line 142)
        move_to_219872 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 16), ctx_219871, 'move_to')
        # Calling move_to(args, kwargs) (line 142)
        move_to_call_result_219875 = invoke(stypy.reporting.localization.Localization(__file__, 142, 16), move_to_219872, *[points_219873], **kwargs_219874)
        
        # SSA branch for the else part of an if statement (line 141)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'code' (line 143)
        code_219876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 17), 'code')
        # Getting the type of 'Path' (line 143)
        Path_219877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 25), 'Path')
        # Obtaining the member 'CLOSEPOLY' of a type (line 143)
        CLOSEPOLY_219878 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 25), Path_219877, 'CLOSEPOLY')
        # Applying the binary operator '==' (line 143)
        result_eq_219879 = python_operator(stypy.reporting.localization.Localization(__file__, 143, 17), '==', code_219876, CLOSEPOLY_219878)
        
        # Testing the type of an if condition (line 143)
        if_condition_219880 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 143, 17), result_eq_219879)
        # Assigning a type to the variable 'if_condition_219880' (line 143)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 17), 'if_condition_219880', if_condition_219880)
        # SSA begins for if statement (line 143)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to close_path(...): (line 144)
        # Processing the call keyword arguments (line 144)
        kwargs_219883 = {}
        # Getting the type of 'ctx' (line 144)
        ctx_219881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 16), 'ctx', False)
        # Obtaining the member 'close_path' of a type (line 144)
        close_path_219882 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 16), ctx_219881, 'close_path')
        # Calling close_path(args, kwargs) (line 144)
        close_path_call_result_219884 = invoke(stypy.reporting.localization.Localization(__file__, 144, 16), close_path_219882, *[], **kwargs_219883)
        
        # SSA branch for the else part of an if statement (line 143)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'code' (line 145)
        code_219885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 17), 'code')
        # Getting the type of 'Path' (line 145)
        Path_219886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 25), 'Path')
        # Obtaining the member 'LINETO' of a type (line 145)
        LINETO_219887 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 25), Path_219886, 'LINETO')
        # Applying the binary operator '==' (line 145)
        result_eq_219888 = python_operator(stypy.reporting.localization.Localization(__file__, 145, 17), '==', code_219885, LINETO_219887)
        
        # Testing the type of an if condition (line 145)
        if_condition_219889 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 145, 17), result_eq_219888)
        # Assigning a type to the variable 'if_condition_219889' (line 145)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 17), 'if_condition_219889', if_condition_219889)
        # SSA begins for if statement (line 145)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to line_to(...): (line 146)
        # Getting the type of 'points' (line 146)
        points_219892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 29), 'points', False)
        # Processing the call keyword arguments (line 146)
        kwargs_219893 = {}
        # Getting the type of 'ctx' (line 146)
        ctx_219890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 16), 'ctx', False)
        # Obtaining the member 'line_to' of a type (line 146)
        line_to_219891 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 16), ctx_219890, 'line_to')
        # Calling line_to(args, kwargs) (line 146)
        line_to_call_result_219894 = invoke(stypy.reporting.localization.Localization(__file__, 146, 16), line_to_219891, *[points_219892], **kwargs_219893)
        
        # SSA branch for the else part of an if statement (line 145)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'code' (line 147)
        code_219895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 17), 'code')
        # Getting the type of 'Path' (line 147)
        Path_219896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 25), 'Path')
        # Obtaining the member 'CURVE3' of a type (line 147)
        CURVE3_219897 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 25), Path_219896, 'CURVE3')
        # Applying the binary operator '==' (line 147)
        result_eq_219898 = python_operator(stypy.reporting.localization.Localization(__file__, 147, 17), '==', code_219895, CURVE3_219897)
        
        # Testing the type of an if condition (line 147)
        if_condition_219899 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 147, 17), result_eq_219898)
        # Assigning a type to the variable 'if_condition_219899' (line 147)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 17), 'if_condition_219899', if_condition_219899)
        # SSA begins for if statement (line 147)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to curve_to(...): (line 148)
        # Processing the call arguments (line 148)
        
        # Obtaining the type of the subscript
        int_219902 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 36), 'int')
        # Getting the type of 'points' (line 148)
        points_219903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 29), 'points', False)
        # Obtaining the member '__getitem__' of a type (line 148)
        getitem___219904 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 29), points_219903, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 148)
        subscript_call_result_219905 = invoke(stypy.reporting.localization.Localization(__file__, 148, 29), getitem___219904, int_219902)
        
        
        # Obtaining the type of the subscript
        int_219906 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 47), 'int')
        # Getting the type of 'points' (line 148)
        points_219907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 40), 'points', False)
        # Obtaining the member '__getitem__' of a type (line 148)
        getitem___219908 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 40), points_219907, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 148)
        subscript_call_result_219909 = invoke(stypy.reporting.localization.Localization(__file__, 148, 40), getitem___219908, int_219906)
        
        
        # Obtaining the type of the subscript
        int_219910 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 36), 'int')
        # Getting the type of 'points' (line 149)
        points_219911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 29), 'points', False)
        # Obtaining the member '__getitem__' of a type (line 149)
        getitem___219912 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 29), points_219911, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 149)
        subscript_call_result_219913 = invoke(stypy.reporting.localization.Localization(__file__, 149, 29), getitem___219912, int_219910)
        
        
        # Obtaining the type of the subscript
        int_219914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 47), 'int')
        # Getting the type of 'points' (line 149)
        points_219915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 40), 'points', False)
        # Obtaining the member '__getitem__' of a type (line 149)
        getitem___219916 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 40), points_219915, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 149)
        subscript_call_result_219917 = invoke(stypy.reporting.localization.Localization(__file__, 149, 40), getitem___219916, int_219914)
        
        
        # Obtaining the type of the subscript
        int_219918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 36), 'int')
        # Getting the type of 'points' (line 150)
        points_219919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 29), 'points', False)
        # Obtaining the member '__getitem__' of a type (line 150)
        getitem___219920 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 29), points_219919, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 150)
        subscript_call_result_219921 = invoke(stypy.reporting.localization.Localization(__file__, 150, 29), getitem___219920, int_219918)
        
        
        # Obtaining the type of the subscript
        int_219922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 47), 'int')
        # Getting the type of 'points' (line 150)
        points_219923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 40), 'points', False)
        # Obtaining the member '__getitem__' of a type (line 150)
        getitem___219924 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 40), points_219923, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 150)
        subscript_call_result_219925 = invoke(stypy.reporting.localization.Localization(__file__, 150, 40), getitem___219924, int_219922)
        
        # Processing the call keyword arguments (line 148)
        kwargs_219926 = {}
        # Getting the type of 'ctx' (line 148)
        ctx_219900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 16), 'ctx', False)
        # Obtaining the member 'curve_to' of a type (line 148)
        curve_to_219901 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 16), ctx_219900, 'curve_to')
        # Calling curve_to(args, kwargs) (line 148)
        curve_to_call_result_219927 = invoke(stypy.reporting.localization.Localization(__file__, 148, 16), curve_to_219901, *[subscript_call_result_219905, subscript_call_result_219909, subscript_call_result_219913, subscript_call_result_219917, subscript_call_result_219921, subscript_call_result_219925], **kwargs_219926)
        
        # SSA branch for the else part of an if statement (line 147)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'code' (line 151)
        code_219928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 17), 'code')
        # Getting the type of 'Path' (line 151)
        Path_219929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 25), 'Path')
        # Obtaining the member 'CURVE4' of a type (line 151)
        CURVE4_219930 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 25), Path_219929, 'CURVE4')
        # Applying the binary operator '==' (line 151)
        result_eq_219931 = python_operator(stypy.reporting.localization.Localization(__file__, 151, 17), '==', code_219928, CURVE4_219930)
        
        # Testing the type of an if condition (line 151)
        if_condition_219932 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 151, 17), result_eq_219931)
        # Assigning a type to the variable 'if_condition_219932' (line 151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 17), 'if_condition_219932', if_condition_219932)
        # SSA begins for if statement (line 151)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to curve_to(...): (line 152)
        # Getting the type of 'points' (line 152)
        points_219935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 30), 'points', False)
        # Processing the call keyword arguments (line 152)
        kwargs_219936 = {}
        # Getting the type of 'ctx' (line 152)
        ctx_219933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 16), 'ctx', False)
        # Obtaining the member 'curve_to' of a type (line 152)
        curve_to_219934 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 16), ctx_219933, 'curve_to')
        # Calling curve_to(args, kwargs) (line 152)
        curve_to_call_result_219937 = invoke(stypy.reporting.localization.Localization(__file__, 152, 16), curve_to_219934, *[points_219935], **kwargs_219936)
        
        # SSA join for if statement (line 151)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 147)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 145)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 143)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 141)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'convert_path(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'convert_path' in the type store
        # Getting the type of 'stypy_return_type' (line 138)
        stypy_return_type_219938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_219938)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'convert_path'
        return stypy_return_type_219938


    @norecursion
    def draw_path(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 154)
        None_219939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 53), 'None')
        defaults = [None_219939]
        # Create a new context for function 'draw_path'
        module_type_store = module_type_store.open_function_context('draw_path', 154, 4, False)
        # Assigning a type to the variable 'self' (line 155)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        RendererCairo.draw_path.__dict__.__setitem__('stypy_localization', localization)
        RendererCairo.draw_path.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        RendererCairo.draw_path.__dict__.__setitem__('stypy_type_store', module_type_store)
        RendererCairo.draw_path.__dict__.__setitem__('stypy_function_name', 'RendererCairo.draw_path')
        RendererCairo.draw_path.__dict__.__setitem__('stypy_param_names_list', ['gc', 'path', 'transform', 'rgbFace'])
        RendererCairo.draw_path.__dict__.__setitem__('stypy_varargs_param_name', None)
        RendererCairo.draw_path.__dict__.__setitem__('stypy_kwargs_param_name', None)
        RendererCairo.draw_path.__dict__.__setitem__('stypy_call_defaults', defaults)
        RendererCairo.draw_path.__dict__.__setitem__('stypy_call_varargs', varargs)
        RendererCairo.draw_path.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        RendererCairo.draw_path.__dict__.__setitem__('stypy_declared_arg_number', 5)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RendererCairo.draw_path', ['gc', 'path', 'transform', 'rgbFace'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'draw_path', localization, ['gc', 'path', 'transform', 'rgbFace'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'draw_path(...)' code ##################

        
        # Assigning a Attribute to a Name (line 155):
        
        # Assigning a Attribute to a Name (line 155):
        # Getting the type of 'gc' (line 155)
        gc_219940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 14), 'gc')
        # Obtaining the member 'ctx' of a type (line 155)
        ctx_219941 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 14), gc_219940, 'ctx')
        # Assigning a type to the variable 'ctx' (line 155)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 8), 'ctx', ctx_219941)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'rgbFace' (line 159)
        rgbFace_219942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 11), 'rgbFace')
        # Getting the type of 'None' (line 159)
        None_219943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 22), 'None')
        # Applying the binary operator 'is' (line 159)
        result_is__219944 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 11), 'is', rgbFace_219942, None_219943)
        
        
        
        # Call to get_hatch(...): (line 159)
        # Processing the call keyword arguments (line 159)
        kwargs_219947 = {}
        # Getting the type of 'gc' (line 159)
        gc_219945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 31), 'gc', False)
        # Obtaining the member 'get_hatch' of a type (line 159)
        get_hatch_219946 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 31), gc_219945, 'get_hatch')
        # Calling get_hatch(args, kwargs) (line 159)
        get_hatch_call_result_219948 = invoke(stypy.reporting.localization.Localization(__file__, 159, 31), get_hatch_219946, *[], **kwargs_219947)
        
        # Getting the type of 'None' (line 159)
        None_219949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 49), 'None')
        # Applying the binary operator 'is' (line 159)
        result_is__219950 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 31), 'is', get_hatch_call_result_219948, None_219949)
        
        # Applying the binary operator 'and' (line 159)
        result_and_keyword_219951 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 11), 'and', result_is__219944, result_is__219950)
        
        # Testing the type of an if condition (line 159)
        if_condition_219952 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 159, 8), result_and_keyword_219951)
        # Assigning a type to the variable 'if_condition_219952' (line 159)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'if_condition_219952', if_condition_219952)
        # SSA begins for if statement (line 159)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 160):
        
        # Assigning a Call to a Name (line 160):
        
        # Call to clip_extents(...): (line 160)
        # Processing the call keyword arguments (line 160)
        kwargs_219955 = {}
        # Getting the type of 'ctx' (line 160)
        ctx_219953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 19), 'ctx', False)
        # Obtaining the member 'clip_extents' of a type (line 160)
        clip_extents_219954 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 19), ctx_219953, 'clip_extents')
        # Calling clip_extents(args, kwargs) (line 160)
        clip_extents_call_result_219956 = invoke(stypy.reporting.localization.Localization(__file__, 160, 19), clip_extents_219954, *[], **kwargs_219955)
        
        # Assigning a type to the variable 'clip' (line 160)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 12), 'clip', clip_extents_call_result_219956)
        # SSA branch for the else part of an if statement (line 159)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Name to a Name (line 162):
        
        # Assigning a Name to a Name (line 162):
        # Getting the type of 'None' (line 162)
        None_219957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 19), 'None')
        # Assigning a type to the variable 'clip' (line 162)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 12), 'clip', None_219957)
        # SSA join for if statement (line 159)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 164):
        
        # Assigning a BinOp to a Name (line 164):
        # Getting the type of 'transform' (line 164)
        transform_219958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 21), 'transform')
        
        # Call to translate(...): (line 165)
        # Processing the call arguments (line 165)
        int_219968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 61), 'int')
        # Getting the type of 'self' (line 165)
        self_219969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 64), 'self', False)
        # Obtaining the member 'height' of a type (line 165)
        height_219970 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 64), self_219969, 'height')
        # Processing the call keyword arguments (line 165)
        kwargs_219971 = {}
        
        # Call to scale(...): (line 165)
        # Processing the call arguments (line 165)
        float_219963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 40), 'float')
        float_219964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 45), 'float')
        # Processing the call keyword arguments (line 165)
        kwargs_219965 = {}
        
        # Call to Affine2D(...): (line 165)
        # Processing the call keyword arguments (line 165)
        kwargs_219960 = {}
        # Getting the type of 'Affine2D' (line 165)
        Affine2D_219959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 23), 'Affine2D', False)
        # Calling Affine2D(args, kwargs) (line 165)
        Affine2D_call_result_219961 = invoke(stypy.reporting.localization.Localization(__file__, 165, 23), Affine2D_219959, *[], **kwargs_219960)
        
        # Obtaining the member 'scale' of a type (line 165)
        scale_219962 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 23), Affine2D_call_result_219961, 'scale')
        # Calling scale(args, kwargs) (line 165)
        scale_call_result_219966 = invoke(stypy.reporting.localization.Localization(__file__, 165, 23), scale_219962, *[float_219963, float_219964], **kwargs_219965)
        
        # Obtaining the member 'translate' of a type (line 165)
        translate_219967 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 23), scale_call_result_219966, 'translate')
        # Calling translate(args, kwargs) (line 165)
        translate_call_result_219972 = invoke(stypy.reporting.localization.Localization(__file__, 165, 23), translate_219967, *[int_219968, height_219970], **kwargs_219971)
        
        # Applying the binary operator '+' (line 164)
        result_add_219973 = python_operator(stypy.reporting.localization.Localization(__file__, 164, 21), '+', transform_219958, translate_call_result_219972)
        
        # Assigning a type to the variable 'transform' (line 164)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 8), 'transform', result_add_219973)
        
        # Call to new_path(...): (line 167)
        # Processing the call keyword arguments (line 167)
        kwargs_219976 = {}
        # Getting the type of 'ctx' (line 167)
        ctx_219974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 'ctx', False)
        # Obtaining the member 'new_path' of a type (line 167)
        new_path_219975 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 8), ctx_219974, 'new_path')
        # Calling new_path(args, kwargs) (line 167)
        new_path_call_result_219977 = invoke(stypy.reporting.localization.Localization(__file__, 167, 8), new_path_219975, *[], **kwargs_219976)
        
        
        # Call to convert_path(...): (line 168)
        # Processing the call arguments (line 168)
        # Getting the type of 'ctx' (line 168)
        ctx_219980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 26), 'ctx', False)
        # Getting the type of 'path' (line 168)
        path_219981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 31), 'path', False)
        # Getting the type of 'transform' (line 168)
        transform_219982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 37), 'transform', False)
        # Getting the type of 'clip' (line 168)
        clip_219983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 48), 'clip', False)
        # Processing the call keyword arguments (line 168)
        kwargs_219984 = {}
        # Getting the type of 'self' (line 168)
        self_219978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 8), 'self', False)
        # Obtaining the member 'convert_path' of a type (line 168)
        convert_path_219979 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 8), self_219978, 'convert_path')
        # Calling convert_path(args, kwargs) (line 168)
        convert_path_call_result_219985 = invoke(stypy.reporting.localization.Localization(__file__, 168, 8), convert_path_219979, *[ctx_219980, path_219981, transform_219982, clip_219983], **kwargs_219984)
        
        
        # Call to _fill_and_stroke(...): (line 170)
        # Processing the call arguments (line 170)
        # Getting the type of 'ctx' (line 171)
        ctx_219988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 12), 'ctx', False)
        # Getting the type of 'rgbFace' (line 171)
        rgbFace_219989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 17), 'rgbFace', False)
        
        # Call to get_alpha(...): (line 171)
        # Processing the call keyword arguments (line 171)
        kwargs_219992 = {}
        # Getting the type of 'gc' (line 171)
        gc_219990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 26), 'gc', False)
        # Obtaining the member 'get_alpha' of a type (line 171)
        get_alpha_219991 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 26), gc_219990, 'get_alpha')
        # Calling get_alpha(args, kwargs) (line 171)
        get_alpha_call_result_219993 = invoke(stypy.reporting.localization.Localization(__file__, 171, 26), get_alpha_219991, *[], **kwargs_219992)
        
        
        # Call to get_forced_alpha(...): (line 171)
        # Processing the call keyword arguments (line 171)
        kwargs_219996 = {}
        # Getting the type of 'gc' (line 171)
        gc_219994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 42), 'gc', False)
        # Obtaining the member 'get_forced_alpha' of a type (line 171)
        get_forced_alpha_219995 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 42), gc_219994, 'get_forced_alpha')
        # Calling get_forced_alpha(args, kwargs) (line 171)
        get_forced_alpha_call_result_219997 = invoke(stypy.reporting.localization.Localization(__file__, 171, 42), get_forced_alpha_219995, *[], **kwargs_219996)
        
        # Processing the call keyword arguments (line 170)
        kwargs_219998 = {}
        # Getting the type of 'self' (line 170)
        self_219986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 8), 'self', False)
        # Obtaining the member '_fill_and_stroke' of a type (line 170)
        _fill_and_stroke_219987 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 8), self_219986, '_fill_and_stroke')
        # Calling _fill_and_stroke(args, kwargs) (line 170)
        _fill_and_stroke_call_result_219999 = invoke(stypy.reporting.localization.Localization(__file__, 170, 8), _fill_and_stroke_219987, *[ctx_219988, rgbFace_219989, get_alpha_call_result_219993, get_forced_alpha_call_result_219997], **kwargs_219998)
        
        
        # ################# End of 'draw_path(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'draw_path' in the type store
        # Getting the type of 'stypy_return_type' (line 154)
        stypy_return_type_220000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_220000)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'draw_path'
        return stypy_return_type_220000


    @norecursion
    def draw_markers(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 174)
        None_220001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 74), 'None')
        defaults = [None_220001]
        # Create a new context for function 'draw_markers'
        module_type_store = module_type_store.open_function_context('draw_markers', 173, 4, False)
        # Assigning a type to the variable 'self' (line 174)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        RendererCairo.draw_markers.__dict__.__setitem__('stypy_localization', localization)
        RendererCairo.draw_markers.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        RendererCairo.draw_markers.__dict__.__setitem__('stypy_type_store', module_type_store)
        RendererCairo.draw_markers.__dict__.__setitem__('stypy_function_name', 'RendererCairo.draw_markers')
        RendererCairo.draw_markers.__dict__.__setitem__('stypy_param_names_list', ['gc', 'marker_path', 'marker_trans', 'path', 'transform', 'rgbFace'])
        RendererCairo.draw_markers.__dict__.__setitem__('stypy_varargs_param_name', None)
        RendererCairo.draw_markers.__dict__.__setitem__('stypy_kwargs_param_name', None)
        RendererCairo.draw_markers.__dict__.__setitem__('stypy_call_defaults', defaults)
        RendererCairo.draw_markers.__dict__.__setitem__('stypy_call_varargs', varargs)
        RendererCairo.draw_markers.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        RendererCairo.draw_markers.__dict__.__setitem__('stypy_declared_arg_number', 7)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RendererCairo.draw_markers', ['gc', 'marker_path', 'marker_trans', 'path', 'transform', 'rgbFace'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'draw_markers', localization, ['gc', 'marker_path', 'marker_trans', 'path', 'transform', 'rgbFace'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'draw_markers(...)' code ##################

        
        # Assigning a Attribute to a Name (line 175):
        
        # Assigning a Attribute to a Name (line 175):
        # Getting the type of 'gc' (line 175)
        gc_220002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 14), 'gc')
        # Obtaining the member 'ctx' of a type (line 175)
        ctx_220003 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 14), gc_220002, 'ctx')
        # Assigning a type to the variable 'ctx' (line 175)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'ctx', ctx_220003)
        
        # Call to new_path(...): (line 177)
        # Processing the call keyword arguments (line 177)
        kwargs_220006 = {}
        # Getting the type of 'ctx' (line 177)
        ctx_220004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), 'ctx', False)
        # Obtaining the member 'new_path' of a type (line 177)
        new_path_220005 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 8), ctx_220004, 'new_path')
        # Calling new_path(args, kwargs) (line 177)
        new_path_call_result_220007 = invoke(stypy.reporting.localization.Localization(__file__, 177, 8), new_path_220005, *[], **kwargs_220006)
        
        
        # Call to convert_path(...): (line 179)
        # Processing the call arguments (line 179)
        # Getting the type of 'ctx' (line 180)
        ctx_220010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 12), 'ctx', False)
        # Getting the type of 'marker_path' (line 180)
        marker_path_220011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 17), 'marker_path', False)
        # Getting the type of 'marker_trans' (line 180)
        marker_trans_220012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 30), 'marker_trans', False)
        
        # Call to scale(...): (line 180)
        # Processing the call arguments (line 180)
        float_220017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 62), 'float')
        float_220018 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 67), 'float')
        # Processing the call keyword arguments (line 180)
        kwargs_220019 = {}
        
        # Call to Affine2D(...): (line 180)
        # Processing the call keyword arguments (line 180)
        kwargs_220014 = {}
        # Getting the type of 'Affine2D' (line 180)
        Affine2D_220013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 45), 'Affine2D', False)
        # Calling Affine2D(args, kwargs) (line 180)
        Affine2D_call_result_220015 = invoke(stypy.reporting.localization.Localization(__file__, 180, 45), Affine2D_220013, *[], **kwargs_220014)
        
        # Obtaining the member 'scale' of a type (line 180)
        scale_220016 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 45), Affine2D_call_result_220015, 'scale')
        # Calling scale(args, kwargs) (line 180)
        scale_call_result_220020 = invoke(stypy.reporting.localization.Localization(__file__, 180, 45), scale_220016, *[float_220017, float_220018], **kwargs_220019)
        
        # Applying the binary operator '+' (line 180)
        result_add_220021 = python_operator(stypy.reporting.localization.Localization(__file__, 180, 30), '+', marker_trans_220012, scale_call_result_220020)
        
        # Processing the call keyword arguments (line 179)
        kwargs_220022 = {}
        # Getting the type of 'self' (line 179)
        self_220008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 8), 'self', False)
        # Obtaining the member 'convert_path' of a type (line 179)
        convert_path_220009 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 8), self_220008, 'convert_path')
        # Calling convert_path(args, kwargs) (line 179)
        convert_path_call_result_220023 = invoke(stypy.reporting.localization.Localization(__file__, 179, 8), convert_path_220009, *[ctx_220010, marker_path_220011, result_add_220021], **kwargs_220022)
        
        
        # Assigning a Call to a Name (line 181):
        
        # Assigning a Call to a Name (line 181):
        
        # Call to copy_path_flat(...): (line 181)
        # Processing the call keyword arguments (line 181)
        kwargs_220026 = {}
        # Getting the type of 'ctx' (line 181)
        ctx_220024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 22), 'ctx', False)
        # Obtaining the member 'copy_path_flat' of a type (line 181)
        copy_path_flat_220025 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 22), ctx_220024, 'copy_path_flat')
        # Calling copy_path_flat(args, kwargs) (line 181)
        copy_path_flat_call_result_220027 = invoke(stypy.reporting.localization.Localization(__file__, 181, 22), copy_path_flat_220025, *[], **kwargs_220026)
        
        # Assigning a type to the variable 'marker_path' (line 181)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 'marker_path', copy_path_flat_call_result_220027)
        
        # Assigning a Call to a Tuple (line 184):
        
        # Assigning a Call to a Name:
        
        # Call to fill_extents(...): (line 184)
        # Processing the call keyword arguments (line 184)
        kwargs_220030 = {}
        # Getting the type of 'ctx' (line 184)
        ctx_220028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 25), 'ctx', False)
        # Obtaining the member 'fill_extents' of a type (line 184)
        fill_extents_220029 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 25), ctx_220028, 'fill_extents')
        # Calling fill_extents(args, kwargs) (line 184)
        fill_extents_call_result_220031 = invoke(stypy.reporting.localization.Localization(__file__, 184, 25), fill_extents_220029, *[], **kwargs_220030)
        
        # Assigning a type to the variable 'call_assignment_219621' (line 184)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 'call_assignment_219621', fill_extents_call_result_220031)
        
        # Assigning a Call to a Name (line 184):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_220034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 8), 'int')
        # Processing the call keyword arguments
        kwargs_220035 = {}
        # Getting the type of 'call_assignment_219621' (line 184)
        call_assignment_219621_220032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 'call_assignment_219621', False)
        # Obtaining the member '__getitem__' of a type (line 184)
        getitem___220033 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 8), call_assignment_219621_220032, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_220036 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___220033, *[int_220034], **kwargs_220035)
        
        # Assigning a type to the variable 'call_assignment_219622' (line 184)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 'call_assignment_219622', getitem___call_result_220036)
        
        # Assigning a Name to a Name (line 184):
        # Getting the type of 'call_assignment_219622' (line 184)
        call_assignment_219622_220037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 'call_assignment_219622')
        # Assigning a type to the variable 'x1' (line 184)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 'x1', call_assignment_219622_220037)
        
        # Assigning a Call to a Name (line 184):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_220040 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 8), 'int')
        # Processing the call keyword arguments
        kwargs_220041 = {}
        # Getting the type of 'call_assignment_219621' (line 184)
        call_assignment_219621_220038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 'call_assignment_219621', False)
        # Obtaining the member '__getitem__' of a type (line 184)
        getitem___220039 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 8), call_assignment_219621_220038, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_220042 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___220039, *[int_220040], **kwargs_220041)
        
        # Assigning a type to the variable 'call_assignment_219623' (line 184)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 'call_assignment_219623', getitem___call_result_220042)
        
        # Assigning a Name to a Name (line 184):
        # Getting the type of 'call_assignment_219623' (line 184)
        call_assignment_219623_220043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 'call_assignment_219623')
        # Assigning a type to the variable 'y1' (line 184)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 12), 'y1', call_assignment_219623_220043)
        
        # Assigning a Call to a Name (line 184):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_220046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 8), 'int')
        # Processing the call keyword arguments
        kwargs_220047 = {}
        # Getting the type of 'call_assignment_219621' (line 184)
        call_assignment_219621_220044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 'call_assignment_219621', False)
        # Obtaining the member '__getitem__' of a type (line 184)
        getitem___220045 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 8), call_assignment_219621_220044, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_220048 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___220045, *[int_220046], **kwargs_220047)
        
        # Assigning a type to the variable 'call_assignment_219624' (line 184)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 'call_assignment_219624', getitem___call_result_220048)
        
        # Assigning a Name to a Name (line 184):
        # Getting the type of 'call_assignment_219624' (line 184)
        call_assignment_219624_220049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 'call_assignment_219624')
        # Assigning a type to the variable 'x2' (line 184)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 16), 'x2', call_assignment_219624_220049)
        
        # Assigning a Call to a Name (line 184):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_220052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 8), 'int')
        # Processing the call keyword arguments
        kwargs_220053 = {}
        # Getting the type of 'call_assignment_219621' (line 184)
        call_assignment_219621_220050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 'call_assignment_219621', False)
        # Obtaining the member '__getitem__' of a type (line 184)
        getitem___220051 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 8), call_assignment_219621_220050, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_220054 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___220051, *[int_220052], **kwargs_220053)
        
        # Assigning a type to the variable 'call_assignment_219625' (line 184)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 'call_assignment_219625', getitem___call_result_220054)
        
        # Assigning a Name to a Name (line 184):
        # Getting the type of 'call_assignment_219625' (line 184)
        call_assignment_219625_220055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 'call_assignment_219625')
        # Assigning a type to the variable 'y2' (line 184)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 20), 'y2', call_assignment_219625_220055)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'x1' (line 185)
        x1_220056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 11), 'x1')
        int_220057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 17), 'int')
        # Applying the binary operator '==' (line 185)
        result_eq_220058 = python_operator(stypy.reporting.localization.Localization(__file__, 185, 11), '==', x1_220056, int_220057)
        
        
        # Getting the type of 'y1' (line 185)
        y1_220059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 23), 'y1')
        int_220060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 29), 'int')
        # Applying the binary operator '==' (line 185)
        result_eq_220061 = python_operator(stypy.reporting.localization.Localization(__file__, 185, 23), '==', y1_220059, int_220060)
        
        # Applying the binary operator 'and' (line 185)
        result_and_keyword_220062 = python_operator(stypy.reporting.localization.Localization(__file__, 185, 11), 'and', result_eq_220058, result_eq_220061)
        
        # Getting the type of 'x2' (line 185)
        x2_220063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 35), 'x2')
        int_220064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 41), 'int')
        # Applying the binary operator '==' (line 185)
        result_eq_220065 = python_operator(stypy.reporting.localization.Localization(__file__, 185, 35), '==', x2_220063, int_220064)
        
        # Applying the binary operator 'and' (line 185)
        result_and_keyword_220066 = python_operator(stypy.reporting.localization.Localization(__file__, 185, 11), 'and', result_and_keyword_220062, result_eq_220065)
        
        # Getting the type of 'y2' (line 185)
        y2_220067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 47), 'y2')
        int_220068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 53), 'int')
        # Applying the binary operator '==' (line 185)
        result_eq_220069 = python_operator(stypy.reporting.localization.Localization(__file__, 185, 47), '==', y2_220067, int_220068)
        
        # Applying the binary operator 'and' (line 185)
        result_and_keyword_220070 = python_operator(stypy.reporting.localization.Localization(__file__, 185, 11), 'and', result_and_keyword_220066, result_eq_220069)
        
        # Testing the type of an if condition (line 185)
        if_condition_220071 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 185, 8), result_and_keyword_220070)
        # Assigning a type to the variable 'if_condition_220071' (line 185)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 8), 'if_condition_220071', if_condition_220071)
        # SSA begins for if statement (line 185)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 186):
        
        # Assigning a Name to a Name (line 186):
        # Getting the type of 'False' (line 186)
        False_220072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 21), 'False')
        # Assigning a type to the variable 'filled' (line 186)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 12), 'filled', False_220072)
        
        # Assigning a Name to a Name (line 188):
        
        # Assigning a Name to a Name (line 188):
        # Getting the type of 'None' (line 188)
        None_220073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 22), 'None')
        # Assigning a type to the variable 'rgbFace' (line 188)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 12), 'rgbFace', None_220073)
        # SSA branch for the else part of an if statement (line 185)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Name to a Name (line 190):
        
        # Assigning a Name to a Name (line 190):
        # Getting the type of 'True' (line 190)
        True_220074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 21), 'True')
        # Assigning a type to the variable 'filled' (line 190)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 12), 'filled', True_220074)
        # SSA join for if statement (line 185)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 192):
        
        # Assigning a BinOp to a Name (line 192):
        # Getting the type of 'transform' (line 192)
        transform_220075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 21), 'transform')
        
        # Call to translate(...): (line 193)
        # Processing the call arguments (line 193)
        int_220085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 61), 'int')
        # Getting the type of 'self' (line 193)
        self_220086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 64), 'self', False)
        # Obtaining the member 'height' of a type (line 193)
        height_220087 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 64), self_220086, 'height')
        # Processing the call keyword arguments (line 193)
        kwargs_220088 = {}
        
        # Call to scale(...): (line 193)
        # Processing the call arguments (line 193)
        float_220080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 40), 'float')
        float_220081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 45), 'float')
        # Processing the call keyword arguments (line 193)
        kwargs_220082 = {}
        
        # Call to Affine2D(...): (line 193)
        # Processing the call keyword arguments (line 193)
        kwargs_220077 = {}
        # Getting the type of 'Affine2D' (line 193)
        Affine2D_220076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 23), 'Affine2D', False)
        # Calling Affine2D(args, kwargs) (line 193)
        Affine2D_call_result_220078 = invoke(stypy.reporting.localization.Localization(__file__, 193, 23), Affine2D_220076, *[], **kwargs_220077)
        
        # Obtaining the member 'scale' of a type (line 193)
        scale_220079 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 23), Affine2D_call_result_220078, 'scale')
        # Calling scale(args, kwargs) (line 193)
        scale_call_result_220083 = invoke(stypy.reporting.localization.Localization(__file__, 193, 23), scale_220079, *[float_220080, float_220081], **kwargs_220082)
        
        # Obtaining the member 'translate' of a type (line 193)
        translate_220084 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 23), scale_call_result_220083, 'translate')
        # Calling translate(args, kwargs) (line 193)
        translate_call_result_220089 = invoke(stypy.reporting.localization.Localization(__file__, 193, 23), translate_220084, *[int_220085, height_220087], **kwargs_220088)
        
        # Applying the binary operator '+' (line 192)
        result_add_220090 = python_operator(stypy.reporting.localization.Localization(__file__, 192, 21), '+', transform_220075, translate_call_result_220089)
        
        # Assigning a type to the variable 'transform' (line 192)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 8), 'transform', result_add_220090)
        
        # Call to new_path(...): (line 195)
        # Processing the call keyword arguments (line 195)
        kwargs_220093 = {}
        # Getting the type of 'ctx' (line 195)
        ctx_220091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 8), 'ctx', False)
        # Obtaining the member 'new_path' of a type (line 195)
        new_path_220092 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 8), ctx_220091, 'new_path')
        # Calling new_path(args, kwargs) (line 195)
        new_path_call_result_220094 = invoke(stypy.reporting.localization.Localization(__file__, 195, 8), new_path_220092, *[], **kwargs_220093)
        
        
        
        # Call to enumerate(...): (line 196)
        # Processing the call arguments (line 196)
        
        # Call to iter_segments(...): (line 197)
        # Processing the call arguments (line 197)
        # Getting the type of 'transform' (line 197)
        transform_220098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 35), 'transform', False)
        # Processing the call keyword arguments (line 197)
        # Getting the type of 'False' (line 197)
        False_220099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 55), 'False', False)
        keyword_220100 = False_220099
        kwargs_220101 = {'simplify': keyword_220100}
        # Getting the type of 'path' (line 197)
        path_220096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 16), 'path', False)
        # Obtaining the member 'iter_segments' of a type (line 197)
        iter_segments_220097 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 16), path_220096, 'iter_segments')
        # Calling iter_segments(args, kwargs) (line 197)
        iter_segments_call_result_220102 = invoke(stypy.reporting.localization.Localization(__file__, 197, 16), iter_segments_220097, *[transform_220098], **kwargs_220101)
        
        # Processing the call keyword arguments (line 196)
        kwargs_220103 = {}
        # Getting the type of 'enumerate' (line 196)
        enumerate_220095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 36), 'enumerate', False)
        # Calling enumerate(args, kwargs) (line 196)
        enumerate_call_result_220104 = invoke(stypy.reporting.localization.Localization(__file__, 196, 36), enumerate_220095, *[iter_segments_call_result_220102], **kwargs_220103)
        
        # Testing the type of a for loop iterable (line 196)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 196, 8), enumerate_call_result_220104)
        # Getting the type of the for loop variable (line 196)
        for_loop_var_220105 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 196, 8), enumerate_call_result_220104)
        # Assigning a type to the variable 'i' (line 196)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 196, 8), for_loop_var_220105))
        # Assigning a type to the variable 'vertices' (line 196)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'vertices', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 196, 8), for_loop_var_220105))
        # Assigning a type to the variable 'codes' (line 196)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'codes', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 196, 8), for_loop_var_220105))
        # SSA begins for a for statement (line 196)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Call to len(...): (line 198)
        # Processing the call arguments (line 198)
        # Getting the type of 'vertices' (line 198)
        vertices_220107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 19), 'vertices', False)
        # Processing the call keyword arguments (line 198)
        kwargs_220108 = {}
        # Getting the type of 'len' (line 198)
        len_220106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 15), 'len', False)
        # Calling len(args, kwargs) (line 198)
        len_call_result_220109 = invoke(stypy.reporting.localization.Localization(__file__, 198, 15), len_220106, *[vertices_220107], **kwargs_220108)
        
        # Testing the type of an if condition (line 198)
        if_condition_220110 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 198, 12), len_call_result_220109)
        # Assigning a type to the variable 'if_condition_220110' (line 198)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 12), 'if_condition_220110', if_condition_220110)
        # SSA begins for if statement (line 198)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Tuple (line 199):
        
        # Assigning a Subscript to a Name (line 199):
        
        # Obtaining the type of the subscript
        int_220111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 16), 'int')
        
        # Obtaining the type of the subscript
        int_220112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 32), 'int')
        slice_220113 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 199, 23), int_220112, None, None)
        # Getting the type of 'vertices' (line 199)
        vertices_220114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 23), 'vertices')
        # Obtaining the member '__getitem__' of a type (line 199)
        getitem___220115 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 23), vertices_220114, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 199)
        subscript_call_result_220116 = invoke(stypy.reporting.localization.Localization(__file__, 199, 23), getitem___220115, slice_220113)
        
        # Obtaining the member '__getitem__' of a type (line 199)
        getitem___220117 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 16), subscript_call_result_220116, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 199)
        subscript_call_result_220118 = invoke(stypy.reporting.localization.Localization(__file__, 199, 16), getitem___220117, int_220111)
        
        # Assigning a type to the variable 'tuple_var_assignment_219626' (line 199)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 16), 'tuple_var_assignment_219626', subscript_call_result_220118)
        
        # Assigning a Subscript to a Name (line 199):
        
        # Obtaining the type of the subscript
        int_220119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 16), 'int')
        
        # Obtaining the type of the subscript
        int_220120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 32), 'int')
        slice_220121 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 199, 23), int_220120, None, None)
        # Getting the type of 'vertices' (line 199)
        vertices_220122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 23), 'vertices')
        # Obtaining the member '__getitem__' of a type (line 199)
        getitem___220123 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 23), vertices_220122, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 199)
        subscript_call_result_220124 = invoke(stypy.reporting.localization.Localization(__file__, 199, 23), getitem___220123, slice_220121)
        
        # Obtaining the member '__getitem__' of a type (line 199)
        getitem___220125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 16), subscript_call_result_220124, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 199)
        subscript_call_result_220126 = invoke(stypy.reporting.localization.Localization(__file__, 199, 16), getitem___220125, int_220119)
        
        # Assigning a type to the variable 'tuple_var_assignment_219627' (line 199)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 16), 'tuple_var_assignment_219627', subscript_call_result_220126)
        
        # Assigning a Name to a Name (line 199):
        # Getting the type of 'tuple_var_assignment_219626' (line 199)
        tuple_var_assignment_219626_220127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 16), 'tuple_var_assignment_219626')
        # Assigning a type to the variable 'x' (line 199)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 16), 'x', tuple_var_assignment_219626_220127)
        
        # Assigning a Name to a Name (line 199):
        # Getting the type of 'tuple_var_assignment_219627' (line 199)
        tuple_var_assignment_219627_220128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 16), 'tuple_var_assignment_219627')
        # Assigning a type to the variable 'y' (line 199)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 19), 'y', tuple_var_assignment_219627_220128)
        
        # Call to save(...): (line 200)
        # Processing the call keyword arguments (line 200)
        kwargs_220131 = {}
        # Getting the type of 'ctx' (line 200)
        ctx_220129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 16), 'ctx', False)
        # Obtaining the member 'save' of a type (line 200)
        save_220130 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 16), ctx_220129, 'save')
        # Calling save(args, kwargs) (line 200)
        save_call_result_220132 = invoke(stypy.reporting.localization.Localization(__file__, 200, 16), save_220130, *[], **kwargs_220131)
        
        
        # Call to translate(...): (line 203)
        # Processing the call arguments (line 203)
        # Getting the type of 'x' (line 203)
        x_220135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 30), 'x', False)
        # Getting the type of 'y' (line 203)
        y_220136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 33), 'y', False)
        # Processing the call keyword arguments (line 203)
        kwargs_220137 = {}
        # Getting the type of 'ctx' (line 203)
        ctx_220133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 16), 'ctx', False)
        # Obtaining the member 'translate' of a type (line 203)
        translate_220134 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 16), ctx_220133, 'translate')
        # Calling translate(args, kwargs) (line 203)
        translate_call_result_220138 = invoke(stypy.reporting.localization.Localization(__file__, 203, 16), translate_220134, *[x_220135, y_220136], **kwargs_220137)
        
        
        # Call to append_path(...): (line 204)
        # Processing the call arguments (line 204)
        # Getting the type of 'marker_path' (line 204)
        marker_path_220141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 32), 'marker_path', False)
        # Processing the call keyword arguments (line 204)
        kwargs_220142 = {}
        # Getting the type of 'ctx' (line 204)
        ctx_220139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 16), 'ctx', False)
        # Obtaining the member 'append_path' of a type (line 204)
        append_path_220140 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 16), ctx_220139, 'append_path')
        # Calling append_path(args, kwargs) (line 204)
        append_path_call_result_220143 = invoke(stypy.reporting.localization.Localization(__file__, 204, 16), append_path_220140, *[marker_path_220141], **kwargs_220142)
        
        
        # Call to restore(...): (line 206)
        # Processing the call keyword arguments (line 206)
        kwargs_220146 = {}
        # Getting the type of 'ctx' (line 206)
        ctx_220144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 16), 'ctx', False)
        # Obtaining the member 'restore' of a type (line 206)
        restore_220145 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 16), ctx_220144, 'restore')
        # Calling restore(args, kwargs) (line 206)
        restore_call_result_220147 = invoke(stypy.reporting.localization.Localization(__file__, 206, 16), restore_220145, *[], **kwargs_220146)
        
        
        
        # Evaluating a boolean operation
        # Getting the type of 'filled' (line 212)
        filled_220148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 19), 'filled')
        
        # Getting the type of 'i' (line 212)
        i_220149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 29), 'i')
        int_220150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 33), 'int')
        # Applying the binary operator '%' (line 212)
        result_mod_220151 = python_operator(stypy.reporting.localization.Localization(__file__, 212, 29), '%', i_220149, int_220150)
        
        int_220152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 41), 'int')
        # Applying the binary operator '==' (line 212)
        result_eq_220153 = python_operator(stypy.reporting.localization.Localization(__file__, 212, 29), '==', result_mod_220151, int_220152)
        
        # Applying the binary operator 'or' (line 212)
        result_or_keyword_220154 = python_operator(stypy.reporting.localization.Localization(__file__, 212, 19), 'or', filled_220148, result_eq_220153)
        
        # Testing the type of an if condition (line 212)
        if_condition_220155 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 212, 16), result_or_keyword_220154)
        # Assigning a type to the variable 'if_condition_220155' (line 212)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 16), 'if_condition_220155', if_condition_220155)
        # SSA begins for if statement (line 212)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to _fill_and_stroke(...): (line 213)
        # Processing the call arguments (line 213)
        # Getting the type of 'ctx' (line 214)
        ctx_220158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 24), 'ctx', False)
        # Getting the type of 'rgbFace' (line 214)
        rgbFace_220159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 29), 'rgbFace', False)
        
        # Call to get_alpha(...): (line 214)
        # Processing the call keyword arguments (line 214)
        kwargs_220162 = {}
        # Getting the type of 'gc' (line 214)
        gc_220160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 38), 'gc', False)
        # Obtaining the member 'get_alpha' of a type (line 214)
        get_alpha_220161 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 38), gc_220160, 'get_alpha')
        # Calling get_alpha(args, kwargs) (line 214)
        get_alpha_call_result_220163 = invoke(stypy.reporting.localization.Localization(__file__, 214, 38), get_alpha_220161, *[], **kwargs_220162)
        
        
        # Call to get_forced_alpha(...): (line 214)
        # Processing the call keyword arguments (line 214)
        kwargs_220166 = {}
        # Getting the type of 'gc' (line 214)
        gc_220164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 54), 'gc', False)
        # Obtaining the member 'get_forced_alpha' of a type (line 214)
        get_forced_alpha_220165 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 54), gc_220164, 'get_forced_alpha')
        # Calling get_forced_alpha(args, kwargs) (line 214)
        get_forced_alpha_call_result_220167 = invoke(stypy.reporting.localization.Localization(__file__, 214, 54), get_forced_alpha_220165, *[], **kwargs_220166)
        
        # Processing the call keyword arguments (line 213)
        kwargs_220168 = {}
        # Getting the type of 'self' (line 213)
        self_220156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 20), 'self', False)
        # Obtaining the member '_fill_and_stroke' of a type (line 213)
        _fill_and_stroke_220157 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 20), self_220156, '_fill_and_stroke')
        # Calling _fill_and_stroke(args, kwargs) (line 213)
        _fill_and_stroke_call_result_220169 = invoke(stypy.reporting.localization.Localization(__file__, 213, 20), _fill_and_stroke_220157, *[ctx_220158, rgbFace_220159, get_alpha_call_result_220163, get_forced_alpha_call_result_220167], **kwargs_220168)
        
        # SSA join for if statement (line 212)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 198)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'filled' (line 217)
        filled_220170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 15), 'filled')
        # Applying the 'not' unary operator (line 217)
        result_not__220171 = python_operator(stypy.reporting.localization.Localization(__file__, 217, 11), 'not', filled_220170)
        
        # Testing the type of an if condition (line 217)
        if_condition_220172 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 217, 8), result_not__220171)
        # Assigning a type to the variable 'if_condition_220172' (line 217)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 8), 'if_condition_220172', if_condition_220172)
        # SSA begins for if statement (line 217)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to _fill_and_stroke(...): (line 218)
        # Processing the call arguments (line 218)
        # Getting the type of 'ctx' (line 219)
        ctx_220175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 16), 'ctx', False)
        # Getting the type of 'rgbFace' (line 219)
        rgbFace_220176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 21), 'rgbFace', False)
        
        # Call to get_alpha(...): (line 219)
        # Processing the call keyword arguments (line 219)
        kwargs_220179 = {}
        # Getting the type of 'gc' (line 219)
        gc_220177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 30), 'gc', False)
        # Obtaining the member 'get_alpha' of a type (line 219)
        get_alpha_220178 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 30), gc_220177, 'get_alpha')
        # Calling get_alpha(args, kwargs) (line 219)
        get_alpha_call_result_220180 = invoke(stypy.reporting.localization.Localization(__file__, 219, 30), get_alpha_220178, *[], **kwargs_220179)
        
        
        # Call to get_forced_alpha(...): (line 219)
        # Processing the call keyword arguments (line 219)
        kwargs_220183 = {}
        # Getting the type of 'gc' (line 219)
        gc_220181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 46), 'gc', False)
        # Obtaining the member 'get_forced_alpha' of a type (line 219)
        get_forced_alpha_220182 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 46), gc_220181, 'get_forced_alpha')
        # Calling get_forced_alpha(args, kwargs) (line 219)
        get_forced_alpha_call_result_220184 = invoke(stypy.reporting.localization.Localization(__file__, 219, 46), get_forced_alpha_220182, *[], **kwargs_220183)
        
        # Processing the call keyword arguments (line 218)
        kwargs_220185 = {}
        # Getting the type of 'self' (line 218)
        self_220173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 12), 'self', False)
        # Obtaining the member '_fill_and_stroke' of a type (line 218)
        _fill_and_stroke_220174 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 12), self_220173, '_fill_and_stroke')
        # Calling _fill_and_stroke(args, kwargs) (line 218)
        _fill_and_stroke_call_result_220186 = invoke(stypy.reporting.localization.Localization(__file__, 218, 12), _fill_and_stroke_220174, *[ctx_220175, rgbFace_220176, get_alpha_call_result_220180, get_forced_alpha_call_result_220184], **kwargs_220185)
        
        # SSA join for if statement (line 217)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'draw_markers(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'draw_markers' in the type store
        # Getting the type of 'stypy_return_type' (line 173)
        stypy_return_type_220187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_220187)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'draw_markers'
        return stypy_return_type_220187


    @norecursion
    def draw_image(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'draw_image'
        module_type_store = module_type_store.open_function_context('draw_image', 221, 4, False)
        # Assigning a type to the variable 'self' (line 222)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        RendererCairo.draw_image.__dict__.__setitem__('stypy_localization', localization)
        RendererCairo.draw_image.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        RendererCairo.draw_image.__dict__.__setitem__('stypy_type_store', module_type_store)
        RendererCairo.draw_image.__dict__.__setitem__('stypy_function_name', 'RendererCairo.draw_image')
        RendererCairo.draw_image.__dict__.__setitem__('stypy_param_names_list', ['gc', 'x', 'y', 'im'])
        RendererCairo.draw_image.__dict__.__setitem__('stypy_varargs_param_name', None)
        RendererCairo.draw_image.__dict__.__setitem__('stypy_kwargs_param_name', None)
        RendererCairo.draw_image.__dict__.__setitem__('stypy_call_defaults', defaults)
        RendererCairo.draw_image.__dict__.__setitem__('stypy_call_varargs', varargs)
        RendererCairo.draw_image.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        RendererCairo.draw_image.__dict__.__setitem__('stypy_declared_arg_number', 5)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RendererCairo.draw_image', ['gc', 'x', 'y', 'im'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'draw_image', localization, ['gc', 'x', 'y', 'im'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'draw_image(...)' code ##################

        
        
        # Getting the type of 'sys' (line 223)
        sys_220188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 11), 'sys')
        # Obtaining the member 'byteorder' of a type (line 223)
        byteorder_220189 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 11), sys_220188, 'byteorder')
        unicode_220190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 28), 'unicode', u'little')
        # Applying the binary operator '==' (line 223)
        result_eq_220191 = python_operator(stypy.reporting.localization.Localization(__file__, 223, 11), '==', byteorder_220189, unicode_220190)
        
        # Testing the type of an if condition (line 223)
        if_condition_220192 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 223, 8), result_eq_220191)
        # Assigning a type to the variable 'if_condition_220192' (line 223)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'if_condition_220192', if_condition_220192)
        # SSA begins for if statement (line 223)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Name (line 224):
        
        # Assigning a Subscript to a Name (line 224):
        
        # Obtaining the type of the subscript
        slice_220193 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 224, 17), None, None, None)
        slice_220194 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 224, 17), None, None, None)
        
        # Obtaining an instance of the builtin type 'tuple' (line 224)
        tuple_220195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 27), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 224)
        # Adding element type (line 224)
        int_220196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 224, 27), tuple_220195, int_220196)
        # Adding element type (line 224)
        int_220197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 224, 27), tuple_220195, int_220197)
        # Adding element type (line 224)
        int_220198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 224, 27), tuple_220195, int_220198)
        # Adding element type (line 224)
        int_220199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 224, 27), tuple_220195, int_220199)
        
        # Getting the type of 'im' (line 224)
        im_220200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 17), 'im')
        # Obtaining the member '__getitem__' of a type (line 224)
        getitem___220201 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 17), im_220200, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 224)
        subscript_call_result_220202 = invoke(stypy.reporting.localization.Localization(__file__, 224, 17), getitem___220201, (slice_220193, slice_220194, tuple_220195))
        
        # Assigning a type to the variable 'im' (line 224)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 12), 'im', subscript_call_result_220202)
        # SSA branch for the else part of an if statement (line 223)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Subscript to a Name (line 226):
        
        # Assigning a Subscript to a Name (line 226):
        
        # Obtaining the type of the subscript
        slice_220203 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 226, 17), None, None, None)
        slice_220204 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 226, 17), None, None, None)
        
        # Obtaining an instance of the builtin type 'tuple' (line 226)
        tuple_220205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 27), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 226)
        # Adding element type (line 226)
        int_220206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 226, 27), tuple_220205, int_220206)
        # Adding element type (line 226)
        int_220207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 226, 27), tuple_220205, int_220207)
        # Adding element type (line 226)
        int_220208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 226, 27), tuple_220205, int_220208)
        # Adding element type (line 226)
        int_220209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 226, 27), tuple_220205, int_220209)
        
        # Getting the type of 'im' (line 226)
        im_220210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 17), 'im')
        # Obtaining the member '__getitem__' of a type (line 226)
        getitem___220211 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 17), im_220210, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 226)
        subscript_call_result_220212 = invoke(stypy.reporting.localization.Localization(__file__, 226, 17), getitem___220211, (slice_220203, slice_220204, tuple_220205))
        
        # Assigning a type to the variable 'im' (line 226)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 12), 'im', subscript_call_result_220212)
        # SSA join for if statement (line 223)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'HAS_CAIRO_CFFI' (line 227)
        HAS_CAIRO_CFFI_220213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 11), 'HAS_CAIRO_CFFI')
        # Testing the type of an if condition (line 227)
        if_condition_220214 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 227, 8), HAS_CAIRO_CFFI_220213)
        # Assigning a type to the variable 'if_condition_220214' (line 227)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'if_condition_220214', if_condition_220214)
        # SSA begins for if statement (line 227)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 233):
        
        # Assigning a Call to a Name (line 233):
        
        # Call to ArrayWrapper(...): (line 233)
        # Processing the call arguments (line 233)
        
        # Call to flatten(...): (line 233)
        # Processing the call keyword arguments (line 233)
        kwargs_220218 = {}
        # Getting the type of 'im' (line 233)
        im_220216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 36), 'im', False)
        # Obtaining the member 'flatten' of a type (line 233)
        flatten_220217 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 36), im_220216, 'flatten')
        # Calling flatten(args, kwargs) (line 233)
        flatten_call_result_220219 = invoke(stypy.reporting.localization.Localization(__file__, 233, 36), flatten_220217, *[], **kwargs_220218)
        
        # Processing the call keyword arguments (line 233)
        kwargs_220220 = {}
        # Getting the type of 'ArrayWrapper' (line 233)
        ArrayWrapper_220215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 23), 'ArrayWrapper', False)
        # Calling ArrayWrapper(args, kwargs) (line 233)
        ArrayWrapper_call_result_220221 = invoke(stypy.reporting.localization.Localization(__file__, 233, 23), ArrayWrapper_220215, *[flatten_call_result_220219], **kwargs_220220)
        
        # Assigning a type to the variable 'imbuffer' (line 233)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 12), 'imbuffer', ArrayWrapper_call_result_220221)
        # SSA branch for the else part of an if statement (line 227)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 240):
        
        # Assigning a Call to a Name (line 240):
        
        # Call to flatten(...): (line 240)
        # Processing the call keyword arguments (line 240)
        kwargs_220224 = {}
        # Getting the type of 'im' (line 240)
        im_220222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 23), 'im', False)
        # Obtaining the member 'flatten' of a type (line 240)
        flatten_220223 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 23), im_220222, 'flatten')
        # Calling flatten(args, kwargs) (line 240)
        flatten_call_result_220225 = invoke(stypy.reporting.localization.Localization(__file__, 240, 23), flatten_220223, *[], **kwargs_220224)
        
        # Assigning a type to the variable 'imbuffer' (line 240)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 12), 'imbuffer', flatten_call_result_220225)
        # SSA join for if statement (line 227)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 241):
        
        # Assigning a Call to a Name (line 241):
        
        # Call to create_for_data(...): (line 241)
        # Processing the call arguments (line 241)
        # Getting the type of 'imbuffer' (line 242)
        imbuffer_220229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 12), 'imbuffer', False)
        # Getting the type of 'cairo' (line 242)
        cairo_220230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 22), 'cairo', False)
        # Obtaining the member 'FORMAT_ARGB32' of a type (line 242)
        FORMAT_ARGB32_220231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 22), cairo_220230, 'FORMAT_ARGB32')
        
        # Obtaining the type of the subscript
        int_220232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 21), 'int')
        # Getting the type of 'im' (line 243)
        im_220233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 12), 'im', False)
        # Obtaining the member 'shape' of a type (line 243)
        shape_220234 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 12), im_220233, 'shape')
        # Obtaining the member '__getitem__' of a type (line 243)
        getitem___220235 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 12), shape_220234, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 243)
        subscript_call_result_220236 = invoke(stypy.reporting.localization.Localization(__file__, 243, 12), getitem___220235, int_220232)
        
        
        # Obtaining the type of the subscript
        int_220237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 34), 'int')
        # Getting the type of 'im' (line 243)
        im_220238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 25), 'im', False)
        # Obtaining the member 'shape' of a type (line 243)
        shape_220239 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 25), im_220238, 'shape')
        # Obtaining the member '__getitem__' of a type (line 243)
        getitem___220240 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 25), shape_220239, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 243)
        subscript_call_result_220241 = invoke(stypy.reporting.localization.Localization(__file__, 243, 25), getitem___220240, int_220237)
        
        
        # Obtaining the type of the subscript
        int_220242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 47), 'int')
        # Getting the type of 'im' (line 243)
        im_220243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 38), 'im', False)
        # Obtaining the member 'shape' of a type (line 243)
        shape_220244 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 38), im_220243, 'shape')
        # Obtaining the member '__getitem__' of a type (line 243)
        getitem___220245 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 38), shape_220244, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 243)
        subscript_call_result_220246 = invoke(stypy.reporting.localization.Localization(__file__, 243, 38), getitem___220245, int_220242)
        
        int_220247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 50), 'int')
        # Applying the binary operator '*' (line 243)
        result_mul_220248 = python_operator(stypy.reporting.localization.Localization(__file__, 243, 38), '*', subscript_call_result_220246, int_220247)
        
        # Processing the call keyword arguments (line 241)
        kwargs_220249 = {}
        # Getting the type of 'cairo' (line 241)
        cairo_220226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 18), 'cairo', False)
        # Obtaining the member 'ImageSurface' of a type (line 241)
        ImageSurface_220227 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 18), cairo_220226, 'ImageSurface')
        # Obtaining the member 'create_for_data' of a type (line 241)
        create_for_data_220228 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 18), ImageSurface_220227, 'create_for_data')
        # Calling create_for_data(args, kwargs) (line 241)
        create_for_data_call_result_220250 = invoke(stypy.reporting.localization.Localization(__file__, 241, 18), create_for_data_220228, *[imbuffer_220229, FORMAT_ARGB32_220231, subscript_call_result_220236, subscript_call_result_220241, result_mul_220248], **kwargs_220249)
        
        # Assigning a type to the variable 'surface' (line 241)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 8), 'surface', create_for_data_call_result_220250)
        
        # Assigning a Attribute to a Name (line 244):
        
        # Assigning a Attribute to a Name (line 244):
        # Getting the type of 'gc' (line 244)
        gc_220251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 14), 'gc')
        # Obtaining the member 'ctx' of a type (line 244)
        ctx_220252 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 14), gc_220251, 'ctx')
        # Assigning a type to the variable 'ctx' (line 244)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 8), 'ctx', ctx_220252)
        
        # Assigning a BinOp to a Name (line 245):
        
        # Assigning a BinOp to a Name (line 245):
        # Getting the type of 'self' (line 245)
        self_220253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 12), 'self')
        # Obtaining the member 'height' of a type (line 245)
        height_220254 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 12), self_220253, 'height')
        # Getting the type of 'y' (line 245)
        y_220255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 26), 'y')
        # Applying the binary operator '-' (line 245)
        result_sub_220256 = python_operator(stypy.reporting.localization.Localization(__file__, 245, 12), '-', height_220254, y_220255)
        
        
        # Obtaining the type of the subscript
        int_220257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 39), 'int')
        # Getting the type of 'im' (line 245)
        im_220258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 30), 'im')
        # Obtaining the member 'shape' of a type (line 245)
        shape_220259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 30), im_220258, 'shape')
        # Obtaining the member '__getitem__' of a type (line 245)
        getitem___220260 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 30), shape_220259, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 245)
        subscript_call_result_220261 = invoke(stypy.reporting.localization.Localization(__file__, 245, 30), getitem___220260, int_220257)
        
        # Applying the binary operator '-' (line 245)
        result_sub_220262 = python_operator(stypy.reporting.localization.Localization(__file__, 245, 28), '-', result_sub_220256, subscript_call_result_220261)
        
        # Assigning a type to the variable 'y' (line 245)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 8), 'y', result_sub_220262)
        
        # Call to save(...): (line 247)
        # Processing the call keyword arguments (line 247)
        kwargs_220265 = {}
        # Getting the type of 'ctx' (line 247)
        ctx_220263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 8), 'ctx', False)
        # Obtaining the member 'save' of a type (line 247)
        save_220264 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 8), ctx_220263, 'save')
        # Calling save(args, kwargs) (line 247)
        save_call_result_220266 = invoke(stypy.reporting.localization.Localization(__file__, 247, 8), save_220264, *[], **kwargs_220265)
        
        
        # Call to set_source_surface(...): (line 248)
        # Processing the call arguments (line 248)
        # Getting the type of 'surface' (line 248)
        surface_220269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 31), 'surface', False)
        
        # Call to float(...): (line 248)
        # Processing the call arguments (line 248)
        # Getting the type of 'x' (line 248)
        x_220271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 46), 'x', False)
        # Processing the call keyword arguments (line 248)
        kwargs_220272 = {}
        # Getting the type of 'float' (line 248)
        float_220270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 40), 'float', False)
        # Calling float(args, kwargs) (line 248)
        float_call_result_220273 = invoke(stypy.reporting.localization.Localization(__file__, 248, 40), float_220270, *[x_220271], **kwargs_220272)
        
        
        # Call to float(...): (line 248)
        # Processing the call arguments (line 248)
        # Getting the type of 'y' (line 248)
        y_220275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 56), 'y', False)
        # Processing the call keyword arguments (line 248)
        kwargs_220276 = {}
        # Getting the type of 'float' (line 248)
        float_220274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 50), 'float', False)
        # Calling float(args, kwargs) (line 248)
        float_call_result_220277 = invoke(stypy.reporting.localization.Localization(__file__, 248, 50), float_220274, *[y_220275], **kwargs_220276)
        
        # Processing the call keyword arguments (line 248)
        kwargs_220278 = {}
        # Getting the type of 'ctx' (line 248)
        ctx_220267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 8), 'ctx', False)
        # Obtaining the member 'set_source_surface' of a type (line 248)
        set_source_surface_220268 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 8), ctx_220267, 'set_source_surface')
        # Calling set_source_surface(args, kwargs) (line 248)
        set_source_surface_call_result_220279 = invoke(stypy.reporting.localization.Localization(__file__, 248, 8), set_source_surface_220268, *[surface_220269, float_call_result_220273, float_call_result_220277], **kwargs_220278)
        
        
        
        
        # Call to get_alpha(...): (line 249)
        # Processing the call keyword arguments (line 249)
        kwargs_220282 = {}
        # Getting the type of 'gc' (line 249)
        gc_220280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 11), 'gc', False)
        # Obtaining the member 'get_alpha' of a type (line 249)
        get_alpha_220281 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 11), gc_220280, 'get_alpha')
        # Calling get_alpha(args, kwargs) (line 249)
        get_alpha_call_result_220283 = invoke(stypy.reporting.localization.Localization(__file__, 249, 11), get_alpha_220281, *[], **kwargs_220282)
        
        float_220284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 29), 'float')
        # Applying the binary operator '!=' (line 249)
        result_ne_220285 = python_operator(stypy.reporting.localization.Localization(__file__, 249, 11), '!=', get_alpha_call_result_220283, float_220284)
        
        # Testing the type of an if condition (line 249)
        if_condition_220286 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 249, 8), result_ne_220285)
        # Assigning a type to the variable 'if_condition_220286' (line 249)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 8), 'if_condition_220286', if_condition_220286)
        # SSA begins for if statement (line 249)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to paint_with_alpha(...): (line 250)
        # Processing the call arguments (line 250)
        
        # Call to get_alpha(...): (line 250)
        # Processing the call keyword arguments (line 250)
        kwargs_220291 = {}
        # Getting the type of 'gc' (line 250)
        gc_220289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 33), 'gc', False)
        # Obtaining the member 'get_alpha' of a type (line 250)
        get_alpha_220290 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 33), gc_220289, 'get_alpha')
        # Calling get_alpha(args, kwargs) (line 250)
        get_alpha_call_result_220292 = invoke(stypy.reporting.localization.Localization(__file__, 250, 33), get_alpha_220290, *[], **kwargs_220291)
        
        # Processing the call keyword arguments (line 250)
        kwargs_220293 = {}
        # Getting the type of 'ctx' (line 250)
        ctx_220287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 12), 'ctx', False)
        # Obtaining the member 'paint_with_alpha' of a type (line 250)
        paint_with_alpha_220288 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 12), ctx_220287, 'paint_with_alpha')
        # Calling paint_with_alpha(args, kwargs) (line 250)
        paint_with_alpha_call_result_220294 = invoke(stypy.reporting.localization.Localization(__file__, 250, 12), paint_with_alpha_220288, *[get_alpha_call_result_220292], **kwargs_220293)
        
        # SSA branch for the else part of an if statement (line 249)
        module_type_store.open_ssa_branch('else')
        
        # Call to paint(...): (line 252)
        # Processing the call keyword arguments (line 252)
        kwargs_220297 = {}
        # Getting the type of 'ctx' (line 252)
        ctx_220295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 12), 'ctx', False)
        # Obtaining the member 'paint' of a type (line 252)
        paint_220296 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 12), ctx_220295, 'paint')
        # Calling paint(args, kwargs) (line 252)
        paint_call_result_220298 = invoke(stypy.reporting.localization.Localization(__file__, 252, 12), paint_220296, *[], **kwargs_220297)
        
        # SSA join for if statement (line 249)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to restore(...): (line 253)
        # Processing the call keyword arguments (line 253)
        kwargs_220301 = {}
        # Getting the type of 'ctx' (line 253)
        ctx_220299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 8), 'ctx', False)
        # Obtaining the member 'restore' of a type (line 253)
        restore_220300 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 8), ctx_220299, 'restore')
        # Calling restore(args, kwargs) (line 253)
        restore_call_result_220302 = invoke(stypy.reporting.localization.Localization(__file__, 253, 8), restore_220300, *[], **kwargs_220301)
        
        
        # ################# End of 'draw_image(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'draw_image' in the type store
        # Getting the type of 'stypy_return_type' (line 221)
        stypy_return_type_220303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_220303)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'draw_image'
        return stypy_return_type_220303


    @norecursion
    def draw_text(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'False' (line 255)
        False_220304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 57), 'False')
        # Getting the type of 'None' (line 255)
        None_220305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 70), 'None')
        defaults = [False_220304, None_220305]
        # Create a new context for function 'draw_text'
        module_type_store = module_type_store.open_function_context('draw_text', 255, 4, False)
        # Assigning a type to the variable 'self' (line 256)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        RendererCairo.draw_text.__dict__.__setitem__('stypy_localization', localization)
        RendererCairo.draw_text.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        RendererCairo.draw_text.__dict__.__setitem__('stypy_type_store', module_type_store)
        RendererCairo.draw_text.__dict__.__setitem__('stypy_function_name', 'RendererCairo.draw_text')
        RendererCairo.draw_text.__dict__.__setitem__('stypy_param_names_list', ['gc', 'x', 'y', 's', 'prop', 'angle', 'ismath', 'mtext'])
        RendererCairo.draw_text.__dict__.__setitem__('stypy_varargs_param_name', None)
        RendererCairo.draw_text.__dict__.__setitem__('stypy_kwargs_param_name', None)
        RendererCairo.draw_text.__dict__.__setitem__('stypy_call_defaults', defaults)
        RendererCairo.draw_text.__dict__.__setitem__('stypy_call_varargs', varargs)
        RendererCairo.draw_text.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        RendererCairo.draw_text.__dict__.__setitem__('stypy_declared_arg_number', 9)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RendererCairo.draw_text', ['gc', 'x', 'y', 's', 'prop', 'angle', 'ismath', 'mtext'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'draw_text', localization, ['gc', 'x', 'y', 's', 'prop', 'angle', 'ismath', 'mtext'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'draw_text(...)' code ##################

        
        # Getting the type of 'ismath' (line 258)
        ismath_220306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 11), 'ismath')
        # Testing the type of an if condition (line 258)
        if_condition_220307 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 258, 8), ismath_220306)
        # Assigning a type to the variable 'if_condition_220307' (line 258)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 8), 'if_condition_220307', if_condition_220307)
        # SSA begins for if statement (line 258)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to _draw_mathtext(...): (line 259)
        # Processing the call arguments (line 259)
        # Getting the type of 'gc' (line 259)
        gc_220310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 32), 'gc', False)
        # Getting the type of 'x' (line 259)
        x_220311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 36), 'x', False)
        # Getting the type of 'y' (line 259)
        y_220312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 39), 'y', False)
        # Getting the type of 's' (line 259)
        s_220313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 42), 's', False)
        # Getting the type of 'prop' (line 259)
        prop_220314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 45), 'prop', False)
        # Getting the type of 'angle' (line 259)
        angle_220315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 51), 'angle', False)
        # Processing the call keyword arguments (line 259)
        kwargs_220316 = {}
        # Getting the type of 'self' (line 259)
        self_220308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 12), 'self', False)
        # Obtaining the member '_draw_mathtext' of a type (line 259)
        _draw_mathtext_220309 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 12), self_220308, '_draw_mathtext')
        # Calling _draw_mathtext(args, kwargs) (line 259)
        _draw_mathtext_call_result_220317 = invoke(stypy.reporting.localization.Localization(__file__, 259, 12), _draw_mathtext_220309, *[gc_220310, x_220311, y_220312, s_220313, prop_220314, angle_220315], **kwargs_220316)
        
        # SSA branch for the else part of an if statement (line 258)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Attribute to a Name (line 262):
        
        # Assigning a Attribute to a Name (line 262):
        # Getting the type of 'gc' (line 262)
        gc_220318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 18), 'gc')
        # Obtaining the member 'ctx' of a type (line 262)
        ctx_220319 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 18), gc_220318, 'ctx')
        # Assigning a type to the variable 'ctx' (line 262)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 12), 'ctx', ctx_220319)
        
        # Call to new_path(...): (line 263)
        # Processing the call keyword arguments (line 263)
        kwargs_220322 = {}
        # Getting the type of 'ctx' (line 263)
        ctx_220320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 12), 'ctx', False)
        # Obtaining the member 'new_path' of a type (line 263)
        new_path_220321 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 12), ctx_220320, 'new_path')
        # Calling new_path(args, kwargs) (line 263)
        new_path_call_result_220323 = invoke(stypy.reporting.localization.Localization(__file__, 263, 12), new_path_220321, *[], **kwargs_220322)
        
        
        # Call to move_to(...): (line 264)
        # Processing the call arguments (line 264)
        # Getting the type of 'x' (line 264)
        x_220326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 24), 'x', False)
        # Getting the type of 'y' (line 264)
        y_220327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 27), 'y', False)
        # Processing the call keyword arguments (line 264)
        kwargs_220328 = {}
        # Getting the type of 'ctx' (line 264)
        ctx_220324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 12), 'ctx', False)
        # Obtaining the member 'move_to' of a type (line 264)
        move_to_220325 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 12), ctx_220324, 'move_to')
        # Calling move_to(args, kwargs) (line 264)
        move_to_call_result_220329 = invoke(stypy.reporting.localization.Localization(__file__, 264, 12), move_to_220325, *[x_220326, y_220327], **kwargs_220328)
        
        
        # Call to select_font_face(...): (line 265)
        # Processing the call arguments (line 265)
        
        # Call to get_name(...): (line 265)
        # Processing the call keyword arguments (line 265)
        kwargs_220334 = {}
        # Getting the type of 'prop' (line 265)
        prop_220332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 33), 'prop', False)
        # Obtaining the member 'get_name' of a type (line 265)
        get_name_220333 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 33), prop_220332, 'get_name')
        # Calling get_name(args, kwargs) (line 265)
        get_name_call_result_220335 = invoke(stypy.reporting.localization.Localization(__file__, 265, 33), get_name_220333, *[], **kwargs_220334)
        
        
        # Obtaining the type of the subscript
        
        # Call to get_style(...): (line 266)
        # Processing the call keyword arguments (line 266)
        kwargs_220338 = {}
        # Getting the type of 'prop' (line 266)
        prop_220336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 49), 'prop', False)
        # Obtaining the member 'get_style' of a type (line 266)
        get_style_220337 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 49), prop_220336, 'get_style')
        # Calling get_style(args, kwargs) (line 266)
        get_style_call_result_220339 = invoke(stypy.reporting.localization.Localization(__file__, 266, 49), get_style_220337, *[], **kwargs_220338)
        
        # Getting the type of 'self' (line 266)
        self_220340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 33), 'self', False)
        # Obtaining the member 'fontangles' of a type (line 266)
        fontangles_220341 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 33), self_220340, 'fontangles')
        # Obtaining the member '__getitem__' of a type (line 266)
        getitem___220342 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 33), fontangles_220341, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 266)
        subscript_call_result_220343 = invoke(stypy.reporting.localization.Localization(__file__, 266, 33), getitem___220342, get_style_call_result_220339)
        
        
        # Obtaining the type of the subscript
        
        # Call to get_weight(...): (line 267)
        # Processing the call keyword arguments (line 267)
        kwargs_220346 = {}
        # Getting the type of 'prop' (line 267)
        prop_220344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 50), 'prop', False)
        # Obtaining the member 'get_weight' of a type (line 267)
        get_weight_220345 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 267, 50), prop_220344, 'get_weight')
        # Calling get_weight(args, kwargs) (line 267)
        get_weight_call_result_220347 = invoke(stypy.reporting.localization.Localization(__file__, 267, 50), get_weight_220345, *[], **kwargs_220346)
        
        # Getting the type of 'self' (line 267)
        self_220348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 33), 'self', False)
        # Obtaining the member 'fontweights' of a type (line 267)
        fontweights_220349 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 267, 33), self_220348, 'fontweights')
        # Obtaining the member '__getitem__' of a type (line 267)
        getitem___220350 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 267, 33), fontweights_220349, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 267)
        subscript_call_result_220351 = invoke(stypy.reporting.localization.Localization(__file__, 267, 33), getitem___220350, get_weight_call_result_220347)
        
        # Processing the call keyword arguments (line 265)
        kwargs_220352 = {}
        # Getting the type of 'ctx' (line 265)
        ctx_220330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 12), 'ctx', False)
        # Obtaining the member 'select_font_face' of a type (line 265)
        select_font_face_220331 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 12), ctx_220330, 'select_font_face')
        # Calling select_font_face(args, kwargs) (line 265)
        select_font_face_call_result_220353 = invoke(stypy.reporting.localization.Localization(__file__, 265, 12), select_font_face_220331, *[get_name_call_result_220335, subscript_call_result_220343, subscript_call_result_220351], **kwargs_220352)
        
        
        # Assigning a BinOp to a Name (line 269):
        
        # Assigning a BinOp to a Name (line 269):
        
        # Call to get_size_in_points(...): (line 269)
        # Processing the call keyword arguments (line 269)
        kwargs_220356 = {}
        # Getting the type of 'prop' (line 269)
        prop_220354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 19), 'prop', False)
        # Obtaining the member 'get_size_in_points' of a type (line 269)
        get_size_in_points_220355 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 19), prop_220354, 'get_size_in_points')
        # Calling get_size_in_points(args, kwargs) (line 269)
        get_size_in_points_call_result_220357 = invoke(stypy.reporting.localization.Localization(__file__, 269, 19), get_size_in_points_220355, *[], **kwargs_220356)
        
        # Getting the type of 'self' (line 269)
        self_220358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 47), 'self')
        # Obtaining the member 'dpi' of a type (line 269)
        dpi_220359 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 47), self_220358, 'dpi')
        # Applying the binary operator '*' (line 269)
        result_mul_220360 = python_operator(stypy.reporting.localization.Localization(__file__, 269, 19), '*', get_size_in_points_call_result_220357, dpi_220359)
        
        float_220361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 58), 'float')
        # Applying the binary operator 'div' (line 269)
        result_div_220362 = python_operator(stypy.reporting.localization.Localization(__file__, 269, 56), 'div', result_mul_220360, float_220361)
        
        # Assigning a type to the variable 'size' (line 269)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 12), 'size', result_div_220362)
        
        # Call to save(...): (line 271)
        # Processing the call keyword arguments (line 271)
        kwargs_220365 = {}
        # Getting the type of 'ctx' (line 271)
        ctx_220363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 12), 'ctx', False)
        # Obtaining the member 'save' of a type (line 271)
        save_220364 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 12), ctx_220363, 'save')
        # Calling save(args, kwargs) (line 271)
        save_call_result_220366 = invoke(stypy.reporting.localization.Localization(__file__, 271, 12), save_220364, *[], **kwargs_220365)
        
        
        # Getting the type of 'angle' (line 272)
        angle_220367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 15), 'angle')
        # Testing the type of an if condition (line 272)
        if_condition_220368 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 272, 12), angle_220367)
        # Assigning a type to the variable 'if_condition_220368' (line 272)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 12), 'if_condition_220368', if_condition_220368)
        # SSA begins for if statement (line 272)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to rotate(...): (line 273)
        # Processing the call arguments (line 273)
        
        # Call to deg2rad(...): (line 273)
        # Processing the call arguments (line 273)
        
        # Getting the type of 'angle' (line 273)
        angle_220373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 39), 'angle', False)
        # Applying the 'usub' unary operator (line 273)
        result___neg___220374 = python_operator(stypy.reporting.localization.Localization(__file__, 273, 38), 'usub', angle_220373)
        
        # Processing the call keyword arguments (line 273)
        kwargs_220375 = {}
        # Getting the type of 'np' (line 273)
        np_220371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 27), 'np', False)
        # Obtaining the member 'deg2rad' of a type (line 273)
        deg2rad_220372 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 27), np_220371, 'deg2rad')
        # Calling deg2rad(args, kwargs) (line 273)
        deg2rad_call_result_220376 = invoke(stypy.reporting.localization.Localization(__file__, 273, 27), deg2rad_220372, *[result___neg___220374], **kwargs_220375)
        
        # Processing the call keyword arguments (line 273)
        kwargs_220377 = {}
        # Getting the type of 'ctx' (line 273)
        ctx_220369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 16), 'ctx', False)
        # Obtaining the member 'rotate' of a type (line 273)
        rotate_220370 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 16), ctx_220369, 'rotate')
        # Calling rotate(args, kwargs) (line 273)
        rotate_call_result_220378 = invoke(stypy.reporting.localization.Localization(__file__, 273, 16), rotate_220370, *[deg2rad_call_result_220376], **kwargs_220377)
        
        # SSA join for if statement (line 272)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to set_font_size(...): (line 274)
        # Processing the call arguments (line 274)
        # Getting the type of 'size' (line 274)
        size_220381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 30), 'size', False)
        # Processing the call keyword arguments (line 274)
        kwargs_220382 = {}
        # Getting the type of 'ctx' (line 274)
        ctx_220379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 12), 'ctx', False)
        # Obtaining the member 'set_font_size' of a type (line 274)
        set_font_size_220380 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 12), ctx_220379, 'set_font_size')
        # Calling set_font_size(args, kwargs) (line 274)
        set_font_size_call_result_220383 = invoke(stypy.reporting.localization.Localization(__file__, 274, 12), set_font_size_220380, *[size_220381], **kwargs_220382)
        
        
        # Getting the type of 'HAS_CAIRO_CFFI' (line 276)
        HAS_CAIRO_CFFI_220384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 15), 'HAS_CAIRO_CFFI')
        # Testing the type of an if condition (line 276)
        if_condition_220385 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 276, 12), HAS_CAIRO_CFFI_220384)
        # Assigning a type to the variable 'if_condition_220385' (line 276)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 12), 'if_condition_220385', if_condition_220385)
        # SSA begins for if statement (line 276)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        
        # Call to isinstance(...): (line 277)
        # Processing the call arguments (line 277)
        # Getting the type of 's' (line 277)
        s_220387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 34), 's', False)
        # Getting the type of 'six' (line 277)
        six_220388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 37), 'six', False)
        # Obtaining the member 'text_type' of a type (line 277)
        text_type_220389 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 37), six_220388, 'text_type')
        # Processing the call keyword arguments (line 277)
        kwargs_220390 = {}
        # Getting the type of 'isinstance' (line 277)
        isinstance_220386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 23), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 277)
        isinstance_call_result_220391 = invoke(stypy.reporting.localization.Localization(__file__, 277, 23), isinstance_220386, *[s_220387, text_type_220389], **kwargs_220390)
        
        # Applying the 'not' unary operator (line 277)
        result_not__220392 = python_operator(stypy.reporting.localization.Localization(__file__, 277, 19), 'not', isinstance_call_result_220391)
        
        # Testing the type of an if condition (line 277)
        if_condition_220393 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 277, 16), result_not__220392)
        # Assigning a type to the variable 'if_condition_220393' (line 277)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 16), 'if_condition_220393', if_condition_220393)
        # SSA begins for if statement (line 277)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 278):
        
        # Assigning a Call to a Name (line 278):
        
        # Call to text_type(...): (line 278)
        # Processing the call arguments (line 278)
        # Getting the type of 's' (line 278)
        s_220396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 38), 's', False)
        # Processing the call keyword arguments (line 278)
        kwargs_220397 = {}
        # Getting the type of 'six' (line 278)
        six_220394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 24), 'six', False)
        # Obtaining the member 'text_type' of a type (line 278)
        text_type_220395 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 24), six_220394, 'text_type')
        # Calling text_type(args, kwargs) (line 278)
        text_type_call_result_220398 = invoke(stypy.reporting.localization.Localization(__file__, 278, 24), text_type_220395, *[s_220396], **kwargs_220397)
        
        # Assigning a type to the variable 's' (line 278)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 20), 's', text_type_call_result_220398)
        # SSA join for if statement (line 277)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 276)
        module_type_store.open_ssa_branch('else')
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'six' (line 280)
        six_220399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 23), 'six')
        # Obtaining the member 'PY3' of a type (line 280)
        PY3_220400 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 23), six_220399, 'PY3')
        # Applying the 'not' unary operator (line 280)
        result_not__220401 = python_operator(stypy.reporting.localization.Localization(__file__, 280, 19), 'not', PY3_220400)
        
        
        # Call to isinstance(...): (line 280)
        # Processing the call arguments (line 280)
        # Getting the type of 's' (line 280)
        s_220403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 46), 's', False)
        # Getting the type of 'six' (line 280)
        six_220404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 49), 'six', False)
        # Obtaining the member 'text_type' of a type (line 280)
        text_type_220405 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 49), six_220404, 'text_type')
        # Processing the call keyword arguments (line 280)
        kwargs_220406 = {}
        # Getting the type of 'isinstance' (line 280)
        isinstance_220402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 35), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 280)
        isinstance_call_result_220407 = invoke(stypy.reporting.localization.Localization(__file__, 280, 35), isinstance_220402, *[s_220403, text_type_220405], **kwargs_220406)
        
        # Applying the binary operator 'and' (line 280)
        result_and_keyword_220408 = python_operator(stypy.reporting.localization.Localization(__file__, 280, 19), 'and', result_not__220401, isinstance_call_result_220407)
        
        # Testing the type of an if condition (line 280)
        if_condition_220409 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 280, 16), result_and_keyword_220408)
        # Assigning a type to the variable 'if_condition_220409' (line 280)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 16), 'if_condition_220409', if_condition_220409)
        # SSA begins for if statement (line 280)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 281):
        
        # Assigning a Call to a Name (line 281):
        
        # Call to encode(...): (line 281)
        # Processing the call arguments (line 281)
        unicode_220412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 33), 'unicode', u'utf-8')
        # Processing the call keyword arguments (line 281)
        kwargs_220413 = {}
        # Getting the type of 's' (line 281)
        s_220410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 24), 's', False)
        # Obtaining the member 'encode' of a type (line 281)
        encode_220411 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 24), s_220410, 'encode')
        # Calling encode(args, kwargs) (line 281)
        encode_call_result_220414 = invoke(stypy.reporting.localization.Localization(__file__, 281, 24), encode_220411, *[unicode_220412], **kwargs_220413)
        
        # Assigning a type to the variable 's' (line 281)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 20), 's', encode_call_result_220414)
        # SSA join for if statement (line 280)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 276)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to show_text(...): (line 283)
        # Processing the call arguments (line 283)
        # Getting the type of 's' (line 283)
        s_220417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 26), 's', False)
        # Processing the call keyword arguments (line 283)
        kwargs_220418 = {}
        # Getting the type of 'ctx' (line 283)
        ctx_220415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 12), 'ctx', False)
        # Obtaining the member 'show_text' of a type (line 283)
        show_text_220416 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 12), ctx_220415, 'show_text')
        # Calling show_text(args, kwargs) (line 283)
        show_text_call_result_220419 = invoke(stypy.reporting.localization.Localization(__file__, 283, 12), show_text_220416, *[s_220417], **kwargs_220418)
        
        
        # Call to restore(...): (line 284)
        # Processing the call keyword arguments (line 284)
        kwargs_220422 = {}
        # Getting the type of 'ctx' (line 284)
        ctx_220420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 12), 'ctx', False)
        # Obtaining the member 'restore' of a type (line 284)
        restore_220421 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 12), ctx_220420, 'restore')
        # Calling restore(args, kwargs) (line 284)
        restore_call_result_220423 = invoke(stypy.reporting.localization.Localization(__file__, 284, 12), restore_220421, *[], **kwargs_220422)
        
        # SSA join for if statement (line 258)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'draw_text(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'draw_text' in the type store
        # Getting the type of 'stypy_return_type' (line 255)
        stypy_return_type_220424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_220424)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'draw_text'
        return stypy_return_type_220424


    @norecursion
    def _draw_mathtext(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_draw_mathtext'
        module_type_store = module_type_store.open_function_context('_draw_mathtext', 286, 4, False)
        # Assigning a type to the variable 'self' (line 287)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        RendererCairo._draw_mathtext.__dict__.__setitem__('stypy_localization', localization)
        RendererCairo._draw_mathtext.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        RendererCairo._draw_mathtext.__dict__.__setitem__('stypy_type_store', module_type_store)
        RendererCairo._draw_mathtext.__dict__.__setitem__('stypy_function_name', 'RendererCairo._draw_mathtext')
        RendererCairo._draw_mathtext.__dict__.__setitem__('stypy_param_names_list', ['gc', 'x', 'y', 's', 'prop', 'angle'])
        RendererCairo._draw_mathtext.__dict__.__setitem__('stypy_varargs_param_name', None)
        RendererCairo._draw_mathtext.__dict__.__setitem__('stypy_kwargs_param_name', None)
        RendererCairo._draw_mathtext.__dict__.__setitem__('stypy_call_defaults', defaults)
        RendererCairo._draw_mathtext.__dict__.__setitem__('stypy_call_varargs', varargs)
        RendererCairo._draw_mathtext.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        RendererCairo._draw_mathtext.__dict__.__setitem__('stypy_declared_arg_number', 7)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RendererCairo._draw_mathtext', ['gc', 'x', 'y', 's', 'prop', 'angle'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_draw_mathtext', localization, ['gc', 'x', 'y', 's', 'prop', 'angle'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_draw_mathtext(...)' code ##################

        
        # Assigning a Attribute to a Name (line 287):
        
        # Assigning a Attribute to a Name (line 287):
        # Getting the type of 'gc' (line 287)
        gc_220425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 14), 'gc')
        # Obtaining the member 'ctx' of a type (line 287)
        ctx_220426 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 14), gc_220425, 'ctx')
        # Assigning a type to the variable 'ctx' (line 287)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 8), 'ctx', ctx_220426)
        
        # Assigning a Call to a Tuple (line 288):
        
        # Assigning a Call to a Name:
        
        # Call to parse(...): (line 288)
        # Processing the call arguments (line 288)
        # Getting the type of 's' (line 289)
        s_220430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 12), 's', False)
        # Getting the type of 'self' (line 289)
        self_220431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 15), 'self', False)
        # Obtaining the member 'dpi' of a type (line 289)
        dpi_220432 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 15), self_220431, 'dpi')
        # Getting the type of 'prop' (line 289)
        prop_220433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 25), 'prop', False)
        # Processing the call keyword arguments (line 288)
        kwargs_220434 = {}
        # Getting the type of 'self' (line 288)
        self_220427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 48), 'self', False)
        # Obtaining the member 'mathtext_parser' of a type (line 288)
        mathtext_parser_220428 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 48), self_220427, 'mathtext_parser')
        # Obtaining the member 'parse' of a type (line 288)
        parse_220429 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 48), mathtext_parser_220428, 'parse')
        # Calling parse(args, kwargs) (line 288)
        parse_call_result_220435 = invoke(stypy.reporting.localization.Localization(__file__, 288, 48), parse_220429, *[s_220430, dpi_220432, prop_220433], **kwargs_220434)
        
        # Assigning a type to the variable 'call_assignment_219628' (line 288)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 8), 'call_assignment_219628', parse_call_result_220435)
        
        # Assigning a Call to a Name (line 288):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_220438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 8), 'int')
        # Processing the call keyword arguments
        kwargs_220439 = {}
        # Getting the type of 'call_assignment_219628' (line 288)
        call_assignment_219628_220436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 8), 'call_assignment_219628', False)
        # Obtaining the member '__getitem__' of a type (line 288)
        getitem___220437 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 8), call_assignment_219628_220436, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_220440 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___220437, *[int_220438], **kwargs_220439)
        
        # Assigning a type to the variable 'call_assignment_219629' (line 288)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 8), 'call_assignment_219629', getitem___call_result_220440)
        
        # Assigning a Name to a Name (line 288):
        # Getting the type of 'call_assignment_219629' (line 288)
        call_assignment_219629_220441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 8), 'call_assignment_219629')
        # Assigning a type to the variable 'width' (line 288)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 8), 'width', call_assignment_219629_220441)
        
        # Assigning a Call to a Name (line 288):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_220444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 8), 'int')
        # Processing the call keyword arguments
        kwargs_220445 = {}
        # Getting the type of 'call_assignment_219628' (line 288)
        call_assignment_219628_220442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 8), 'call_assignment_219628', False)
        # Obtaining the member '__getitem__' of a type (line 288)
        getitem___220443 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 8), call_assignment_219628_220442, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_220446 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___220443, *[int_220444], **kwargs_220445)
        
        # Assigning a type to the variable 'call_assignment_219630' (line 288)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 8), 'call_assignment_219630', getitem___call_result_220446)
        
        # Assigning a Name to a Name (line 288):
        # Getting the type of 'call_assignment_219630' (line 288)
        call_assignment_219630_220447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 8), 'call_assignment_219630')
        # Assigning a type to the variable 'height' (line 288)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 15), 'height', call_assignment_219630_220447)
        
        # Assigning a Call to a Name (line 288):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_220450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 8), 'int')
        # Processing the call keyword arguments
        kwargs_220451 = {}
        # Getting the type of 'call_assignment_219628' (line 288)
        call_assignment_219628_220448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 8), 'call_assignment_219628', False)
        # Obtaining the member '__getitem__' of a type (line 288)
        getitem___220449 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 8), call_assignment_219628_220448, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_220452 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___220449, *[int_220450], **kwargs_220451)
        
        # Assigning a type to the variable 'call_assignment_219631' (line 288)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 8), 'call_assignment_219631', getitem___call_result_220452)
        
        # Assigning a Name to a Name (line 288):
        # Getting the type of 'call_assignment_219631' (line 288)
        call_assignment_219631_220453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 8), 'call_assignment_219631')
        # Assigning a type to the variable 'descent' (line 288)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 23), 'descent', call_assignment_219631_220453)
        
        # Assigning a Call to a Name (line 288):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_220456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 8), 'int')
        # Processing the call keyword arguments
        kwargs_220457 = {}
        # Getting the type of 'call_assignment_219628' (line 288)
        call_assignment_219628_220454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 8), 'call_assignment_219628', False)
        # Obtaining the member '__getitem__' of a type (line 288)
        getitem___220455 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 8), call_assignment_219628_220454, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_220458 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___220455, *[int_220456], **kwargs_220457)
        
        # Assigning a type to the variable 'call_assignment_219632' (line 288)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 8), 'call_assignment_219632', getitem___call_result_220458)
        
        # Assigning a Name to a Name (line 288):
        # Getting the type of 'call_assignment_219632' (line 288)
        call_assignment_219632_220459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 8), 'call_assignment_219632')
        # Assigning a type to the variable 'glyphs' (line 288)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 32), 'glyphs', call_assignment_219632_220459)
        
        # Assigning a Call to a Name (line 288):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_220462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 8), 'int')
        # Processing the call keyword arguments
        kwargs_220463 = {}
        # Getting the type of 'call_assignment_219628' (line 288)
        call_assignment_219628_220460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 8), 'call_assignment_219628', False)
        # Obtaining the member '__getitem__' of a type (line 288)
        getitem___220461 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 8), call_assignment_219628_220460, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_220464 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___220461, *[int_220462], **kwargs_220463)
        
        # Assigning a type to the variable 'call_assignment_219633' (line 288)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 8), 'call_assignment_219633', getitem___call_result_220464)
        
        # Assigning a Name to a Name (line 288):
        # Getting the type of 'call_assignment_219633' (line 288)
        call_assignment_219633_220465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 8), 'call_assignment_219633')
        # Assigning a type to the variable 'rects' (line 288)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 40), 'rects', call_assignment_219633_220465)
        
        # Call to save(...): (line 291)
        # Processing the call keyword arguments (line 291)
        kwargs_220468 = {}
        # Getting the type of 'ctx' (line 291)
        ctx_220466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 8), 'ctx', False)
        # Obtaining the member 'save' of a type (line 291)
        save_220467 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 8), ctx_220466, 'save')
        # Calling save(args, kwargs) (line 291)
        save_call_result_220469 = invoke(stypy.reporting.localization.Localization(__file__, 291, 8), save_220467, *[], **kwargs_220468)
        
        
        # Call to translate(...): (line 292)
        # Processing the call arguments (line 292)
        # Getting the type of 'x' (line 292)
        x_220472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 22), 'x', False)
        # Getting the type of 'y' (line 292)
        y_220473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 25), 'y', False)
        # Processing the call keyword arguments (line 292)
        kwargs_220474 = {}
        # Getting the type of 'ctx' (line 292)
        ctx_220470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 8), 'ctx', False)
        # Obtaining the member 'translate' of a type (line 292)
        translate_220471 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 8), ctx_220470, 'translate')
        # Calling translate(args, kwargs) (line 292)
        translate_call_result_220475 = invoke(stypy.reporting.localization.Localization(__file__, 292, 8), translate_220471, *[x_220472, y_220473], **kwargs_220474)
        
        
        # Getting the type of 'angle' (line 293)
        angle_220476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 11), 'angle')
        # Testing the type of an if condition (line 293)
        if_condition_220477 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 293, 8), angle_220476)
        # Assigning a type to the variable 'if_condition_220477' (line 293)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 8), 'if_condition_220477', if_condition_220477)
        # SSA begins for if statement (line 293)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to rotate(...): (line 294)
        # Processing the call arguments (line 294)
        
        # Call to deg2rad(...): (line 294)
        # Processing the call arguments (line 294)
        
        # Getting the type of 'angle' (line 294)
        angle_220482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 35), 'angle', False)
        # Applying the 'usub' unary operator (line 294)
        result___neg___220483 = python_operator(stypy.reporting.localization.Localization(__file__, 294, 34), 'usub', angle_220482)
        
        # Processing the call keyword arguments (line 294)
        kwargs_220484 = {}
        # Getting the type of 'np' (line 294)
        np_220480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 23), 'np', False)
        # Obtaining the member 'deg2rad' of a type (line 294)
        deg2rad_220481 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 23), np_220480, 'deg2rad')
        # Calling deg2rad(args, kwargs) (line 294)
        deg2rad_call_result_220485 = invoke(stypy.reporting.localization.Localization(__file__, 294, 23), deg2rad_220481, *[result___neg___220483], **kwargs_220484)
        
        # Processing the call keyword arguments (line 294)
        kwargs_220486 = {}
        # Getting the type of 'ctx' (line 294)
        ctx_220478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 12), 'ctx', False)
        # Obtaining the member 'rotate' of a type (line 294)
        rotate_220479 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 12), ctx_220478, 'rotate')
        # Calling rotate(args, kwargs) (line 294)
        rotate_call_result_220487 = invoke(stypy.reporting.localization.Localization(__file__, 294, 12), rotate_220479, *[deg2rad_call_result_220485], **kwargs_220486)
        
        # SSA join for if statement (line 293)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'glyphs' (line 296)
        glyphs_220488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 41), 'glyphs')
        # Testing the type of a for loop iterable (line 296)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 296, 8), glyphs_220488)
        # Getting the type of the for loop variable (line 296)
        for_loop_var_220489 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 296, 8), glyphs_220488)
        # Assigning a type to the variable 'font' (line 296)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 296, 8), 'font', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 296, 8), for_loop_var_220489))
        # Assigning a type to the variable 'fontsize' (line 296)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 296, 8), 'fontsize', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 296, 8), for_loop_var_220489))
        # Assigning a type to the variable 's' (line 296)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 296, 8), 's', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 296, 8), for_loop_var_220489))
        # Assigning a type to the variable 'ox' (line 296)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 296, 8), 'ox', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 296, 8), for_loop_var_220489))
        # Assigning a type to the variable 'oy' (line 296)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 296, 8), 'oy', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 296, 8), for_loop_var_220489))
        # SSA begins for a for statement (line 296)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to new_path(...): (line 297)
        # Processing the call keyword arguments (line 297)
        kwargs_220492 = {}
        # Getting the type of 'ctx' (line 297)
        ctx_220490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 12), 'ctx', False)
        # Obtaining the member 'new_path' of a type (line 297)
        new_path_220491 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 297, 12), ctx_220490, 'new_path')
        # Calling new_path(args, kwargs) (line 297)
        new_path_call_result_220493 = invoke(stypy.reporting.localization.Localization(__file__, 297, 12), new_path_220491, *[], **kwargs_220492)
        
        
        # Call to move_to(...): (line 298)
        # Processing the call arguments (line 298)
        # Getting the type of 'ox' (line 298)
        ox_220496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 24), 'ox', False)
        # Getting the type of 'oy' (line 298)
        oy_220497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 28), 'oy', False)
        # Processing the call keyword arguments (line 298)
        kwargs_220498 = {}
        # Getting the type of 'ctx' (line 298)
        ctx_220494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 12), 'ctx', False)
        # Obtaining the member 'move_to' of a type (line 298)
        move_to_220495 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 12), ctx_220494, 'move_to')
        # Calling move_to(args, kwargs) (line 298)
        move_to_call_result_220499 = invoke(stypy.reporting.localization.Localization(__file__, 298, 12), move_to_220495, *[ox_220496, oy_220497], **kwargs_220498)
        
        
        # Assigning a Call to a Name (line 300):
        
        # Assigning a Call to a Name (line 300):
        
        # Call to ttfFontProperty(...): (line 300)
        # Processing the call arguments (line 300)
        # Getting the type of 'font' (line 300)
        font_220501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 39), 'font', False)
        # Processing the call keyword arguments (line 300)
        kwargs_220502 = {}
        # Getting the type of 'ttfFontProperty' (line 300)
        ttfFontProperty_220500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 23), 'ttfFontProperty', False)
        # Calling ttfFontProperty(args, kwargs) (line 300)
        ttfFontProperty_call_result_220503 = invoke(stypy.reporting.localization.Localization(__file__, 300, 23), ttfFontProperty_220500, *[font_220501], **kwargs_220502)
        
        # Assigning a type to the variable 'fontProp' (line 300)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 12), 'fontProp', ttfFontProperty_call_result_220503)
        
        # Call to save(...): (line 301)
        # Processing the call keyword arguments (line 301)
        kwargs_220506 = {}
        # Getting the type of 'ctx' (line 301)
        ctx_220504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 12), 'ctx', False)
        # Obtaining the member 'save' of a type (line 301)
        save_220505 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 12), ctx_220504, 'save')
        # Calling save(args, kwargs) (line 301)
        save_call_result_220507 = invoke(stypy.reporting.localization.Localization(__file__, 301, 12), save_220505, *[], **kwargs_220506)
        
        
        # Call to select_font_face(...): (line 302)
        # Processing the call arguments (line 302)
        # Getting the type of 'fontProp' (line 302)
        fontProp_220510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 33), 'fontProp', False)
        # Obtaining the member 'name' of a type (line 302)
        name_220511 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 33), fontProp_220510, 'name')
        
        # Obtaining the type of the subscript
        # Getting the type of 'fontProp' (line 303)
        fontProp_220512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 49), 'fontProp', False)
        # Obtaining the member 'style' of a type (line 303)
        style_220513 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 49), fontProp_220512, 'style')
        # Getting the type of 'self' (line 303)
        self_220514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 33), 'self', False)
        # Obtaining the member 'fontangles' of a type (line 303)
        fontangles_220515 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 33), self_220514, 'fontangles')
        # Obtaining the member '__getitem__' of a type (line 303)
        getitem___220516 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 33), fontangles_220515, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 303)
        subscript_call_result_220517 = invoke(stypy.reporting.localization.Localization(__file__, 303, 33), getitem___220516, style_220513)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'fontProp' (line 304)
        fontProp_220518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 50), 'fontProp', False)
        # Obtaining the member 'weight' of a type (line 304)
        weight_220519 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 50), fontProp_220518, 'weight')
        # Getting the type of 'self' (line 304)
        self_220520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 33), 'self', False)
        # Obtaining the member 'fontweights' of a type (line 304)
        fontweights_220521 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 33), self_220520, 'fontweights')
        # Obtaining the member '__getitem__' of a type (line 304)
        getitem___220522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 33), fontweights_220521, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 304)
        subscript_call_result_220523 = invoke(stypy.reporting.localization.Localization(__file__, 304, 33), getitem___220522, weight_220519)
        
        # Processing the call keyword arguments (line 302)
        kwargs_220524 = {}
        # Getting the type of 'ctx' (line 302)
        ctx_220508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 12), 'ctx', False)
        # Obtaining the member 'select_font_face' of a type (line 302)
        select_font_face_220509 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 12), ctx_220508, 'select_font_face')
        # Calling select_font_face(args, kwargs) (line 302)
        select_font_face_call_result_220525 = invoke(stypy.reporting.localization.Localization(__file__, 302, 12), select_font_face_220509, *[name_220511, subscript_call_result_220517, subscript_call_result_220523], **kwargs_220524)
        
        
        # Assigning a BinOp to a Name (line 306):
        
        # Assigning a BinOp to a Name (line 306):
        # Getting the type of 'fontsize' (line 306)
        fontsize_220526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 19), 'fontsize')
        # Getting the type of 'self' (line 306)
        self_220527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 30), 'self')
        # Obtaining the member 'dpi' of a type (line 306)
        dpi_220528 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 30), self_220527, 'dpi')
        # Applying the binary operator '*' (line 306)
        result_mul_220529 = python_operator(stypy.reporting.localization.Localization(__file__, 306, 19), '*', fontsize_220526, dpi_220528)
        
        float_220530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 41), 'float')
        # Applying the binary operator 'div' (line 306)
        result_div_220531 = python_operator(stypy.reporting.localization.Localization(__file__, 306, 39), 'div', result_mul_220529, float_220530)
        
        # Assigning a type to the variable 'size' (line 306)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 12), 'size', result_div_220531)
        
        # Call to set_font_size(...): (line 307)
        # Processing the call arguments (line 307)
        # Getting the type of 'size' (line 307)
        size_220534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 30), 'size', False)
        # Processing the call keyword arguments (line 307)
        kwargs_220535 = {}
        # Getting the type of 'ctx' (line 307)
        ctx_220532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 12), 'ctx', False)
        # Obtaining the member 'set_font_size' of a type (line 307)
        set_font_size_220533 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 12), ctx_220532, 'set_font_size')
        # Calling set_font_size(args, kwargs) (line 307)
        set_font_size_call_result_220536 = invoke(stypy.reporting.localization.Localization(__file__, 307, 12), set_font_size_220533, *[size_220534], **kwargs_220535)
        
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'six' (line 308)
        six_220537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 19), 'six')
        # Obtaining the member 'PY3' of a type (line 308)
        PY3_220538 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 19), six_220537, 'PY3')
        # Applying the 'not' unary operator (line 308)
        result_not__220539 = python_operator(stypy.reporting.localization.Localization(__file__, 308, 15), 'not', PY3_220538)
        
        
        # Call to isinstance(...): (line 308)
        # Processing the call arguments (line 308)
        # Getting the type of 's' (line 308)
        s_220541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 42), 's', False)
        # Getting the type of 'six' (line 308)
        six_220542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 45), 'six', False)
        # Obtaining the member 'text_type' of a type (line 308)
        text_type_220543 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 45), six_220542, 'text_type')
        # Processing the call keyword arguments (line 308)
        kwargs_220544 = {}
        # Getting the type of 'isinstance' (line 308)
        isinstance_220540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 31), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 308)
        isinstance_call_result_220545 = invoke(stypy.reporting.localization.Localization(__file__, 308, 31), isinstance_220540, *[s_220541, text_type_220543], **kwargs_220544)
        
        # Applying the binary operator 'and' (line 308)
        result_and_keyword_220546 = python_operator(stypy.reporting.localization.Localization(__file__, 308, 15), 'and', result_not__220539, isinstance_call_result_220545)
        
        # Testing the type of an if condition (line 308)
        if_condition_220547 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 308, 12), result_and_keyword_220546)
        # Assigning a type to the variable 'if_condition_220547' (line 308)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 308, 12), 'if_condition_220547', if_condition_220547)
        # SSA begins for if statement (line 308)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 309):
        
        # Assigning a Call to a Name (line 309):
        
        # Call to encode(...): (line 309)
        # Processing the call arguments (line 309)
        unicode_220550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 29), 'unicode', u'utf-8')
        # Processing the call keyword arguments (line 309)
        kwargs_220551 = {}
        # Getting the type of 's' (line 309)
        s_220548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 20), 's', False)
        # Obtaining the member 'encode' of a type (line 309)
        encode_220549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 20), s_220548, 'encode')
        # Calling encode(args, kwargs) (line 309)
        encode_call_result_220552 = invoke(stypy.reporting.localization.Localization(__file__, 309, 20), encode_220549, *[unicode_220550], **kwargs_220551)
        
        # Assigning a type to the variable 's' (line 309)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 16), 's', encode_call_result_220552)
        # SSA join for if statement (line 308)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to show_text(...): (line 310)
        # Processing the call arguments (line 310)
        # Getting the type of 's' (line 310)
        s_220555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 26), 's', False)
        # Processing the call keyword arguments (line 310)
        kwargs_220556 = {}
        # Getting the type of 'ctx' (line 310)
        ctx_220553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 12), 'ctx', False)
        # Obtaining the member 'show_text' of a type (line 310)
        show_text_220554 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 12), ctx_220553, 'show_text')
        # Calling show_text(args, kwargs) (line 310)
        show_text_call_result_220557 = invoke(stypy.reporting.localization.Localization(__file__, 310, 12), show_text_220554, *[s_220555], **kwargs_220556)
        
        
        # Call to restore(...): (line 311)
        # Processing the call keyword arguments (line 311)
        kwargs_220560 = {}
        # Getting the type of 'ctx' (line 311)
        ctx_220558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 12), 'ctx', False)
        # Obtaining the member 'restore' of a type (line 311)
        restore_220559 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 12), ctx_220558, 'restore')
        # Calling restore(args, kwargs) (line 311)
        restore_call_result_220561 = invoke(stypy.reporting.localization.Localization(__file__, 311, 12), restore_220559, *[], **kwargs_220560)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'rects' (line 313)
        rects_220562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 28), 'rects')
        # Testing the type of a for loop iterable (line 313)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 313, 8), rects_220562)
        # Getting the type of the for loop variable (line 313)
        for_loop_var_220563 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 313, 8), rects_220562)
        # Assigning a type to the variable 'ox' (line 313)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 313, 8), 'ox', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 313, 8), for_loop_var_220563))
        # Assigning a type to the variable 'oy' (line 313)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 313, 8), 'oy', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 313, 8), for_loop_var_220563))
        # Assigning a type to the variable 'w' (line 313)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 313, 8), 'w', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 313, 8), for_loop_var_220563))
        # Assigning a type to the variable 'h' (line 313)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 313, 8), 'h', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 313, 8), for_loop_var_220563))
        # SSA begins for a for statement (line 313)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to new_path(...): (line 314)
        # Processing the call keyword arguments (line 314)
        kwargs_220566 = {}
        # Getting the type of 'ctx' (line 314)
        ctx_220564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 12), 'ctx', False)
        # Obtaining the member 'new_path' of a type (line 314)
        new_path_220565 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 12), ctx_220564, 'new_path')
        # Calling new_path(args, kwargs) (line 314)
        new_path_call_result_220567 = invoke(stypy.reporting.localization.Localization(__file__, 314, 12), new_path_220565, *[], **kwargs_220566)
        
        
        # Call to rectangle(...): (line 315)
        # Processing the call arguments (line 315)
        # Getting the type of 'ox' (line 315)
        ox_220570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 26), 'ox', False)
        # Getting the type of 'oy' (line 315)
        oy_220571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 30), 'oy', False)
        # Getting the type of 'w' (line 315)
        w_220572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 34), 'w', False)
        # Getting the type of 'h' (line 315)
        h_220573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 37), 'h', False)
        # Processing the call keyword arguments (line 315)
        kwargs_220574 = {}
        # Getting the type of 'ctx' (line 315)
        ctx_220568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 12), 'ctx', False)
        # Obtaining the member 'rectangle' of a type (line 315)
        rectangle_220569 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 315, 12), ctx_220568, 'rectangle')
        # Calling rectangle(args, kwargs) (line 315)
        rectangle_call_result_220575 = invoke(stypy.reporting.localization.Localization(__file__, 315, 12), rectangle_220569, *[ox_220570, oy_220571, w_220572, h_220573], **kwargs_220574)
        
        
        # Call to set_source_rgb(...): (line 316)
        # Processing the call arguments (line 316)
        int_220578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 316, 31), 'int')
        int_220579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 316, 34), 'int')
        int_220580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 316, 37), 'int')
        # Processing the call keyword arguments (line 316)
        kwargs_220581 = {}
        # Getting the type of 'ctx' (line 316)
        ctx_220576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 12), 'ctx', False)
        # Obtaining the member 'set_source_rgb' of a type (line 316)
        set_source_rgb_220577 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 12), ctx_220576, 'set_source_rgb')
        # Calling set_source_rgb(args, kwargs) (line 316)
        set_source_rgb_call_result_220582 = invoke(stypy.reporting.localization.Localization(__file__, 316, 12), set_source_rgb_220577, *[int_220578, int_220579, int_220580], **kwargs_220581)
        
        
        # Call to fill_preserve(...): (line 317)
        # Processing the call keyword arguments (line 317)
        kwargs_220585 = {}
        # Getting the type of 'ctx' (line 317)
        ctx_220583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 12), 'ctx', False)
        # Obtaining the member 'fill_preserve' of a type (line 317)
        fill_preserve_220584 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 12), ctx_220583, 'fill_preserve')
        # Calling fill_preserve(args, kwargs) (line 317)
        fill_preserve_call_result_220586 = invoke(stypy.reporting.localization.Localization(__file__, 317, 12), fill_preserve_220584, *[], **kwargs_220585)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to restore(...): (line 319)
        # Processing the call keyword arguments (line 319)
        kwargs_220589 = {}
        # Getting the type of 'ctx' (line 319)
        ctx_220587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 8), 'ctx', False)
        # Obtaining the member 'restore' of a type (line 319)
        restore_220588 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 319, 8), ctx_220587, 'restore')
        # Calling restore(args, kwargs) (line 319)
        restore_call_result_220590 = invoke(stypy.reporting.localization.Localization(__file__, 319, 8), restore_220588, *[], **kwargs_220589)
        
        
        # ################# End of '_draw_mathtext(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_draw_mathtext' in the type store
        # Getting the type of 'stypy_return_type' (line 286)
        stypy_return_type_220591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_220591)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_draw_mathtext'
        return stypy_return_type_220591


    @norecursion
    def flipy(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'flipy'
        module_type_store = module_type_store.open_function_context('flipy', 321, 4, False)
        # Assigning a type to the variable 'self' (line 322)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 322, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        RendererCairo.flipy.__dict__.__setitem__('stypy_localization', localization)
        RendererCairo.flipy.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        RendererCairo.flipy.__dict__.__setitem__('stypy_type_store', module_type_store)
        RendererCairo.flipy.__dict__.__setitem__('stypy_function_name', 'RendererCairo.flipy')
        RendererCairo.flipy.__dict__.__setitem__('stypy_param_names_list', [])
        RendererCairo.flipy.__dict__.__setitem__('stypy_varargs_param_name', None)
        RendererCairo.flipy.__dict__.__setitem__('stypy_kwargs_param_name', None)
        RendererCairo.flipy.__dict__.__setitem__('stypy_call_defaults', defaults)
        RendererCairo.flipy.__dict__.__setitem__('stypy_call_varargs', varargs)
        RendererCairo.flipy.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        RendererCairo.flipy.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RendererCairo.flipy', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'flipy', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'flipy(...)' code ##################

        # Getting the type of 'True' (line 322)
        True_220592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 15), 'True')
        # Assigning a type to the variable 'stypy_return_type' (line 322)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 322, 8), 'stypy_return_type', True_220592)
        
        # ################# End of 'flipy(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'flipy' in the type store
        # Getting the type of 'stypy_return_type' (line 321)
        stypy_return_type_220593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_220593)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'flipy'
        return stypy_return_type_220593


    @norecursion
    def get_canvas_width_height(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_canvas_width_height'
        module_type_store = module_type_store.open_function_context('get_canvas_width_height', 326, 4, False)
        # Assigning a type to the variable 'self' (line 327)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 327, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        RendererCairo.get_canvas_width_height.__dict__.__setitem__('stypy_localization', localization)
        RendererCairo.get_canvas_width_height.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        RendererCairo.get_canvas_width_height.__dict__.__setitem__('stypy_type_store', module_type_store)
        RendererCairo.get_canvas_width_height.__dict__.__setitem__('stypy_function_name', 'RendererCairo.get_canvas_width_height')
        RendererCairo.get_canvas_width_height.__dict__.__setitem__('stypy_param_names_list', [])
        RendererCairo.get_canvas_width_height.__dict__.__setitem__('stypy_varargs_param_name', None)
        RendererCairo.get_canvas_width_height.__dict__.__setitem__('stypy_kwargs_param_name', None)
        RendererCairo.get_canvas_width_height.__dict__.__setitem__('stypy_call_defaults', defaults)
        RendererCairo.get_canvas_width_height.__dict__.__setitem__('stypy_call_varargs', varargs)
        RendererCairo.get_canvas_width_height.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        RendererCairo.get_canvas_width_height.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RendererCairo.get_canvas_width_height', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_canvas_width_height', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_canvas_width_height(...)' code ##################

        
        # Obtaining an instance of the builtin type 'tuple' (line 327)
        tuple_220594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 327)
        # Adding element type (line 327)
        # Getting the type of 'self' (line 327)
        self_220595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 15), 'self')
        # Obtaining the member 'width' of a type (line 327)
        width_220596 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 15), self_220595, 'width')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 327, 15), tuple_220594, width_220596)
        # Adding element type (line 327)
        # Getting the type of 'self' (line 327)
        self_220597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 27), 'self')
        # Obtaining the member 'height' of a type (line 327)
        height_220598 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 27), self_220597, 'height')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 327, 15), tuple_220594, height_220598)
        
        # Assigning a type to the variable 'stypy_return_type' (line 327)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 327, 8), 'stypy_return_type', tuple_220594)
        
        # ################# End of 'get_canvas_width_height(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_canvas_width_height' in the type store
        # Getting the type of 'stypy_return_type' (line 326)
        stypy_return_type_220599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_220599)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_canvas_width_height'
        return stypy_return_type_220599


    @norecursion
    def get_text_width_height_descent(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_text_width_height_descent'
        module_type_store = module_type_store.open_function_context('get_text_width_height_descent', 329, 4, False)
        # Assigning a type to the variable 'self' (line 330)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        RendererCairo.get_text_width_height_descent.__dict__.__setitem__('stypy_localization', localization)
        RendererCairo.get_text_width_height_descent.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        RendererCairo.get_text_width_height_descent.__dict__.__setitem__('stypy_type_store', module_type_store)
        RendererCairo.get_text_width_height_descent.__dict__.__setitem__('stypy_function_name', 'RendererCairo.get_text_width_height_descent')
        RendererCairo.get_text_width_height_descent.__dict__.__setitem__('stypy_param_names_list', ['s', 'prop', 'ismath'])
        RendererCairo.get_text_width_height_descent.__dict__.__setitem__('stypy_varargs_param_name', None)
        RendererCairo.get_text_width_height_descent.__dict__.__setitem__('stypy_kwargs_param_name', None)
        RendererCairo.get_text_width_height_descent.__dict__.__setitem__('stypy_call_defaults', defaults)
        RendererCairo.get_text_width_height_descent.__dict__.__setitem__('stypy_call_varargs', varargs)
        RendererCairo.get_text_width_height_descent.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        RendererCairo.get_text_width_height_descent.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RendererCairo.get_text_width_height_descent', ['s', 'prop', 'ismath'], None, None, defaults, varargs, kwargs)

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

        
        # Getting the type of 'ismath' (line 330)
        ismath_220600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 11), 'ismath')
        # Testing the type of an if condition (line 330)
        if_condition_220601 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 330, 8), ismath_220600)
        # Assigning a type to the variable 'if_condition_220601' (line 330)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 8), 'if_condition_220601', if_condition_220601)
        # SSA begins for if statement (line 330)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Tuple (line 331):
        
        # Assigning a Call to a Name:
        
        # Call to parse(...): (line 331)
        # Processing the call arguments (line 331)
        # Getting the type of 's' (line 332)
        s_220605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 15), 's', False)
        # Getting the type of 'self' (line 332)
        self_220606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 18), 'self', False)
        # Obtaining the member 'dpi' of a type (line 332)
        dpi_220607 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 18), self_220606, 'dpi')
        # Getting the type of 'prop' (line 332)
        prop_220608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 28), 'prop', False)
        # Processing the call keyword arguments (line 331)
        kwargs_220609 = {}
        # Getting the type of 'self' (line 331)
        self_220602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 61), 'self', False)
        # Obtaining the member 'mathtext_parser' of a type (line 331)
        mathtext_parser_220603 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 61), self_220602, 'mathtext_parser')
        # Obtaining the member 'parse' of a type (line 331)
        parse_220604 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 61), mathtext_parser_220603, 'parse')
        # Calling parse(args, kwargs) (line 331)
        parse_call_result_220610 = invoke(stypy.reporting.localization.Localization(__file__, 331, 61), parse_220604, *[s_220605, dpi_220607, prop_220608], **kwargs_220609)
        
        # Assigning a type to the variable 'call_assignment_219634' (line 331)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 12), 'call_assignment_219634', parse_call_result_220610)
        
        # Assigning a Call to a Name (line 331):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_220613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 12), 'int')
        # Processing the call keyword arguments
        kwargs_220614 = {}
        # Getting the type of 'call_assignment_219634' (line 331)
        call_assignment_219634_220611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 12), 'call_assignment_219634', False)
        # Obtaining the member '__getitem__' of a type (line 331)
        getitem___220612 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 12), call_assignment_219634_220611, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_220615 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___220612, *[int_220613], **kwargs_220614)
        
        # Assigning a type to the variable 'call_assignment_219635' (line 331)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 12), 'call_assignment_219635', getitem___call_result_220615)
        
        # Assigning a Name to a Name (line 331):
        # Getting the type of 'call_assignment_219635' (line 331)
        call_assignment_219635_220616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 12), 'call_assignment_219635')
        # Assigning a type to the variable 'width' (line 331)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 12), 'width', call_assignment_219635_220616)
        
        # Assigning a Call to a Name (line 331):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_220619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 12), 'int')
        # Processing the call keyword arguments
        kwargs_220620 = {}
        # Getting the type of 'call_assignment_219634' (line 331)
        call_assignment_219634_220617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 12), 'call_assignment_219634', False)
        # Obtaining the member '__getitem__' of a type (line 331)
        getitem___220618 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 12), call_assignment_219634_220617, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_220621 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___220618, *[int_220619], **kwargs_220620)
        
        # Assigning a type to the variable 'call_assignment_219636' (line 331)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 12), 'call_assignment_219636', getitem___call_result_220621)
        
        # Assigning a Name to a Name (line 331):
        # Getting the type of 'call_assignment_219636' (line 331)
        call_assignment_219636_220622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 12), 'call_assignment_219636')
        # Assigning a type to the variable 'height' (line 331)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 19), 'height', call_assignment_219636_220622)
        
        # Assigning a Call to a Name (line 331):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_220625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 12), 'int')
        # Processing the call keyword arguments
        kwargs_220626 = {}
        # Getting the type of 'call_assignment_219634' (line 331)
        call_assignment_219634_220623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 12), 'call_assignment_219634', False)
        # Obtaining the member '__getitem__' of a type (line 331)
        getitem___220624 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 12), call_assignment_219634_220623, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_220627 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___220624, *[int_220625], **kwargs_220626)
        
        # Assigning a type to the variable 'call_assignment_219637' (line 331)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 12), 'call_assignment_219637', getitem___call_result_220627)
        
        # Assigning a Name to a Name (line 331):
        # Getting the type of 'call_assignment_219637' (line 331)
        call_assignment_219637_220628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 12), 'call_assignment_219637')
        # Assigning a type to the variable 'descent' (line 331)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 27), 'descent', call_assignment_219637_220628)
        
        # Assigning a Call to a Name (line 331):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_220631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 12), 'int')
        # Processing the call keyword arguments
        kwargs_220632 = {}
        # Getting the type of 'call_assignment_219634' (line 331)
        call_assignment_219634_220629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 12), 'call_assignment_219634', False)
        # Obtaining the member '__getitem__' of a type (line 331)
        getitem___220630 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 12), call_assignment_219634_220629, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_220633 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___220630, *[int_220631], **kwargs_220632)
        
        # Assigning a type to the variable 'call_assignment_219638' (line 331)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 12), 'call_assignment_219638', getitem___call_result_220633)
        
        # Assigning a Name to a Name (line 331):
        # Getting the type of 'call_assignment_219638' (line 331)
        call_assignment_219638_220634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 12), 'call_assignment_219638')
        # Assigning a type to the variable 'fonts' (line 331)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 36), 'fonts', call_assignment_219638_220634)
        
        # Assigning a Call to a Name (line 331):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_220637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 12), 'int')
        # Processing the call keyword arguments
        kwargs_220638 = {}
        # Getting the type of 'call_assignment_219634' (line 331)
        call_assignment_219634_220635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 12), 'call_assignment_219634', False)
        # Obtaining the member '__getitem__' of a type (line 331)
        getitem___220636 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 12), call_assignment_219634_220635, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_220639 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___220636, *[int_220637], **kwargs_220638)
        
        # Assigning a type to the variable 'call_assignment_219639' (line 331)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 12), 'call_assignment_219639', getitem___call_result_220639)
        
        # Assigning a Name to a Name (line 331):
        # Getting the type of 'call_assignment_219639' (line 331)
        call_assignment_219639_220640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 12), 'call_assignment_219639')
        # Assigning a type to the variable 'used_characters' (line 331)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 43), 'used_characters', call_assignment_219639_220640)
        
        # Obtaining an instance of the builtin type 'tuple' (line 333)
        tuple_220641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 19), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 333)
        # Adding element type (line 333)
        # Getting the type of 'width' (line 333)
        width_220642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 19), 'width')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 333, 19), tuple_220641, width_220642)
        # Adding element type (line 333)
        # Getting the type of 'height' (line 333)
        height_220643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 26), 'height')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 333, 19), tuple_220641, height_220643)
        # Adding element type (line 333)
        # Getting the type of 'descent' (line 333)
        descent_220644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 34), 'descent')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 333, 19), tuple_220641, descent_220644)
        
        # Assigning a type to the variable 'stypy_return_type' (line 333)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 333, 12), 'stypy_return_type', tuple_220641)
        # SSA join for if statement (line 330)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Attribute to a Name (line 335):
        
        # Assigning a Attribute to a Name (line 335):
        # Getting the type of 'self' (line 335)
        self_220645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 14), 'self')
        # Obtaining the member 'text_ctx' of a type (line 335)
        text_ctx_220646 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 14), self_220645, 'text_ctx')
        # Assigning a type to the variable 'ctx' (line 335)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 8), 'ctx', text_ctx_220646)
        
        # Call to save(...): (line 336)
        # Processing the call keyword arguments (line 336)
        kwargs_220649 = {}
        # Getting the type of 'ctx' (line 336)
        ctx_220647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 8), 'ctx', False)
        # Obtaining the member 'save' of a type (line 336)
        save_220648 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 8), ctx_220647, 'save')
        # Calling save(args, kwargs) (line 336)
        save_call_result_220650 = invoke(stypy.reporting.localization.Localization(__file__, 336, 8), save_220648, *[], **kwargs_220649)
        
        
        # Call to select_font_face(...): (line 337)
        # Processing the call arguments (line 337)
        
        # Call to get_name(...): (line 337)
        # Processing the call keyword arguments (line 337)
        kwargs_220655 = {}
        # Getting the type of 'prop' (line 337)
        prop_220653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 29), 'prop', False)
        # Obtaining the member 'get_name' of a type (line 337)
        get_name_220654 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 29), prop_220653, 'get_name')
        # Calling get_name(args, kwargs) (line 337)
        get_name_call_result_220656 = invoke(stypy.reporting.localization.Localization(__file__, 337, 29), get_name_220654, *[], **kwargs_220655)
        
        
        # Obtaining the type of the subscript
        
        # Call to get_style(...): (line 338)
        # Processing the call keyword arguments (line 338)
        kwargs_220659 = {}
        # Getting the type of 'prop' (line 338)
        prop_220657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 45), 'prop', False)
        # Obtaining the member 'get_style' of a type (line 338)
        get_style_220658 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 45), prop_220657, 'get_style')
        # Calling get_style(args, kwargs) (line 338)
        get_style_call_result_220660 = invoke(stypy.reporting.localization.Localization(__file__, 338, 45), get_style_220658, *[], **kwargs_220659)
        
        # Getting the type of 'self' (line 338)
        self_220661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 29), 'self', False)
        # Obtaining the member 'fontangles' of a type (line 338)
        fontangles_220662 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 29), self_220661, 'fontangles')
        # Obtaining the member '__getitem__' of a type (line 338)
        getitem___220663 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 29), fontangles_220662, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 338)
        subscript_call_result_220664 = invoke(stypy.reporting.localization.Localization(__file__, 338, 29), getitem___220663, get_style_call_result_220660)
        
        
        # Obtaining the type of the subscript
        
        # Call to get_weight(...): (line 339)
        # Processing the call keyword arguments (line 339)
        kwargs_220667 = {}
        # Getting the type of 'prop' (line 339)
        prop_220665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 46), 'prop', False)
        # Obtaining the member 'get_weight' of a type (line 339)
        get_weight_220666 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 339, 46), prop_220665, 'get_weight')
        # Calling get_weight(args, kwargs) (line 339)
        get_weight_call_result_220668 = invoke(stypy.reporting.localization.Localization(__file__, 339, 46), get_weight_220666, *[], **kwargs_220667)
        
        # Getting the type of 'self' (line 339)
        self_220669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 29), 'self', False)
        # Obtaining the member 'fontweights' of a type (line 339)
        fontweights_220670 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 339, 29), self_220669, 'fontweights')
        # Obtaining the member '__getitem__' of a type (line 339)
        getitem___220671 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 339, 29), fontweights_220670, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 339)
        subscript_call_result_220672 = invoke(stypy.reporting.localization.Localization(__file__, 339, 29), getitem___220671, get_weight_call_result_220668)
        
        # Processing the call keyword arguments (line 337)
        kwargs_220673 = {}
        # Getting the type of 'ctx' (line 337)
        ctx_220651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 8), 'ctx', False)
        # Obtaining the member 'select_font_face' of a type (line 337)
        select_font_face_220652 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 8), ctx_220651, 'select_font_face')
        # Calling select_font_face(args, kwargs) (line 337)
        select_font_face_call_result_220674 = invoke(stypy.reporting.localization.Localization(__file__, 337, 8), select_font_face_220652, *[get_name_call_result_220656, subscript_call_result_220664, subscript_call_result_220672], **kwargs_220673)
        
        
        # Assigning a BinOp to a Name (line 343):
        
        # Assigning a BinOp to a Name (line 343):
        
        # Call to get_size_in_points(...): (line 343)
        # Processing the call keyword arguments (line 343)
        kwargs_220677 = {}
        # Getting the type of 'prop' (line 343)
        prop_220675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 15), 'prop', False)
        # Obtaining the member 'get_size_in_points' of a type (line 343)
        get_size_in_points_220676 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 15), prop_220675, 'get_size_in_points')
        # Calling get_size_in_points(args, kwargs) (line 343)
        get_size_in_points_call_result_220678 = invoke(stypy.reporting.localization.Localization(__file__, 343, 15), get_size_in_points_220676, *[], **kwargs_220677)
        
        # Getting the type of 'self' (line 343)
        self_220679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 43), 'self')
        # Obtaining the member 'dpi' of a type (line 343)
        dpi_220680 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 43), self_220679, 'dpi')
        # Applying the binary operator '*' (line 343)
        result_mul_220681 = python_operator(stypy.reporting.localization.Localization(__file__, 343, 15), '*', get_size_in_points_call_result_220678, dpi_220680)
        
        int_220682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 54), 'int')
        # Applying the binary operator 'div' (line 343)
        result_div_220683 = python_operator(stypy.reporting.localization.Localization(__file__, 343, 52), 'div', result_mul_220681, int_220682)
        
        # Assigning a type to the variable 'size' (line 343)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 8), 'size', result_div_220683)
        
        # Call to set_font_size(...): (line 348)
        # Processing the call arguments (line 348)
        # Getting the type of 'size' (line 348)
        size_220686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 26), 'size', False)
        # Processing the call keyword arguments (line 348)
        kwargs_220687 = {}
        # Getting the type of 'ctx' (line 348)
        ctx_220684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 8), 'ctx', False)
        # Obtaining the member 'set_font_size' of a type (line 348)
        set_font_size_220685 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 8), ctx_220684, 'set_font_size')
        # Calling set_font_size(args, kwargs) (line 348)
        set_font_size_call_result_220688 = invoke(stypy.reporting.localization.Localization(__file__, 348, 8), set_font_size_220685, *[size_220686], **kwargs_220687)
        
        
        # Assigning a Subscript to a Tuple (line 350):
        
        # Assigning a Subscript to a Name (line 350):
        
        # Obtaining the type of the subscript
        int_220689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 8), 'int')
        
        # Obtaining the type of the subscript
        int_220690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 46), 'int')
        int_220691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 48), 'int')
        slice_220692 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 350, 26), int_220690, int_220691, None)
        
        # Call to text_extents(...): (line 350)
        # Processing the call arguments (line 350)
        # Getting the type of 's' (line 350)
        s_220695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 43), 's', False)
        # Processing the call keyword arguments (line 350)
        kwargs_220696 = {}
        # Getting the type of 'ctx' (line 350)
        ctx_220693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 26), 'ctx', False)
        # Obtaining the member 'text_extents' of a type (line 350)
        text_extents_220694 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 26), ctx_220693, 'text_extents')
        # Calling text_extents(args, kwargs) (line 350)
        text_extents_call_result_220697 = invoke(stypy.reporting.localization.Localization(__file__, 350, 26), text_extents_220694, *[s_220695], **kwargs_220696)
        
        # Obtaining the member '__getitem__' of a type (line 350)
        getitem___220698 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 26), text_extents_call_result_220697, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 350)
        subscript_call_result_220699 = invoke(stypy.reporting.localization.Localization(__file__, 350, 26), getitem___220698, slice_220692)
        
        # Obtaining the member '__getitem__' of a type (line 350)
        getitem___220700 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 8), subscript_call_result_220699, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 350)
        subscript_call_result_220701 = invoke(stypy.reporting.localization.Localization(__file__, 350, 8), getitem___220700, int_220689)
        
        # Assigning a type to the variable 'tuple_var_assignment_219640' (line 350)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 8), 'tuple_var_assignment_219640', subscript_call_result_220701)
        
        # Assigning a Subscript to a Name (line 350):
        
        # Obtaining the type of the subscript
        int_220702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 8), 'int')
        
        # Obtaining the type of the subscript
        int_220703 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 46), 'int')
        int_220704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 48), 'int')
        slice_220705 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 350, 26), int_220703, int_220704, None)
        
        # Call to text_extents(...): (line 350)
        # Processing the call arguments (line 350)
        # Getting the type of 's' (line 350)
        s_220708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 43), 's', False)
        # Processing the call keyword arguments (line 350)
        kwargs_220709 = {}
        # Getting the type of 'ctx' (line 350)
        ctx_220706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 26), 'ctx', False)
        # Obtaining the member 'text_extents' of a type (line 350)
        text_extents_220707 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 26), ctx_220706, 'text_extents')
        # Calling text_extents(args, kwargs) (line 350)
        text_extents_call_result_220710 = invoke(stypy.reporting.localization.Localization(__file__, 350, 26), text_extents_220707, *[s_220708], **kwargs_220709)
        
        # Obtaining the member '__getitem__' of a type (line 350)
        getitem___220711 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 26), text_extents_call_result_220710, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 350)
        subscript_call_result_220712 = invoke(stypy.reporting.localization.Localization(__file__, 350, 26), getitem___220711, slice_220705)
        
        # Obtaining the member '__getitem__' of a type (line 350)
        getitem___220713 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 8), subscript_call_result_220712, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 350)
        subscript_call_result_220714 = invoke(stypy.reporting.localization.Localization(__file__, 350, 8), getitem___220713, int_220702)
        
        # Assigning a type to the variable 'tuple_var_assignment_219641' (line 350)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 8), 'tuple_var_assignment_219641', subscript_call_result_220714)
        
        # Assigning a Subscript to a Name (line 350):
        
        # Obtaining the type of the subscript
        int_220715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 8), 'int')
        
        # Obtaining the type of the subscript
        int_220716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 46), 'int')
        int_220717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 48), 'int')
        slice_220718 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 350, 26), int_220716, int_220717, None)
        
        # Call to text_extents(...): (line 350)
        # Processing the call arguments (line 350)
        # Getting the type of 's' (line 350)
        s_220721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 43), 's', False)
        # Processing the call keyword arguments (line 350)
        kwargs_220722 = {}
        # Getting the type of 'ctx' (line 350)
        ctx_220719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 26), 'ctx', False)
        # Obtaining the member 'text_extents' of a type (line 350)
        text_extents_220720 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 26), ctx_220719, 'text_extents')
        # Calling text_extents(args, kwargs) (line 350)
        text_extents_call_result_220723 = invoke(stypy.reporting.localization.Localization(__file__, 350, 26), text_extents_220720, *[s_220721], **kwargs_220722)
        
        # Obtaining the member '__getitem__' of a type (line 350)
        getitem___220724 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 26), text_extents_call_result_220723, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 350)
        subscript_call_result_220725 = invoke(stypy.reporting.localization.Localization(__file__, 350, 26), getitem___220724, slice_220718)
        
        # Obtaining the member '__getitem__' of a type (line 350)
        getitem___220726 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 8), subscript_call_result_220725, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 350)
        subscript_call_result_220727 = invoke(stypy.reporting.localization.Localization(__file__, 350, 8), getitem___220726, int_220715)
        
        # Assigning a type to the variable 'tuple_var_assignment_219642' (line 350)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 8), 'tuple_var_assignment_219642', subscript_call_result_220727)
        
        # Assigning a Name to a Name (line 350):
        # Getting the type of 'tuple_var_assignment_219640' (line 350)
        tuple_var_assignment_219640_220728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 8), 'tuple_var_assignment_219640')
        # Assigning a type to the variable 'y_bearing' (line 350)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 8), 'y_bearing', tuple_var_assignment_219640_220728)
        
        # Assigning a Name to a Name (line 350):
        # Getting the type of 'tuple_var_assignment_219641' (line 350)
        tuple_var_assignment_219641_220729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 8), 'tuple_var_assignment_219641')
        # Assigning a type to the variable 'w' (line 350)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 19), 'w', tuple_var_assignment_219641_220729)
        
        # Assigning a Name to a Name (line 350):
        # Getting the type of 'tuple_var_assignment_219642' (line 350)
        tuple_var_assignment_219642_220730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 8), 'tuple_var_assignment_219642')
        # Assigning a type to the variable 'h' (line 350)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 22), 'h', tuple_var_assignment_219642_220730)
        
        # Call to restore(...): (line 351)
        # Processing the call keyword arguments (line 351)
        kwargs_220733 = {}
        # Getting the type of 'ctx' (line 351)
        ctx_220731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 8), 'ctx', False)
        # Obtaining the member 'restore' of a type (line 351)
        restore_220732 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 351, 8), ctx_220731, 'restore')
        # Calling restore(args, kwargs) (line 351)
        restore_call_result_220734 = invoke(stypy.reporting.localization.Localization(__file__, 351, 8), restore_220732, *[], **kwargs_220733)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 353)
        tuple_220735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 353, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 353)
        # Adding element type (line 353)
        # Getting the type of 'w' (line 353)
        w_220736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 15), 'w')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 353, 15), tuple_220735, w_220736)
        # Adding element type (line 353)
        # Getting the type of 'h' (line 353)
        h_220737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 18), 'h')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 353, 15), tuple_220735, h_220737)
        # Adding element type (line 353)
        # Getting the type of 'h' (line 353)
        h_220738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 21), 'h')
        # Getting the type of 'y_bearing' (line 353)
        y_bearing_220739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 25), 'y_bearing')
        # Applying the binary operator '+' (line 353)
        result_add_220740 = python_operator(stypy.reporting.localization.Localization(__file__, 353, 21), '+', h_220738, y_bearing_220739)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 353, 15), tuple_220735, result_add_220740)
        
        # Assigning a type to the variable 'stypy_return_type' (line 353)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 353, 8), 'stypy_return_type', tuple_220735)
        
        # ################# End of 'get_text_width_height_descent(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_text_width_height_descent' in the type store
        # Getting the type of 'stypy_return_type' (line 329)
        stypy_return_type_220741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_220741)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_text_width_height_descent'
        return stypy_return_type_220741


    @norecursion
    def new_gc(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'new_gc'
        module_type_store = module_type_store.open_function_context('new_gc', 355, 4, False)
        # Assigning a type to the variable 'self' (line 356)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 356, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        RendererCairo.new_gc.__dict__.__setitem__('stypy_localization', localization)
        RendererCairo.new_gc.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        RendererCairo.new_gc.__dict__.__setitem__('stypy_type_store', module_type_store)
        RendererCairo.new_gc.__dict__.__setitem__('stypy_function_name', 'RendererCairo.new_gc')
        RendererCairo.new_gc.__dict__.__setitem__('stypy_param_names_list', [])
        RendererCairo.new_gc.__dict__.__setitem__('stypy_varargs_param_name', None)
        RendererCairo.new_gc.__dict__.__setitem__('stypy_kwargs_param_name', None)
        RendererCairo.new_gc.__dict__.__setitem__('stypy_call_defaults', defaults)
        RendererCairo.new_gc.__dict__.__setitem__('stypy_call_varargs', varargs)
        RendererCairo.new_gc.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        RendererCairo.new_gc.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RendererCairo.new_gc', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'new_gc', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'new_gc(...)' code ##################

        
        # Call to save(...): (line 356)
        # Processing the call keyword arguments (line 356)
        kwargs_220746 = {}
        # Getting the type of 'self' (line 356)
        self_220742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 8), 'self', False)
        # Obtaining the member 'gc' of a type (line 356)
        gc_220743 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 356, 8), self_220742, 'gc')
        # Obtaining the member 'ctx' of a type (line 356)
        ctx_220744 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 356, 8), gc_220743, 'ctx')
        # Obtaining the member 'save' of a type (line 356)
        save_220745 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 356, 8), ctx_220744, 'save')
        # Calling save(args, kwargs) (line 356)
        save_call_result_220747 = invoke(stypy.reporting.localization.Localization(__file__, 356, 8), save_220745, *[], **kwargs_220746)
        
        
        # Assigning a Num to a Attribute (line 357):
        
        # Assigning a Num to a Attribute (line 357):
        float_220748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 357, 25), 'float')
        # Getting the type of 'self' (line 357)
        self_220749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 8), 'self')
        # Obtaining the member 'gc' of a type (line 357)
        gc_220750 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 357, 8), self_220749, 'gc')
        # Setting the type of the member '_alpha' of a type (line 357)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 357, 8), gc_220750, '_alpha', float_220748)
        
        # Assigning a Name to a Attribute (line 358):
        
        # Assigning a Name to a Attribute (line 358):
        # Getting the type of 'False' (line 358)
        False_220751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 32), 'False')
        # Getting the type of 'self' (line 358)
        self_220752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 8), 'self')
        # Obtaining the member 'gc' of a type (line 358)
        gc_220753 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 8), self_220752, 'gc')
        # Setting the type of the member '_forced_alpha' of a type (line 358)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 8), gc_220753, '_forced_alpha', False_220751)
        # Getting the type of 'self' (line 359)
        self_220754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 15), 'self')
        # Obtaining the member 'gc' of a type (line 359)
        gc_220755 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 15), self_220754, 'gc')
        # Assigning a type to the variable 'stypy_return_type' (line 359)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 359, 8), 'stypy_return_type', gc_220755)
        
        # ################# End of 'new_gc(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'new_gc' in the type store
        # Getting the type of 'stypy_return_type' (line 355)
        stypy_return_type_220756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_220756)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'new_gc'
        return stypy_return_type_220756


    @norecursion
    def points_to_pixels(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'points_to_pixels'
        module_type_store = module_type_store.open_function_context('points_to_pixels', 361, 4, False)
        # Assigning a type to the variable 'self' (line 362)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        RendererCairo.points_to_pixels.__dict__.__setitem__('stypy_localization', localization)
        RendererCairo.points_to_pixels.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        RendererCairo.points_to_pixels.__dict__.__setitem__('stypy_type_store', module_type_store)
        RendererCairo.points_to_pixels.__dict__.__setitem__('stypy_function_name', 'RendererCairo.points_to_pixels')
        RendererCairo.points_to_pixels.__dict__.__setitem__('stypy_param_names_list', ['points'])
        RendererCairo.points_to_pixels.__dict__.__setitem__('stypy_varargs_param_name', None)
        RendererCairo.points_to_pixels.__dict__.__setitem__('stypy_kwargs_param_name', None)
        RendererCairo.points_to_pixels.__dict__.__setitem__('stypy_call_defaults', defaults)
        RendererCairo.points_to_pixels.__dict__.__setitem__('stypy_call_varargs', varargs)
        RendererCairo.points_to_pixels.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        RendererCairo.points_to_pixels.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RendererCairo.points_to_pixels', ['points'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'points_to_pixels', localization, ['points'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'points_to_pixels(...)' code ##################

        # Getting the type of 'points' (line 362)
        points_220757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 15), 'points')
        int_220758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 24), 'int')
        # Applying the binary operator 'div' (line 362)
        result_div_220759 = python_operator(stypy.reporting.localization.Localization(__file__, 362, 15), 'div', points_220757, int_220758)
        
        # Getting the type of 'self' (line 362)
        self_220760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 29), 'self')
        # Obtaining the member 'dpi' of a type (line 362)
        dpi_220761 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 29), self_220760, 'dpi')
        # Applying the binary operator '*' (line 362)
        result_mul_220762 = python_operator(stypy.reporting.localization.Localization(__file__, 362, 27), '*', result_div_220759, dpi_220761)
        
        # Assigning a type to the variable 'stypy_return_type' (line 362)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 8), 'stypy_return_type', result_mul_220762)
        
        # ################# End of 'points_to_pixels(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'points_to_pixels' in the type store
        # Getting the type of 'stypy_return_type' (line 361)
        stypy_return_type_220763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_220763)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'points_to_pixels'
        return stypy_return_type_220763


# Assigning a type to the variable 'RendererCairo' (line 79)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 0), 'RendererCairo', RendererCairo)

# Assigning a Dict to a Name (line 80):

# Obtaining an instance of the builtin type 'dict' (line 80)
dict_220764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 18), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 80)
# Adding element type (key, value) (line 80)
int_220765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 8), 'int')
# Getting the type of 'cairo' (line 81)
cairo_220766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 23), 'cairo')
# Obtaining the member 'FONT_WEIGHT_NORMAL' of a type (line 81)
FONT_WEIGHT_NORMAL_220767 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 23), cairo_220766, 'FONT_WEIGHT_NORMAL')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 18), dict_220764, (int_220765, FONT_WEIGHT_NORMAL_220767))
# Adding element type (key, value) (line 80)
int_220768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 8), 'int')
# Getting the type of 'cairo' (line 82)
cairo_220769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 23), 'cairo')
# Obtaining the member 'FONT_WEIGHT_NORMAL' of a type (line 82)
FONT_WEIGHT_NORMAL_220770 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 23), cairo_220769, 'FONT_WEIGHT_NORMAL')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 18), dict_220764, (int_220768, FONT_WEIGHT_NORMAL_220770))
# Adding element type (key, value) (line 80)
int_220771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 8), 'int')
# Getting the type of 'cairo' (line 83)
cairo_220772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 23), 'cairo')
# Obtaining the member 'FONT_WEIGHT_NORMAL' of a type (line 83)
FONT_WEIGHT_NORMAL_220773 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 23), cairo_220772, 'FONT_WEIGHT_NORMAL')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 18), dict_220764, (int_220771, FONT_WEIGHT_NORMAL_220773))
# Adding element type (key, value) (line 80)
int_220774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 8), 'int')
# Getting the type of 'cairo' (line 84)
cairo_220775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 23), 'cairo')
# Obtaining the member 'FONT_WEIGHT_NORMAL' of a type (line 84)
FONT_WEIGHT_NORMAL_220776 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 23), cairo_220775, 'FONT_WEIGHT_NORMAL')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 18), dict_220764, (int_220774, FONT_WEIGHT_NORMAL_220776))
# Adding element type (key, value) (line 80)
int_220777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 8), 'int')
# Getting the type of 'cairo' (line 85)
cairo_220778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 23), 'cairo')
# Obtaining the member 'FONT_WEIGHT_NORMAL' of a type (line 85)
FONT_WEIGHT_NORMAL_220779 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 23), cairo_220778, 'FONT_WEIGHT_NORMAL')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 18), dict_220764, (int_220777, FONT_WEIGHT_NORMAL_220779))
# Adding element type (key, value) (line 80)
int_220780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 8), 'int')
# Getting the type of 'cairo' (line 86)
cairo_220781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 23), 'cairo')
# Obtaining the member 'FONT_WEIGHT_BOLD' of a type (line 86)
FONT_WEIGHT_BOLD_220782 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 23), cairo_220781, 'FONT_WEIGHT_BOLD')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 18), dict_220764, (int_220780, FONT_WEIGHT_BOLD_220782))
# Adding element type (key, value) (line 80)
int_220783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 8), 'int')
# Getting the type of 'cairo' (line 87)
cairo_220784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 23), 'cairo')
# Obtaining the member 'FONT_WEIGHT_BOLD' of a type (line 87)
FONT_WEIGHT_BOLD_220785 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 23), cairo_220784, 'FONT_WEIGHT_BOLD')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 18), dict_220764, (int_220783, FONT_WEIGHT_BOLD_220785))
# Adding element type (key, value) (line 80)
int_220786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 8), 'int')
# Getting the type of 'cairo' (line 88)
cairo_220787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 23), 'cairo')
# Obtaining the member 'FONT_WEIGHT_BOLD' of a type (line 88)
FONT_WEIGHT_BOLD_220788 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 23), cairo_220787, 'FONT_WEIGHT_BOLD')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 18), dict_220764, (int_220786, FONT_WEIGHT_BOLD_220788))
# Adding element type (key, value) (line 80)
int_220789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 8), 'int')
# Getting the type of 'cairo' (line 89)
cairo_220790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 23), 'cairo')
# Obtaining the member 'FONT_WEIGHT_BOLD' of a type (line 89)
FONT_WEIGHT_BOLD_220791 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 23), cairo_220790, 'FONT_WEIGHT_BOLD')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 18), dict_220764, (int_220789, FONT_WEIGHT_BOLD_220791))
# Adding element type (key, value) (line 80)
unicode_220792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 8), 'unicode', u'ultralight')
# Getting the type of 'cairo' (line 90)
cairo_220793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 23), 'cairo')
# Obtaining the member 'FONT_WEIGHT_NORMAL' of a type (line 90)
FONT_WEIGHT_NORMAL_220794 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 23), cairo_220793, 'FONT_WEIGHT_NORMAL')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 18), dict_220764, (unicode_220792, FONT_WEIGHT_NORMAL_220794))
# Adding element type (key, value) (line 80)
unicode_220795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 8), 'unicode', u'light')
# Getting the type of 'cairo' (line 91)
cairo_220796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 23), 'cairo')
# Obtaining the member 'FONT_WEIGHT_NORMAL' of a type (line 91)
FONT_WEIGHT_NORMAL_220797 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 23), cairo_220796, 'FONT_WEIGHT_NORMAL')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 18), dict_220764, (unicode_220795, FONT_WEIGHT_NORMAL_220797))
# Adding element type (key, value) (line 80)
unicode_220798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 8), 'unicode', u'normal')
# Getting the type of 'cairo' (line 92)
cairo_220799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 23), 'cairo')
# Obtaining the member 'FONT_WEIGHT_NORMAL' of a type (line 92)
FONT_WEIGHT_NORMAL_220800 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 23), cairo_220799, 'FONT_WEIGHT_NORMAL')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 18), dict_220764, (unicode_220798, FONT_WEIGHT_NORMAL_220800))
# Adding element type (key, value) (line 80)
unicode_220801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 8), 'unicode', u'medium')
# Getting the type of 'cairo' (line 93)
cairo_220802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 23), 'cairo')
# Obtaining the member 'FONT_WEIGHT_NORMAL' of a type (line 93)
FONT_WEIGHT_NORMAL_220803 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 23), cairo_220802, 'FONT_WEIGHT_NORMAL')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 18), dict_220764, (unicode_220801, FONT_WEIGHT_NORMAL_220803))
# Adding element type (key, value) (line 80)
unicode_220804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 8), 'unicode', u'regular')
# Getting the type of 'cairo' (line 94)
cairo_220805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 23), 'cairo')
# Obtaining the member 'FONT_WEIGHT_NORMAL' of a type (line 94)
FONT_WEIGHT_NORMAL_220806 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 23), cairo_220805, 'FONT_WEIGHT_NORMAL')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 18), dict_220764, (unicode_220804, FONT_WEIGHT_NORMAL_220806))
# Adding element type (key, value) (line 80)
unicode_220807 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 8), 'unicode', u'semibold')
# Getting the type of 'cairo' (line 95)
cairo_220808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 23), 'cairo')
# Obtaining the member 'FONT_WEIGHT_BOLD' of a type (line 95)
FONT_WEIGHT_BOLD_220809 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 23), cairo_220808, 'FONT_WEIGHT_BOLD')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 18), dict_220764, (unicode_220807, FONT_WEIGHT_BOLD_220809))
# Adding element type (key, value) (line 80)
unicode_220810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 8), 'unicode', u'bold')
# Getting the type of 'cairo' (line 96)
cairo_220811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 23), 'cairo')
# Obtaining the member 'FONT_WEIGHT_BOLD' of a type (line 96)
FONT_WEIGHT_BOLD_220812 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 23), cairo_220811, 'FONT_WEIGHT_BOLD')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 18), dict_220764, (unicode_220810, FONT_WEIGHT_BOLD_220812))
# Adding element type (key, value) (line 80)
unicode_220813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 8), 'unicode', u'heavy')
# Getting the type of 'cairo' (line 97)
cairo_220814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 23), 'cairo')
# Obtaining the member 'FONT_WEIGHT_BOLD' of a type (line 97)
FONT_WEIGHT_BOLD_220815 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 23), cairo_220814, 'FONT_WEIGHT_BOLD')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 18), dict_220764, (unicode_220813, FONT_WEIGHT_BOLD_220815))
# Adding element type (key, value) (line 80)
unicode_220816 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 8), 'unicode', u'ultrabold')
# Getting the type of 'cairo' (line 98)
cairo_220817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 23), 'cairo')
# Obtaining the member 'FONT_WEIGHT_BOLD' of a type (line 98)
FONT_WEIGHT_BOLD_220818 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 23), cairo_220817, 'FONT_WEIGHT_BOLD')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 18), dict_220764, (unicode_220816, FONT_WEIGHT_BOLD_220818))
# Adding element type (key, value) (line 80)
unicode_220819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 8), 'unicode', u'black')
# Getting the type of 'cairo' (line 99)
cairo_220820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 23), 'cairo')
# Obtaining the member 'FONT_WEIGHT_BOLD' of a type (line 99)
FONT_WEIGHT_BOLD_220821 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 23), cairo_220820, 'FONT_WEIGHT_BOLD')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 18), dict_220764, (unicode_220819, FONT_WEIGHT_BOLD_220821))

# Getting the type of 'RendererCairo'
RendererCairo_220822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'RendererCairo')
# Setting the type of the member 'fontweights' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), RendererCairo_220822, 'fontweights', dict_220764)

# Assigning a Dict to a Name (line 101):

# Obtaining an instance of the builtin type 'dict' (line 101)
dict_220823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 17), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 101)
# Adding element type (key, value) (line 101)
unicode_220824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 8), 'unicode', u'italic')
# Getting the type of 'cairo' (line 102)
cairo_220825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 20), 'cairo')
# Obtaining the member 'FONT_SLANT_ITALIC' of a type (line 102)
FONT_SLANT_ITALIC_220826 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 20), cairo_220825, 'FONT_SLANT_ITALIC')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 17), dict_220823, (unicode_220824, FONT_SLANT_ITALIC_220826))
# Adding element type (key, value) (line 101)
unicode_220827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 8), 'unicode', u'normal')
# Getting the type of 'cairo' (line 103)
cairo_220828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 20), 'cairo')
# Obtaining the member 'FONT_SLANT_NORMAL' of a type (line 103)
FONT_SLANT_NORMAL_220829 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 20), cairo_220828, 'FONT_SLANT_NORMAL')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 17), dict_220823, (unicode_220827, FONT_SLANT_NORMAL_220829))
# Adding element type (key, value) (line 101)
unicode_220830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 8), 'unicode', u'oblique')
# Getting the type of 'cairo' (line 104)
cairo_220831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 20), 'cairo')
# Obtaining the member 'FONT_SLANT_OBLIQUE' of a type (line 104)
FONT_SLANT_OBLIQUE_220832 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 20), cairo_220831, 'FONT_SLANT_OBLIQUE')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 17), dict_220823, (unicode_220830, FONT_SLANT_OBLIQUE_220832))

# Getting the type of 'RendererCairo'
RendererCairo_220833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'RendererCairo')
# Setting the type of the member 'fontangles' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), RendererCairo_220833, 'fontangles', dict_220823)
# Declaration of the 'GraphicsContextCairo' class
# Getting the type of 'GraphicsContextBase' (line 365)
GraphicsContextBase_220834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 27), 'GraphicsContextBase')

class GraphicsContextCairo(GraphicsContextBase_220834, ):
    
    # Assigning a Dict to a Name (line 366):
    
    # Assigning a Dict to a Name (line 372):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 378, 4, False)
        # Assigning a type to the variable 'self' (line 379)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 379, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'GraphicsContextCairo.__init__', ['renderer'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['renderer'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Call to __init__(...): (line 379)
        # Processing the call arguments (line 379)
        # Getting the type of 'self' (line 379)
        self_220837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 37), 'self', False)
        # Processing the call keyword arguments (line 379)
        kwargs_220838 = {}
        # Getting the type of 'GraphicsContextBase' (line 379)
        GraphicsContextBase_220835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 8), 'GraphicsContextBase', False)
        # Obtaining the member '__init__' of a type (line 379)
        init___220836 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 379, 8), GraphicsContextBase_220835, '__init__')
        # Calling __init__(args, kwargs) (line 379)
        init___call_result_220839 = invoke(stypy.reporting.localization.Localization(__file__, 379, 8), init___220836, *[self_220837], **kwargs_220838)
        
        
        # Assigning a Name to a Attribute (line 380):
        
        # Assigning a Name to a Attribute (line 380):
        # Getting the type of 'renderer' (line 380)
        renderer_220840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 24), 'renderer')
        # Getting the type of 'self' (line 380)
        self_220841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 8), 'self')
        # Setting the type of the member 'renderer' of a type (line 380)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 380, 8), self_220841, 'renderer', renderer_220840)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def restore(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'restore'
        module_type_store = module_type_store.open_function_context('restore', 382, 4, False)
        # Assigning a type to the variable 'self' (line 383)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        GraphicsContextCairo.restore.__dict__.__setitem__('stypy_localization', localization)
        GraphicsContextCairo.restore.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        GraphicsContextCairo.restore.__dict__.__setitem__('stypy_type_store', module_type_store)
        GraphicsContextCairo.restore.__dict__.__setitem__('stypy_function_name', 'GraphicsContextCairo.restore')
        GraphicsContextCairo.restore.__dict__.__setitem__('stypy_param_names_list', [])
        GraphicsContextCairo.restore.__dict__.__setitem__('stypy_varargs_param_name', None)
        GraphicsContextCairo.restore.__dict__.__setitem__('stypy_kwargs_param_name', None)
        GraphicsContextCairo.restore.__dict__.__setitem__('stypy_call_defaults', defaults)
        GraphicsContextCairo.restore.__dict__.__setitem__('stypy_call_varargs', varargs)
        GraphicsContextCairo.restore.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        GraphicsContextCairo.restore.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'GraphicsContextCairo.restore', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'restore', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'restore(...)' code ##################

        
        # Call to restore(...): (line 383)
        # Processing the call keyword arguments (line 383)
        kwargs_220845 = {}
        # Getting the type of 'self' (line 383)
        self_220842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 8), 'self', False)
        # Obtaining the member 'ctx' of a type (line 383)
        ctx_220843 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 383, 8), self_220842, 'ctx')
        # Obtaining the member 'restore' of a type (line 383)
        restore_220844 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 383, 8), ctx_220843, 'restore')
        # Calling restore(args, kwargs) (line 383)
        restore_call_result_220846 = invoke(stypy.reporting.localization.Localization(__file__, 383, 8), restore_220844, *[], **kwargs_220845)
        
        
        # ################# End of 'restore(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'restore' in the type store
        # Getting the type of 'stypy_return_type' (line 382)
        stypy_return_type_220847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_220847)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'restore'
        return stypy_return_type_220847


    @norecursion
    def set_alpha(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_alpha'
        module_type_store = module_type_store.open_function_context('set_alpha', 385, 4, False)
        # Assigning a type to the variable 'self' (line 386)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 386, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        GraphicsContextCairo.set_alpha.__dict__.__setitem__('stypy_localization', localization)
        GraphicsContextCairo.set_alpha.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        GraphicsContextCairo.set_alpha.__dict__.__setitem__('stypy_type_store', module_type_store)
        GraphicsContextCairo.set_alpha.__dict__.__setitem__('stypy_function_name', 'GraphicsContextCairo.set_alpha')
        GraphicsContextCairo.set_alpha.__dict__.__setitem__('stypy_param_names_list', ['alpha'])
        GraphicsContextCairo.set_alpha.__dict__.__setitem__('stypy_varargs_param_name', None)
        GraphicsContextCairo.set_alpha.__dict__.__setitem__('stypy_kwargs_param_name', None)
        GraphicsContextCairo.set_alpha.__dict__.__setitem__('stypy_call_defaults', defaults)
        GraphicsContextCairo.set_alpha.__dict__.__setitem__('stypy_call_varargs', varargs)
        GraphicsContextCairo.set_alpha.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        GraphicsContextCairo.set_alpha.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'GraphicsContextCairo.set_alpha', ['alpha'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_alpha', localization, ['alpha'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_alpha(...)' code ##################

        
        # Call to set_alpha(...): (line 386)
        # Processing the call arguments (line 386)
        # Getting the type of 'self' (line 386)
        self_220850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 38), 'self', False)
        # Getting the type of 'alpha' (line 386)
        alpha_220851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 44), 'alpha', False)
        # Processing the call keyword arguments (line 386)
        kwargs_220852 = {}
        # Getting the type of 'GraphicsContextBase' (line 386)
        GraphicsContextBase_220848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 8), 'GraphicsContextBase', False)
        # Obtaining the member 'set_alpha' of a type (line 386)
        set_alpha_220849 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 386, 8), GraphicsContextBase_220848, 'set_alpha')
        # Calling set_alpha(args, kwargs) (line 386)
        set_alpha_call_result_220853 = invoke(stypy.reporting.localization.Localization(__file__, 386, 8), set_alpha_220849, *[self_220850, alpha_220851], **kwargs_220852)
        
        
        # Assigning a Call to a Name (line 387):
        
        # Assigning a Call to a Name (line 387):
        
        # Call to get_alpha(...): (line 387)
        # Processing the call keyword arguments (line 387)
        kwargs_220856 = {}
        # Getting the type of 'self' (line 387)
        self_220854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 17), 'self', False)
        # Obtaining the member 'get_alpha' of a type (line 387)
        get_alpha_220855 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 387, 17), self_220854, 'get_alpha')
        # Calling get_alpha(args, kwargs) (line 387)
        get_alpha_call_result_220857 = invoke(stypy.reporting.localization.Localization(__file__, 387, 17), get_alpha_220855, *[], **kwargs_220856)
        
        # Assigning a type to the variable '_alpha' (line 387)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 387, 8), '_alpha', get_alpha_call_result_220857)
        
        # Assigning a Attribute to a Name (line 388):
        
        # Assigning a Attribute to a Name (line 388):
        # Getting the type of 'self' (line 388)
        self_220858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 14), 'self')
        # Obtaining the member '_rgb' of a type (line 388)
        _rgb_220859 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 14), self_220858, '_rgb')
        # Assigning a type to the variable 'rgb' (line 388)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 388, 8), 'rgb', _rgb_220859)
        
        
        # Call to get_forced_alpha(...): (line 389)
        # Processing the call keyword arguments (line 389)
        kwargs_220862 = {}
        # Getting the type of 'self' (line 389)
        self_220860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 11), 'self', False)
        # Obtaining the member 'get_forced_alpha' of a type (line 389)
        get_forced_alpha_220861 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 11), self_220860, 'get_forced_alpha')
        # Calling get_forced_alpha(args, kwargs) (line 389)
        get_forced_alpha_call_result_220863 = invoke(stypy.reporting.localization.Localization(__file__, 389, 11), get_forced_alpha_220861, *[], **kwargs_220862)
        
        # Testing the type of an if condition (line 389)
        if_condition_220864 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 389, 8), get_forced_alpha_call_result_220863)
        # Assigning a type to the variable 'if_condition_220864' (line 389)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 389, 8), 'if_condition_220864', if_condition_220864)
        # SSA begins for if statement (line 389)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to set_source_rgba(...): (line 390)
        # Processing the call arguments (line 390)
        
        # Obtaining the type of the subscript
        int_220868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 41), 'int')
        # Getting the type of 'rgb' (line 390)
        rgb_220869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 37), 'rgb', False)
        # Obtaining the member '__getitem__' of a type (line 390)
        getitem___220870 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 37), rgb_220869, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 390)
        subscript_call_result_220871 = invoke(stypy.reporting.localization.Localization(__file__, 390, 37), getitem___220870, int_220868)
        
        
        # Obtaining the type of the subscript
        int_220872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 49), 'int')
        # Getting the type of 'rgb' (line 390)
        rgb_220873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 45), 'rgb', False)
        # Obtaining the member '__getitem__' of a type (line 390)
        getitem___220874 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 45), rgb_220873, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 390)
        subscript_call_result_220875 = invoke(stypy.reporting.localization.Localization(__file__, 390, 45), getitem___220874, int_220872)
        
        
        # Obtaining the type of the subscript
        int_220876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 57), 'int')
        # Getting the type of 'rgb' (line 390)
        rgb_220877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 53), 'rgb', False)
        # Obtaining the member '__getitem__' of a type (line 390)
        getitem___220878 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 53), rgb_220877, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 390)
        subscript_call_result_220879 = invoke(stypy.reporting.localization.Localization(__file__, 390, 53), getitem___220878, int_220876)
        
        # Getting the type of '_alpha' (line 390)
        _alpha_220880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 61), '_alpha', False)
        # Processing the call keyword arguments (line 390)
        kwargs_220881 = {}
        # Getting the type of 'self' (line 390)
        self_220865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 12), 'self', False)
        # Obtaining the member 'ctx' of a type (line 390)
        ctx_220866 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 12), self_220865, 'ctx')
        # Obtaining the member 'set_source_rgba' of a type (line 390)
        set_source_rgba_220867 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 12), ctx_220866, 'set_source_rgba')
        # Calling set_source_rgba(args, kwargs) (line 390)
        set_source_rgba_call_result_220882 = invoke(stypy.reporting.localization.Localization(__file__, 390, 12), set_source_rgba_220867, *[subscript_call_result_220871, subscript_call_result_220875, subscript_call_result_220879, _alpha_220880], **kwargs_220881)
        
        # SSA branch for the else part of an if statement (line 389)
        module_type_store.open_ssa_branch('else')
        
        # Call to set_source_rgba(...): (line 392)
        # Processing the call arguments (line 392)
        
        # Obtaining the type of the subscript
        int_220886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 392, 41), 'int')
        # Getting the type of 'rgb' (line 392)
        rgb_220887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 37), 'rgb', False)
        # Obtaining the member '__getitem__' of a type (line 392)
        getitem___220888 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 392, 37), rgb_220887, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 392)
        subscript_call_result_220889 = invoke(stypy.reporting.localization.Localization(__file__, 392, 37), getitem___220888, int_220886)
        
        
        # Obtaining the type of the subscript
        int_220890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 392, 49), 'int')
        # Getting the type of 'rgb' (line 392)
        rgb_220891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 45), 'rgb', False)
        # Obtaining the member '__getitem__' of a type (line 392)
        getitem___220892 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 392, 45), rgb_220891, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 392)
        subscript_call_result_220893 = invoke(stypy.reporting.localization.Localization(__file__, 392, 45), getitem___220892, int_220890)
        
        
        # Obtaining the type of the subscript
        int_220894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 392, 57), 'int')
        # Getting the type of 'rgb' (line 392)
        rgb_220895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 53), 'rgb', False)
        # Obtaining the member '__getitem__' of a type (line 392)
        getitem___220896 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 392, 53), rgb_220895, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 392)
        subscript_call_result_220897 = invoke(stypy.reporting.localization.Localization(__file__, 392, 53), getitem___220896, int_220894)
        
        
        # Obtaining the type of the subscript
        int_220898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 392, 65), 'int')
        # Getting the type of 'rgb' (line 392)
        rgb_220899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 61), 'rgb', False)
        # Obtaining the member '__getitem__' of a type (line 392)
        getitem___220900 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 392, 61), rgb_220899, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 392)
        subscript_call_result_220901 = invoke(stypy.reporting.localization.Localization(__file__, 392, 61), getitem___220900, int_220898)
        
        # Processing the call keyword arguments (line 392)
        kwargs_220902 = {}
        # Getting the type of 'self' (line 392)
        self_220883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 12), 'self', False)
        # Obtaining the member 'ctx' of a type (line 392)
        ctx_220884 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 392, 12), self_220883, 'ctx')
        # Obtaining the member 'set_source_rgba' of a type (line 392)
        set_source_rgba_220885 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 392, 12), ctx_220884, 'set_source_rgba')
        # Calling set_source_rgba(args, kwargs) (line 392)
        set_source_rgba_call_result_220903 = invoke(stypy.reporting.localization.Localization(__file__, 392, 12), set_source_rgba_220885, *[subscript_call_result_220889, subscript_call_result_220893, subscript_call_result_220897, subscript_call_result_220901], **kwargs_220902)
        
        # SSA join for if statement (line 389)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'set_alpha(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_alpha' in the type store
        # Getting the type of 'stypy_return_type' (line 385)
        stypy_return_type_220904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_220904)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_alpha'
        return stypy_return_type_220904


    @norecursion
    def set_capstyle(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_capstyle'
        module_type_store = module_type_store.open_function_context('set_capstyle', 397, 4, False)
        # Assigning a type to the variable 'self' (line 398)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 398, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        GraphicsContextCairo.set_capstyle.__dict__.__setitem__('stypy_localization', localization)
        GraphicsContextCairo.set_capstyle.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        GraphicsContextCairo.set_capstyle.__dict__.__setitem__('stypy_type_store', module_type_store)
        GraphicsContextCairo.set_capstyle.__dict__.__setitem__('stypy_function_name', 'GraphicsContextCairo.set_capstyle')
        GraphicsContextCairo.set_capstyle.__dict__.__setitem__('stypy_param_names_list', ['cs'])
        GraphicsContextCairo.set_capstyle.__dict__.__setitem__('stypy_varargs_param_name', None)
        GraphicsContextCairo.set_capstyle.__dict__.__setitem__('stypy_kwargs_param_name', None)
        GraphicsContextCairo.set_capstyle.__dict__.__setitem__('stypy_call_defaults', defaults)
        GraphicsContextCairo.set_capstyle.__dict__.__setitem__('stypy_call_varargs', varargs)
        GraphicsContextCairo.set_capstyle.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        GraphicsContextCairo.set_capstyle.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'GraphicsContextCairo.set_capstyle', ['cs'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_capstyle', localization, ['cs'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_capstyle(...)' code ##################

        
        
        # Getting the type of 'cs' (line 398)
        cs_220905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 11), 'cs')
        
        # Obtaining an instance of the builtin type 'tuple' (line 398)
        tuple_220906 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 18), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 398)
        # Adding element type (line 398)
        unicode_220907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 18), 'unicode', u'butt')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 398, 18), tuple_220906, unicode_220907)
        # Adding element type (line 398)
        unicode_220908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 26), 'unicode', u'round')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 398, 18), tuple_220906, unicode_220908)
        # Adding element type (line 398)
        unicode_220909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 35), 'unicode', u'projecting')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 398, 18), tuple_220906, unicode_220909)
        
        # Applying the binary operator 'in' (line 398)
        result_contains_220910 = python_operator(stypy.reporting.localization.Localization(__file__, 398, 11), 'in', cs_220905, tuple_220906)
        
        # Testing the type of an if condition (line 398)
        if_condition_220911 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 398, 8), result_contains_220910)
        # Assigning a type to the variable 'if_condition_220911' (line 398)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 398, 8), 'if_condition_220911', if_condition_220911)
        # SSA begins for if statement (line 398)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Attribute (line 399):
        
        # Assigning a Name to a Attribute (line 399):
        # Getting the type of 'cs' (line 399)
        cs_220912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 29), 'cs')
        # Getting the type of 'self' (line 399)
        self_220913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 12), 'self')
        # Setting the type of the member '_capstyle' of a type (line 399)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 399, 12), self_220913, '_capstyle', cs_220912)
        
        # Call to set_line_cap(...): (line 400)
        # Processing the call arguments (line 400)
        
        # Obtaining the type of the subscript
        # Getting the type of 'cs' (line 400)
        cs_220917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 45), 'cs', False)
        # Getting the type of 'self' (line 400)
        self_220918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 34), 'self', False)
        # Obtaining the member '_capd' of a type (line 400)
        _capd_220919 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 400, 34), self_220918, '_capd')
        # Obtaining the member '__getitem__' of a type (line 400)
        getitem___220920 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 400, 34), _capd_220919, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 400)
        subscript_call_result_220921 = invoke(stypy.reporting.localization.Localization(__file__, 400, 34), getitem___220920, cs_220917)
        
        # Processing the call keyword arguments (line 400)
        kwargs_220922 = {}
        # Getting the type of 'self' (line 400)
        self_220914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 12), 'self', False)
        # Obtaining the member 'ctx' of a type (line 400)
        ctx_220915 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 400, 12), self_220914, 'ctx')
        # Obtaining the member 'set_line_cap' of a type (line 400)
        set_line_cap_220916 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 400, 12), ctx_220915, 'set_line_cap')
        # Calling set_line_cap(args, kwargs) (line 400)
        set_line_cap_call_result_220923 = invoke(stypy.reporting.localization.Localization(__file__, 400, 12), set_line_cap_220916, *[subscript_call_result_220921], **kwargs_220922)
        
        # SSA branch for the else part of an if statement (line 398)
        module_type_store.open_ssa_branch('else')
        
        # Call to ValueError(...): (line 402)
        # Processing the call arguments (line 402)
        unicode_220925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 402, 29), 'unicode', u'Unrecognized cap style.  Found %s')
        # Getting the type of 'cs' (line 402)
        cs_220926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 67), 'cs', False)
        # Applying the binary operator '%' (line 402)
        result_mod_220927 = python_operator(stypy.reporting.localization.Localization(__file__, 402, 29), '%', unicode_220925, cs_220926)
        
        # Processing the call keyword arguments (line 402)
        kwargs_220928 = {}
        # Getting the type of 'ValueError' (line 402)
        ValueError_220924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 402)
        ValueError_call_result_220929 = invoke(stypy.reporting.localization.Localization(__file__, 402, 18), ValueError_220924, *[result_mod_220927], **kwargs_220928)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 402, 12), ValueError_call_result_220929, 'raise parameter', BaseException)
        # SSA join for if statement (line 398)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'set_capstyle(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_capstyle' in the type store
        # Getting the type of 'stypy_return_type' (line 397)
        stypy_return_type_220930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_220930)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_capstyle'
        return stypy_return_type_220930


    @norecursion
    def set_clip_rectangle(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_clip_rectangle'
        module_type_store = module_type_store.open_function_context('set_clip_rectangle', 404, 4, False)
        # Assigning a type to the variable 'self' (line 405)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 405, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        GraphicsContextCairo.set_clip_rectangle.__dict__.__setitem__('stypy_localization', localization)
        GraphicsContextCairo.set_clip_rectangle.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        GraphicsContextCairo.set_clip_rectangle.__dict__.__setitem__('stypy_type_store', module_type_store)
        GraphicsContextCairo.set_clip_rectangle.__dict__.__setitem__('stypy_function_name', 'GraphicsContextCairo.set_clip_rectangle')
        GraphicsContextCairo.set_clip_rectangle.__dict__.__setitem__('stypy_param_names_list', ['rectangle'])
        GraphicsContextCairo.set_clip_rectangle.__dict__.__setitem__('stypy_varargs_param_name', None)
        GraphicsContextCairo.set_clip_rectangle.__dict__.__setitem__('stypy_kwargs_param_name', None)
        GraphicsContextCairo.set_clip_rectangle.__dict__.__setitem__('stypy_call_defaults', defaults)
        GraphicsContextCairo.set_clip_rectangle.__dict__.__setitem__('stypy_call_varargs', varargs)
        GraphicsContextCairo.set_clip_rectangle.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        GraphicsContextCairo.set_clip_rectangle.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'GraphicsContextCairo.set_clip_rectangle', ['rectangle'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_clip_rectangle', localization, ['rectangle'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_clip_rectangle(...)' code ##################

        
        
        # Getting the type of 'rectangle' (line 405)
        rectangle_220931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 15), 'rectangle')
        # Applying the 'not' unary operator (line 405)
        result_not__220932 = python_operator(stypy.reporting.localization.Localization(__file__, 405, 11), 'not', rectangle_220931)
        
        # Testing the type of an if condition (line 405)
        if_condition_220933 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 405, 8), result_not__220932)
        # Assigning a type to the variable 'if_condition_220933' (line 405)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 405, 8), 'if_condition_220933', if_condition_220933)
        # SSA begins for if statement (line 405)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Assigning a type to the variable 'stypy_return_type' (line 406)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 406, 12), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 405)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Attribute to a Tuple (line 407):
        
        # Assigning a Subscript to a Name (line 407):
        
        # Obtaining the type of the subscript
        int_220934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 407, 8), 'int')
        # Getting the type of 'rectangle' (line 407)
        rectangle_220935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 21), 'rectangle')
        # Obtaining the member 'bounds' of a type (line 407)
        bounds_220936 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 407, 21), rectangle_220935, 'bounds')
        # Obtaining the member '__getitem__' of a type (line 407)
        getitem___220937 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 407, 8), bounds_220936, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 407)
        subscript_call_result_220938 = invoke(stypy.reporting.localization.Localization(__file__, 407, 8), getitem___220937, int_220934)
        
        # Assigning a type to the variable 'tuple_var_assignment_219643' (line 407)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 407, 8), 'tuple_var_assignment_219643', subscript_call_result_220938)
        
        # Assigning a Subscript to a Name (line 407):
        
        # Obtaining the type of the subscript
        int_220939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 407, 8), 'int')
        # Getting the type of 'rectangle' (line 407)
        rectangle_220940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 21), 'rectangle')
        # Obtaining the member 'bounds' of a type (line 407)
        bounds_220941 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 407, 21), rectangle_220940, 'bounds')
        # Obtaining the member '__getitem__' of a type (line 407)
        getitem___220942 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 407, 8), bounds_220941, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 407)
        subscript_call_result_220943 = invoke(stypy.reporting.localization.Localization(__file__, 407, 8), getitem___220942, int_220939)
        
        # Assigning a type to the variable 'tuple_var_assignment_219644' (line 407)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 407, 8), 'tuple_var_assignment_219644', subscript_call_result_220943)
        
        # Assigning a Subscript to a Name (line 407):
        
        # Obtaining the type of the subscript
        int_220944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 407, 8), 'int')
        # Getting the type of 'rectangle' (line 407)
        rectangle_220945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 21), 'rectangle')
        # Obtaining the member 'bounds' of a type (line 407)
        bounds_220946 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 407, 21), rectangle_220945, 'bounds')
        # Obtaining the member '__getitem__' of a type (line 407)
        getitem___220947 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 407, 8), bounds_220946, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 407)
        subscript_call_result_220948 = invoke(stypy.reporting.localization.Localization(__file__, 407, 8), getitem___220947, int_220944)
        
        # Assigning a type to the variable 'tuple_var_assignment_219645' (line 407)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 407, 8), 'tuple_var_assignment_219645', subscript_call_result_220948)
        
        # Assigning a Subscript to a Name (line 407):
        
        # Obtaining the type of the subscript
        int_220949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 407, 8), 'int')
        # Getting the type of 'rectangle' (line 407)
        rectangle_220950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 21), 'rectangle')
        # Obtaining the member 'bounds' of a type (line 407)
        bounds_220951 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 407, 21), rectangle_220950, 'bounds')
        # Obtaining the member '__getitem__' of a type (line 407)
        getitem___220952 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 407, 8), bounds_220951, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 407)
        subscript_call_result_220953 = invoke(stypy.reporting.localization.Localization(__file__, 407, 8), getitem___220952, int_220949)
        
        # Assigning a type to the variable 'tuple_var_assignment_219646' (line 407)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 407, 8), 'tuple_var_assignment_219646', subscript_call_result_220953)
        
        # Assigning a Name to a Name (line 407):
        # Getting the type of 'tuple_var_assignment_219643' (line 407)
        tuple_var_assignment_219643_220954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 8), 'tuple_var_assignment_219643')
        # Assigning a type to the variable 'x' (line 407)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 407, 8), 'x', tuple_var_assignment_219643_220954)
        
        # Assigning a Name to a Name (line 407):
        # Getting the type of 'tuple_var_assignment_219644' (line 407)
        tuple_var_assignment_219644_220955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 8), 'tuple_var_assignment_219644')
        # Assigning a type to the variable 'y' (line 407)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 407, 11), 'y', tuple_var_assignment_219644_220955)
        
        # Assigning a Name to a Name (line 407):
        # Getting the type of 'tuple_var_assignment_219645' (line 407)
        tuple_var_assignment_219645_220956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 8), 'tuple_var_assignment_219645')
        # Assigning a type to the variable 'w' (line 407)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 407, 14), 'w', tuple_var_assignment_219645_220956)
        
        # Assigning a Name to a Name (line 407):
        # Getting the type of 'tuple_var_assignment_219646' (line 407)
        tuple_var_assignment_219646_220957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 8), 'tuple_var_assignment_219646')
        # Assigning a type to the variable 'h' (line 407)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 407, 17), 'h', tuple_var_assignment_219646_220957)
        
        # Assigning a Tuple to a Tuple (line 409):
        
        # Assigning a Call to a Name (line 409):
        
        # Call to round(...): (line 409)
        # Processing the call arguments (line 409)
        # Getting the type of 'x' (line 409)
        x_220960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 27), 'x', False)
        # Processing the call keyword arguments (line 409)
        kwargs_220961 = {}
        # Getting the type of 'np' (line 409)
        np_220958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 18), 'np', False)
        # Obtaining the member 'round' of a type (line 409)
        round_220959 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 409, 18), np_220958, 'round')
        # Calling round(args, kwargs) (line 409)
        round_call_result_220962 = invoke(stypy.reporting.localization.Localization(__file__, 409, 18), round_220959, *[x_220960], **kwargs_220961)
        
        # Assigning a type to the variable 'tuple_assignment_219647' (line 409)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 409, 8), 'tuple_assignment_219647', round_call_result_220962)
        
        # Assigning a Call to a Name (line 409):
        
        # Call to round(...): (line 409)
        # Processing the call arguments (line 409)
        # Getting the type of 'y' (line 409)
        y_220965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 40), 'y', False)
        # Processing the call keyword arguments (line 409)
        kwargs_220966 = {}
        # Getting the type of 'np' (line 409)
        np_220963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 31), 'np', False)
        # Obtaining the member 'round' of a type (line 409)
        round_220964 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 409, 31), np_220963, 'round')
        # Calling round(args, kwargs) (line 409)
        round_call_result_220967 = invoke(stypy.reporting.localization.Localization(__file__, 409, 31), round_220964, *[y_220965], **kwargs_220966)
        
        # Assigning a type to the variable 'tuple_assignment_219648' (line 409)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 409, 8), 'tuple_assignment_219648', round_call_result_220967)
        
        # Assigning a Call to a Name (line 409):
        
        # Call to round(...): (line 409)
        # Processing the call arguments (line 409)
        # Getting the type of 'w' (line 409)
        w_220970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 53), 'w', False)
        # Processing the call keyword arguments (line 409)
        kwargs_220971 = {}
        # Getting the type of 'np' (line 409)
        np_220968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 44), 'np', False)
        # Obtaining the member 'round' of a type (line 409)
        round_220969 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 409, 44), np_220968, 'round')
        # Calling round(args, kwargs) (line 409)
        round_call_result_220972 = invoke(stypy.reporting.localization.Localization(__file__, 409, 44), round_220969, *[w_220970], **kwargs_220971)
        
        # Assigning a type to the variable 'tuple_assignment_219649' (line 409)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 409, 8), 'tuple_assignment_219649', round_call_result_220972)
        
        # Assigning a Call to a Name (line 409):
        
        # Call to round(...): (line 409)
        # Processing the call arguments (line 409)
        # Getting the type of 'h' (line 409)
        h_220975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 66), 'h', False)
        # Processing the call keyword arguments (line 409)
        kwargs_220976 = {}
        # Getting the type of 'np' (line 409)
        np_220973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 57), 'np', False)
        # Obtaining the member 'round' of a type (line 409)
        round_220974 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 409, 57), np_220973, 'round')
        # Calling round(args, kwargs) (line 409)
        round_call_result_220977 = invoke(stypy.reporting.localization.Localization(__file__, 409, 57), round_220974, *[h_220975], **kwargs_220976)
        
        # Assigning a type to the variable 'tuple_assignment_219650' (line 409)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 409, 8), 'tuple_assignment_219650', round_call_result_220977)
        
        # Assigning a Name to a Name (line 409):
        # Getting the type of 'tuple_assignment_219647' (line 409)
        tuple_assignment_219647_220978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 8), 'tuple_assignment_219647')
        # Assigning a type to the variable 'x' (line 409)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 409, 8), 'x', tuple_assignment_219647_220978)
        
        # Assigning a Name to a Name (line 409):
        # Getting the type of 'tuple_assignment_219648' (line 409)
        tuple_assignment_219648_220979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 8), 'tuple_assignment_219648')
        # Assigning a type to the variable 'y' (line 409)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 409, 10), 'y', tuple_assignment_219648_220979)
        
        # Assigning a Name to a Name (line 409):
        # Getting the type of 'tuple_assignment_219649' (line 409)
        tuple_assignment_219649_220980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 8), 'tuple_assignment_219649')
        # Assigning a type to the variable 'w' (line 409)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 409, 12), 'w', tuple_assignment_219649_220980)
        
        # Assigning a Name to a Name (line 409):
        # Getting the type of 'tuple_assignment_219650' (line 409)
        tuple_assignment_219650_220981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 8), 'tuple_assignment_219650')
        # Assigning a type to the variable 'h' (line 409)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 409, 14), 'h', tuple_assignment_219650_220981)
        
        # Assigning a Attribute to a Name (line 410):
        
        # Assigning a Attribute to a Name (line 410):
        # Getting the type of 'self' (line 410)
        self_220982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 14), 'self')
        # Obtaining the member 'ctx' of a type (line 410)
        ctx_220983 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 410, 14), self_220982, 'ctx')
        # Assigning a type to the variable 'ctx' (line 410)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 410, 8), 'ctx', ctx_220983)
        
        # Call to new_path(...): (line 411)
        # Processing the call keyword arguments (line 411)
        kwargs_220986 = {}
        # Getting the type of 'ctx' (line 411)
        ctx_220984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 8), 'ctx', False)
        # Obtaining the member 'new_path' of a type (line 411)
        new_path_220985 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 411, 8), ctx_220984, 'new_path')
        # Calling new_path(args, kwargs) (line 411)
        new_path_call_result_220987 = invoke(stypy.reporting.localization.Localization(__file__, 411, 8), new_path_220985, *[], **kwargs_220986)
        
        
        # Call to rectangle(...): (line 412)
        # Processing the call arguments (line 412)
        # Getting the type of 'x' (line 412)
        x_220990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 22), 'x', False)
        # Getting the type of 'self' (line 412)
        self_220991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 25), 'self', False)
        # Obtaining the member 'renderer' of a type (line 412)
        renderer_220992 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 412, 25), self_220991, 'renderer')
        # Obtaining the member 'height' of a type (line 412)
        height_220993 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 412, 25), renderer_220992, 'height')
        # Getting the type of 'h' (line 412)
        h_220994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 48), 'h', False)
        # Applying the binary operator '-' (line 412)
        result_sub_220995 = python_operator(stypy.reporting.localization.Localization(__file__, 412, 25), '-', height_220993, h_220994)
        
        # Getting the type of 'y' (line 412)
        y_220996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 52), 'y', False)
        # Applying the binary operator '-' (line 412)
        result_sub_220997 = python_operator(stypy.reporting.localization.Localization(__file__, 412, 50), '-', result_sub_220995, y_220996)
        
        # Getting the type of 'w' (line 412)
        w_220998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 55), 'w', False)
        # Getting the type of 'h' (line 412)
        h_220999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 58), 'h', False)
        # Processing the call keyword arguments (line 412)
        kwargs_221000 = {}
        # Getting the type of 'ctx' (line 412)
        ctx_220988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 8), 'ctx', False)
        # Obtaining the member 'rectangle' of a type (line 412)
        rectangle_220989 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 412, 8), ctx_220988, 'rectangle')
        # Calling rectangle(args, kwargs) (line 412)
        rectangle_call_result_221001 = invoke(stypy.reporting.localization.Localization(__file__, 412, 8), rectangle_220989, *[x_220990, result_sub_220997, w_220998, h_220999], **kwargs_221000)
        
        
        # Call to clip(...): (line 413)
        # Processing the call keyword arguments (line 413)
        kwargs_221004 = {}
        # Getting the type of 'ctx' (line 413)
        ctx_221002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 8), 'ctx', False)
        # Obtaining the member 'clip' of a type (line 413)
        clip_221003 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 413, 8), ctx_221002, 'clip')
        # Calling clip(args, kwargs) (line 413)
        clip_call_result_221005 = invoke(stypy.reporting.localization.Localization(__file__, 413, 8), clip_221003, *[], **kwargs_221004)
        
        
        # ################# End of 'set_clip_rectangle(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_clip_rectangle' in the type store
        # Getting the type of 'stypy_return_type' (line 404)
        stypy_return_type_221006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_221006)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_clip_rectangle'
        return stypy_return_type_221006


    @norecursion
    def set_clip_path(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_clip_path'
        module_type_store = module_type_store.open_function_context('set_clip_path', 415, 4, False)
        # Assigning a type to the variable 'self' (line 416)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 416, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        GraphicsContextCairo.set_clip_path.__dict__.__setitem__('stypy_localization', localization)
        GraphicsContextCairo.set_clip_path.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        GraphicsContextCairo.set_clip_path.__dict__.__setitem__('stypy_type_store', module_type_store)
        GraphicsContextCairo.set_clip_path.__dict__.__setitem__('stypy_function_name', 'GraphicsContextCairo.set_clip_path')
        GraphicsContextCairo.set_clip_path.__dict__.__setitem__('stypy_param_names_list', ['path'])
        GraphicsContextCairo.set_clip_path.__dict__.__setitem__('stypy_varargs_param_name', None)
        GraphicsContextCairo.set_clip_path.__dict__.__setitem__('stypy_kwargs_param_name', None)
        GraphicsContextCairo.set_clip_path.__dict__.__setitem__('stypy_call_defaults', defaults)
        GraphicsContextCairo.set_clip_path.__dict__.__setitem__('stypy_call_varargs', varargs)
        GraphicsContextCairo.set_clip_path.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        GraphicsContextCairo.set_clip_path.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'GraphicsContextCairo.set_clip_path', ['path'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_clip_path', localization, ['path'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_clip_path(...)' code ##################

        
        
        # Getting the type of 'path' (line 416)
        path_221007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 15), 'path')
        # Applying the 'not' unary operator (line 416)
        result_not__221008 = python_operator(stypy.reporting.localization.Localization(__file__, 416, 11), 'not', path_221007)
        
        # Testing the type of an if condition (line 416)
        if_condition_221009 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 416, 8), result_not__221008)
        # Assigning a type to the variable 'if_condition_221009' (line 416)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 416, 8), 'if_condition_221009', if_condition_221009)
        # SSA begins for if statement (line 416)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Assigning a type to the variable 'stypy_return_type' (line 417)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 417, 12), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 416)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Tuple (line 418):
        
        # Assigning a Call to a Name:
        
        # Call to get_transformed_path_and_affine(...): (line 418)
        # Processing the call keyword arguments (line 418)
        kwargs_221012 = {}
        # Getting the type of 'path' (line 418)
        path_221010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 24), 'path', False)
        # Obtaining the member 'get_transformed_path_and_affine' of a type (line 418)
        get_transformed_path_and_affine_221011 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 418, 24), path_221010, 'get_transformed_path_and_affine')
        # Calling get_transformed_path_and_affine(args, kwargs) (line 418)
        get_transformed_path_and_affine_call_result_221013 = invoke(stypy.reporting.localization.Localization(__file__, 418, 24), get_transformed_path_and_affine_221011, *[], **kwargs_221012)
        
        # Assigning a type to the variable 'call_assignment_219651' (line 418)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 418, 8), 'call_assignment_219651', get_transformed_path_and_affine_call_result_221013)
        
        # Assigning a Call to a Name (line 418):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_221016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 8), 'int')
        # Processing the call keyword arguments
        kwargs_221017 = {}
        # Getting the type of 'call_assignment_219651' (line 418)
        call_assignment_219651_221014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 8), 'call_assignment_219651', False)
        # Obtaining the member '__getitem__' of a type (line 418)
        getitem___221015 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 418, 8), call_assignment_219651_221014, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_221018 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___221015, *[int_221016], **kwargs_221017)
        
        # Assigning a type to the variable 'call_assignment_219652' (line 418)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 418, 8), 'call_assignment_219652', getitem___call_result_221018)
        
        # Assigning a Name to a Name (line 418):
        # Getting the type of 'call_assignment_219652' (line 418)
        call_assignment_219652_221019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 8), 'call_assignment_219652')
        # Assigning a type to the variable 'tpath' (line 418)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 418, 8), 'tpath', call_assignment_219652_221019)
        
        # Assigning a Call to a Name (line 418):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_221022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 8), 'int')
        # Processing the call keyword arguments
        kwargs_221023 = {}
        # Getting the type of 'call_assignment_219651' (line 418)
        call_assignment_219651_221020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 8), 'call_assignment_219651', False)
        # Obtaining the member '__getitem__' of a type (line 418)
        getitem___221021 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 418, 8), call_assignment_219651_221020, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_221024 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___221021, *[int_221022], **kwargs_221023)
        
        # Assigning a type to the variable 'call_assignment_219653' (line 418)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 418, 8), 'call_assignment_219653', getitem___call_result_221024)
        
        # Assigning a Name to a Name (line 418):
        # Getting the type of 'call_assignment_219653' (line 418)
        call_assignment_219653_221025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 8), 'call_assignment_219653')
        # Assigning a type to the variable 'affine' (line 418)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 418, 15), 'affine', call_assignment_219653_221025)
        
        # Assigning a Attribute to a Name (line 419):
        
        # Assigning a Attribute to a Name (line 419):
        # Getting the type of 'self' (line 419)
        self_221026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 14), 'self')
        # Obtaining the member 'ctx' of a type (line 419)
        ctx_221027 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 419, 14), self_221026, 'ctx')
        # Assigning a type to the variable 'ctx' (line 419)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 419, 8), 'ctx', ctx_221027)
        
        # Call to new_path(...): (line 420)
        # Processing the call keyword arguments (line 420)
        kwargs_221030 = {}
        # Getting the type of 'ctx' (line 420)
        ctx_221028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 8), 'ctx', False)
        # Obtaining the member 'new_path' of a type (line 420)
        new_path_221029 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 420, 8), ctx_221028, 'new_path')
        # Calling new_path(args, kwargs) (line 420)
        new_path_call_result_221031 = invoke(stypy.reporting.localization.Localization(__file__, 420, 8), new_path_221029, *[], **kwargs_221030)
        
        
        # Assigning a BinOp to a Name (line 421):
        
        # Assigning a BinOp to a Name (line 421):
        # Getting the type of 'affine' (line 421)
        affine_221032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 18), 'affine')
        
        # Call to translate(...): (line 422)
        # Processing the call arguments (line 422)
        int_221042 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 422, 54), 'int')
        # Getting the type of 'self' (line 422)
        self_221043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 57), 'self', False)
        # Obtaining the member 'renderer' of a type (line 422)
        renderer_221044 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 422, 57), self_221043, 'renderer')
        # Obtaining the member 'height' of a type (line 422)
        height_221045 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 422, 57), renderer_221044, 'height')
        # Processing the call keyword arguments (line 422)
        kwargs_221046 = {}
        
        # Call to scale(...): (line 422)
        # Processing the call arguments (line 422)
        int_221037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 422, 37), 'int')
        int_221038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 422, 40), 'int')
        # Processing the call keyword arguments (line 422)
        kwargs_221039 = {}
        
        # Call to Affine2D(...): (line 422)
        # Processing the call keyword arguments (line 422)
        kwargs_221034 = {}
        # Getting the type of 'Affine2D' (line 422)
        Affine2D_221033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 20), 'Affine2D', False)
        # Calling Affine2D(args, kwargs) (line 422)
        Affine2D_call_result_221035 = invoke(stypy.reporting.localization.Localization(__file__, 422, 20), Affine2D_221033, *[], **kwargs_221034)
        
        # Obtaining the member 'scale' of a type (line 422)
        scale_221036 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 422, 20), Affine2D_call_result_221035, 'scale')
        # Calling scale(args, kwargs) (line 422)
        scale_call_result_221040 = invoke(stypy.reporting.localization.Localization(__file__, 422, 20), scale_221036, *[int_221037, int_221038], **kwargs_221039)
        
        # Obtaining the member 'translate' of a type (line 422)
        translate_221041 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 422, 20), scale_call_result_221040, 'translate')
        # Calling translate(args, kwargs) (line 422)
        translate_call_result_221047 = invoke(stypy.reporting.localization.Localization(__file__, 422, 20), translate_221041, *[int_221042, height_221045], **kwargs_221046)
        
        # Applying the binary operator '+' (line 421)
        result_add_221048 = python_operator(stypy.reporting.localization.Localization(__file__, 421, 18), '+', affine_221032, translate_call_result_221047)
        
        # Assigning a type to the variable 'affine' (line 421)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 421, 8), 'affine', result_add_221048)
        
        # Call to convert_path(...): (line 423)
        # Processing the call arguments (line 423)
        # Getting the type of 'ctx' (line 423)
        ctx_221051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 35), 'ctx', False)
        # Getting the type of 'tpath' (line 423)
        tpath_221052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 40), 'tpath', False)
        # Getting the type of 'affine' (line 423)
        affine_221053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 47), 'affine', False)
        # Processing the call keyword arguments (line 423)
        kwargs_221054 = {}
        # Getting the type of 'RendererCairo' (line 423)
        RendererCairo_221049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 8), 'RendererCairo', False)
        # Obtaining the member 'convert_path' of a type (line 423)
        convert_path_221050 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 423, 8), RendererCairo_221049, 'convert_path')
        # Calling convert_path(args, kwargs) (line 423)
        convert_path_call_result_221055 = invoke(stypy.reporting.localization.Localization(__file__, 423, 8), convert_path_221050, *[ctx_221051, tpath_221052, affine_221053], **kwargs_221054)
        
        
        # Call to clip(...): (line 424)
        # Processing the call keyword arguments (line 424)
        kwargs_221058 = {}
        # Getting the type of 'ctx' (line 424)
        ctx_221056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 8), 'ctx', False)
        # Obtaining the member 'clip' of a type (line 424)
        clip_221057 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 424, 8), ctx_221056, 'clip')
        # Calling clip(args, kwargs) (line 424)
        clip_call_result_221059 = invoke(stypy.reporting.localization.Localization(__file__, 424, 8), clip_221057, *[], **kwargs_221058)
        
        
        # ################# End of 'set_clip_path(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_clip_path' in the type store
        # Getting the type of 'stypy_return_type' (line 415)
        stypy_return_type_221060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_221060)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_clip_path'
        return stypy_return_type_221060


    @norecursion
    def set_dashes(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_dashes'
        module_type_store = module_type_store.open_function_context('set_dashes', 426, 4, False)
        # Assigning a type to the variable 'self' (line 427)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 427, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        GraphicsContextCairo.set_dashes.__dict__.__setitem__('stypy_localization', localization)
        GraphicsContextCairo.set_dashes.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        GraphicsContextCairo.set_dashes.__dict__.__setitem__('stypy_type_store', module_type_store)
        GraphicsContextCairo.set_dashes.__dict__.__setitem__('stypy_function_name', 'GraphicsContextCairo.set_dashes')
        GraphicsContextCairo.set_dashes.__dict__.__setitem__('stypy_param_names_list', ['offset', 'dashes'])
        GraphicsContextCairo.set_dashes.__dict__.__setitem__('stypy_varargs_param_name', None)
        GraphicsContextCairo.set_dashes.__dict__.__setitem__('stypy_kwargs_param_name', None)
        GraphicsContextCairo.set_dashes.__dict__.__setitem__('stypy_call_defaults', defaults)
        GraphicsContextCairo.set_dashes.__dict__.__setitem__('stypy_call_varargs', varargs)
        GraphicsContextCairo.set_dashes.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        GraphicsContextCairo.set_dashes.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'GraphicsContextCairo.set_dashes', ['offset', 'dashes'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_dashes', localization, ['offset', 'dashes'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_dashes(...)' code ##################

        
        # Assigning a Tuple to a Attribute (line 427):
        
        # Assigning a Tuple to a Attribute (line 427):
        
        # Obtaining an instance of the builtin type 'tuple' (line 427)
        tuple_221061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 427, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 427)
        # Adding element type (line 427)
        # Getting the type of 'offset' (line 427)
        offset_221062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 23), 'offset')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 427, 23), tuple_221061, offset_221062)
        # Adding element type (line 427)
        # Getting the type of 'dashes' (line 427)
        dashes_221063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 31), 'dashes')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 427, 23), tuple_221061, dashes_221063)
        
        # Getting the type of 'self' (line 427)
        self_221064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 8), 'self')
        # Setting the type of the member '_dashes' of a type (line 427)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 427, 8), self_221064, '_dashes', tuple_221061)
        
        # Type idiom detected: calculating its left and rigth part (line 428)
        # Getting the type of 'dashes' (line 428)
        dashes_221065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 8), 'dashes')
        # Getting the type of 'None' (line 428)
        None_221066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 21), 'None')
        
        (may_be_221067, more_types_in_union_221068) = may_be_none(dashes_221065, None_221066)

        if may_be_221067:

            if more_types_in_union_221068:
                # Runtime conditional SSA (line 428)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to set_dash(...): (line 429)
            # Processing the call arguments (line 429)
            
            # Obtaining an instance of the builtin type 'list' (line 429)
            list_221072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 429, 30), 'list')
            # Adding type elements to the builtin type 'list' instance (line 429)
            
            int_221073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 429, 34), 'int')
            # Processing the call keyword arguments (line 429)
            kwargs_221074 = {}
            # Getting the type of 'self' (line 429)
            self_221069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 12), 'self', False)
            # Obtaining the member 'ctx' of a type (line 429)
            ctx_221070 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 429, 12), self_221069, 'ctx')
            # Obtaining the member 'set_dash' of a type (line 429)
            set_dash_221071 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 429, 12), ctx_221070, 'set_dash')
            # Calling set_dash(args, kwargs) (line 429)
            set_dash_call_result_221075 = invoke(stypy.reporting.localization.Localization(__file__, 429, 12), set_dash_221071, *[list_221072, int_221073], **kwargs_221074)
            

            if more_types_in_union_221068:
                # Runtime conditional SSA for else branch (line 428)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_221067) or more_types_in_union_221068):
            
            # Call to set_dash(...): (line 431)
            # Processing the call arguments (line 431)
            
            # Call to list(...): (line 432)
            # Processing the call arguments (line 432)
            
            # Call to points_to_pixels(...): (line 432)
            # Processing the call arguments (line 432)
            
            # Call to asarray(...): (line 432)
            # Processing the call arguments (line 432)
            # Getting the type of 'dashes' (line 432)
            dashes_221085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 63), 'dashes', False)
            # Processing the call keyword arguments (line 432)
            kwargs_221086 = {}
            # Getting the type of 'np' (line 432)
            np_221083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 52), 'np', False)
            # Obtaining the member 'asarray' of a type (line 432)
            asarray_221084 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 432, 52), np_221083, 'asarray')
            # Calling asarray(args, kwargs) (line 432)
            asarray_call_result_221087 = invoke(stypy.reporting.localization.Localization(__file__, 432, 52), asarray_221084, *[dashes_221085], **kwargs_221086)
            
            # Processing the call keyword arguments (line 432)
            kwargs_221088 = {}
            # Getting the type of 'self' (line 432)
            self_221080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 21), 'self', False)
            # Obtaining the member 'renderer' of a type (line 432)
            renderer_221081 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 432, 21), self_221080, 'renderer')
            # Obtaining the member 'points_to_pixels' of a type (line 432)
            points_to_pixels_221082 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 432, 21), renderer_221081, 'points_to_pixels')
            # Calling points_to_pixels(args, kwargs) (line 432)
            points_to_pixels_call_result_221089 = invoke(stypy.reporting.localization.Localization(__file__, 432, 21), points_to_pixels_221082, *[asarray_call_result_221087], **kwargs_221088)
            
            # Processing the call keyword arguments (line 432)
            kwargs_221090 = {}
            # Getting the type of 'list' (line 432)
            list_221079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 16), 'list', False)
            # Calling list(args, kwargs) (line 432)
            list_call_result_221091 = invoke(stypy.reporting.localization.Localization(__file__, 432, 16), list_221079, *[points_to_pixels_call_result_221089], **kwargs_221090)
            
            # Getting the type of 'offset' (line 433)
            offset_221092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 16), 'offset', False)
            # Processing the call keyword arguments (line 431)
            kwargs_221093 = {}
            # Getting the type of 'self' (line 431)
            self_221076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 12), 'self', False)
            # Obtaining the member 'ctx' of a type (line 431)
            ctx_221077 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 431, 12), self_221076, 'ctx')
            # Obtaining the member 'set_dash' of a type (line 431)
            set_dash_221078 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 431, 12), ctx_221077, 'set_dash')
            # Calling set_dash(args, kwargs) (line 431)
            set_dash_call_result_221094 = invoke(stypy.reporting.localization.Localization(__file__, 431, 12), set_dash_221078, *[list_call_result_221091, offset_221092], **kwargs_221093)
            

            if (may_be_221067 and more_types_in_union_221068):
                # SSA join for if statement (line 428)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # ################# End of 'set_dashes(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_dashes' in the type store
        # Getting the type of 'stypy_return_type' (line 426)
        stypy_return_type_221095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_221095)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_dashes'
        return stypy_return_type_221095


    @norecursion
    def set_foreground(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 435)
        None_221096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 40), 'None')
        defaults = [None_221096]
        # Create a new context for function 'set_foreground'
        module_type_store = module_type_store.open_function_context('set_foreground', 435, 4, False)
        # Assigning a type to the variable 'self' (line 436)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 436, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        GraphicsContextCairo.set_foreground.__dict__.__setitem__('stypy_localization', localization)
        GraphicsContextCairo.set_foreground.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        GraphicsContextCairo.set_foreground.__dict__.__setitem__('stypy_type_store', module_type_store)
        GraphicsContextCairo.set_foreground.__dict__.__setitem__('stypy_function_name', 'GraphicsContextCairo.set_foreground')
        GraphicsContextCairo.set_foreground.__dict__.__setitem__('stypy_param_names_list', ['fg', 'isRGBA'])
        GraphicsContextCairo.set_foreground.__dict__.__setitem__('stypy_varargs_param_name', None)
        GraphicsContextCairo.set_foreground.__dict__.__setitem__('stypy_kwargs_param_name', None)
        GraphicsContextCairo.set_foreground.__dict__.__setitem__('stypy_call_defaults', defaults)
        GraphicsContextCairo.set_foreground.__dict__.__setitem__('stypy_call_varargs', varargs)
        GraphicsContextCairo.set_foreground.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        GraphicsContextCairo.set_foreground.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'GraphicsContextCairo.set_foreground', ['fg', 'isRGBA'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_foreground', localization, ['fg', 'isRGBA'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_foreground(...)' code ##################

        
        # Call to set_foreground(...): (line 436)
        # Processing the call arguments (line 436)
        # Getting the type of 'self' (line 436)
        self_221099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 43), 'self', False)
        # Getting the type of 'fg' (line 436)
        fg_221100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 49), 'fg', False)
        # Getting the type of 'isRGBA' (line 436)
        isRGBA_221101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 53), 'isRGBA', False)
        # Processing the call keyword arguments (line 436)
        kwargs_221102 = {}
        # Getting the type of 'GraphicsContextBase' (line 436)
        GraphicsContextBase_221097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 8), 'GraphicsContextBase', False)
        # Obtaining the member 'set_foreground' of a type (line 436)
        set_foreground_221098 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 436, 8), GraphicsContextBase_221097, 'set_foreground')
        # Calling set_foreground(args, kwargs) (line 436)
        set_foreground_call_result_221103 = invoke(stypy.reporting.localization.Localization(__file__, 436, 8), set_foreground_221098, *[self_221099, fg_221100, isRGBA_221101], **kwargs_221102)
        
        
        
        
        # Call to len(...): (line 437)
        # Processing the call arguments (line 437)
        # Getting the type of 'self' (line 437)
        self_221105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 15), 'self', False)
        # Obtaining the member '_rgb' of a type (line 437)
        _rgb_221106 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 437, 15), self_221105, '_rgb')
        # Processing the call keyword arguments (line 437)
        kwargs_221107 = {}
        # Getting the type of 'len' (line 437)
        len_221104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 11), 'len', False)
        # Calling len(args, kwargs) (line 437)
        len_call_result_221108 = invoke(stypy.reporting.localization.Localization(__file__, 437, 11), len_221104, *[_rgb_221106], **kwargs_221107)
        
        int_221109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 437, 29), 'int')
        # Applying the binary operator '==' (line 437)
        result_eq_221110 = python_operator(stypy.reporting.localization.Localization(__file__, 437, 11), '==', len_call_result_221108, int_221109)
        
        # Testing the type of an if condition (line 437)
        if_condition_221111 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 437, 8), result_eq_221110)
        # Assigning a type to the variable 'if_condition_221111' (line 437)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 437, 8), 'if_condition_221111', if_condition_221111)
        # SSA begins for if statement (line 437)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to set_source_rgb(...): (line 438)
        # Getting the type of 'self' (line 438)
        self_221115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 37), 'self', False)
        # Obtaining the member '_rgb' of a type (line 438)
        _rgb_221116 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 438, 37), self_221115, '_rgb')
        # Processing the call keyword arguments (line 438)
        kwargs_221117 = {}
        # Getting the type of 'self' (line 438)
        self_221112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 12), 'self', False)
        # Obtaining the member 'ctx' of a type (line 438)
        ctx_221113 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 438, 12), self_221112, 'ctx')
        # Obtaining the member 'set_source_rgb' of a type (line 438)
        set_source_rgb_221114 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 438, 12), ctx_221113, 'set_source_rgb')
        # Calling set_source_rgb(args, kwargs) (line 438)
        set_source_rgb_call_result_221118 = invoke(stypy.reporting.localization.Localization(__file__, 438, 12), set_source_rgb_221114, *[_rgb_221116], **kwargs_221117)
        
        # SSA branch for the else part of an if statement (line 437)
        module_type_store.open_ssa_branch('else')
        
        # Call to set_source_rgba(...): (line 440)
        # Getting the type of 'self' (line 440)
        self_221122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 38), 'self', False)
        # Obtaining the member '_rgb' of a type (line 440)
        _rgb_221123 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 440, 38), self_221122, '_rgb')
        # Processing the call keyword arguments (line 440)
        kwargs_221124 = {}
        # Getting the type of 'self' (line 440)
        self_221119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 12), 'self', False)
        # Obtaining the member 'ctx' of a type (line 440)
        ctx_221120 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 440, 12), self_221119, 'ctx')
        # Obtaining the member 'set_source_rgba' of a type (line 440)
        set_source_rgba_221121 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 440, 12), ctx_221120, 'set_source_rgba')
        # Calling set_source_rgba(args, kwargs) (line 440)
        set_source_rgba_call_result_221125 = invoke(stypy.reporting.localization.Localization(__file__, 440, 12), set_source_rgba_221121, *[_rgb_221123], **kwargs_221124)
        
        # SSA join for if statement (line 437)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'set_foreground(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_foreground' in the type store
        # Getting the type of 'stypy_return_type' (line 435)
        stypy_return_type_221126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_221126)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_foreground'
        return stypy_return_type_221126


    @norecursion
    def get_rgb(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_rgb'
        module_type_store = module_type_store.open_function_context('get_rgb', 442, 4, False)
        # Assigning a type to the variable 'self' (line 443)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 443, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        GraphicsContextCairo.get_rgb.__dict__.__setitem__('stypy_localization', localization)
        GraphicsContextCairo.get_rgb.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        GraphicsContextCairo.get_rgb.__dict__.__setitem__('stypy_type_store', module_type_store)
        GraphicsContextCairo.get_rgb.__dict__.__setitem__('stypy_function_name', 'GraphicsContextCairo.get_rgb')
        GraphicsContextCairo.get_rgb.__dict__.__setitem__('stypy_param_names_list', [])
        GraphicsContextCairo.get_rgb.__dict__.__setitem__('stypy_varargs_param_name', None)
        GraphicsContextCairo.get_rgb.__dict__.__setitem__('stypy_kwargs_param_name', None)
        GraphicsContextCairo.get_rgb.__dict__.__setitem__('stypy_call_defaults', defaults)
        GraphicsContextCairo.get_rgb.__dict__.__setitem__('stypy_call_varargs', varargs)
        GraphicsContextCairo.get_rgb.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        GraphicsContextCairo.get_rgb.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'GraphicsContextCairo.get_rgb', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_rgb', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_rgb(...)' code ##################

        
        # Obtaining the type of the subscript
        int_221127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 443, 49), 'int')
        slice_221128 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 443, 15), None, int_221127, None)
        
        # Call to get_rgba(...): (line 443)
        # Processing the call keyword arguments (line 443)
        kwargs_221135 = {}
        
        # Call to get_source(...): (line 443)
        # Processing the call keyword arguments (line 443)
        kwargs_221132 = {}
        # Getting the type of 'self' (line 443)
        self_221129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 15), 'self', False)
        # Obtaining the member 'ctx' of a type (line 443)
        ctx_221130 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 443, 15), self_221129, 'ctx')
        # Obtaining the member 'get_source' of a type (line 443)
        get_source_221131 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 443, 15), ctx_221130, 'get_source')
        # Calling get_source(args, kwargs) (line 443)
        get_source_call_result_221133 = invoke(stypy.reporting.localization.Localization(__file__, 443, 15), get_source_221131, *[], **kwargs_221132)
        
        # Obtaining the member 'get_rgba' of a type (line 443)
        get_rgba_221134 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 443, 15), get_source_call_result_221133, 'get_rgba')
        # Calling get_rgba(args, kwargs) (line 443)
        get_rgba_call_result_221136 = invoke(stypy.reporting.localization.Localization(__file__, 443, 15), get_rgba_221134, *[], **kwargs_221135)
        
        # Obtaining the member '__getitem__' of a type (line 443)
        getitem___221137 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 443, 15), get_rgba_call_result_221136, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 443)
        subscript_call_result_221138 = invoke(stypy.reporting.localization.Localization(__file__, 443, 15), getitem___221137, slice_221128)
        
        # Assigning a type to the variable 'stypy_return_type' (line 443)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 443, 8), 'stypy_return_type', subscript_call_result_221138)
        
        # ################# End of 'get_rgb(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_rgb' in the type store
        # Getting the type of 'stypy_return_type' (line 442)
        stypy_return_type_221139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_221139)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_rgb'
        return stypy_return_type_221139


    @norecursion
    def set_joinstyle(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_joinstyle'
        module_type_store = module_type_store.open_function_context('set_joinstyle', 445, 4, False)
        # Assigning a type to the variable 'self' (line 446)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 446, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        GraphicsContextCairo.set_joinstyle.__dict__.__setitem__('stypy_localization', localization)
        GraphicsContextCairo.set_joinstyle.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        GraphicsContextCairo.set_joinstyle.__dict__.__setitem__('stypy_type_store', module_type_store)
        GraphicsContextCairo.set_joinstyle.__dict__.__setitem__('stypy_function_name', 'GraphicsContextCairo.set_joinstyle')
        GraphicsContextCairo.set_joinstyle.__dict__.__setitem__('stypy_param_names_list', ['js'])
        GraphicsContextCairo.set_joinstyle.__dict__.__setitem__('stypy_varargs_param_name', None)
        GraphicsContextCairo.set_joinstyle.__dict__.__setitem__('stypy_kwargs_param_name', None)
        GraphicsContextCairo.set_joinstyle.__dict__.__setitem__('stypy_call_defaults', defaults)
        GraphicsContextCairo.set_joinstyle.__dict__.__setitem__('stypy_call_varargs', varargs)
        GraphicsContextCairo.set_joinstyle.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        GraphicsContextCairo.set_joinstyle.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'GraphicsContextCairo.set_joinstyle', ['js'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_joinstyle', localization, ['js'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_joinstyle(...)' code ##################

        
        
        # Getting the type of 'js' (line 446)
        js_221140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 11), 'js')
        
        # Obtaining an instance of the builtin type 'tuple' (line 446)
        tuple_221141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 446, 18), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 446)
        # Adding element type (line 446)
        unicode_221142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 446, 18), 'unicode', u'miter')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 446, 18), tuple_221141, unicode_221142)
        # Adding element type (line 446)
        unicode_221143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 446, 27), 'unicode', u'round')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 446, 18), tuple_221141, unicode_221143)
        # Adding element type (line 446)
        unicode_221144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 446, 36), 'unicode', u'bevel')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 446, 18), tuple_221141, unicode_221144)
        
        # Applying the binary operator 'in' (line 446)
        result_contains_221145 = python_operator(stypy.reporting.localization.Localization(__file__, 446, 11), 'in', js_221140, tuple_221141)
        
        # Testing the type of an if condition (line 446)
        if_condition_221146 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 446, 8), result_contains_221145)
        # Assigning a type to the variable 'if_condition_221146' (line 446)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 446, 8), 'if_condition_221146', if_condition_221146)
        # SSA begins for if statement (line 446)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Attribute (line 447):
        
        # Assigning a Name to a Attribute (line 447):
        # Getting the type of 'js' (line 447)
        js_221147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 30), 'js')
        # Getting the type of 'self' (line 447)
        self_221148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 12), 'self')
        # Setting the type of the member '_joinstyle' of a type (line 447)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 447, 12), self_221148, '_joinstyle', js_221147)
        
        # Call to set_line_join(...): (line 448)
        # Processing the call arguments (line 448)
        
        # Obtaining the type of the subscript
        # Getting the type of 'js' (line 448)
        js_221152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 47), 'js', False)
        # Getting the type of 'self' (line 448)
        self_221153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 35), 'self', False)
        # Obtaining the member '_joind' of a type (line 448)
        _joind_221154 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 448, 35), self_221153, '_joind')
        # Obtaining the member '__getitem__' of a type (line 448)
        getitem___221155 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 448, 35), _joind_221154, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 448)
        subscript_call_result_221156 = invoke(stypy.reporting.localization.Localization(__file__, 448, 35), getitem___221155, js_221152)
        
        # Processing the call keyword arguments (line 448)
        kwargs_221157 = {}
        # Getting the type of 'self' (line 448)
        self_221149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 12), 'self', False)
        # Obtaining the member 'ctx' of a type (line 448)
        ctx_221150 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 448, 12), self_221149, 'ctx')
        # Obtaining the member 'set_line_join' of a type (line 448)
        set_line_join_221151 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 448, 12), ctx_221150, 'set_line_join')
        # Calling set_line_join(args, kwargs) (line 448)
        set_line_join_call_result_221158 = invoke(stypy.reporting.localization.Localization(__file__, 448, 12), set_line_join_221151, *[subscript_call_result_221156], **kwargs_221157)
        
        # SSA branch for the else part of an if statement (line 446)
        module_type_store.open_ssa_branch('else')
        
        # Call to ValueError(...): (line 450)
        # Processing the call arguments (line 450)
        unicode_221160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 450, 29), 'unicode', u'Unrecognized join style.  Found %s')
        # Getting the type of 'js' (line 450)
        js_221161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 68), 'js', False)
        # Applying the binary operator '%' (line 450)
        result_mod_221162 = python_operator(stypy.reporting.localization.Localization(__file__, 450, 29), '%', unicode_221160, js_221161)
        
        # Processing the call keyword arguments (line 450)
        kwargs_221163 = {}
        # Getting the type of 'ValueError' (line 450)
        ValueError_221159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 450)
        ValueError_call_result_221164 = invoke(stypy.reporting.localization.Localization(__file__, 450, 18), ValueError_221159, *[result_mod_221162], **kwargs_221163)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 450, 12), ValueError_call_result_221164, 'raise parameter', BaseException)
        # SSA join for if statement (line 446)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'set_joinstyle(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_joinstyle' in the type store
        # Getting the type of 'stypy_return_type' (line 445)
        stypy_return_type_221165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_221165)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_joinstyle'
        return stypy_return_type_221165


    @norecursion
    def set_linewidth(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_linewidth'
        module_type_store = module_type_store.open_function_context('set_linewidth', 452, 4, False)
        # Assigning a type to the variable 'self' (line 453)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 453, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        GraphicsContextCairo.set_linewidth.__dict__.__setitem__('stypy_localization', localization)
        GraphicsContextCairo.set_linewidth.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        GraphicsContextCairo.set_linewidth.__dict__.__setitem__('stypy_type_store', module_type_store)
        GraphicsContextCairo.set_linewidth.__dict__.__setitem__('stypy_function_name', 'GraphicsContextCairo.set_linewidth')
        GraphicsContextCairo.set_linewidth.__dict__.__setitem__('stypy_param_names_list', ['w'])
        GraphicsContextCairo.set_linewidth.__dict__.__setitem__('stypy_varargs_param_name', None)
        GraphicsContextCairo.set_linewidth.__dict__.__setitem__('stypy_kwargs_param_name', None)
        GraphicsContextCairo.set_linewidth.__dict__.__setitem__('stypy_call_defaults', defaults)
        GraphicsContextCairo.set_linewidth.__dict__.__setitem__('stypy_call_varargs', varargs)
        GraphicsContextCairo.set_linewidth.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        GraphicsContextCairo.set_linewidth.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'GraphicsContextCairo.set_linewidth', ['w'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_linewidth', localization, ['w'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_linewidth(...)' code ##################

        
        # Assigning a Call to a Attribute (line 453):
        
        # Assigning a Call to a Attribute (line 453):
        
        # Call to float(...): (line 453)
        # Processing the call arguments (line 453)
        # Getting the type of 'w' (line 453)
        w_221167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 32), 'w', False)
        # Processing the call keyword arguments (line 453)
        kwargs_221168 = {}
        # Getting the type of 'float' (line 453)
        float_221166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 26), 'float', False)
        # Calling float(args, kwargs) (line 453)
        float_call_result_221169 = invoke(stypy.reporting.localization.Localization(__file__, 453, 26), float_221166, *[w_221167], **kwargs_221168)
        
        # Getting the type of 'self' (line 453)
        self_221170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 8), 'self')
        # Setting the type of the member '_linewidth' of a type (line 453)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 453, 8), self_221170, '_linewidth', float_call_result_221169)
        
        # Call to set_line_width(...): (line 454)
        # Processing the call arguments (line 454)
        
        # Call to points_to_pixels(...): (line 454)
        # Processing the call arguments (line 454)
        # Getting the type of 'w' (line 454)
        w_221177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 63), 'w', False)
        # Processing the call keyword arguments (line 454)
        kwargs_221178 = {}
        # Getting the type of 'self' (line 454)
        self_221174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 32), 'self', False)
        # Obtaining the member 'renderer' of a type (line 454)
        renderer_221175 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 454, 32), self_221174, 'renderer')
        # Obtaining the member 'points_to_pixels' of a type (line 454)
        points_to_pixels_221176 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 454, 32), renderer_221175, 'points_to_pixels')
        # Calling points_to_pixels(args, kwargs) (line 454)
        points_to_pixels_call_result_221179 = invoke(stypy.reporting.localization.Localization(__file__, 454, 32), points_to_pixels_221176, *[w_221177], **kwargs_221178)
        
        # Processing the call keyword arguments (line 454)
        kwargs_221180 = {}
        # Getting the type of 'self' (line 454)
        self_221171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 8), 'self', False)
        # Obtaining the member 'ctx' of a type (line 454)
        ctx_221172 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 454, 8), self_221171, 'ctx')
        # Obtaining the member 'set_line_width' of a type (line 454)
        set_line_width_221173 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 454, 8), ctx_221172, 'set_line_width')
        # Calling set_line_width(args, kwargs) (line 454)
        set_line_width_call_result_221181 = invoke(stypy.reporting.localization.Localization(__file__, 454, 8), set_line_width_221173, *[points_to_pixels_call_result_221179], **kwargs_221180)
        
        
        # ################# End of 'set_linewidth(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_linewidth' in the type store
        # Getting the type of 'stypy_return_type' (line 452)
        stypy_return_type_221182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_221182)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_linewidth'
        return stypy_return_type_221182


# Assigning a type to the variable 'GraphicsContextCairo' (line 365)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 365, 0), 'GraphicsContextCairo', GraphicsContextCairo)

# Assigning a Dict to a Name (line 366):

# Obtaining an instance of the builtin type 'dict' (line 366)
dict_221183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 13), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 366)
# Adding element type (key, value) (line 366)
unicode_221184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 367, 8), 'unicode', u'bevel')
# Getting the type of 'cairo' (line 367)
cairo_221185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 18), 'cairo')
# Obtaining the member 'LINE_JOIN_BEVEL' of a type (line 367)
LINE_JOIN_BEVEL_221186 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 367, 18), cairo_221185, 'LINE_JOIN_BEVEL')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 366, 13), dict_221183, (unicode_221184, LINE_JOIN_BEVEL_221186))
# Adding element type (key, value) (line 366)
unicode_221187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 368, 8), 'unicode', u'miter')
# Getting the type of 'cairo' (line 368)
cairo_221188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 18), 'cairo')
# Obtaining the member 'LINE_JOIN_MITER' of a type (line 368)
LINE_JOIN_MITER_221189 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 368, 18), cairo_221188, 'LINE_JOIN_MITER')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 366, 13), dict_221183, (unicode_221187, LINE_JOIN_MITER_221189))
# Adding element type (key, value) (line 366)
unicode_221190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 369, 8), 'unicode', u'round')
# Getting the type of 'cairo' (line 369)
cairo_221191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 18), 'cairo')
# Obtaining the member 'LINE_JOIN_ROUND' of a type (line 369)
LINE_JOIN_ROUND_221192 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 369, 18), cairo_221191, 'LINE_JOIN_ROUND')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 366, 13), dict_221183, (unicode_221190, LINE_JOIN_ROUND_221192))

# Getting the type of 'GraphicsContextCairo'
GraphicsContextCairo_221193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'GraphicsContextCairo')
# Setting the type of the member '_joind' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), GraphicsContextCairo_221193, '_joind', dict_221183)

# Assigning a Dict to a Name (line 372):

# Obtaining an instance of the builtin type 'dict' (line 372)
dict_221194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 372, 12), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 372)
# Adding element type (key, value) (line 372)
unicode_221195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, 8), 'unicode', u'butt')
# Getting the type of 'cairo' (line 373)
cairo_221196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 23), 'cairo')
# Obtaining the member 'LINE_CAP_BUTT' of a type (line 373)
LINE_CAP_BUTT_221197 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 23), cairo_221196, 'LINE_CAP_BUTT')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 372, 12), dict_221194, (unicode_221195, LINE_CAP_BUTT_221197))
# Adding element type (key, value) (line 372)
unicode_221198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, 8), 'unicode', u'projecting')
# Getting the type of 'cairo' (line 374)
cairo_221199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 23), 'cairo')
# Obtaining the member 'LINE_CAP_SQUARE' of a type (line 374)
LINE_CAP_SQUARE_221200 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 374, 23), cairo_221199, 'LINE_CAP_SQUARE')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 372, 12), dict_221194, (unicode_221198, LINE_CAP_SQUARE_221200))
# Adding element type (key, value) (line 372)
unicode_221201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 8), 'unicode', u'round')
# Getting the type of 'cairo' (line 375)
cairo_221202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 23), 'cairo')
# Obtaining the member 'LINE_CAP_ROUND' of a type (line 375)
LINE_CAP_ROUND_221203 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 375, 23), cairo_221202, 'LINE_CAP_ROUND')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 372, 12), dict_221194, (unicode_221201, LINE_CAP_ROUND_221203))

# Getting the type of 'GraphicsContextCairo'
GraphicsContextCairo_221204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'GraphicsContextCairo')
# Setting the type of the member '_capd' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), GraphicsContextCairo_221204, '_capd', dict_221194)
# Declaration of the 'FigureCanvasCairo' class
# Getting the type of 'FigureCanvasBase' (line 457)
FigureCanvasBase_221205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 24), 'FigureCanvasBase')

class FigureCanvasCairo(FigureCanvasBase_221205, ):

    @norecursion
    def print_png(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'print_png'
        module_type_store = module_type_store.open_function_context('print_png', 458, 4, False)
        # Assigning a type to the variable 'self' (line 459)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 459, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasCairo.print_png.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasCairo.print_png.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasCairo.print_png.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasCairo.print_png.__dict__.__setitem__('stypy_function_name', 'FigureCanvasCairo.print_png')
        FigureCanvasCairo.print_png.__dict__.__setitem__('stypy_param_names_list', ['fobj'])
        FigureCanvasCairo.print_png.__dict__.__setitem__('stypy_varargs_param_name', 'args')
        FigureCanvasCairo.print_png.__dict__.__setitem__('stypy_kwargs_param_name', 'kwargs')
        FigureCanvasCairo.print_png.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasCairo.print_png.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasCairo.print_png.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasCairo.print_png.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasCairo.print_png', ['fobj'], 'args', 'kwargs', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'print_png', localization, ['fobj'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'print_png(...)' code ##################

        
        # Assigning a Call to a Tuple (line 459):
        
        # Assigning a Call to a Name:
        
        # Call to get_width_height(...): (line 459)
        # Processing the call keyword arguments (line 459)
        kwargs_221208 = {}
        # Getting the type of 'self' (line 459)
        self_221206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 24), 'self', False)
        # Obtaining the member 'get_width_height' of a type (line 459)
        get_width_height_221207 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 459, 24), self_221206, 'get_width_height')
        # Calling get_width_height(args, kwargs) (line 459)
        get_width_height_call_result_221209 = invoke(stypy.reporting.localization.Localization(__file__, 459, 24), get_width_height_221207, *[], **kwargs_221208)
        
        # Assigning a type to the variable 'call_assignment_219654' (line 459)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 459, 8), 'call_assignment_219654', get_width_height_call_result_221209)
        
        # Assigning a Call to a Name (line 459):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_221212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 459, 8), 'int')
        # Processing the call keyword arguments
        kwargs_221213 = {}
        # Getting the type of 'call_assignment_219654' (line 459)
        call_assignment_219654_221210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 8), 'call_assignment_219654', False)
        # Obtaining the member '__getitem__' of a type (line 459)
        getitem___221211 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 459, 8), call_assignment_219654_221210, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_221214 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___221211, *[int_221212], **kwargs_221213)
        
        # Assigning a type to the variable 'call_assignment_219655' (line 459)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 459, 8), 'call_assignment_219655', getitem___call_result_221214)
        
        # Assigning a Name to a Name (line 459):
        # Getting the type of 'call_assignment_219655' (line 459)
        call_assignment_219655_221215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 8), 'call_assignment_219655')
        # Assigning a type to the variable 'width' (line 459)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 459, 8), 'width', call_assignment_219655_221215)
        
        # Assigning a Call to a Name (line 459):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_221218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 459, 8), 'int')
        # Processing the call keyword arguments
        kwargs_221219 = {}
        # Getting the type of 'call_assignment_219654' (line 459)
        call_assignment_219654_221216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 8), 'call_assignment_219654', False)
        # Obtaining the member '__getitem__' of a type (line 459)
        getitem___221217 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 459, 8), call_assignment_219654_221216, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_221220 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___221217, *[int_221218], **kwargs_221219)
        
        # Assigning a type to the variable 'call_assignment_219656' (line 459)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 459, 8), 'call_assignment_219656', getitem___call_result_221220)
        
        # Assigning a Name to a Name (line 459):
        # Getting the type of 'call_assignment_219656' (line 459)
        call_assignment_219656_221221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 8), 'call_assignment_219656')
        # Assigning a type to the variable 'height' (line 459)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 459, 15), 'height', call_assignment_219656_221221)
        
        # Assigning a Call to a Name (line 461):
        
        # Assigning a Call to a Name (line 461):
        
        # Call to RendererCairo(...): (line 461)
        # Processing the call arguments (line 461)
        # Getting the type of 'self' (line 461)
        self_221223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 33), 'self', False)
        # Obtaining the member 'figure' of a type (line 461)
        figure_221224 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 461, 33), self_221223, 'figure')
        # Obtaining the member 'dpi' of a type (line 461)
        dpi_221225 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 461, 33), figure_221224, 'dpi')
        # Processing the call keyword arguments (line 461)
        kwargs_221226 = {}
        # Getting the type of 'RendererCairo' (line 461)
        RendererCairo_221222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 19), 'RendererCairo', False)
        # Calling RendererCairo(args, kwargs) (line 461)
        RendererCairo_call_result_221227 = invoke(stypy.reporting.localization.Localization(__file__, 461, 19), RendererCairo_221222, *[dpi_221225], **kwargs_221226)
        
        # Assigning a type to the variable 'renderer' (line 461)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 461, 8), 'renderer', RendererCairo_call_result_221227)
        
        # Call to set_width_height(...): (line 462)
        # Processing the call arguments (line 462)
        # Getting the type of 'width' (line 462)
        width_221230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 34), 'width', False)
        # Getting the type of 'height' (line 462)
        height_221231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 41), 'height', False)
        # Processing the call keyword arguments (line 462)
        kwargs_221232 = {}
        # Getting the type of 'renderer' (line 462)
        renderer_221228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 8), 'renderer', False)
        # Obtaining the member 'set_width_height' of a type (line 462)
        set_width_height_221229 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 462, 8), renderer_221228, 'set_width_height')
        # Calling set_width_height(args, kwargs) (line 462)
        set_width_height_call_result_221233 = invoke(stypy.reporting.localization.Localization(__file__, 462, 8), set_width_height_221229, *[width_221230, height_221231], **kwargs_221232)
        
        
        # Assigning a Call to a Name (line 463):
        
        # Assigning a Call to a Name (line 463):
        
        # Call to ImageSurface(...): (line 463)
        # Processing the call arguments (line 463)
        # Getting the type of 'cairo' (line 463)
        cairo_221236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 37), 'cairo', False)
        # Obtaining the member 'FORMAT_ARGB32' of a type (line 463)
        FORMAT_ARGB32_221237 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 463, 37), cairo_221236, 'FORMAT_ARGB32')
        # Getting the type of 'width' (line 463)
        width_221238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 58), 'width', False)
        # Getting the type of 'height' (line 463)
        height_221239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 65), 'height', False)
        # Processing the call keyword arguments (line 463)
        kwargs_221240 = {}
        # Getting the type of 'cairo' (line 463)
        cairo_221234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 18), 'cairo', False)
        # Obtaining the member 'ImageSurface' of a type (line 463)
        ImageSurface_221235 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 463, 18), cairo_221234, 'ImageSurface')
        # Calling ImageSurface(args, kwargs) (line 463)
        ImageSurface_call_result_221241 = invoke(stypy.reporting.localization.Localization(__file__, 463, 18), ImageSurface_221235, *[FORMAT_ARGB32_221237, width_221238, height_221239], **kwargs_221240)
        
        # Assigning a type to the variable 'surface' (line 463)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 463, 8), 'surface', ImageSurface_call_result_221241)
        
        # Call to set_ctx_from_surface(...): (line 464)
        # Processing the call arguments (line 464)
        # Getting the type of 'surface' (line 464)
        surface_221244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 38), 'surface', False)
        # Processing the call keyword arguments (line 464)
        kwargs_221245 = {}
        # Getting the type of 'renderer' (line 464)
        renderer_221242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 8), 'renderer', False)
        # Obtaining the member 'set_ctx_from_surface' of a type (line 464)
        set_ctx_from_surface_221243 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 464, 8), renderer_221242, 'set_ctx_from_surface')
        # Calling set_ctx_from_surface(args, kwargs) (line 464)
        set_ctx_from_surface_call_result_221246 = invoke(stypy.reporting.localization.Localization(__file__, 464, 8), set_ctx_from_surface_221243, *[surface_221244], **kwargs_221245)
        
        
        # Call to draw(...): (line 466)
        # Processing the call arguments (line 466)
        # Getting the type of 'renderer' (line 466)
        renderer_221250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 25), 'renderer', False)
        # Processing the call keyword arguments (line 466)
        kwargs_221251 = {}
        # Getting the type of 'self' (line 466)
        self_221247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 8), 'self', False)
        # Obtaining the member 'figure' of a type (line 466)
        figure_221248 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 466, 8), self_221247, 'figure')
        # Obtaining the member 'draw' of a type (line 466)
        draw_221249 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 466, 8), figure_221248, 'draw')
        # Calling draw(args, kwargs) (line 466)
        draw_call_result_221252 = invoke(stypy.reporting.localization.Localization(__file__, 466, 8), draw_221249, *[renderer_221250], **kwargs_221251)
        
        
        # Call to write_to_png(...): (line 467)
        # Processing the call arguments (line 467)
        # Getting the type of 'fobj' (line 467)
        fobj_221255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 29), 'fobj', False)
        # Processing the call keyword arguments (line 467)
        kwargs_221256 = {}
        # Getting the type of 'surface' (line 467)
        surface_221253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 8), 'surface', False)
        # Obtaining the member 'write_to_png' of a type (line 467)
        write_to_png_221254 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 467, 8), surface_221253, 'write_to_png')
        # Calling write_to_png(args, kwargs) (line 467)
        write_to_png_call_result_221257 = invoke(stypy.reporting.localization.Localization(__file__, 467, 8), write_to_png_221254, *[fobj_221255], **kwargs_221256)
        
        
        # ################# End of 'print_png(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'print_png' in the type store
        # Getting the type of 'stypy_return_type' (line 458)
        stypy_return_type_221258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_221258)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'print_png'
        return stypy_return_type_221258


    @norecursion
    def print_pdf(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'print_pdf'
        module_type_store = module_type_store.open_function_context('print_pdf', 469, 4, False)
        # Assigning a type to the variable 'self' (line 470)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 470, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasCairo.print_pdf.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasCairo.print_pdf.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasCairo.print_pdf.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasCairo.print_pdf.__dict__.__setitem__('stypy_function_name', 'FigureCanvasCairo.print_pdf')
        FigureCanvasCairo.print_pdf.__dict__.__setitem__('stypy_param_names_list', ['fobj'])
        FigureCanvasCairo.print_pdf.__dict__.__setitem__('stypy_varargs_param_name', 'args')
        FigureCanvasCairo.print_pdf.__dict__.__setitem__('stypy_kwargs_param_name', 'kwargs')
        FigureCanvasCairo.print_pdf.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasCairo.print_pdf.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasCairo.print_pdf.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasCairo.print_pdf.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasCairo.print_pdf', ['fobj'], 'args', 'kwargs', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'print_pdf', localization, ['fobj'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'print_pdf(...)' code ##################

        
        # Call to _save(...): (line 470)
        # Processing the call arguments (line 470)
        # Getting the type of 'fobj' (line 470)
        fobj_221261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 26), 'fobj', False)
        unicode_221262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 470, 32), 'unicode', u'pdf')
        # Getting the type of 'args' (line 470)
        args_221263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 40), 'args', False)
        # Processing the call keyword arguments (line 470)
        # Getting the type of 'kwargs' (line 470)
        kwargs_221264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 48), 'kwargs', False)
        kwargs_221265 = {'kwargs_221264': kwargs_221264}
        # Getting the type of 'self' (line 470)
        self_221259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 15), 'self', False)
        # Obtaining the member '_save' of a type (line 470)
        _save_221260 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 470, 15), self_221259, '_save')
        # Calling _save(args, kwargs) (line 470)
        _save_call_result_221266 = invoke(stypy.reporting.localization.Localization(__file__, 470, 15), _save_221260, *[fobj_221261, unicode_221262, args_221263], **kwargs_221265)
        
        # Assigning a type to the variable 'stypy_return_type' (line 470)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 470, 8), 'stypy_return_type', _save_call_result_221266)
        
        # ################# End of 'print_pdf(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'print_pdf' in the type store
        # Getting the type of 'stypy_return_type' (line 469)
        stypy_return_type_221267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_221267)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'print_pdf'
        return stypy_return_type_221267


    @norecursion
    def print_ps(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'print_ps'
        module_type_store = module_type_store.open_function_context('print_ps', 472, 4, False)
        # Assigning a type to the variable 'self' (line 473)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 473, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasCairo.print_ps.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasCairo.print_ps.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasCairo.print_ps.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasCairo.print_ps.__dict__.__setitem__('stypy_function_name', 'FigureCanvasCairo.print_ps')
        FigureCanvasCairo.print_ps.__dict__.__setitem__('stypy_param_names_list', ['fobj'])
        FigureCanvasCairo.print_ps.__dict__.__setitem__('stypy_varargs_param_name', 'args')
        FigureCanvasCairo.print_ps.__dict__.__setitem__('stypy_kwargs_param_name', 'kwargs')
        FigureCanvasCairo.print_ps.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasCairo.print_ps.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasCairo.print_ps.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasCairo.print_ps.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasCairo.print_ps', ['fobj'], 'args', 'kwargs', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'print_ps', localization, ['fobj'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'print_ps(...)' code ##################

        
        # Call to _save(...): (line 473)
        # Processing the call arguments (line 473)
        # Getting the type of 'fobj' (line 473)
        fobj_221270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 26), 'fobj', False)
        unicode_221271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 473, 32), 'unicode', u'ps')
        # Getting the type of 'args' (line 473)
        args_221272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 39), 'args', False)
        # Processing the call keyword arguments (line 473)
        # Getting the type of 'kwargs' (line 473)
        kwargs_221273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 47), 'kwargs', False)
        kwargs_221274 = {'kwargs_221273': kwargs_221273}
        # Getting the type of 'self' (line 473)
        self_221268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 15), 'self', False)
        # Obtaining the member '_save' of a type (line 473)
        _save_221269 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 473, 15), self_221268, '_save')
        # Calling _save(args, kwargs) (line 473)
        _save_call_result_221275 = invoke(stypy.reporting.localization.Localization(__file__, 473, 15), _save_221269, *[fobj_221270, unicode_221271, args_221272], **kwargs_221274)
        
        # Assigning a type to the variable 'stypy_return_type' (line 473)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 473, 8), 'stypy_return_type', _save_call_result_221275)
        
        # ################# End of 'print_ps(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'print_ps' in the type store
        # Getting the type of 'stypy_return_type' (line 472)
        stypy_return_type_221276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_221276)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'print_ps'
        return stypy_return_type_221276


    @norecursion
    def print_svg(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'print_svg'
        module_type_store = module_type_store.open_function_context('print_svg', 475, 4, False)
        # Assigning a type to the variable 'self' (line 476)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 476, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasCairo.print_svg.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasCairo.print_svg.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasCairo.print_svg.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasCairo.print_svg.__dict__.__setitem__('stypy_function_name', 'FigureCanvasCairo.print_svg')
        FigureCanvasCairo.print_svg.__dict__.__setitem__('stypy_param_names_list', ['fobj'])
        FigureCanvasCairo.print_svg.__dict__.__setitem__('stypy_varargs_param_name', 'args')
        FigureCanvasCairo.print_svg.__dict__.__setitem__('stypy_kwargs_param_name', 'kwargs')
        FigureCanvasCairo.print_svg.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasCairo.print_svg.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasCairo.print_svg.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasCairo.print_svg.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasCairo.print_svg', ['fobj'], 'args', 'kwargs', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'print_svg', localization, ['fobj'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'print_svg(...)' code ##################

        
        # Call to _save(...): (line 476)
        # Processing the call arguments (line 476)
        # Getting the type of 'fobj' (line 476)
        fobj_221279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 26), 'fobj', False)
        unicode_221280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 476, 32), 'unicode', u'svg')
        # Getting the type of 'args' (line 476)
        args_221281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 40), 'args', False)
        # Processing the call keyword arguments (line 476)
        # Getting the type of 'kwargs' (line 476)
        kwargs_221282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 48), 'kwargs', False)
        kwargs_221283 = {'kwargs_221282': kwargs_221282}
        # Getting the type of 'self' (line 476)
        self_221277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 15), 'self', False)
        # Obtaining the member '_save' of a type (line 476)
        _save_221278 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 476, 15), self_221277, '_save')
        # Calling _save(args, kwargs) (line 476)
        _save_call_result_221284 = invoke(stypy.reporting.localization.Localization(__file__, 476, 15), _save_221278, *[fobj_221279, unicode_221280, args_221281], **kwargs_221283)
        
        # Assigning a type to the variable 'stypy_return_type' (line 476)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 476, 8), 'stypy_return_type', _save_call_result_221284)
        
        # ################# End of 'print_svg(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'print_svg' in the type store
        # Getting the type of 'stypy_return_type' (line 475)
        stypy_return_type_221285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_221285)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'print_svg'
        return stypy_return_type_221285


    @norecursion
    def print_svgz(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'print_svgz'
        module_type_store = module_type_store.open_function_context('print_svgz', 478, 4, False)
        # Assigning a type to the variable 'self' (line 479)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 479, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasCairo.print_svgz.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasCairo.print_svgz.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasCairo.print_svgz.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasCairo.print_svgz.__dict__.__setitem__('stypy_function_name', 'FigureCanvasCairo.print_svgz')
        FigureCanvasCairo.print_svgz.__dict__.__setitem__('stypy_param_names_list', ['fobj'])
        FigureCanvasCairo.print_svgz.__dict__.__setitem__('stypy_varargs_param_name', 'args')
        FigureCanvasCairo.print_svgz.__dict__.__setitem__('stypy_kwargs_param_name', 'kwargs')
        FigureCanvasCairo.print_svgz.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasCairo.print_svgz.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasCairo.print_svgz.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasCairo.print_svgz.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasCairo.print_svgz', ['fobj'], 'args', 'kwargs', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'print_svgz', localization, ['fobj'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'print_svgz(...)' code ##################

        
        # Call to _save(...): (line 479)
        # Processing the call arguments (line 479)
        # Getting the type of 'fobj' (line 479)
        fobj_221288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 26), 'fobj', False)
        unicode_221289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 479, 32), 'unicode', u'svgz')
        # Getting the type of 'args' (line 479)
        args_221290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 41), 'args', False)
        # Processing the call keyword arguments (line 479)
        # Getting the type of 'kwargs' (line 479)
        kwargs_221291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 49), 'kwargs', False)
        kwargs_221292 = {'kwargs_221291': kwargs_221291}
        # Getting the type of 'self' (line 479)
        self_221286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 15), 'self', False)
        # Obtaining the member '_save' of a type (line 479)
        _save_221287 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 479, 15), self_221286, '_save')
        # Calling _save(args, kwargs) (line 479)
        _save_call_result_221293 = invoke(stypy.reporting.localization.Localization(__file__, 479, 15), _save_221287, *[fobj_221288, unicode_221289, args_221290], **kwargs_221292)
        
        # Assigning a type to the variable 'stypy_return_type' (line 479)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 479, 8), 'stypy_return_type', _save_call_result_221293)
        
        # ################# End of 'print_svgz(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'print_svgz' in the type store
        # Getting the type of 'stypy_return_type' (line 478)
        stypy_return_type_221294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_221294)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'print_svgz'
        return stypy_return_type_221294


    @norecursion
    def _save(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_save'
        module_type_store = module_type_store.open_function_context('_save', 481, 4, False)
        # Assigning a type to the variable 'self' (line 482)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 482, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasCairo._save.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasCairo._save.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasCairo._save.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasCairo._save.__dict__.__setitem__('stypy_function_name', 'FigureCanvasCairo._save')
        FigureCanvasCairo._save.__dict__.__setitem__('stypy_param_names_list', ['fo', 'fmt'])
        FigureCanvasCairo._save.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureCanvasCairo._save.__dict__.__setitem__('stypy_kwargs_param_name', 'kwargs')
        FigureCanvasCairo._save.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasCairo._save.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasCairo._save.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasCairo._save.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasCairo._save', ['fo', 'fmt'], None, 'kwargs', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_save', localization, ['fo', 'fmt'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_save(...)' code ##################

        
        # Assigning a Call to a Name (line 483):
        
        # Assigning a Call to a Name (line 483):
        
        # Call to get(...): (line 483)
        # Processing the call arguments (line 483)
        unicode_221297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 483, 33), 'unicode', u'orientation')
        unicode_221298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 483, 48), 'unicode', u'portrait')
        # Processing the call keyword arguments (line 483)
        kwargs_221299 = {}
        # Getting the type of 'kwargs' (line 483)
        kwargs_221295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 22), 'kwargs', False)
        # Obtaining the member 'get' of a type (line 483)
        get_221296 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 483, 22), kwargs_221295, 'get')
        # Calling get(args, kwargs) (line 483)
        get_call_result_221300 = invoke(stypy.reporting.localization.Localization(__file__, 483, 22), get_221296, *[unicode_221297, unicode_221298], **kwargs_221299)
        
        # Assigning a type to the variable 'orientation' (line 483)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 483, 8), 'orientation', get_call_result_221300)
        
        # Assigning a Num to a Name (line 485):
        
        # Assigning a Num to a Name (line 485):
        int_221301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 485, 14), 'int')
        # Assigning a type to the variable 'dpi' (line 485)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 485, 8), 'dpi', int_221301)
        
        # Assigning a Name to a Attribute (line 486):
        
        # Assigning a Name to a Attribute (line 486):
        # Getting the type of 'dpi' (line 486)
        dpi_221302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 26), 'dpi')
        # Getting the type of 'self' (line 486)
        self_221303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 8), 'self')
        # Obtaining the member 'figure' of a type (line 486)
        figure_221304 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 486, 8), self_221303, 'figure')
        # Setting the type of the member 'dpi' of a type (line 486)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 486, 8), figure_221304, 'dpi', dpi_221302)
        
        # Assigning a Call to a Tuple (line 487):
        
        # Assigning a Call to a Name:
        
        # Call to get_size_inches(...): (line 487)
        # Processing the call keyword arguments (line 487)
        kwargs_221308 = {}
        # Getting the type of 'self' (line 487)
        self_221305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 21), 'self', False)
        # Obtaining the member 'figure' of a type (line 487)
        figure_221306 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 487, 21), self_221305, 'figure')
        # Obtaining the member 'get_size_inches' of a type (line 487)
        get_size_inches_221307 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 487, 21), figure_221306, 'get_size_inches')
        # Calling get_size_inches(args, kwargs) (line 487)
        get_size_inches_call_result_221309 = invoke(stypy.reporting.localization.Localization(__file__, 487, 21), get_size_inches_221307, *[], **kwargs_221308)
        
        # Assigning a type to the variable 'call_assignment_219657' (line 487)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 487, 8), 'call_assignment_219657', get_size_inches_call_result_221309)
        
        # Assigning a Call to a Name (line 487):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_221312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 487, 8), 'int')
        # Processing the call keyword arguments
        kwargs_221313 = {}
        # Getting the type of 'call_assignment_219657' (line 487)
        call_assignment_219657_221310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 8), 'call_assignment_219657', False)
        # Obtaining the member '__getitem__' of a type (line 487)
        getitem___221311 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 487, 8), call_assignment_219657_221310, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_221314 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___221311, *[int_221312], **kwargs_221313)
        
        # Assigning a type to the variable 'call_assignment_219658' (line 487)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 487, 8), 'call_assignment_219658', getitem___call_result_221314)
        
        # Assigning a Name to a Name (line 487):
        # Getting the type of 'call_assignment_219658' (line 487)
        call_assignment_219658_221315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 8), 'call_assignment_219658')
        # Assigning a type to the variable 'w_in' (line 487)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 487, 8), 'w_in', call_assignment_219658_221315)
        
        # Assigning a Call to a Name (line 487):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_221318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 487, 8), 'int')
        # Processing the call keyword arguments
        kwargs_221319 = {}
        # Getting the type of 'call_assignment_219657' (line 487)
        call_assignment_219657_221316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 8), 'call_assignment_219657', False)
        # Obtaining the member '__getitem__' of a type (line 487)
        getitem___221317 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 487, 8), call_assignment_219657_221316, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_221320 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___221317, *[int_221318], **kwargs_221319)
        
        # Assigning a type to the variable 'call_assignment_219659' (line 487)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 487, 8), 'call_assignment_219659', getitem___call_result_221320)
        
        # Assigning a Name to a Name (line 487):
        # Getting the type of 'call_assignment_219659' (line 487)
        call_assignment_219659_221321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 8), 'call_assignment_219659')
        # Assigning a type to the variable 'h_in' (line 487)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 487, 14), 'h_in', call_assignment_219659_221321)
        
        # Assigning a Tuple to a Tuple (line 488):
        
        # Assigning a BinOp to a Name (line 488):
        # Getting the type of 'w_in' (line 488)
        w_in_221322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 44), 'w_in')
        # Getting the type of 'dpi' (line 488)
        dpi_221323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 51), 'dpi')
        # Applying the binary operator '*' (line 488)
        result_mul_221324 = python_operator(stypy.reporting.localization.Localization(__file__, 488, 44), '*', w_in_221322, dpi_221323)
        
        # Assigning a type to the variable 'tuple_assignment_219660' (line 488)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 488, 8), 'tuple_assignment_219660', result_mul_221324)
        
        # Assigning a BinOp to a Name (line 488):
        # Getting the type of 'h_in' (line 488)
        h_in_221325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 56), 'h_in')
        # Getting the type of 'dpi' (line 488)
        dpi_221326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 63), 'dpi')
        # Applying the binary operator '*' (line 488)
        result_mul_221327 = python_operator(stypy.reporting.localization.Localization(__file__, 488, 56), '*', h_in_221325, dpi_221326)
        
        # Assigning a type to the variable 'tuple_assignment_219661' (line 488)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 488, 8), 'tuple_assignment_219661', result_mul_221327)
        
        # Assigning a Name to a Name (line 488):
        # Getting the type of 'tuple_assignment_219660' (line 488)
        tuple_assignment_219660_221328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 8), 'tuple_assignment_219660')
        # Assigning a type to the variable 'width_in_points' (line 488)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 488, 8), 'width_in_points', tuple_assignment_219660_221328)
        
        # Assigning a Name to a Name (line 488):
        # Getting the type of 'tuple_assignment_219661' (line 488)
        tuple_assignment_219661_221329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 8), 'tuple_assignment_219661')
        # Assigning a type to the variable 'height_in_points' (line 488)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 488, 25), 'height_in_points', tuple_assignment_219661_221329)
        
        
        # Getting the type of 'orientation' (line 490)
        orientation_221330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 11), 'orientation')
        unicode_221331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 490, 26), 'unicode', u'landscape')
        # Applying the binary operator '==' (line 490)
        result_eq_221332 = python_operator(stypy.reporting.localization.Localization(__file__, 490, 11), '==', orientation_221330, unicode_221331)
        
        # Testing the type of an if condition (line 490)
        if_condition_221333 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 490, 8), result_eq_221332)
        # Assigning a type to the variable 'if_condition_221333' (line 490)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 490, 8), 'if_condition_221333', if_condition_221333)
        # SSA begins for if statement (line 490)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Tuple to a Tuple (line 491):
        
        # Assigning a Name to a Name (line 491):
        # Getting the type of 'height_in_points' (line 492)
        height_in_points_221334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 16), 'height_in_points')
        # Assigning a type to the variable 'tuple_assignment_219662' (line 491)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 491, 12), 'tuple_assignment_219662', height_in_points_221334)
        
        # Assigning a Name to a Name (line 491):
        # Getting the type of 'width_in_points' (line 492)
        width_in_points_221335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 34), 'width_in_points')
        # Assigning a type to the variable 'tuple_assignment_219663' (line 491)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 491, 12), 'tuple_assignment_219663', width_in_points_221335)
        
        # Assigning a Name to a Name (line 491):
        # Getting the type of 'tuple_assignment_219662' (line 491)
        tuple_assignment_219662_221336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 12), 'tuple_assignment_219662')
        # Assigning a type to the variable 'width_in_points' (line 491)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 491, 12), 'width_in_points', tuple_assignment_219662_221336)
        
        # Assigning a Name to a Name (line 491):
        # Getting the type of 'tuple_assignment_219663' (line 491)
        tuple_assignment_219663_221337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 12), 'tuple_assignment_219663')
        # Assigning a type to the variable 'height_in_points' (line 491)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 491, 29), 'height_in_points', tuple_assignment_219663_221337)
        # SSA join for if statement (line 490)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'fmt' (line 494)
        fmt_221338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 11), 'fmt')
        unicode_221339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 494, 18), 'unicode', u'ps')
        # Applying the binary operator '==' (line 494)
        result_eq_221340 = python_operator(stypy.reporting.localization.Localization(__file__, 494, 11), '==', fmt_221338, unicode_221339)
        
        # Testing the type of an if condition (line 494)
        if_condition_221341 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 494, 8), result_eq_221340)
        # Assigning a type to the variable 'if_condition_221341' (line 494)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 494, 8), 'if_condition_221341', if_condition_221341)
        # SSA begins for if statement (line 494)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Type idiom detected: calculating its left and rigth part (line 495)
        unicode_221342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 495, 34), 'unicode', u'PSSurface')
        # Getting the type of 'cairo' (line 495)
        cairo_221343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 495, 27), 'cairo')
        
        (may_be_221344, more_types_in_union_221345) = may_not_provide_member(unicode_221342, cairo_221343)

        if may_be_221344:

            if more_types_in_union_221345:
                # Runtime conditional SSA (line 495)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'cairo' (line 495)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 495, 12), 'cairo', remove_member_provider_from_union(cairo_221343, u'PSSurface'))
            
            # Call to RuntimeError(...): (line 496)
            # Processing the call arguments (line 496)
            unicode_221347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 496, 35), 'unicode', u'cairo has not been compiled with PS support enabled')
            # Processing the call keyword arguments (line 496)
            kwargs_221348 = {}
            # Getting the type of 'RuntimeError' (line 496)
            RuntimeError_221346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 22), 'RuntimeError', False)
            # Calling RuntimeError(args, kwargs) (line 496)
            RuntimeError_call_result_221349 = invoke(stypy.reporting.localization.Localization(__file__, 496, 22), RuntimeError_221346, *[unicode_221347], **kwargs_221348)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 496, 16), RuntimeError_call_result_221349, 'raise parameter', BaseException)

            if more_types_in_union_221345:
                # SSA join for if statement (line 495)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Name (line 498):
        
        # Assigning a Call to a Name (line 498):
        
        # Call to PSSurface(...): (line 498)
        # Processing the call arguments (line 498)
        # Getting the type of 'fo' (line 498)
        fo_221352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 38), 'fo', False)
        # Getting the type of 'width_in_points' (line 498)
        width_in_points_221353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 42), 'width_in_points', False)
        # Getting the type of 'height_in_points' (line 498)
        height_in_points_221354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 59), 'height_in_points', False)
        # Processing the call keyword arguments (line 498)
        kwargs_221355 = {}
        # Getting the type of 'cairo' (line 498)
        cairo_221350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 22), 'cairo', False)
        # Obtaining the member 'PSSurface' of a type (line 498)
        PSSurface_221351 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 498, 22), cairo_221350, 'PSSurface')
        # Calling PSSurface(args, kwargs) (line 498)
        PSSurface_call_result_221356 = invoke(stypy.reporting.localization.Localization(__file__, 498, 22), PSSurface_221351, *[fo_221352, width_in_points_221353, height_in_points_221354], **kwargs_221355)
        
        # Assigning a type to the variable 'surface' (line 498)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 498, 12), 'surface', PSSurface_call_result_221356)
        # SSA branch for the else part of an if statement (line 494)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'fmt' (line 499)
        fmt_221357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 13), 'fmt')
        unicode_221358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 499, 20), 'unicode', u'pdf')
        # Applying the binary operator '==' (line 499)
        result_eq_221359 = python_operator(stypy.reporting.localization.Localization(__file__, 499, 13), '==', fmt_221357, unicode_221358)
        
        # Testing the type of an if condition (line 499)
        if_condition_221360 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 499, 13), result_eq_221359)
        # Assigning a type to the variable 'if_condition_221360' (line 499)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 499, 13), 'if_condition_221360', if_condition_221360)
        # SSA begins for if statement (line 499)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Type idiom detected: calculating its left and rigth part (line 500)
        unicode_221361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 500, 34), 'unicode', u'PDFSurface')
        # Getting the type of 'cairo' (line 500)
        cairo_221362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 27), 'cairo')
        
        (may_be_221363, more_types_in_union_221364) = may_not_provide_member(unicode_221361, cairo_221362)

        if may_be_221363:

            if more_types_in_union_221364:
                # Runtime conditional SSA (line 500)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'cairo' (line 500)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 500, 12), 'cairo', remove_member_provider_from_union(cairo_221362, u'PDFSurface'))
            
            # Call to RuntimeError(...): (line 501)
            # Processing the call arguments (line 501)
            unicode_221366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 501, 35), 'unicode', u'cairo has not been compiled with PDF support enabled')
            # Processing the call keyword arguments (line 501)
            kwargs_221367 = {}
            # Getting the type of 'RuntimeError' (line 501)
            RuntimeError_221365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 22), 'RuntimeError', False)
            # Calling RuntimeError(args, kwargs) (line 501)
            RuntimeError_call_result_221368 = invoke(stypy.reporting.localization.Localization(__file__, 501, 22), RuntimeError_221365, *[unicode_221366], **kwargs_221367)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 501, 16), RuntimeError_call_result_221368, 'raise parameter', BaseException)

            if more_types_in_union_221364:
                # SSA join for if statement (line 500)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Name (line 503):
        
        # Assigning a Call to a Name (line 503):
        
        # Call to PDFSurface(...): (line 503)
        # Processing the call arguments (line 503)
        # Getting the type of 'fo' (line 503)
        fo_221371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 39), 'fo', False)
        # Getting the type of 'width_in_points' (line 503)
        width_in_points_221372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 43), 'width_in_points', False)
        # Getting the type of 'height_in_points' (line 503)
        height_in_points_221373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 60), 'height_in_points', False)
        # Processing the call keyword arguments (line 503)
        kwargs_221374 = {}
        # Getting the type of 'cairo' (line 503)
        cairo_221369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 22), 'cairo', False)
        # Obtaining the member 'PDFSurface' of a type (line 503)
        PDFSurface_221370 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 503, 22), cairo_221369, 'PDFSurface')
        # Calling PDFSurface(args, kwargs) (line 503)
        PDFSurface_call_result_221375 = invoke(stypy.reporting.localization.Localization(__file__, 503, 22), PDFSurface_221370, *[fo_221371, width_in_points_221372, height_in_points_221373], **kwargs_221374)
        
        # Assigning a type to the variable 'surface' (line 503)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 503, 12), 'surface', PDFSurface_call_result_221375)
        # SSA branch for the else part of an if statement (line 499)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'fmt' (line 504)
        fmt_221376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 13), 'fmt')
        
        # Obtaining an instance of the builtin type 'tuple' (line 504)
        tuple_221377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 504, 21), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 504)
        # Adding element type (line 504)
        unicode_221378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 504, 21), 'unicode', u'svg')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 504, 21), tuple_221377, unicode_221378)
        # Adding element type (line 504)
        unicode_221379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 504, 28), 'unicode', u'svgz')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 504, 21), tuple_221377, unicode_221379)
        
        # Applying the binary operator 'in' (line 504)
        result_contains_221380 = python_operator(stypy.reporting.localization.Localization(__file__, 504, 13), 'in', fmt_221376, tuple_221377)
        
        # Testing the type of an if condition (line 504)
        if_condition_221381 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 504, 13), result_contains_221380)
        # Assigning a type to the variable 'if_condition_221381' (line 504)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 504, 13), 'if_condition_221381', if_condition_221381)
        # SSA begins for if statement (line 504)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Type idiom detected: calculating its left and rigth part (line 505)
        unicode_221382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 505, 34), 'unicode', u'SVGSurface')
        # Getting the type of 'cairo' (line 505)
        cairo_221383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 27), 'cairo')
        
        (may_be_221384, more_types_in_union_221385) = may_not_provide_member(unicode_221382, cairo_221383)

        if may_be_221384:

            if more_types_in_union_221385:
                # Runtime conditional SSA (line 505)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'cairo' (line 505)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 505, 12), 'cairo', remove_member_provider_from_union(cairo_221383, u'SVGSurface'))
            
            # Call to RuntimeError(...): (line 506)
            # Processing the call arguments (line 506)
            unicode_221387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 506, 35), 'unicode', u'cairo has not been compiled with SVG support enabled')
            # Processing the call keyword arguments (line 506)
            kwargs_221388 = {}
            # Getting the type of 'RuntimeError' (line 506)
            RuntimeError_221386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 22), 'RuntimeError', False)
            # Calling RuntimeError(args, kwargs) (line 506)
            RuntimeError_call_result_221389 = invoke(stypy.reporting.localization.Localization(__file__, 506, 22), RuntimeError_221386, *[unicode_221387], **kwargs_221388)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 506, 16), RuntimeError_call_result_221389, 'raise parameter', BaseException)

            if more_types_in_union_221385:
                # SSA join for if statement (line 505)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        # Getting the type of 'fmt' (line 508)
        fmt_221390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 15), 'fmt')
        unicode_221391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 508, 22), 'unicode', u'svgz')
        # Applying the binary operator '==' (line 508)
        result_eq_221392 = python_operator(stypy.reporting.localization.Localization(__file__, 508, 15), '==', fmt_221390, unicode_221391)
        
        # Testing the type of an if condition (line 508)
        if_condition_221393 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 508, 12), result_eq_221392)
        # Assigning a type to the variable 'if_condition_221393' (line 508)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 508, 12), 'if_condition_221393', if_condition_221393)
        # SSA begins for if statement (line 508)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Call to isinstance(...): (line 509)
        # Processing the call arguments (line 509)
        # Getting the type of 'fo' (line 509)
        fo_221395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 30), 'fo', False)
        # Getting the type of 'six' (line 509)
        six_221396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 34), 'six', False)
        # Obtaining the member 'string_types' of a type (line 509)
        string_types_221397 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 509, 34), six_221396, 'string_types')
        # Processing the call keyword arguments (line 509)
        kwargs_221398 = {}
        # Getting the type of 'isinstance' (line 509)
        isinstance_221394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 19), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 509)
        isinstance_call_result_221399 = invoke(stypy.reporting.localization.Localization(__file__, 509, 19), isinstance_221394, *[fo_221395, string_types_221397], **kwargs_221398)
        
        # Testing the type of an if condition (line 509)
        if_condition_221400 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 509, 16), isinstance_call_result_221399)
        # Assigning a type to the variable 'if_condition_221400' (line 509)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 509, 16), 'if_condition_221400', if_condition_221400)
        # SSA begins for if statement (line 509)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 510):
        
        # Assigning a Call to a Name (line 510):
        
        # Call to GzipFile(...): (line 510)
        # Processing the call arguments (line 510)
        # Getting the type of 'fo' (line 510)
        fo_221403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 39), 'fo', False)
        unicode_221404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 510, 43), 'unicode', u'wb')
        # Processing the call keyword arguments (line 510)
        kwargs_221405 = {}
        # Getting the type of 'gzip' (line 510)
        gzip_221401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 25), 'gzip', False)
        # Obtaining the member 'GzipFile' of a type (line 510)
        GzipFile_221402 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 510, 25), gzip_221401, 'GzipFile')
        # Calling GzipFile(args, kwargs) (line 510)
        GzipFile_call_result_221406 = invoke(stypy.reporting.localization.Localization(__file__, 510, 25), GzipFile_221402, *[fo_221403, unicode_221404], **kwargs_221405)
        
        # Assigning a type to the variable 'fo' (line 510)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 510, 20), 'fo', GzipFile_call_result_221406)
        # SSA branch for the else part of an if statement (line 509)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 512):
        
        # Assigning a Call to a Name (line 512):
        
        # Call to GzipFile(...): (line 512)
        # Processing the call arguments (line 512)
        # Getting the type of 'None' (line 512)
        None_221409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 39), 'None', False)
        unicode_221410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 512, 45), 'unicode', u'wb')
        # Processing the call keyword arguments (line 512)
        # Getting the type of 'fo' (line 512)
        fo_221411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 59), 'fo', False)
        keyword_221412 = fo_221411
        kwargs_221413 = {'fileobj': keyword_221412}
        # Getting the type of 'gzip' (line 512)
        gzip_221407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 25), 'gzip', False)
        # Obtaining the member 'GzipFile' of a type (line 512)
        GzipFile_221408 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 512, 25), gzip_221407, 'GzipFile')
        # Calling GzipFile(args, kwargs) (line 512)
        GzipFile_call_result_221414 = invoke(stypy.reporting.localization.Localization(__file__, 512, 25), GzipFile_221408, *[None_221409, unicode_221410], **kwargs_221413)
        
        # Assigning a type to the variable 'fo' (line 512)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 512, 20), 'fo', GzipFile_call_result_221414)
        # SSA join for if statement (line 509)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 508)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 513):
        
        # Assigning a Call to a Name (line 513):
        
        # Call to SVGSurface(...): (line 513)
        # Processing the call arguments (line 513)
        # Getting the type of 'fo' (line 513)
        fo_221417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 39), 'fo', False)
        # Getting the type of 'width_in_points' (line 513)
        width_in_points_221418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 43), 'width_in_points', False)
        # Getting the type of 'height_in_points' (line 513)
        height_in_points_221419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 60), 'height_in_points', False)
        # Processing the call keyword arguments (line 513)
        kwargs_221420 = {}
        # Getting the type of 'cairo' (line 513)
        cairo_221415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 22), 'cairo', False)
        # Obtaining the member 'SVGSurface' of a type (line 513)
        SVGSurface_221416 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 513, 22), cairo_221415, 'SVGSurface')
        # Calling SVGSurface(args, kwargs) (line 513)
        SVGSurface_call_result_221421 = invoke(stypy.reporting.localization.Localization(__file__, 513, 22), SVGSurface_221416, *[fo_221417, width_in_points_221418, height_in_points_221419], **kwargs_221420)
        
        # Assigning a type to the variable 'surface' (line 513)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 513, 12), 'surface', SVGSurface_call_result_221421)
        # SSA branch for the else part of an if statement (line 504)
        module_type_store.open_ssa_branch('else')
        
        # Call to warn(...): (line 515)
        # Processing the call arguments (line 515)
        unicode_221424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 515, 26), 'unicode', u'unknown format: %s')
        # Getting the type of 'fmt' (line 515)
        fmt_221425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 49), 'fmt', False)
        # Applying the binary operator '%' (line 515)
        result_mod_221426 = python_operator(stypy.reporting.localization.Localization(__file__, 515, 26), '%', unicode_221424, fmt_221425)
        
        # Processing the call keyword arguments (line 515)
        kwargs_221427 = {}
        # Getting the type of 'warnings' (line 515)
        warnings_221422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 12), 'warnings', False)
        # Obtaining the member 'warn' of a type (line 515)
        warn_221423 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 515, 12), warnings_221422, 'warn')
        # Calling warn(args, kwargs) (line 515)
        warn_call_result_221428 = invoke(stypy.reporting.localization.Localization(__file__, 515, 12), warn_221423, *[result_mod_221426], **kwargs_221427)
        
        # Assigning a type to the variable 'stypy_return_type' (line 516)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 516, 12), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 504)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 499)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 494)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 519):
        
        # Assigning a Call to a Name (line 519):
        
        # Call to RendererCairo(...): (line 519)
        # Processing the call arguments (line 519)
        # Getting the type of 'self' (line 519)
        self_221430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 33), 'self', False)
        # Obtaining the member 'figure' of a type (line 519)
        figure_221431 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 519, 33), self_221430, 'figure')
        # Obtaining the member 'dpi' of a type (line 519)
        dpi_221432 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 519, 33), figure_221431, 'dpi')
        # Processing the call keyword arguments (line 519)
        kwargs_221433 = {}
        # Getting the type of 'RendererCairo' (line 519)
        RendererCairo_221429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 19), 'RendererCairo', False)
        # Calling RendererCairo(args, kwargs) (line 519)
        RendererCairo_call_result_221434 = invoke(stypy.reporting.localization.Localization(__file__, 519, 19), RendererCairo_221429, *[dpi_221432], **kwargs_221433)
        
        # Assigning a type to the variable 'renderer' (line 519)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 519, 8), 'renderer', RendererCairo_call_result_221434)
        
        # Call to set_width_height(...): (line 520)
        # Processing the call arguments (line 520)
        # Getting the type of 'width_in_points' (line 520)
        width_in_points_221437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 34), 'width_in_points', False)
        # Getting the type of 'height_in_points' (line 520)
        height_in_points_221438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 51), 'height_in_points', False)
        # Processing the call keyword arguments (line 520)
        kwargs_221439 = {}
        # Getting the type of 'renderer' (line 520)
        renderer_221435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 8), 'renderer', False)
        # Obtaining the member 'set_width_height' of a type (line 520)
        set_width_height_221436 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 520, 8), renderer_221435, 'set_width_height')
        # Calling set_width_height(args, kwargs) (line 520)
        set_width_height_call_result_221440 = invoke(stypy.reporting.localization.Localization(__file__, 520, 8), set_width_height_221436, *[width_in_points_221437, height_in_points_221438], **kwargs_221439)
        
        
        # Call to set_ctx_from_surface(...): (line 521)
        # Processing the call arguments (line 521)
        # Getting the type of 'surface' (line 521)
        surface_221443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 38), 'surface', False)
        # Processing the call keyword arguments (line 521)
        kwargs_221444 = {}
        # Getting the type of 'renderer' (line 521)
        renderer_221441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 8), 'renderer', False)
        # Obtaining the member 'set_ctx_from_surface' of a type (line 521)
        set_ctx_from_surface_221442 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 521, 8), renderer_221441, 'set_ctx_from_surface')
        # Calling set_ctx_from_surface(args, kwargs) (line 521)
        set_ctx_from_surface_call_result_221445 = invoke(stypy.reporting.localization.Localization(__file__, 521, 8), set_ctx_from_surface_221442, *[surface_221443], **kwargs_221444)
        
        
        # Assigning a Attribute to a Name (line 522):
        
        # Assigning a Attribute to a Name (line 522):
        # Getting the type of 'renderer' (line 522)
        renderer_221446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 14), 'renderer')
        # Obtaining the member 'gc' of a type (line 522)
        gc_221447 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 522, 14), renderer_221446, 'gc')
        # Obtaining the member 'ctx' of a type (line 522)
        ctx_221448 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 522, 14), gc_221447, 'ctx')
        # Assigning a type to the variable 'ctx' (line 522)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 522, 8), 'ctx', ctx_221448)
        
        
        # Getting the type of 'orientation' (line 524)
        orientation_221449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 11), 'orientation')
        unicode_221450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 524, 26), 'unicode', u'landscape')
        # Applying the binary operator '==' (line 524)
        result_eq_221451 = python_operator(stypy.reporting.localization.Localization(__file__, 524, 11), '==', orientation_221449, unicode_221450)
        
        # Testing the type of an if condition (line 524)
        if_condition_221452 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 524, 8), result_eq_221451)
        # Assigning a type to the variable 'if_condition_221452' (line 524)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 524, 8), 'if_condition_221452', if_condition_221452)
        # SSA begins for if statement (line 524)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to rotate(...): (line 525)
        # Processing the call arguments (line 525)
        # Getting the type of 'np' (line 525)
        np_221455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 23), 'np', False)
        # Obtaining the member 'pi' of a type (line 525)
        pi_221456 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 525, 23), np_221455, 'pi')
        int_221457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 525, 29), 'int')
        # Applying the binary operator 'div' (line 525)
        result_div_221458 = python_operator(stypy.reporting.localization.Localization(__file__, 525, 23), 'div', pi_221456, int_221457)
        
        # Processing the call keyword arguments (line 525)
        kwargs_221459 = {}
        # Getting the type of 'ctx' (line 525)
        ctx_221453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 12), 'ctx', False)
        # Obtaining the member 'rotate' of a type (line 525)
        rotate_221454 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 525, 12), ctx_221453, 'rotate')
        # Calling rotate(args, kwargs) (line 525)
        rotate_call_result_221460 = invoke(stypy.reporting.localization.Localization(__file__, 525, 12), rotate_221454, *[result_div_221458], **kwargs_221459)
        
        
        # Call to translate(...): (line 526)
        # Processing the call arguments (line 526)
        int_221463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 526, 26), 'int')
        
        # Getting the type of 'height_in_points' (line 526)
        height_in_points_221464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 30), 'height_in_points', False)
        # Applying the 'usub' unary operator (line 526)
        result___neg___221465 = python_operator(stypy.reporting.localization.Localization(__file__, 526, 29), 'usub', height_in_points_221464)
        
        # Processing the call keyword arguments (line 526)
        kwargs_221466 = {}
        # Getting the type of 'ctx' (line 526)
        ctx_221461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 12), 'ctx', False)
        # Obtaining the member 'translate' of a type (line 526)
        translate_221462 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 526, 12), ctx_221461, 'translate')
        # Calling translate(args, kwargs) (line 526)
        translate_call_result_221467 = invoke(stypy.reporting.localization.Localization(__file__, 526, 12), translate_221462, *[int_221463, result___neg___221465], **kwargs_221466)
        
        # SSA join for if statement (line 524)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to draw(...): (line 534)
        # Processing the call arguments (line 534)
        # Getting the type of 'renderer' (line 534)
        renderer_221471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 25), 'renderer', False)
        # Processing the call keyword arguments (line 534)
        kwargs_221472 = {}
        # Getting the type of 'self' (line 534)
        self_221468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 8), 'self', False)
        # Obtaining the member 'figure' of a type (line 534)
        figure_221469 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 534, 8), self_221468, 'figure')
        # Obtaining the member 'draw' of a type (line 534)
        draw_221470 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 534, 8), figure_221469, 'draw')
        # Calling draw(args, kwargs) (line 534)
        draw_call_result_221473 = invoke(stypy.reporting.localization.Localization(__file__, 534, 8), draw_221470, *[renderer_221471], **kwargs_221472)
        
        
        # Call to show_page(...): (line 536)
        # Processing the call keyword arguments (line 536)
        kwargs_221476 = {}
        # Getting the type of 'ctx' (line 536)
        ctx_221474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 8), 'ctx', False)
        # Obtaining the member 'show_page' of a type (line 536)
        show_page_221475 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 536, 8), ctx_221474, 'show_page')
        # Calling show_page(args, kwargs) (line 536)
        show_page_call_result_221477 = invoke(stypy.reporting.localization.Localization(__file__, 536, 8), show_page_221475, *[], **kwargs_221476)
        
        
        # Call to finish(...): (line 537)
        # Processing the call keyword arguments (line 537)
        kwargs_221480 = {}
        # Getting the type of 'surface' (line 537)
        surface_221478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 8), 'surface', False)
        # Obtaining the member 'finish' of a type (line 537)
        finish_221479 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 537, 8), surface_221478, 'finish')
        # Calling finish(args, kwargs) (line 537)
        finish_call_result_221481 = invoke(stypy.reporting.localization.Localization(__file__, 537, 8), finish_221479, *[], **kwargs_221480)
        
        
        
        # Getting the type of 'fmt' (line 538)
        fmt_221482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 11), 'fmt')
        unicode_221483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 538, 18), 'unicode', u'svgz')
        # Applying the binary operator '==' (line 538)
        result_eq_221484 = python_operator(stypy.reporting.localization.Localization(__file__, 538, 11), '==', fmt_221482, unicode_221483)
        
        # Testing the type of an if condition (line 538)
        if_condition_221485 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 538, 8), result_eq_221484)
        # Assigning a type to the variable 'if_condition_221485' (line 538)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 538, 8), 'if_condition_221485', if_condition_221485)
        # SSA begins for if statement (line 538)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to close(...): (line 539)
        # Processing the call keyword arguments (line 539)
        kwargs_221488 = {}
        # Getting the type of 'fo' (line 539)
        fo_221486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 12), 'fo', False)
        # Obtaining the member 'close' of a type (line 539)
        close_221487 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 539, 12), fo_221486, 'close')
        # Calling close(args, kwargs) (line 539)
        close_call_result_221489 = invoke(stypy.reporting.localization.Localization(__file__, 539, 12), close_221487, *[], **kwargs_221488)
        
        # SSA join for if statement (line 538)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_save(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_save' in the type store
        # Getting the type of 'stypy_return_type' (line 481)
        stypy_return_type_221490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_221490)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_save'
        return stypy_return_type_221490


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 457, 0, False)
        # Assigning a type to the variable 'self' (line 458)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 458, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasCairo.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'FigureCanvasCairo' (line 457)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 457, 0), 'FigureCanvasCairo', FigureCanvasCairo)
# Declaration of the '_BackendCairo' class
# Getting the type of '_Backend' (line 543)
_Backend_221491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 20), '_Backend')

class _BackendCairo(_Backend_221491, ):
    
    # Assigning a Name to a Name (line 544):
    
    # Assigning a Name to a Name (line 545):
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 542, 0, False)
        # Assigning a type to the variable 'self' (line 543)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 543, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_BackendCairo.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable '_BackendCairo' (line 542)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 542, 0), '_BackendCairo', _BackendCairo)

# Assigning a Name to a Name (line 544):
# Getting the type of 'FigureCanvasCairo' (line 544)
FigureCanvasCairo_221492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 19), 'FigureCanvasCairo')
# Getting the type of '_BackendCairo'
_BackendCairo_221493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), '_BackendCairo')
# Setting the type of the member 'FigureCanvas' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), _BackendCairo_221493, 'FigureCanvas', FigureCanvasCairo_221492)

# Assigning a Name to a Name (line 545):
# Getting the type of 'FigureManagerBase' (line 545)
FigureManagerBase_221494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 20), 'FigureManagerBase')
# Getting the type of '_BackendCairo'
_BackendCairo_221495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), '_BackendCairo')
# Setting the type of the member 'FigureManager' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), _BackendCairo_221495, 'FigureManager', FigureManagerBase_221494)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
